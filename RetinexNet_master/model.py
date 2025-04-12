from __future__ import print_function
import os
import time
import random
import tensorflow as tf
import numpy as np
#from utils import save_images, data_augmentation
from .utils import save_images, data_augmentation
class DecomNet(tf.keras.Model):
    def __init__(self, layer_num=5, channel=64, kernel_size=3):
        super(DecomNet, self).__init__()
        self.layer_num = layer_num
        self.channel = channel
        self.kernel_size = kernel_size
        self.shallow_feature_extraction = tf.keras.layers.Conv2D(channel, kernel_size * 3, padding='same', activation=None)
        self.activated_layers = [tf.keras.layers.Conv2D(channel, kernel_size, padding='same', activation=tf.nn.relu) for _ in range(layer_num)]
        self.recon_layer = tf.keras.layers.Conv2D(4, kernel_size, padding='same', activation=None)

    def call(self, input_im):
        input_max = tf.reduce_max(input_im, axis=3, keepdims=True)
        input_im = tf.concat([input_max, input_im], axis=3)
        conv = self.shallow_feature_extraction(input_im)
        for layer in self.activated_layers:
            conv = layer(conv)
        conv = self.recon_layer(conv)
        R = tf.sigmoid(conv[:, :, :, 0:3])
        L = tf.sigmoid(conv[:, :, :, 3:4])
        return R, L

class RelightNet(tf.keras.Model):
    def __init__(self, channel=64, kernel_size=3):
        super(RelightNet, self).__init__()
        self.channel = channel
        self.kernel_size = kernel_size
        self.conv0 = tf.keras.layers.Conv2D(channel, kernel_size, padding='same', activation=None)
        self.conv1 = tf.keras.layers.Conv2D(channel, kernel_size, strides=2, padding='same', activation=tf.nn.relu)
        self.conv2 = tf.keras.layers.Conv2D(channel, kernel_size, strides=2, padding='same', activation=tf.nn.relu)
        self.conv3 = tf.keras.layers.Conv2D(channel, kernel_size, strides=2, padding='same', activation=tf.nn.relu)
        self.deconv1 = tf.keras.layers.Conv2D(channel, kernel_size, padding='same', activation=tf.nn.relu)
        self.deconv2 = tf.keras.layers.Conv2D(channel, kernel_size, padding='same', activation=tf.nn.relu)
        self.deconv3 = tf.keras.layers.Conv2D(channel, kernel_size, padding='same', activation=tf.nn.relu)
        self.feature_fusion = tf.keras.layers.Conv2D(channel, 1, padding='same', activation=None)
        self.output_layer = tf.keras.layers.Conv2D(1, 3, padding='same', activation=None)

    def call(self, input_L, input_R):
        input_im = tf.concat([input_R, input_L], axis=3)
        conv0 = self.conv0(input_im)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        up1 = tf.image.resize(conv3, [tf.shape(conv2)[1], tf.shape(conv2)[2]], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        deconv1 = self.deconv1(up1) + conv2
        up2 = tf.image.resize(deconv1, [tf.shape(conv1)[1], tf.shape(conv1)[2]], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        deconv2 = self.deconv2(up2) + conv1
        up3 = tf.image.resize(deconv2, [tf.shape(conv0)[1], tf.shape(conv0)[2]], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        deconv3 = self.deconv3(up3) + conv0

        deconv1_resize = tf.image.resize(deconv1, [tf.shape(deconv3)[1], tf.shape(deconv3)[2]], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        deconv2_resize = tf.image.resize(deconv2, [tf.shape(deconv3)[1], tf.shape(deconv3)[2]], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        feature_gather = tf.concat([deconv1_resize, deconv2_resize, deconv3], axis=3)
        feature_fusion = self.feature_fusion(feature_gather)
        output = self.output_layer(feature_fusion)
        return output

class LowLightEnhance:
    def __init__(self):
        self.DecomNet_layer_num = 5
        self.decom_net = DecomNet(layer_num=self.DecomNet_layer_num)
        self.relight_net = RelightNet()
        self.optimizer = tf.keras.optimizers.Adam()
        self.saver_Decom = tf.keras.callbacks.ModelCheckpoint('./model/Decom', save_weights_only=True)
        self.saver_Relight = tf.keras.callbacks.ModelCheckpoint('./model/Relight', save_weights_only=True)
        print("[*] Initialize model successfully...")

    def gradient(self, input_tensor, direction):
        smooth_kernel_x = tf.reshape(tf.constant([[0, 0], [-1, 1]], tf.float32), [2, 2, 1, 1])
        smooth_kernel_y = tf.transpose(smooth_kernel_x, [1, 0, 2, 3])
        if direction == "x":
            kernel = smooth_kernel_x
        elif direction == "y":
            kernel = smooth_kernel_y
        return tf.abs(tf.nn.conv2d(input_tensor, kernel, strides=[1, 1, 1, 1], padding='SAME'))

    def ave_gradient(self, input_tensor, direction):
        return tf.nn.avg_pool2d(self.gradient(input_tensor, direction), ksize=3, strides=1, padding='SAME')

    def smooth(self, input_I, input_R):
        input_R = tf.image.rgb_to_grayscale(input_R)
        return tf.reduce_mean(self.gradient(input_I, "x") * tf.exp(-10 * self.ave_gradient(input_R, "x")) +
                              self.gradient(input_I, "y") * tf.exp(-10 * self.ave_gradient(input_R, "y")))

    def evaluate(self, epoch_num, eval_low_data, sample_dir, train_phase):
        print("[*] Evaluating for phase %s / epoch %d..." % (train_phase, epoch_num))
        for idx in range(len(eval_low_data)):
            input_low_eval = np.expand_dims(eval_low_data[idx], axis=0)
            if train_phase == "Decom":
                R_low, I_low = self.decom_net(input_low_eval)
                result_1, result_2 = R_low, tf.concat([I_low, I_low, I_low], axis=3)
            elif train_phase == "Relight":
                R_low, I_low = self.decom_net(input_low_eval)
                I_delta = self.relight_net(I_low, R_low)
                S = R_low * tf.concat([I_delta, I_delta, I_delta], axis=3)
                result_1, result_2 = S, tf.concat([I_delta, I_delta, I_delta], axis=3)
            save_images(os.path.join(sample_dir, 'eval_%s_%d_%d.png' % (train_phase, idx + 1, epoch_num)), result_1, result_2)

    def train(self, train_low_data, train_high_data, eval_low_data, batch_size, patch_size, epoch, lr, sample_dir, ckpt_dir, eval_every_epoch, train_phase):
        assert len(train_low_data) == len(train_high_data)
        numBatch = len(train_low_data) // int(batch_size)
        if train_phase == "Decom":
            model = self.decom_net
            saver = self.saver_Decom
        elif train_phase == "Relight":
            model = self.relight_net
            saver = self.saver_Relight

        try:
            model.load_weights(ckpt_dir)
            print("[*] Model restore success!")
        except:
            print("[*] Not find pretrained model!")

        print("[*] Start training for phase %s" % train_phase)
        start_time = time.time()
        image_id = 0

        for epoch in range(epoch):
            for batch_id in range(numBatch):
                batch_input_low = np.zeros((batch_size, patch_size, patch_size, 3), dtype="float32")
                batch_input_high = np.zeros((batch_size, patch_size, patch_size, 3), dtype="float32")
                for patch_id in range(batch_size):
                    h, w, _ = train_low_data[image_id].shape
                    x = random.randint(0, h - patch_size)
                    y = random.randint(0, w - patch_size)
                    rand_mode = random.randint(0, 7)
                    batch_input_low[patch_id, :, :, :] = data_augmentation(train_low_data[image_id][x: x + patch_size, y: y + patch_size, :], rand_mode)
                    batch_input_high[patch_id, :, :, :] = data_augmentation(train_high_data[image_id][x: x + patch_size, y: y + patch_size, :], rand_mode)
                    image_id = (image_id + 1) % len(train_low_data)
                    if image_id == 0:
                        tmp = list(zip(train_low_data, train_high_data))
                        random.shuffle(tmp)
                        train_low_data, train_high_data = zip(*tmp)

                with tf.GradientTape() as tape:
                    if train_phase == "Decom":
                        R_low, I_low = model(batch_input_low)
                        R_high, I_high = model(batch_input_high)
                        I_low_3 = tf.concat([I_low, I_low, I_low], axis=3)
                        I_high_3 = tf.concat([I_high, I_high, I_high], axis=3)
                        recon_loss_low = tf.reduce_mean(tf.abs(R_low * I_low_3 - batch_input_low))
                        recon_loss_high = tf.reduce_mean(tf.abs(R_high * I_high_3 - batch_input_high))
                        recon_loss_mutal_low = tf.reduce_mean(tf.abs(R_high * I_low_3 - batch_input_low))
                        recon_loss_mutal_high = tf.reduce_mean(tf.abs(R_low * I_high_3 - batch_input_high))
                        equal_R_loss = tf.reduce_mean(tf.abs(R_low - R_high))
                        Ismooth_loss_low = self.smooth(I_low, R_low)
                        Ismooth_loss_high = self.smooth(I_high, R_high)
                        loss = recon_loss_low + recon_loss_high + 0.001 * recon_loss_mutal_low + 0.001 * recon_loss_mutal_high + 0.1 * Ismooth_loss_low + 0.1 * Ismooth_loss_high