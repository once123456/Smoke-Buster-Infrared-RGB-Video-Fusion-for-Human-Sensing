import torch
import numpy as np
from lightdehazeNet import LightDehaze_Net


def image_haze_removel(input_image):
    hazy_image = (np.asarray(input_image) / 255.0)
    hazy_image = torch.from_numpy(hazy_image).float()
    hazy_image = hazy_image.permute(2, 0, 1)
    hazy_image = hazy_image.cuda().unsqueeze(0)

    ld_net = LightDehaze_Net().cuda()
    ld_net.load_state_dict(torch.load('trained_weights/trained_LDNet.pth'))

    dehaze_image = ld_net(hazy_image)
    return dehaze_image
    