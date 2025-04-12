import numpy as np
from PIL import Image

def data_augmentation(image: np.ndarray, mode: int) -> np.ndarray:
    """
    对输入的图像进行数据增强操作。

    参数:
    image (np.ndarray): 输入的图像数组。
    mode (int): 数据增强的模式，范围从 0 到 7。

    返回:
    np.ndarray: 经过数据增强后的图像数组。
    """
    if mode == 0:
        # original
        return image
    elif mode == 1:
        # flip up and down
        return np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        return np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        image = np.rot90(image)
        return np.flipud(image)
    elif mode == 4:
        # rotate 180 degree
        return np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        image = np.rot90(image, k=2)
        return np.flipud(image)
    elif mode == 6:
        # rotate 270 degree
        return np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        image = np.rot90(image, k=3)
        return np.flipud(image)

def load_images(file: str) -> np.ndarray:
    """
    从指定文件路径加载图像，并将其转换为 numpy 数组，同时将像素值归一化到 [0, 1] 范围。

    参数:
    file (str): 图像文件的路径。

    返回:
    np.ndarray: 加载并归一化后的图像数组。
    """
    try:
        im = Image.open(file)
        return np.array(im, dtype="float32") / 255.0
    except FileNotFoundError:
        print(f"文件 {file} 未找到。")
        return np.array([])
    except Exception as e:
        print(f"加载图像时出现错误: {e}")
        return np.array([])

def save_images(filepath: str, result_1: np.ndarray, result_2: np.ndarray = None) -> None:
    """
    将处理后的图像保存为 PNG 格式文件。

    参数:
    filepath (str): 保存图像的文件路径。
    result_1 (np.ndarray): 第一个图像结果。
    result_2 (np.ndarray, 可选): 第二个图像结果。默认为 None。
    """
    result_1 = np.squeeze(result_1)
    if result_2 is not None:
        result_2 = np.squeeze(result_2)

    if result_2 is None or not result_2.any():
        cat_image = result_1
    else:
        cat_image = np.concatenate([result_1, result_2], axis=1)

    im = Image.fromarray(np.clip(cat_image * 255.0, 0, 255.0).astype('uint8'))
    im.save(filepath, 'png')