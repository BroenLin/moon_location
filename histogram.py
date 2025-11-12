import cv2
import numpy as np
import pdb

def hist_guide(img,img_target):

    hist1, bins1 = np.histogram(img.flatten(), 256, [0,256])
    hist2, bins2 = np.histogram(img_target.flatten(), 256, [0,256])

    # 计算原始图像和目标图像的累积分布函数
    cdf1 = hist1.cumsum()
    cdf2 = hist2.cumsum()

    # 归一化累积分布函数
    cdf1_normalized = cdf1 * hist1.max() / cdf1.max()
    cdf2_normalized = cdf2 * hist1.max() / cdf2.max()

    # 创建映射表
    lut = [] 
    for i in range(256):
        lut.append(np.argmin(np.abs(cdf1_normalized[i] - cdf2_normalized)))

    # 应用映射表进行直方图规定化
    img_specified = np.array([lut[pixel] for pixel in img.flatten()], dtype=np.uint8)
    img_specified = img_specified.reshape(img.shape)
    return img_specified


