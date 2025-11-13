import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from imgclass.descentImageClass import DescentImage, load_images_from_folder
from utils.histogramSpecification import histogramSpecification, histogramSpecificationForList


def getDescentImageTinyListfromOriginData(
        inputDir: str,
        outputDir: str,
        nameTxtPath: str,
        howManyFrameBetween: int = 40,
        beginFrame: int = 0,
        endFrame: int = 1e5,
        useDetectionOverlapArea: bool = True,
        OverlapAreaRate: float = 0.75,
        renameHead: str = "ce4_test_",
        BinaryMask: np.array = None) -> list:
    '''
    给定原始降落图像的文件夹，抽帧获取后续使用的降落图像序列。可以指定最小重叠比例以提高快速机动时的采样密度。


    :param inputDir: 原始降落图像的文件夹。
    :param outputDir: 抽帧后重命名的降落图像输出文件夹。
    :param nameTxtPath: 重命名的映射文件路径，格式为每一行 “id 原名字 重命名”。
    :param howManyFrameBetween: 每隔多少帧抽取一张数据，默认为40。
    :param beginFrame: 从第几帧开始，手动去除过于倾斜的图像，默认为0。
    :param endFrame:第几帧结束，手动去除最后静止的图像，默认为最后一帧。
    :param useDetectionOverlapArea: 是否检测重叠区域，用以增加采样密度。默认为真
    :param OverlapAreaRate:检测重叠区域时，最小的重叠比例。
    :param renameHead:重命名的开头
    :param BinaryMask:检测重叠区域时二进制掩码。和图像同样大小的0-1，设为0的区域不计入计算。
    :return:descentImgs  : 列表，每一个是DescentImage类的数据。数据中只设置了data和图像、文件路径相关参数。
    '''


    # 获取文件夹中的文件名，并排序
    filenames = sorted(os.listdir(inputDir))
    descentImgs = []

    # 初始化前一帧的数据和最后一次保存的图像的索引
    prev_img_data = None
    i = beginFrame
    last_saved_idx = -1  # 初始设置为-1，因为还没有任何图像被保存
    descent_img_idx = 0  # 用于为DescentImage实例编号
    mapping = []

    while i < min(endFrame, len(filenames)):
        filename = filenames[i]
        img_data = cv2.imread(os.path.join(inputDir, filename), cv2.IMREAD_GRAYSCALE)

        # 如果使用重叠区域检测，计算SSIM来评估图像相似度
        if useDetectionOverlapArea and prev_img_data is not None:
            similarity_index, _ = ssim(img_data, prev_img_data, full=True)

            # 使用阈值来决定是否两幅图像相似
            threshold = OverlapAreaRate

            if similarity_index < threshold:
                # 如果图像不相似，则向前抽帧直到它们相似且索引大于上一张图像的索引
                while similarity_index < threshold and i > last_saved_idx + 1:
                    i -= 1
                    img_data = cv2.imread(os.path.join(inputDir, filenames[i]), cv2.IMREAD_GRAYSCALE)
                    if BinaryMask is not None:
                        # 仅对掩码为非零的区域进行计算
                        img_data_masked = img_data * BinaryMask
                        prev_img_data_masked = prev_img_data * BinaryMask
                        similarity_index, _ = ssim(img_data_masked, prev_img_data_masked, full=True)
                    else:
                        similarity_index, _ = ssim(img_data, prev_img_data, full=True)
            else:
                i += howManyFrameBetween

        idxstr = str(descent_img_idx).zfill(5)
        newName = renameHead + f"{idxstr}.jpg"

        # 创建DescentImage实例
        attributes = {
            'data': img_data,
            'path': os.path.join(outputDir, newName),
            'originName': filename,
            'name': newName,
            'id': descent_img_idx
        }

        descentImage = DescentImage(attributes=attributes)
        descentImgs.append(descentImage)

        # 保存图像到输出目录
        cv2.imwrite(descentImage.path, (descentImage.data).astype(np.uint8))

        # 更新重新命名的映射信息
        mapping.append(f"{descent_img_idx} {filename} {newName}")

        prev_img_data = img_data
        last_saved_idx = i
        descent_img_idx += 1

        # 为下一次迭代准备
        i += 1

    # 将映射信息保存到文本文件
    with open(nameTxtPath, 'w') as f:
        for line in mapping:
            f.write(line + '\n')

    return descentImgs


if __name__ == "__main__":
    inputdir = "../data/ce4/descentimgs/all"
    outputdir = "../output/test/images"

    textpath = "../output/test/nameOrigin2Tiny.txt"
    maskpath = "../data/ce4/descentimgs/mask.tif"
    beginFrame = 1650  # 手动指定
    endFrame = 3340  # 手动指定
    renameHead = "ce4_test_"
    howManyFrameBetween = (endFrame - beginFrame) // 50
    OverlapAreaRate=0.6 #0.75

    os.makedirs(outputdir, exist_ok=True)
    mask = cv2.imread(maskpath, 0)
    # files=os.listdir(inputdir)
    descentimgs = getDescentImageTinyListfromOriginData(inputDir=inputdir, outputDir=outputdir, nameTxtPath=textpath,
                                                        howManyFrameBetween=howManyFrameBetween, renameHead=renameHead,
                                                        beginFrame=beginFrame, endFrame=endFrame,
                                                        BinaryMask=mask, useDetectionOverlapArea=True,
                                                        OverlapAreaRate=OverlapAreaRate)
