import os

import cv2
import numpy as np

from imgclass.descentImageClass import DescentImage, load_images_from_folder

import numpy as np
import cv2


def histogramSpecification(srcImage: np.array, referImage: np.array, binaryMask: np.array = None, usePrint: bool=True) -> [np.array]:
    '''
    对图像进行规定化处理。

    :param
    srcImage: 进行处理的原图像，一般是降落图像。
    referImage: 参考图像，一般是分割后底图。
    binaryMask: 二进制掩码，数据格式为srcImage同尺寸一通道，置为0的像素不参与规定化计算，置为1或非0的区域参与计算。默认设置为Null，表示所有区域均参与计算。

    :return
    dstImage: 进行规定化后的图像，一般是处理后的降落图像。
    '''

    def ensure_grayscale(image,name="image"):
        '''确保图像为单通道灰度图，并返回灰度图像'''
        if len(image.shape) > 2 and image.shape[2] > 1:
            if usePrint:
                print(image.shape)
                print(f"Warning: def histogramSpecification( {name} )is not a grayscale image. Converting it to grayscale.")
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    # 确保参考图像为灰度图
    srcImage = ensure_grayscale(srcImage,"srcImage")
    referImage = ensure_grayscale(referImage,"referImage")

    # 如果提供了binaryMask，则确保它也是灰度图
    if binaryMask is not None:
        binaryMask = ensure_grayscale(binaryMask,"Mask")
    else:
        binaryMask = np.ones_like(srcImage) * 255

    # 只在掩膜区域外计算直方图
    hist1, bins1 = np.histogram(srcImage[binaryMask != 0].flatten(), 256, [0, 256])
    hist2, bins2 = np.histogram(referImage.flatten(), 256, [0, 256])

    # 计算累积分布函数
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
    dstImage = srcImage.copy()
    for i in range(srcImage.shape[0]):
        for j in range(srcImage.shape[1]):
            if binaryMask[i, j] != 0:  # 只改变掩膜区域外的像素
                dstImage[i, j] = lut[srcImage[i, j]]

    return dstImage



def histogramSpecificationForList(srcImageList: list, referImage: np.array, binaryMask: np.array = None,usePrint:bool=True) -> list:
    '''
    对图像序列进行规定化处理。对list中每一个图像进行histogramSpecification函数。

    :param srcImageList: 进行处理的原图像序列，每个元素为DescentImage类，需要保证所有图像大小一致。一般是降落图像。
    :param referImage: 参考图像，一般是分割后底图。
    :param binaryMask: 对所有图像的统一的二进制掩码，数据格式为srcImage同尺寸一通道，置为0的像素不参与规定化计算，置为1或非0的区域参与计算。

    :return dstImageList: 进行规定化后的图像序列，每个元素为DescentImage类，一般是处理后的降落图像。
    '''

    def histogramSpecification(srcImage: np.array, referImage: np.array, binaryMask: np.array = None,
                               usePrint: bool = True) -> [np.array]:
        '''
        对图像进行规定化处理。

        :param
        srcImage: 进行处理的原图像，一般是降落图像。
        referImage: 参考图像，一般是分割后底图。
        binaryMask: 二进制掩码，数据格式为srcImage同尺寸一通道，置为0的像素不参与规定化计算，置为1或非0的区域参与计算。默认设置为Null，表示所有区域均参与计算。

        :return
        dstImage: 进行规定化后的图像，一般是处理后的降落图像。
        '''

        def ensure_grayscale(image, name="image"):
            '''确保图像为单通道灰度图，并返回灰度图像'''
            if len(image.shape) > 2 and image.shape[2] > 1:
                if usePrint:
                    print(
                        f"Warning: def histogramSpecification( {name} )is not a grayscale image. Converting it to grayscale.")
                return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return image

        # 确保参考图像为灰度图
        srcImage = ensure_grayscale(srcImage, "srcImage")
        referImage = ensure_grayscale(referImage, "referImage")

        # 如果提供了binaryMask，则确保它也是灰度图
        if binaryMask is not None:
            binaryMask = ensure_grayscale(binaryMask, "Mask")
        else:
            binaryMask = np.ones_like(srcImage) * 255

        # 只在掩膜区域外计算直方图
        hist1, bins1 = np.histogram(srcImage[binaryMask != 0].flatten(), 256, [0, 256])
        hist2, bins2 = np.histogram(referImage.flatten(), 256, [0, 256])

        # 计算累积分布函数
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
        dstImage = srcImage.copy()
        for i in range(srcImage.shape[0]):
            for j in range(srcImage.shape[1]):
                if binaryMask[i, j] != 0:  # 只改变掩膜区域外的像素
                    dstImage[i, j] = lut[srcImage[i, j]]

        return dstImage


    dstImageList = []

    while srcImageList:
        srcImageInstance = srcImageList[0]

        # 使用histogramSpecification处理每个DescentImage实例的data属性
        if usePrint:
            print(f"histogramSpecification for {srcImageInstance.name}")
        processed_data = histogramSpecification(srcImage= srcImageInstance.data, referImage=refimg,binaryMask=binaryMask,usePrint=False)

        # 创建一个新的DescentImage实例，设置其data属性为processed_data，其他属性和srcImageInstance保持一致
        new_image_instance = srcImageInstance.copy()
        new_image_instance.data = processed_data

        dstImageList.append(new_image_instance)

        # 从srcImageList中删除已处理的实例
        del srcImageList[0]

    return dstImageList

def histogramSpecificationAndMaskForList(srcImageList: list, referImage: np.array, outputdir:str ,binaryMask: np.array = None,usePrint:bool=True) -> list:
    '''
    对图像序列进行规定化处理和增加Mask掩码处理，保存到outputdir。对list中每一个图像进行histogramSpecification函数和mask。

    :param srcImageList: 进行处理的原图像序列，每个元素为DescentImage类，需要保证所有图像大小一致。一般是降落图像。
    :param referImage: 参考图像，一般是分割后底图。
    :param binaryMask: 对所有图像的统一的二进制掩码，数据格式为srcImage同尺寸一通道，置为0的像素不参与规定化计算，置为1或非0的区域参与计算。

    :return dstImageList: 进行规定化后的图像序列，每个元素为DescentImage类，一般是处理后的降落图像。
    '''

    def ensure_grayscale(image, name:str="image",usePrint:bool=True):
        '''确保图像为单通道灰度图，并返回灰度图像'''
        if len(image.shape) > 2 and image.shape[2] > 1:
            print(image.shape)
            if usePrint:
                print(
                    f"Warning: def histogramSpecification( {name} )is not a grayscale image. Converting it to grayscale.")
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def histogramSpecification(srcImage: np.array, referImage: np.array, binaryMask: np.array = None,
                               usePrint: bool = True) -> [np.array]:
        '''
        对图像进行规定化处理。

        :param
        srcImage: 进行处理的原图像，一般是降落图像。
        referImage: 参考图像，一般是分割后底图。
        binaryMask: 二进制掩码，数据格式为srcImage同尺寸一通道，置为0的像素不参与规定化计算，置为1或非0的区域参与计算。默认设置为Null，表示所有区域均参与计算。

        :return
        dstImage: 进行规定化后的图像，一般是处理后的降落图像。
        '''



        # 确保参考图像为灰度图
        srcImage = ensure_grayscale(srcImage, "srcImage",usePrint)
        referImage = ensure_grayscale(referImage, "referImage",usePrint)

        # 如果提供了binaryMask，则确保它也是灰度图
        if binaryMask is not None:
            binaryMask = ensure_grayscale(binaryMask, "Mask",usePrint)
        else:
            binaryMask = ensure_grayscale(np.ones_like(srcImage) * 255,"Mask",usePrint)

        # 只在掩膜区域外计算直方图
        hist1, bins1 = np.histogram(srcImage[binaryMask != 0].flatten(), 256, [0, 256])
        hist2, bins2 = np.histogram(referImage.flatten(), 256, [0, 256])

        # 计算累积分布函数
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
        dstImage = srcImage.copy()
        for i in range(srcImage.shape[0]):
            for j in range(srcImage.shape[1]):
                if binaryMask[i, j] != 0:  # 只改变掩膜区域外的像素
                    dstImage[i, j] = lut[srcImage[i, j]]

        return dstImage

    dstImageList = []
    while srcImageList:
        srcImageInstance = srcImageList[0]

        # 使用histogramSpecification处理每个DescentImage实例的data属性
        if usePrint:
            print(f"histogramSpecification and Mask for {srcImageInstance.name}")
        processed_data = histogramSpecification(srcImage=srcImageInstance.data, referImage=referImage,
                                                binaryMask=binaryMask, usePrint=False)
        processed_data=processed_data*(ensure_grayscale(binaryMask/255, "Mask",usePrint))

        # 创建一个新的DescentImage实例，设置其data属性为processed_data，其他属性和srcImageInstance保持一致
        new_image_instance = srcImageInstance.copy()
        new_image_instance.data = processed_data
        dstImageList.append(new_image_instance)
        savepath=os.path.join(outputdir,srcImageInstance.name)
        cv2.imwrite(savepath,processed_data)
        # 从srcImageList中删除已处理的实例
        del srcImageList[0]

    return dstImageList

if __name__=="__main__":
    # 读入数据
    srcimgpath=r"../data/ce4/ce4_tiny_02040.jpg"
    refimgpath=r"../data/ce4/ce4split2048_003_004.jpg"
    maskpath=r"../data/ce4/descentimgs/mask.tif"
    srcimg=cv2.imread(srcimgpath,0)
    refimg=cv2.imread(refimgpath,0)
    mask=cv2.imread(maskpath,0)
    # resimg=histogramSpecification(srcImage=srcimg,referImage=refimg,binaryMask=mask,usePrint=True )
    # resimg2=histogramSpecification(srcImage=srcimg,referImage=refimg,binaryMask=None,usePrint=True )
    # cv2.imshow("res1",resimg)
    # cv2.imshow("res2",resimg2)
    # cv2.waitKey()

    # dirpath = r"D:\locate\moonlocate/data/ce4/images+bestbaseimg/descentimages"
    dirpath=r"../output\test\images"
    outputdir=r"../output\test\images_mask"
    all_images = load_images_from_folder(dirpath,imgGray=True)
    # hs_images=histogramSpecificationForList(srcImageList=all_images,referImage=refimg,binaryMask=mask)
    histogramSpecificationAndMaskForList(srcImageList=all_images,referImage=refimg,binaryMask=mask,outputdir=outputdir,usePrint=True)