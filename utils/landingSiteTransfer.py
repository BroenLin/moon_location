import cv2
import numpy as np
import os

from imgclass.descentImageClass import DescentImage, load_images_from_folder
from utils.sieveforBestBaseimgs import image_pair_matching_forImgClass


# 计算转换矩阵
def compute_transformation_matrix(mkpts1, mkpts0):
    '''H, _ = cv2.findHomography(mkpts1, mkpts0, method=cv2.RANSAC,
                              ransacReprojThreshold=0.25,
                              confidence=0.99999, maxIters=10000)'''
    H, _ = cv2.findHomography(mkpts1, mkpts0, method=cv2.RANSAC)
    return H


# 应用转换矩阵
def apply_transformation(H, point):
    point = np.array([[point[0]], [point[1]], [1]])
    transformed_point = np.dot(H, point)
    x_transformed = float(transformed_point[0, 0] / transformed_point[2, 0])
    y_transformed = float(transformed_point[1, 0] / transformed_point[2, 0])
    return (x_transformed, y_transformed)


def map_coordinates_to_original(coord, original_shape, resized_shape):
    """
    将缩放后的图像坐标映射到原始图像坐标。

    参数：
    coord (tuple): 缩放后的图像中的坐标点 (x, y)
    original_shape (tuple): 原始图像的形状 (height, width)
    resized_shape (tuple): 缩放后的图像的形状 (height, width)

    返回：
    tuple: 原始图像中的坐标点 (x, y)
    """
    # 解包坐标和形状
    x, y = coord
    original_height, original_width = original_shape
    resized_height, resized_width = resized_shape

    # 计算映射因子
    x_scale = original_width / resized_width
    y_scale = original_height / resized_height

    # 映射到原始图像坐标
    x_original = x * x_scale
    y_original = y * y_scale

    return x_original, y_original

# 主函数
def landingSiteTransfer(bestbaseimg, bestdescentimg, descentimgs,
                        outputdir, superglue="indoor", resize=[1280,960], device="cuda",
                        match_threshold=0.01, useviz=True, landingSitePixelinLastimg=None, binaryMask=None):
    """

    从最后一张着陆图像开始，逆顺序进行相邻帧匹配，将着陆点从最后一张图像传递到最佳着陆图像，然后再传递到最佳底图。

    参数:
    bestbaseimg (object): 最佳底图，是一个特定的图像类实例。
    bestdescentimg (object): 最佳着陆图像，是一个特定的图像类实例。
    descentimgs (list): 着陆图像列表。
    outputdir (str): 输出目录路径。
    superglue (str): SuperGlue配置。
    resize (list): 图像重设大小的维度。
    device (str): 使用的设备（'cuda' 或 'cpu'）。
    match_threshold (float): 匹配阈值。
    useviz (bool): 是否使用可视化。
    landingSitePixelinLastimg (tuple): 最后一张图像中的着陆点像素坐标（x, y）。
    binaryMask (array): 二进制掩码。

    返回:
    (float, float): 返回一个元组，表示着陆点在最佳底图中的像素坐标（x, y）。

    """

    # lastKnownLandingSite = np.array(landingSitePixelinLastimg, dtype=np.float32)

    # 初始化最初的着陆点为最后一张图像的中心点
    #顺带获取精定位的控制点文件
    contrlpoint=[]
    contrlpoints=[]
    workspacedir=r'D:\moonlocate'
    if landingSitePixelinLastimg is None:
        w, h = resize # 获取图像高度和宽度
        lastKnownLandingSite = (w // 2, h // 2)  # 设置为中心点
        print(lastKnownLandingSite)
        lastKnownLandingSite_origin=map_coordinates_to_original(lastKnownLandingSite,
                                                                resized_shape=[resize[1], resize[0]],
                                                                original_shape=descentimgs[-1].data.shape)
        print(f"当前着陆点像素：{lastKnownLandingSite_origin}")
    # 逆序遍历descentimgs，每张图像与它前一张进行匹配
    for i in range(len(descentimgs) - 1, 0, -1):
        current_img = descentimgs[i]
        previous_img = descentimgs[i - 1]
        print(current_img.name+'.jpg', bestdescentimg.name)
        if current_img.name+'.jpg'==bestdescentimg.name:
            break
        save_path = os.path.join(outputdir, f"{previous_img.name}_{current_img.name}_match.jpg")
        print(f"{previous_img.name}_{current_img.name}")

        mkpts_previous, mkpts_current, _ = image_pair_matching_forImgClass(
            previous_img,current_img, save_path=save_path,
            superglue=superglue, resize=resize, device=device,
            match_threshold=match_threshold, useviz=False
        )

        # 如果匹配点的数量足够多，则进行坐标转换
        if len(mkpts_current) > 7:
            H = compute_transformation_matrix(mkpts_current, mkpts_previous)
            lastKnownLandingSite = apply_transformation(H, lastKnownLandingSite)
            print(lastKnownLandingSite)
            if useviz:
                print('resize:', resize, 'current_img.data.shape', current_img.data.shape)
                lastKnownLandingSite_origin=map_coordinates_to_original(lastKnownLandingSite,
                                                                        resized_shape=[resize[1], resize[0]],original_shape=current_img.data.shape)
                print(f"当前着陆点像素：{lastKnownLandingSite_origin}")
                if i<=len(descentimgs)-1 and i>=len(descentimgs)-3:
                    measurement={'x':format(lastKnownLandingSite_origin[0],'.1f'),
                    'y': format(lastKnownLandingSite_origin[1],'.1f'),
                    'PhotoId': i}
                    contrlpoint.append(measurement)
                img_rgb = cv2.cvtColor(previous_img.data, cv2.COLOR_GRAY2BGR)
                cpx_transformed, cpy_transformed = lastKnownLandingSite_origin
                if 0 < cpx_transformed < previous_img.width and 0 < cpy_transformed < previous_img.height:
                    cv2.circle(img_rgb, (int(cpx_transformed), int(cpy_transformed)), 9, (255, 0, 0), -1)  # 蓝
                cv2.imwrite(f"{outputdir}/{current_img.name}_{previous_img.name}_landsite_{int(cpx_transformed)}_{int(cpy_transformed)}.jpg", img_rgb)
    contrlpoints.append(contrlpoint)
    #写入匹配得到的精定位控制点
    with open(os.path.join(workspacedir,"output/test_precise/cc/Suvery.txt"), 'w') as f1:
                f1.write(str(contrlpoints))
    #与底图匹配
    save_path = os.path.join(outputdir, f"{bestdescentimg.name}_{bestbaseimg.name}_match.jpg")
    mkpts0, mkpts1, _ = image_pair_matching_forImgClass(
        bestbaseimg, bestdescentimg,  save_path=save_path,
        superglue=superglue, resize=resize, device=device,
        match_threshold=match_threshold, useviz=useviz
    )
    if len(mkpts0) > 7:
        H = compute_transformation_matrix(mkpts1, mkpts0)
        lastKnownLandingSite = apply_transformation(H, lastKnownLandingSite)
        # print(lastKnownLandingSite)
        # 可视化底图着陆点
        if useviz:
            lastKnownLandingSite_origin = map_coordinates_to_original(lastKnownLandingSite, resized_shape=[resize[1], resize[0]],
                                                                      original_shape=bestbaseimg.data.shape)
            # print(f"最终着陆点像素：{lastKnownLandingSite_origin}")
            img_rgb = cv2.cvtColor(bestbaseimg.data, cv2.COLOR_GRAY2BGR)
            cpx_transformed, cpy_transformed = lastKnownLandingSite_origin
            if 0 < cpx_transformed < bestbaseimg.width and 0 < cpy_transformed < bestbaseimg.height:
                cv2.circle(img_rgb, (int(cpx_transformed), int(cpy_transformed)), 5, (255, 0, 0), -1)  # 蓝色
            cv2.imwrite(f"{outputdir}/landing_site_{int(cpx_transformed)}_{int(cpy_transformed)}.jpg", img_rgb)
        return lastKnownLandingSite_origin[0], lastKnownLandingSite_origin[1]
    else:
        print('定位失败')
        return None
    

if __name__=="__main__":
    # 设置参数
    workspacedir = r"D:/locate/moonlocate"
    # dirpath =os.path.join( workspacedir, "output/test/images_mask2")
    dirpath=r"D:\locate\ce4\data\mask_image"
    outputdir =os.path.join( workspacedir, "output/test/sp/landingSite3" ) # 输出目录
    os.makedirs(outputdir,exist_ok=True)
    superglue = 'indoor'  # SuperGlue配置
    resize =[1280, 960]#[640, 480]  # 重新调整图像的大小
    device = 'cuda'  # 运行设备
    match_threshold = 0.01  # 匹配阈值
    useviz = True  # 是否进行可视化

    # 加载着陆图像
    descentimgs = load_images_from_folder(dirpath, imgGray=True)

    # 加载最佳着陆图像和最佳底图
    # image_path1 = os.path.join( workspacedir,'output/test/images_mask2/ce4_test2_00011.jpg')
    image_path1 = r"D:\locate\moonlocate\data\ce4\ce4_tiny_02040.jpg"

    image_path2 = os.path.join( workspacedir,'data/ce4/ce4split2048_003_004.jpg')

    img1 = cv2.imread(image_path1, 0)  # 以灰度模式读取
    img2 = cv2.imread(image_path2, 0)

    bestdescentimg = DescentImage(img1)
    bestdescentimg.set_attributes(path=image_path1, originName="ce4_test2_00011", name="ce4_tiny_02040")

    bestbaseimg = DescentImage(img2)
    bestbaseimg.set_attributes(path=image_path2, originName="ce4split2048_003_004", name="ce4split2048_003_004")

    # # 设置最后一张着陆图像的中心点作为着陆点
    # h, w = descentimgs[-1].data.shape  # 获取图像高度和宽度
    # landingSitePixelinLastimg = (w // 2, h // 2)  # 设置为中心点

    # 调用函数
    new_x, new_y = landingSiteTransfer(
        bestbaseimg, bestdescentimg, descentimgs,
        outputdir, superglue, resize, device,
        match_threshold, useviz
    )

    # 打印新的着陆点坐标
    print(f"The new landing site coordinates are: x = {new_x}, y = {new_y}")
