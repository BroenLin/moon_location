import numpy as np
import cv2
from imgclass.descentImageClass import DescentImage, BestDescentImg


# class BestDescentImg(DescentImage):
#     """
#     继承自DescentImage，用以匹配的最佳降落图像，为其中一张降落图像或者拼接后的大图。
#     若为拼接大图，则属性stitch=True，其次在属性stemDescentImgName，stemDescentImgId，stemDescentImgWH，
#     stemDescentImgOriginalpoint中记录作为主干降落图像的信息。内外参等信息将与stemDescentImg保持一致。
#     id则是len(descentimgs)。
#     Attributes:
#     ...
#     stitch:  bool 是否为拼接大图
#     stemDescentImgName: str 作为主干降落图像的名称
#     stemDescentImgId: str 作为主干降落图像的索引
#     stemDescentImgWH: (int,int)主干降落图像的长宽
#     stemDescentImgOriginalpoint: 主干降落图像在拼接大图中的原点（row，col）格式
#     path:  str 文件路径
#     originName: str 主干降落图像抽帧前文件名称
#     name: str 图像主名称
#     id: int 设置为抽帧后序列的最后一个索引加一
#     width: int 大图的宽
#     height: int 大图高
#     channel: int 通道数，1或者3
#     data: np.array 图像数据
#     innerParam: np.matrix 主干降落图像3*3的内参矩阵
#     R: np.array 主干降落图像3*3的旋转矩阵
#     T: np.array 主干降落图像3*1的归一化后的平移向量
#     highfromMoon：float 主干降落图像到月面的距离，单位m
#     resolution：float 主干降落图像空间分辨率，单位cm"""
#
#     def __init__(self, descent_img: DescentImage):
#         super().__init__(data=descent_img.data, attributes=descent_img.get_attributes())
#         self.stitch = False  # 是否为拼接大图
#         self.stemDescentImgName = ""  # 作为主干降落图像的名称
#         self.stemDescentImgId = -1  # 作为主干降落图像的索引
#         self.stemDescentImgWH = (-1, -1)  # 主干降落图像的长宽
#         self.stemDescentImgOriginalpoint = (-1, -1)  # 主干降落图像在拼接大图中的原点（row，col）格式


def getBestDescentImgfromDescentImgs(descentimgs,
                                     resolutionforBaseimage: float,
                                     numofdescentimages: int = 1,
                                     useStitching: bool = False,
                                     idforUser: int = -1) :
    """
    选择和给定分辨率相近的一张或多张降落图像，选择多张时将拼接为一张大图。
    # **************************hcf********************
    拼接时取分辨率最接近的降落图作为主干降落图像，其它分辨率相近的图像通过SIFT特征匹配和单应矩阵映射到主干降落图像的平面
    # **************************hcf********************
    :param
    descentimgs : 降落图像列表，每个元素是DescentImage类，需要使用data、 resolution属性,属性列表。
    resolutionforBaseimage :指定的空间分辨率，一般为底图分辨率。
    numofdescentimages:所需的降落图像数量，默认为1。
    useStitching :是否使用拼接，多张图时可选用，返回一张拼接后的图，默认为多张时开启。
    idforUser：直接指定分辨率最接近于给定分辨率的降落图像

    :return
    bestdescentimg: BestDescentImg类，返回一张最佳降落图像用以后续匹配。
    """
    # 指定分辨率最接近于给定分辨率的降落图像时，直接返回
    if idforUser != -1:
        for descentImg in descentimgs:
            if descentImg.id == idforUser:
                descentImg_temp = descentImg.copy()
                bestdescentimg = BestDescentImg(descentImg_temp)
                return bestdescentimg
    # 未指定降落图像
    # 统计分辨率的差值
    resolutions_list = []
    for descentImg in descentimgs:
        resolutions_list.append(descentImg.resolution)
    resolutions_abs_diff = np.absolute(np.array(resolutions_list) - resolutionforBaseimage)
    args_resolutions_abs_diff = np.argsort(resolutions_abs_diff)  # 按升序排列后的索引
    # 不进行拼接
    if not useStitching:
        descentImg_temp = descentimgs[args_resolutions_abs_diff[0]].copy()
        bestdescentimg = BestDescentImg(descentImg_temp)
        return bestdescentimg
    # 进行拼接
    # **************************hcf********************
    # 必须保证useStitching=True时，numofdescentimages > 1！！！！
    assert numofdescentimages > 1, "getBestDescentImgfromDescentImgs: useStitching==True but numofdescentimages<2"
    # **************************hcf********************
    bestdescentimgs_dict = {}
    # 对主干降落图像进行仿射不变的SIFT特征提取
    sift_detector = cv2.SIFT_create()
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    descentImg_stem = descentimgs[args_resolutions_abs_diff[0]]
    if descentImg_stem.path != "":
        image_temp = cv2.imread(descentImg_stem.path)
    else:
        image_temp = descentImg_stem.data
    h_stem, w_stem = image_temp.shape[:-1]
    # **************************hcf********************
    # 必须保证降落图像的id是已知的，并且id按照图像拍摄时间排序，id小的图像拍摄早
    # ！！！！因为后续会根据降落图像的顺序对图像进行叠加
    id_stem = descentImg_stem.id
    # **************************hcf********************
    kps_stem, des_stem = sift_detector.detectAndCompute(image_temp, None)
    # 对其它降落图像进行仿射不变的SIFT特征提取，并与主干图像进行匹配，计算单应矩阵H，最终投影到主干图像的平面
    for i_temp in range(numofdescentimages):
        descentImg = descentimgs[args_resolutions_abs_diff[i_temp]]
        if descentImg.path != "":
            image_temp = cv2.imread(descentImg.path)
        else:
            image_temp = descentImg.data
        kps, des = sift_detector.detectAndCompute(image_temp, None)
        # 进行SIFT特征匹配，计算H
        matches = bf.match(des_stem, des)
        matches = sorted(matches, key=lambda x: x.distance)  # 按照匹配度排序
        kp1_list, kp2_list = [], []
        # **************************hcf********************
        for match in matches[:20]:  # 只选取匹配度最高的前20个匹配点对，将坐标存储在kp1_list和kp2_list
        # **************************hcf********************
            kp_temp = kps_stem[match.queryIdx]
            #  +w_stem/2和+h_stem/2相当于将主干图添加边界框，使主干图尺寸为原来的两倍，主干图置于正中间
            kp1_list.append([kp_temp.pt[0] + w_stem/2, kp_temp.pt[1]+h_stem/2])
            kp_temp = kps[match.trainIdx]
            kp2_list.append([kp_temp.pt[0], kp_temp.pt[1]])
        kp1_np, kp2_np = np.array(kp1_list), np.array(kp2_list)
        # 计算单应矩阵并投影到主干降落图像的平面
        H, mask_H = cv2.findHomography(srcPoints=kp2_np, dstPoints=kp1_np, method=cv2.RANSAC)
        perspective_img = cv2.warpPerspective(image_temp, H, dsize=(w_stem*2, h_stem*2))
        # **************************hcf********************
        # 必须保证降落图像的id是已知的，并且id按照图像拍摄时间排序，id小的图像拍摄早
        # ！！！！因为后续会根据降落图像的顺序对图像进行叠加
        bestdescentimgs_dict[descentImg.id] = perspective_img
        # **************************hcf********************
    img_merge = np.zeros(bestdescentimgs_dict[0].shape, dtype=np.uint8)
    bestdescentimgs_id_sorted = sorted(bestdescentimgs_dict.keys())
    bestdescentimgs_id_sorted.append(id_stem)  # 主干图像在最后叠加
    # 按照时间顺序叠加图片，主干图像在最后叠加
    for id_temp in bestdescentimgs_id_sorted:
        # 必须保证尺寸相同，否则cv2.copyTo返回空图
        assert bestdescentimgs_dict[id_temp].shape == img_merge.shape, "必须保证尺寸相同，否则cv2.copyTo返回空图"
        # 通过检测轮廓来生成拼接所需的mask
        if len(bestdescentimgs_dict[id_temp].shape) == 3 and bestdescentimgs_dict[id_temp].shape[-1] == 3:
            # 若为三通道图像，转为单通道
            img_temp = cv2.cvtColor(bestdescentimgs_dict[id_temp], code=cv2.COLOR_BGR2GRAY)
            # cv2.RETR_EXTERNAL只检测外轮廓；
            # cv2.CHAIN_APPROX_SIMPLE压缩水平方向、垂直方向、对角线方向的元素，只保留该方向的终点坐标，例如一个矩形轮廓只需要4个点来保存轮廓信息
            contours, _ = cv2.findContours(image=img_temp, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
            mask_temp = np.zeros(img_temp.shape, dtype=np.uint8)
        else:
            contours, _ = cv2.findContours(image=bestdescentimgs_dict[id_temp], mode=cv2.RETR_EXTERNAL,
                                           method=cv2.CHAIN_APPROX_SIMPLE)
            mask_temp = np.zeros(bestdescentimgs_dict[id_temp].shape, dtype=np.uint8)
        # thickness=-1表示填充
        cv2.drawContours(image=mask_temp, contours=contours, contourIdx=0, color=255, thickness=-1)
        # 由于存在黑边，因此进行腐蚀
        kernel = np.ones((3, 3), np.uint8)
        mask_temp = cv2.erode(mask_temp, kernel, iterations=1)
        cv2.copyTo(src=bestdescentimgs_dict[id_temp], dst=img_merge, mask=mask_temp)
    descentImg_temp = DescentImage(data=img_merge)
    descentImg_temp.set_id(len(descentimgs))
    descentImg_temp.set_innerParam(descentImg_stem.innerParam)
    descentImg_temp.set_R(descentImg_stem.R)
    descentImg_temp.set_T(descentImg_stem.T)
    descentImg_temp.set_highfromMoon(descentImg_stem.highfromMoon)
    descentImg_temp.set_resolution(descentImg_stem.resolution)
    bestDescentImg = BestDescentImg(descentImg_temp)
    bestDescentImg.stemDescentImgId = descentImg_stem.id
    bestDescentImg.stemDescentImgName = descentImg_stem.name
    bestDescentImg.stemDescentImgWH = (descentImg_stem.width, descentImg_stem.height)
    bestDescentImg.stemDescentImgOriginalpoint = (w_stem//2, h_stem//2)
    return bestDescentImg





