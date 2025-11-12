import os

import cv2
import numpy as np


class DescentImage():
    '''
    单张降落图像的全部信息。包括初始命名，抽帧后命名和路径，图像的长宽通道，相机内参外参，高度和空间分辨率。

    Attributes:
    path: str 文件路径
    originName: str 抽帧前文件名称
    name: str 图像主名称
    id: int 在抽帧后的序列中的文件索引
    width: int 宽
    height: int 高
    channel: int 通道数，1或者3
    data: np.array 图像数据
    innerParam: np.matrix 3*3的内参矩阵
    R: np.array 3*3的旋转矩阵
    T: np.array 3*1的归一化后的平移向量
    highfromMoon: float 到月面的距离，单位m
    resolution: float 空间分辨率，单位cm
    '''

    def __init__(self, data=None, attributes=None):
        '''使用图像的data或属性列表进行初始化。'''
        if attributes:
            for key, value in attributes.items():
                setattr(self, key, value)
        elif data is not None:
            self.data = data
            if len(data.shape)==3:
                self.height, self.width, self.channel = data.shape
            else:
                self.height, self.width = data.shape
                self.channel=1
                # 初始化其他属性
            self.path = ""
            self.originName = ""
            self.name = ""
            self.id = 0
            self.innerParam = np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            self.R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            self.T = np.array([[0], [0], [0]])
            self.highfromMoon = 0.0
            self.resolution = 0.0
            self.angle=0.0

    def set_path(self, path):
        '''设置文件路径'''
        self.path = path

    def set_originName(self, originName):
        '''设置抽帧前的文件名称'''
        self.originName = originName

    def set_name(self, name):
        '''设置图像主名称'''
        self.name = name

    def set_id(self, id_):
        '''设置在抽帧后的序列中的文件索引'''
        self.id = id_

    def set_innerParam(self, innerParam):
        '''设置3*3的内参矩阵'''
        if innerParam.shape == (3, 3):
            self.innerParam = innerParam
        else:
            raise ValueError("The innerParam should be a 3x3 matrix.")

    def set_R(self, R):
        '''设置3*3的旋转矩阵'''
        if R.shape == (3, 3):
            self.R = R
        else:
            raise ValueError("R should be a 3x3 matrix.")

    def set_T(self, T):
        '''设置3*1的归一化后的平移向量'''
        if T.shape == (3, 1):
            self.T = T
        else:
            raise ValueError("T should be a 3x1 vector.")

    def set_highfromMoon(self, highfromMoon):
        '''设置到月面的距离'''
        self.highfromMoon = highfromMoon

    def set_resolution(self, resolution):
        '''设置空间分辨率'''
        self.resolution = resolution

    def set_angle(self,angle):
        self.angle=angle

    def set_attributes(self, **kwargs):
        '''
        一次性设置多个属性。

        参数:
        kwargs: 关键字参数。有效的关键字有：path, originName, name, id,
                innerParam, R, T, highfromMoon, resolution
        '''
        if 'path' in kwargs:
            self.set_path(kwargs['path'])

        if 'originName' in kwargs:
            self.set_originName(kwargs['originName'])

        if 'name' in kwargs:
            self.set_name(kwargs['name'])

        if 'id' in kwargs:
            self.set_id(kwargs['id'])

        if 'innerParam' in kwargs:
            self.set_innerParam(kwargs['innerParam'])

        if 'R' in kwargs:
            self.set_R(kwargs['R'])

        if 'T' in kwargs:
            self.set_T(kwargs['T'])

        if 'highfromMoon' in kwargs:
            self.set_highfromMoon(kwargs['highfromMoon'])

        if 'resolution' in kwargs:
            self.set_resolution(kwargs['resolution'])

    def get_attributes(self):
        '''返回类中所有属性的字典'''
        attributes = {attr: getattr(self, attr) for attr in dir(self) if
                      not callable(getattr(self, attr)) and not attr.startswith("__")}
        return attributes

    def copy(self):
        '''返回当前实例的深拷贝'''
        attributes = self.get_attributes()
        return DescentImage(attributes=attributes)

    def print_attributes(self):
        attributes = self.get_attributes()
        print(attributes)
#计算旋转矩阵
def handle_rotate_val(x,y,rotate):
  cos_val = np.cos(np.deg2rad(rotate))
  sin_val = np.sin(np.deg2rad(rotate))
  return np.float32([
      [cos_val, sin_val, x * (1 - cos_val) - y * sin_val],
      [-sin_val, cos_val, x * sin_val + y * (1 - cos_val)]
    ])


def image_rotate(src, rotate=0):
  h,w = src.shape
  M = handle_rotate_val(w//2,h//2,rotate)
  img = cv2.warpAffine(src, M, (w,h))
  return img


def load_images_from_folder(folder,imgGray:bool= False):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):  # 可按需求增加或减少文件类型
            img_path = os.path.join(folder, filename)
            #都以灰度图形式读入
            img = cv2.imread(img_path,0)
                #试试旋转之后的图
            if img is not None:
                image_obj = DescentImage(img)
                image_obj.set_path(img_path)
                image_obj.set_name(os.path.splitext(filename)[0])  # 文件名，不包括扩展名
                image_obj.set_originName(os.path.splitext(filename)[0])  # 可按需求修改
                images.append(image_obj)
    return images

class BestDescentImg(DescentImage):
    """
    继承自DescentImage，用以匹配的最佳降落图像，为其中一张降落图像或者拼接后的大图。
    若为拼接大图，则属性stitch=True，其次在属性stemDescentImgName，stemDescentImgId，stemDescentImgWH，
    stemDescentImgOriginalpoint中记录作为主干降落图像的信息。内外参等信息将与stemDescentImg保持一致。
    id则是len(descentimgs)。
    Attributes:
    ...
    stitch:  bool 是否为拼接大图
    stemDescentImgName: str 作为主干降落图像的名称
    stemDescentImgId: str 作为主干降落图像的索引
    stemDescentImgWH: (int,int)主干降落图像的长宽
    stemDescentImgOriginalpoint: 主干降落图像在拼接大图中的原点（row，col）格式
    path:  str 文件路径
    originName: str 主干降落图像抽帧前文件名称
    name: str 图像主名称
    id: int 设置为抽帧后序列的最后一个索引加一
    width: int 大图的宽
    height: int 大图高
    channel: int 通道数，1或者3
    data: np.array 图像数据
    innerParam: np.matrix 主干降落图像3*3的内参矩阵
    R: np.array 主干降落图像3*3的旋转矩阵
    T: np.array 主干降落图像3*1的归一化后的平移向量
    highfromMoon：float 主干降落图像到月面的距离，单位m
    resolution：float 主干降落图像空间分辨率，单位cm"""

    def __init__(self, descent_img: DescentImage):
        super().__init__(data=descent_img.data, attributes=descent_img.get_attributes())
        self.stitch = False  # 是否为拼接大图
        self.stemDescentImgName = ""  # 作为主干降落图像的名称
        self.stemDescentImgId = -1  # 作为主干降落图像的索引
        self.stemDescentImgWH = (-1, -1)  # 主干降落图像的长宽
        self.stemDescentImgOriginalpoint = (-1, -1)  # 主干降落图像在拼接大图中的原点（row，col）格式

if __name__=="__main__":
    # 使用示例：
    imgpath="./data/ce4/ce4_tiny_02040.jpg"
    img = cv2.imread(imgpath,0)
    # image_data = np.array([[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 0]]])
    descent_image = DescentImage(img)
    descent_image.set_attributes(path=imgpath, originName="ce4_tiny_02040", name="ce4_tiny_02040")
    arrlist=descent_image.get_attributes()
    # print(arrlist)
    new=DescentImage()
    new.set_id("1")
    arrlist2=new.get_attributes()
    for key, value in arrlist2.items():
        if key=="data":
            continue
        print(f"{key}: {value}")

    dirpath = "./data/ce4/images+bestbaseimg/descentimages"
    all_images = load_images_from_folder(dirpath)
    print()

