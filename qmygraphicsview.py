from PyQt5 import QtCore, QtGui, QtWidgets

from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QLabel
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt,pyqtSignal,QPoint,QRectF,Qt
import numpy as np
import cv2


class QMyGraphicsView(QGraphicsView):
   # 发送鼠标点击的时候的坐标
    sigMousePressPoint = pyqtSignal(QPoint)  # 发送鼠标点击时的坐标
    #自定义信号sigMouseMovePoint，当鼠标点击时，在mouseMoveEvent事件中，将当前的鼠标位置发送出去
    #QPoint--传递的是view坐标
    
    def __init__(self,parent=None):
        super(QMyGraphicsView,self).__init__(parent)
        #设置鼠标形状为十字形方便点控制点对
        self.setCursor(Qt.CrossCursor)


    def mousePressEvent(self, evt):
        if evt.buttons() == Qt.LeftButton:
            pt=evt.pos()  #获取鼠标坐标--view坐标
            self.sigMousePressPoint.emit(pt) #发送鼠标位置
            QGraphicsView.mousePressEvent(self, evt)



#控制点图像类
class contrl_imgClass():
    def __init__(self,data,name):
        #保留原始下降图像类的data
        self.data=data
        self.name=name
        #保存原始的名字
        #保留画上控制点的data
        self.resetdata=data
        self.pointx=0.0
        self.pointy=0.0
        self.flag=True
        self.measure={'x': 0.0,
                    'y': 0.0,
                    'PhotoId':0}
    def set_flag(self,flag):
        self.flag=flag
    #更新控制点
    def set_measure(self,id):
        self.measure['x']=self.pointx
        self.measure['y']=self.pointy
        self.measure['PhotoId']=id
    #控制点选错时，方便重置控制点
    def reset_point(self):
        self.pointx=0.0
        self.pointy=0.0
        self.data=self.resetdata
        self.set_measure(0)
        
    def setpoint(self,posx,posy):
        if self.flag==True:
            self.pointx=posx
            self.pointy=posy
#每次都重置data，并重新画点
    def paint_point(self):
        if self.flag==True:
            self.data=np.expand_dims(self.resetdata,2)
            print(self.data)
            self.data=np.repeat(self.data,3,2)
            print(self.data)
            cv2.circle(self.data,(int(self.pointx),int(self.pointy)),5,(0,0,255),-1)
            print(self.data)

class matchpoint_Class():
    def __init__(self,data):
        #保留原始下降图像类的data
        #先扩展好通道方便后续画匹配点
        self.data=np.repeat(np.expand_dims(data,2),3,2)
        self.fontStyle = cv2.FONT_HERSHEY_TRIPLEX
        #保存原始的名字
        #保留画上控制点的data
        self.resetdata=data
        self.pointx=0.0
        self.pointy=0.0
        self.match_points=[]
    #重置匹配点信息
    def reset_point(self):
        self.pointx=0.0
        self.pointy=0.0
        self.data=np.repeat(np.expand_dims(self.resetdata,2),3,2)
        self.match_points=[]
    #添加匹配点
    def add_point(self,posx,posy):
        self.pointx=posx
        self.pointy=posy
        self.match_points.append([self.pointx,self.pointy])
#绘制点并标记坐标
    def paint_point(self):
        x=int(self.pointx)
        y=int(self.pointy)
        cv2.circle(self.data,(x,y),2,(0,0,255),-1)
        cv2.putText(self.data, "[{},{}]".format(self.pointx,self.pointy), (x,y), self.fontStyle, 0.5, (0, 0, 255))


if __name__=='__main__':
    img=contrl_imgClass(1,'hhh')
    img.setpoint(100,200)
    img.set_measure(5)
    print(img.measure['PhotoId'],img.measure['x'])
        



