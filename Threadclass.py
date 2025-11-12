from PyQt5 import QtCore
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from Ui_main import *
import os
from lashen import *
from ImageSplit import *
from retinex import *
from vignetting_correction import *
from histogram import *
import cv2
import time
import matplotlib.pyplot as plt
import shutil
from qdialogclass import *

#功能文件
# from utils.automaster import bundleAdjustment
from utils.getDescentImageTinyList2 import getDescentImageTinyListfromOriginData
from imgclass.descentImageClass import DescentImage, load_images_from_folder ,BestDescentImg
from utils.getUseTiePointsforLandingSite import getUseTiePointsforLandingSite, tiePointXMLSaved
from utils.histogramSpecification import  histogramSpecification, histogramSpecificationForList,histogramSpecificationAndMaskForList
from utils.auto_reconstruction import mask_images,run_colmap_automatic_reconstructor
from utils.highandResolutionEstimation import highandResolutionEstimation, highandResolutionEstimation_2, get_KRT_from_txt
from utils.landingSiteTransfer import landingSiteTransfer
from utils.reconstructforDescentImagesandBestBaseImage import convert_colmap_results, convert_txt_and_extract_images, convert_txt_and_extract_images_from_npy
from utils.sieveforBestBaseimgs import match_and_rank_best_images,visualize_matches,image_pair_matching
from utils.sieveforBestBaseimgs import *
from utils.stdout_redirect import *

#图像预处理和粗定位、精定位的子线程类
#预处理都读入灰度图
class Thread_retinex_SSR(QtCore.QThread):
    #信号传递cv2读入的图片
    signal = QtCore.pyqtSignal(np.ndarray) # 实例化一个信号
 
    def __init__(self,img,sigma,path):
        super(Thread_retinex_SSR, self).__init__()
        self.img=img
        self.sigma=sigma
        self.path=path
 
    def run(self):
        img=np.expand_dims(self.img,2)
        img=np.repeat(img,3,2)
        img_ssr=SSR(img,self.sigma)
        cv2.imwrite(self.path+'\pretreat.jpg',img_ssr)
        self.signal.emit(img_ssr)


class Thread_retinex_MSR(QtCore.QThread):
    #信号传递cv2读入的图片
    signal = QtCore.pyqtSignal(np.ndarray) # 实例化一个信号
 
    def __init__(self,img,sigma,path):
        super(Thread_retinex_MSR, self).__init__()
        self.img=img
        self.sigma=sigma
        self.path=path
 
    def run(self):
        img=np.expand_dims(self.img,2)
        img=np.repeat(img,3,2)
        img_msr=MSR(img,self.sigma)
        cv2.imwrite(self.path+'\pretreat.jpg',img_msr)
        self.signal.emit(img_msr)

#渐晕校正子线程
class Thread_vignetting_correction(QtCore.QThread):
    #信号传递cv2读入的图片
    signal = QtCore.pyqtSignal(np.ndarray) # 实例化一个信号
 
    def __init__(self,img,path):
        super(Thread_vignetting_correction, self).__init__()
        self.img=img
        self.path=path
 
    def run(self):
        height, width=self.img.shape[0],self.img.shape[1]
        img=cv2.resize(self.img,(256,256))
        img_cor=vignetting_correction(img)
        img_cor=cv2.resize(img_cor,[width,height])
        cv2.imwrite(self.path+'\pretreat.jpg',img_cor)
        self.signal.emit(img_cor)

class Thread_histogram_guide(QtCore.QThread):
    #信号传递cv2读入的图片
    signal = QtCore.pyqtSignal(np.ndarray) # 实例化一个信号
 
    def __init__(self,img_name,target_name,path):
        super(Thread_histogram_guide, self).__init__()
        self.img_name=img_name
        self.target_name=target_name
        self.path=path
 
    def run(self):
        img=cv2.imread(self.img_name,0)
        img_target=cv2.imread(self.target_name,0)
        img_guide=hist_guide(img,img_target)
        cv2.imwrite(self.path+'\pretreat.jpg',img_guide)
        self.signal.emit(img_guide)

class Thread_histogram_equal(QtCore.QThread):
    #信号传递cv2读入的图片
    signal = QtCore.pyqtSignal(np.ndarray) # 实例化一个信号
 
    def __init__(self,img_name,path):
        super(Thread_histogram_equal, self).__init__()
        self.img_name=img_name
        self.path=path
 
    def run(self):
        img=cv2.imread(self.img_name,0)
        img_equal=cv2.equalizeHist(img)
        cv2.imwrite(self.path+'\pretreat.jpg',img_equal)
        self.signal.emit(img_equal)


# 限制对比度的自适应阈值均衡化
class Thread_histogram_adaptive_equal(QtCore.QThread):
    #信号传递cv2读入的图片
    signal = QtCore.pyqtSignal(np.ndarray) # 实例化一个信号
 
    def __init__(self,img_name,limit,gridsize,path):
        super(Thread_histogram_adaptive_equal, self).__init__()
        self.img_name=img_name
        self.limit=limit
        self.gridsize=gridsize
        self.path=path
 
    def run(self):
        img=cv2.imread(self.img_name,0)
        clahe = cv2.createCLAHE(clipLimit=self.limit, tileGridSize=(self.gridsize, self.gridsize))
        img_dst = clahe.apply(img) 
        cv2.imwrite(self.path+'\pretreat.jpg',img_dst )
        self.signal.emit(img_dst)

#粗定位colmap估计内外参、高度估计和下降图像筛选
class colmap_height_estimation(QtCore.QThread):
    #信号传递cv2读入的图片
    signal = QtCore.pyqtSignal(list) # 实例化一个信号
    dir_signal=QtCore.pyqtSignal(str)
    colmap_img_signal=QtCore.pyqtSignal(list)
 
    def __init__(self,inputdir,
    workplace,beginFrame,endFrame,howManyFrameBetween,OverlapAreaRate,mask_signal):
        super(colmap_height_estimation, self).__init__()
        self.inputdir=inputdir
        self.workplace=workplace
        self.beginFrame=beginFrame
        self.endFrame=endFrame
        self.howManyFrameBetween=howManyFrameBetween
        self.OverlapAreaRate = OverlapAreaRate
        self.mask_signal=mask_signal

    def run(self):
        time_llq = time.time()
        inputdir=self.inputdir
        outputdir =  os.path.join(self.workplace,r"output\test\images2")
        textpath =  os.path.join(self.workplace,r"output\test\nameOrigin2Tiny2.txt")
        #掩膜图面路径
        maskpath =  os.path.join(self.workplace,r"data\mask\mask.tif")
        renameHead = "ce_test2_"
        if os.path.exists(outputdir):
            shutil.rmtree(outputdir,ignore_errors=True)
        os.makedirs(outputdir)
        if self.mask_signal:
            mask = cv2.imread(maskpath, 0)
        else:
            img_label_path=os.path.join(inputdir,os.listdir(inputdir)[0])
            img_label=cv2.imread(img_label_path,0)
            mask = 255*np.ones(shape=img_label.shape,dtype=np.uint8)
            #替换掉mask
            cv2.imwrite(maskpath,mask)
        print('开始进行下降图像抽帧')
        #判断掩膜与下降图像shape是否一样
        imglist=sorted(os.listdir(inputdir))
        descent_image1=cv2.imread(os.path.join(inputdir,imglist[0]),0)
        if mask.shape==descent_image1.shape:
        #抽帧加掩膜
            descentimgs_or = getDescentImageTinyListfromOriginData(inputDir=inputdir, outputDir=outputdir, nameTxtPath=textpath,
                                                                howManyFrameBetween=self.howManyFrameBetween, renameHead=renameHead,
                                                                beginFrame=self.beginFrame, endFrame=self.endFrame,
                                                                BinaryMask=mask, useDetectionOverlapArea=True,
                                                                OverlapAreaRate=self.OverlapAreaRate)

            print('下降图像抽帧完毕')
 
            image_path =  os.path.join(self.workplace,r"output\test\images2")
            output_path =  os.path.join( self.workplace,r"output\test\colmap")
            # if os.path.exists(output_path):
            #     shutil.rmtree(output_path,ignore_errors=True)
            # os.makedirs(output_path)
            os.makedirs(output_path, exist_ok=True)

            print('开始进行下降轨迹重建')
            command = f'D:/anaconda/envs/moonlocatefine/python.exe ./locate_descent.py {image_path} {output_path} {output_path}/KRT.npy'
            external_cmd(command, code="utf-8")
            # 读取重建出来的KRT.npy
            KRT_dict = np.load(f'{output_path}/KRT.npy',allow_pickle=True).item()
            descentimgs = descentimgs_or
            for descentimg in descentimgs:
                descentimg_name = descentimg.name
                K_temp, R_temp, T_temp = KRT_dict['K'], KRT_dict[descentimg_name]['R'], KRT_dict[descentimg_name]['T']
                descentimg.set_innerParam(K_temp)
                descentimg.set_R(R_temp)
                descentimg.set_T(T_temp)
            self.dir_signal.emit(f'{output_path}/KRT.npy')
            self.colmap_img_signal.emit(descentimgs)
            self.signal.emit(descentimgs_or)
            print('下降轨迹重建完毕，请关闭对话框')
            print('下降轨迹重建时间', time.time()-time_llq)
        else:
            print('输入的掩膜图像与下降图像大小不一致，请修改掩膜图像后重试')
        
class Thread_descentimg_select(QtCore.QThread):
    signal_bestdescentimg=QtCore.pyqtSignal(list)

    def __init__(self,workspacedir:str,colmap_descentimgs,initHigh,base_resolution,height_dir):
        super(Thread_descentimg_select, self).__init__()
        self.workspacedir=workspacedir
        self.colmap_descentimgs=colmap_descentimgs
        self.initHigh=initHigh
        self.base_resolution=base_resolution
        #保存高度估计结果的路径
        self.height_dir=height_dir

       
    def run(self):
        time_llq = time.time()
        # --------------------------------------------highandResolutionEstimation高度估计-----------------------------#
        #这里应该也是传入初始高度参数
        processbar=[]
        descentimgs=self.colmap_descentimgs
        initHigh = self.initHigh 
        imgIdxforInitHigh = 0
        imgWidth = descentimgs[0].data.shape[0]
        print('开始进行下降图像高度估计')
        # 初始高度initHigh单位为m
        descentimgs=highandResolutionEstimation_2(descentimgs, initHigh=initHigh,
                                    imgIdxforInitHigh=imgIdxforInitHigh,
                                    imgWidth=imgWidth)

        for idx_temp in range(imgIdxforInitHigh, len(descentimgs)-1):
            print(descentimgs[idx_temp].highfromMoon)
    

        heights=[]
        for i in range(imgIdxforInitHigh,len(descentimgs)-1):
            heights.append(descentimgs[i].highfromMoon)
        plt.figure(figsize=(7,7),dpi=100)
        plt.scatter(range(len(heights)),heights,s=50,c='r')
        plt.savefig(os.path.join(self.height_dir,'height.jpg'))
        print('下降图像高度估计完毕')

        
        

        # ----------------------------------------getBestDescentImgfromDescentImgs 选择最佳匹配----------------------------#
        from utils.getBestDescentImgfromDescentImgs import getBestDescentImgfromDescentImgs

        # 最后一帧默认高度为0，不参与计算，故取descentimgs[imgIdxforInitHigh:-1]
        # resolution单位为m
        #这里应该设置为传入参数
        resolutionforBaseimage = self.base_resolution
        print('正在筛选分辨率最接近底图的下降图像.....')
        bestDescentImg = getBestDescentImgfromDescentImgs(descentimgs[imgIdxforInitHigh:-1],
                                                        resolutionforBaseimage=resolutionforBaseimage)
        #输出分辨率最接近底图的下降图像
        #最佳下降图像放在列表末尾，子线程传递list信号
        processbar.append(bestDescentImg)
        self.signal_bestdescentimg.emit(processbar)
        print('下降图像筛选完毕,请关闭对话框')
        print('高度与分辨率估计、下降图像筛选时间', time.time()-time_llq)






class Thread_cyclic_matching(QtCore.QThread):
    #信号传递cv2读入的图片
    signal = QtCore.pyqtSignal(list) # 实例化一个信号
    signal_bestdescentimg=QtCore.pyqtSignal(list)
    def __init__(self,workspacedir:str,bestdescentimg,match_threshold:float,bestdescentimg_name,guide_dir):
        super(Thread_cyclic_matching, self).__init__()
        self.bestdescentimg=bestdescentimg
        self.workspacedir=workspacedir
        self.match_threshold =match_threshold
        self.bestdescentimg_name=bestdescentimg_name
        self.guide_dir=guide_dir
       
    def run(self):
    #先判断最佳下降图像是否为是否手动输入
        time_llq = time.time()
        if self.bestdescentimg.name!= self.bestdescentimg_name:
            #从预处理之后的文件夹读入掩膜和规定化之后的最佳下降图像
            bestdescentimg_path=os.path.join(self.workspacedir,r'output\test\images2',self.bestdescentimg_name)
            bestdescentimg_sub=cv2.imread(bestdescentimg_path,0)
            #实例化最佳下降图像
            bestdescentimg_sub=DescentImage(bestdescentimg_sub)
            bestdescentimg_sub.set_name(self.bestdescentimg_name)
            self.bestdescentimg=bestdescentimg_sub
        processbar=[]
    #-----------------------------match_and_rank_best_images 循环匹配得到最佳底图-----------------------------------#
    # 循环匹配
        print('初始化循环匹配参数')
        superglue = 'indoor'
        resize = [640, 480]
        device = 'cuda'
        match_threshold = self.match_threshold
        useviz = False
        workspacedir=self.workspacedir
        bestdescentimg_list=[]
        self.bestdescentimg.data=self.bestdescentimg.data.astype(np.uint8)
        bestdescentimg_list.append(self.bestdescentimg)
        #传出改变后的的最佳下降图像实例
        refimg = cv2.imread(self.guide_dir, 0)
        outputdir = os.path.join(workspacedir, r"output\test\images_mask2")
        if os.path.exists(outputdir):
            shutil.rmtree(outputdir,ignore_errors=True)
        os.makedirs(outputdir)
        maskpath =  os.path.join(workspacedir,r"data\mask\mask.tif")
        mask=cv2.imread(maskpath,0)
        #对最佳下降图像直方图规定化
        print('对最佳下降图像进行直方图规定化')
        bestdescentimg_list=histogramSpecificationAndMaskForList(srcImageList=bestdescentimg_list, referImage=refimg, binaryMask=mask,
                                          outputdir=outputdir, usePrint=True)
        print('最佳下降图像直方图规定化完毕')
        self.signal_bestdescentimg.emit(bestdescentimg_list)
        baseimgspath = os.path.join(workspacedir,r"output\test\baseimgs_split")
        baseimgs = load_images_from_folder(baseimgspath, imgGray=True)
        outputdir = os.path.join( workspacedir,"output/test/sp/loop_rotate/")
        if os.path.exists(outputdir):
            shutil.rmtree(outputdir,ignore_errors=True)
        os.makedirs(outputdir)
        bestbaseimg_savepath=os.path.join( workspacedir,r"output\test\bestbaseimg")
        if os.path.exists(bestbaseimg_savepath):
            shutil.rmtree(bestbaseimg_savepath,ignore_errors=True)
        os.makedirs( bestbaseimg_savepath)
        print('最佳下降图像与底图循环匹配')
        bestbaseimg, top5_scores, bestbaseimg_angle = match_and_rank_best_images(bestdescentimg_list[0], baseimgs, outputdir,
                                                                             superglue=superglue, resize=resize,
                                                                             device=device,
                                                                             match_threshold=match_threshold,
                                                                           useviz=useviz)
        # bestbaseimg是descentimg类
        #保存旋转后的最佳底图
        cv2.imwrite(os.path.join(bestbaseimg_savepath,bestbaseimg.name+'.png'), (bestbaseimg.data).astype(np.uint8))
        bestbaseimg.set_angle(bestbaseimg_angle)
        print("best Matches:",top5_scores[0])
        processbar.append(bestbaseimg)
        #传出旋转后的最佳底图实例
        self.signal.emit(processbar)
        print('最佳下降图像与底图循环匹配完毕，请关闭对话框')
        print('最佳下降图像与底图循环匹配时间', time.time()-time_llq)

class Thread_landsite_transfer(QtCore.QThread):
    #信号传递cv2读入的图片
    signal = QtCore.pyqtSignal(list) # 实例化一个信号
    signal_bestbaseimg = QtCore.pyqtSignal(list)
    signal_bestdescentimg=QtCore.pyqtSignal(list)
 
    def __init__(self,workspacedir:str,bestdescentimg
                 ,bestbaseimg,match_threshold:float,df:int,pcs,gcs,geo,position_dir,bestbaseimg_name,bestbaseimg_angle:int
                 ,bestdescentimg_name,guide_dir):
        super(Thread_landsite_transfer, self).__init__()
        self.bestdescentimg=bestdescentimg
        self.workspacedir=workspacedir
        self.bestbaseimg=bestbaseimg
        self.match_threshold=match_threshold
        self.df=df#底图分割尺度
        self.pcs=pcs
        self.gcs=gcs
        self.geo=geo
        self.position_dir=position_dir
        self.bestbaseimg_name=bestbaseimg_name
        self.bestbaseimg_angle=bestbaseimg_angle
        self.bestdescentimg_name=bestdescentimg_name
        self.guide_dir=guide_dir
        
    #---------------------------------landingSiteTransfer逐帧传递--------------------------------#

    def run(self):
        #判断是否为手动输入最佳底图
        #并传出信号更新bestbaseimg
        time_llq = time.time()
        bestbaseimg_list=[]
        bestdescentimg_list=[]
        if self.bestbaseimg.name!=self.bestbaseimg_name:
            #把手动选择的最佳底图放在对应文件夹下
            bestbaseimg_savepath=os.path.join(self.workspacedir,r"output\test\bestbaseimg")
            if os.path.exists(bestbaseimg_savepath):
                shutil.rmtree(bestbaseimg_savepath,ignore_errors=True)
            os.makedirs(bestbaseimg_savepath)
            #从split读入手动输入的底图
            path=os.path.join(self.workspacedir,r"output\test\baseimgs_split",self.bestbaseimg_name+'.png')
            bestbaseimg_sub=cv2.imread(path,0)
            #自动旋转
            bestbaseimg_sub=rotate(bestbaseimg_sub,self.bestbaseimg_angle)
            #旋转后的底图替换掉原来的最佳底图
            output_path=os.path.join(bestbaseimg_savepath,self.bestbaseimg_name+'.png')
            cv2.imwrite(output_path,bestbaseimg_sub)
            bestbaseimg_sub=DescentImage(bestbaseimg_sub)
            #name不能有扩展名
            #主要需要name和angle属性
            bestbaseimg_sub.set_name(self.bestbaseimg_name)
            bestbaseimg_sub.set_angle(self.bestbaseimg_angle)
            #替换bestbaseimg
            self.bestbaseimg=bestbaseimg_sub
            bestbaseimg_list.append(bestbaseimg_sub)
        #判断最佳下降图像是否为手动输入
        if self.bestdescentimg.name!=self.bestdescentimg_name:
            path=os.path.join(self.workspacedir,r"output\test\images2",self.bestdescentimg_name)
            bestdescentimg_sub=cv2.imread(path,0)
            bestdescentimg_sub=DescentImage(bestdescentimg_sub)
            bestdescentimg_sub.set_name(self.bestdescentimg_name)
            #替换bestdescentimg属性
            bestdescentimg_list.append(bestdescentimg_sub)
            outputdir = os.path.join(self.workspacedir, r"output\test\images_mask2")
            if os.path.exists(outputdir):
                shutil.rmtree(outputdir,ignore_errors=True)
            os.makedirs(outputdir)
            maskpath =  os.path.join(self.workspacedir,r"data\mask\mask.tif")
            mask=cv2.imread(maskpath,0)
            refimg=cv2.imread(self.guide_dir,0)
            print('对最佳下降图像进行直方图规定化')
            #输出规定化后的列表
            bestdescentimg_list=histogramSpecificationAndMaskForList(srcImageList=bestdescentimg_list, referImage=refimg, binaryMask=mask,
                                            outputdir=outputdir, usePrint=True)
            print('最佳下降图像直方图规定化完毕')
            #修改bestdescentimg属性
            self.bestdescentimg=bestdescentimg_list[0]
        self.signal_bestdescentimg.emit(bestdescentimg_list)
        self.signal_bestbaseimg.emit(bestbaseimg_list)
        landsite=[]
        workspacedir=self.workspacedir
        #重新读入抽帧和掩膜之后的下降图像
        dirpath =os.path.join(workspacedir, "output/test/images2")
        descentimgs = load_images_from_folder(dirpath, imgGray=True)
        t=time.localtime()
        year=t.tm_year
        mon=t.tm_mon
        day=t.tm_mday
        hour=t.tm_hour
        minu=t.tm_min
        name=str(year)+'_'+str(mon)+'_'+str(day)+'_'+str(hour)+'_'+str(minu)+'_'
        outputdir = os.path.join(workspacedir, "output/test/sp",name+'landingSite')  # 输出目录
        os.makedirs(outputdir)
        superglue = 'indoor'  # SuperGlue配置
        resize = [1280, 960]  # [640, 480]  # 重新调整图像的大小
        device = 'cuda'  # 运行设备
        #这里设置为传入参数
        match_threshold = self.match_threshold  # 匹配阈值
        useviz = True  # 是否进行可视化
        #获取的是抽帧和预处理之后的所有下降图像
        bestdescentimg=self.bestdescentimg
        print('输入的最佳下降图像为:'+bestdescentimg.name)
        bestbaseimg=self.bestbaseimg
        print('输入的最佳底图为:'+bestbaseimg.name+'.png')
        print('输入最佳底图的旋转角度为:'+str(bestbaseimg.angle)+'°')
        print('开始进行着陆点位置传递')
        new_x, new_y = landingSiteTransfer(
            bestbaseimg, bestdescentimg, descentimgs,
            outputdir, superglue, resize, device,
            match_threshold, useviz
        )
        #读入原始未经旋转的最佳底图
        # bestbaseimg是一个实例
        bestbaseimg_splitpath=os.path.join(self.workspacedir,r'output\test\baseimgs_split', bestbaseimg.name+'.png')
        bestbaseimg_or=cv2.imread(bestbaseimg_splitpath,0)
        org_shape=bestbaseimg_or.shape
        #旋转前最佳底图尺寸
        shape= bestbaseimg.data.shape
        #获取原图顺时针旋转的角度
        angle=bestbaseimg.angle
        #逆时针旋转，得到在最佳底图的像素坐标
        x_or,y_or=derotate_pt(org_shape,shape,[new_x,new_y],angle)
        print('着陆点在旋转后的最佳底图上的像素坐标为:({},{})'.format(new_x,new_y))
        print('着陆点在旋转前的最佳底图上的像素坐标为:({},{})'.format(x_or,y_or))
        t=time.localtime()
        with open(f"{outputdir}/coarse_xy.txt", 'w') as file:
            file.write(str(x_or)+' '+str(y_or))
        #输出像素坐标
        bestbaseimg_or=np.expand_dims(bestbaseimg_or,2)
        bestbaseimg_or=np.repeat(bestbaseimg_or,3,2)
        cv2.circle(bestbaseimg_or,(int(x_or),int(y_or)),5,(0,0,255),-1)
        cv2.imwrite(os.path.join(self.position_dir,'coarse_position.jpg'),bestbaseimg_or)
        baseimg_name=bestbaseimg.name
        num_h=baseimg_name.split('_')[1]
        num_h=int(num_h)
        num_w=baseimg_name.split('_')[2]
        num_w=int(num_w)
        #还原完整底图上的像素坐标
        x_or=x_or+(self.df//2)*num_w
        y_or=y_or+(self.df//2)*num_h

        #x_or是列号，y_or是行号，注意输入顺序
        x_tou,y_tou=CoordTransf(y_or,x_or,self.geo)
        #返回经纬度
        lon,lat=geo2lonlat(self.pcs,self.gcs,x_tou,y_tou)
        #经纬度保留六位小数
        lon=format(lon,'.6f')
        lat=format(lat,'.6f')
        #输出最终的粗定位经纬度为txt文件
        with open(f"{outputdir}/coarse_position.txt", 'w') as file:
            file.write('粗定位着陆点月球经度为：'+str(lon)+'°\n')
            file.write('粗定位着陆点月球纬度为：'+str(lat)+'°')
        #软件运行时提示输出
        print('着陆点在月球的粗定位坐标经度为:{}°,纬度为{}°'.format(lon,lat))
        landsite.append(lon)
        landsite.append(lat)
        self.signal.emit(landsite)
        print('着陆点位置传递完毕，请关闭对话框')
        print('粗定位着陆点传递时间', time.time()-time_llq)




class Thread_extrinsic_estimate(QtCore.QThread):
    signal = QtCore.pyqtSignal(str) # 实例化一个信号
    signal_ccimgs=QtCore.pyqtSignal(list)
 
    def __init__(self,workspacedir,bestbaseimg_name):
        super(Thread_extrinsic_estimate, self).__init__()
        self.workspacedir=workspacedir
        self.bestbaseimg_name=bestbaseimg_name

    def run(self):
        time_llq = time.time()
        workspacedir=self.workspacedir
        #抽帧和预处理后的下降图像和最佳底图移入同一文件夹下
        image_path = os.path.join(workspacedir, "data/images+bestbaseimg")
        #清空重建输入文件夹
        if os.path.exists(image_path):
            shutil.rmtree(image_path,ignore_errors=True)
        os.makedirs(image_path)
        dirpath =os.path.join(workspacedir,r'output\test\images2')
        #复制粗定位抽帧和掩膜预处理之后的下降图像到指定文件夹
        img_list=os.listdir(dirpath)
        for i in img_list:
            shutil.copyfile(os.path.join(dirpath,i),os.path.join(image_path,i))
        bestbaseimg_path=os.path.join(self.workspacedir,r"output\test\bestbaseimg",self.bestbaseimg_name+'.png')
        #复制最佳底图到精定位输入文件夹下
        shutil.copyfile(bestbaseimg_path,os.path.join(image_path,self.bestbaseimg_name+'.png'))
        #存储在精定位测试文件夹下
        output_path = os.path.join(workspacedir, "output/test_precise/colmap")
        os.makedirs(output_path, exist_ok=True) 


        # 在下降图里添加底图，输出底图外参
        command = f'D:/anaconda/envs/moonlocatefine/python.exe ./locate_descent.py {image_path} {output_path} {output_path}/KRT.npy --bottom'
        external_cmd(command, code="utf-8")
        cams_path_precise=f'{output_path}/KRT.npy'  # 新的外参文件
        #把图像放入cc_input文件夹下
        photosDirPath = os.path.join(workspacedir, "output/test_precise/ccinput_images")
        if os.path.exists(photosDirPath):
            shutil.rmtree(photosDirPath,ignore_errors=True)
        os.makedirs(photosDirPath)
        images_list=os.listdir(image_path)
        for img in images_list:
            shutil.copyfile(os.path.join(image_path,img),os.path.join(photosDirPath,img))
        #读取输入cc的下降图像用来生成控制点
        ccinput_imgs=load_images_from_folder(photosDirPath)[:-1]
        self.signal_ccimgs.emit(ccinput_imgs)
        self.signal.emit(cams_path_precise)
        print('精定位外参估计完毕,请关闭对话框')  
        print('精定位外参估计时间', time.time()-time_llq)

    
        


class Thread_precise_position(QtCore.QThread):
    #信号传递cv2读入的图片
    signal = QtCore.pyqtSignal(list) # 实例化一个信号
 
    def __init__(self,workspacedir,bestbaseimg,descentcontrlimgs,df,pcs,gcs,geo,precise_position_dir,contrlsignal,cams_path):
        super(Thread_precise_position, self).__init__()
        self.workspacedir=workspacedir
        self.descentcontrlimgs=descentcontrlimgs
        #传入旋转后的底图
        self.bestbaseimg=bestbaseimg
        self.df=df
        self.pcs=pcs
        self.gcs=gcs
        self.geo=geo
        self.precise_position_dir=precise_position_dir
        self.contrlsignal=contrlsignal
        self.cams_path=cams_path
    

    def run(self):
        time_llq = time.time()
        descentcontrlimgs=self.descentcontrlimgs
        if self.contrlsignal:
            contrlpoint=[]
            for img in descentcontrlimgs:
                if img.measure['PhotoId']!=0:
                    contrlpoint.append(img.measure)
                else:
                    continue
            contrlpoints=[]
            contrlpoints.append(contrlpoint)
            #控制点信息写入控制点文件
            with open(os.path.join(self.workspacedir, "output/test_precise/cc/Suvery.txt"), 'w') as f1:
                    f1.write(str(contrlpoints))
        landsite=[]
        workspacedir=self.workspacedir
        bestbaseimg=self.bestbaseimg

        #---------------------------------------------集束调整-----------------------------------------#
        camdir=self.cams_path
        txtpath=os.path.join(self.workspacedir, "output/test_precise/cc/txt_cam.txt")
        photosDirPath = os.path.join(workspacedir, "output/test_precise/ccinput_images")
        # convert_txt_and_extract_images(cam_dir=camdir,txt_output_path=txtpath,images_dir=photosDirPath)
        convert_txt_and_extract_images_from_npy(KRT_path=camdir, txt_output_path=txtpath, images_dir=photosDirPath)

        #----------------------getUseTiePointsforLandingSite获取连接点--------------------#
        # user_tie_points_list=getUseTiePointsforLandingSite(descentimgs)
        # xmlpath=os.path.join(workspacedir, "output/test_precise/cc/Suvery.xml")
        #保存连接点文件
        # tiePointXMLSaved(user_tie_points_list,xmlpath)
        #运行cc软件进行集束调整，输出precise_xy.txt
        print('开始进行集束调整，等待中......')
        external_cmd('D:/anaconda/envs/moonlocatefine/python.exe ./locate_precise.py', code="utf-8")
        print('集束调整完毕')
        #读入原始未旋转的底图     
        projectDir =os.path.join(workspacedir, "output/test_precise/cc") 
        bestbaseimg_splitpath=os.path.join(self.workspacedir,r'output\test\baseimgs_split', bestbaseimg.name+'.png')
        bestbaseimg_or=cv2.imread(bestbaseimg_splitpath,0)
        org_shape=bestbaseimg_or.shape
        #旋转前最佳底图尺寸
        shape= bestbaseimg.data.shape
        #获取原图顺时针旋转的角度
        angle=bestbaseimg.angle
        #读入集束调整得到的旋转后最佳底图上的着陆点坐标
        with open(f"{projectDir}/precise_xy.txt", 'r') as f:
            line = f.readlines()
            new_x=line[0].split(' ')[0]
            new_y=line[0].split(' ')[1]
        new_x=float(new_x)
        new_y=float(new_y)
        #逆时针旋转，得到在最佳底图的像素坐标
        x_or,y_or=derotate_pt(org_shape,shape,[new_x,new_y],angle)
        #输出像素坐标
        with open(f"{projectDir}/precise_xyor.txt", 'w') as file:
            file.write(str(x_or)+" "+str(y_or))
        bestbaseimg_or=np.expand_dims(bestbaseimg_or,2)
        bestbaseimg_or=np.repeat(bestbaseimg_or,3,2)
        cv2.circle(bestbaseimg_or,(int(x_or),int(y_or)),2,(0,0,255),2)
        #着陆点示意图保存在预设路径 
        cv2.imwrite(os.path.join(self.precise_position_dir,'precise_position.jpg'),bestbaseimg_or)
        baseimg_name=bestbaseimg.name
        num_h=baseimg_name.split('_')[1]
        num_h=int(num_h)
        num_w=baseimg_name.split('_')[2]
        num_w=int(num_w)
        #还原完整底图上的像素坐标
        x_or=x_or+(self.df//2)*num_w
        y_or=y_or+(self.df//2)*num_h
        #x_or是列号，y_or是行号，注意输入顺序
        x_tou,y_tou=CoordTransf(y_or,x_or,self.geo)
        #返回经纬度
        lon,lat=geo2lonlat(self.pcs,self.gcs,x_tou,y_tou)
        lon=format(lon,'.6f')
        lat=format(lat,'.6f')
        #输出最终的精定位经纬度为txt文件
        with open(f"{projectDir}/precise_position.txt", 'w') as file:
            file.write('精定位着陆点月球经度为：'+str(lon)+'°\n')
            file.write('精定位着陆点月球纬度为：'+str(lat)+'°')
        #软件运行时提示输出
        print('着陆点在月球的精定位坐标经度为:{}°,纬度为{}°'.format(lon,lat))
        landsite.append(lon)
        landsite.append(lat)
        self.signal.emit(landsite)
        print('着陆点精定位完毕，请关闭对话框')
        print('着陆点精定位时间', time.time()-time_llq)





        

                                    


        
            
