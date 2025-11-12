from osgeo import gdal
from osgeo import osr
import numpy as np
import cv2
import os

#反旋转
def derotate_pt(org_shape,shape, pt, angle):
    or_h, or_w = org_shape[0], org_shape[1]
    h,w=shape[0],shape[1]
    x=pt[0]-w/2
    y=pt[1]-h/2
    M = cv2.getRotationMatrix2D(center=(0, 0), angle=angle, scale=1.0)
    x1=M[0,0]*x+M[0,1]*y
    y1=M[1,0]*x+M[1,1]*y
    x_or=x1+or_w/2+M[0,2]
    y_or=y1+or_h/2+M[1,2]

    return x_or,y_or
#像素到投影坐标
def CoordTransf(row, col, GeoTransform):
    #param row: 像素的行号
    #param col: 像素的列号
    XGeo = GeoTransform[0] + GeoTransform[1] * col + row * GeoTransform[2]
    YGeo = GeoTransform[3] + GeoTransform[4] * col + row * GeoTransform[5]
    return XGeo, YGeo

def geo2imgxy(x,y,GeoTransform):
    a=np.array([[GeoTransform[1],GeoTransform[2]],[GeoTransform[4],GeoTransform[5]]])
    b=np.array([x-GeoTransform[0],y-GeoTransform[3]])
    imgx,imgy=np.linalg.solve(a,b)
    #求解方程返回行列号
    #imgx为列号
    #imgy为行号
    return imgx,imgy
#pcs投影坐标系
#gcs地理坐标系
def geo2lonlat(pcs,gcs,x,y):
    ct = osr.CoordinateTransformation(pcs,gcs)
    coords = ct.TransformPoint(x, y)
    return coords[:2]

def lonlat2geo(pcs,gcs,lon,lat):
    ct = osr.CoordinateTransformation(gcs,pcs)
    coords = ct.TransformPoint(lon, lat)
    return coords[:2]

class GRID():
    #读图像文件
    def read_img(self,filename):
        dataset=gdal.Open(filename)       #打开文件
        pcs = osr.SpatialReference()
        pcs.ImportFromWkt(dataset.GetProjection())
        gcs = pcs.CloneGeogCS()
        #pcs和gcs为经纬度坐标系信息
        im_width = dataset.RasterXSize    #栅格矩阵的列数
        im_height = dataset.RasterYSize   #栅格矩阵的行数
        rand=dataset.RasterCount
        #像素坐标与投影坐标的仿射矩阵
        im_geotrans = dataset.GetGeoTransform()  #仿射矩阵
        im_data = dataset.ReadAsArray(0,0,im_width,im_height) #将数据写成数组，对应栅格矩阵
        del dataset
       #返回pcs,gcs
        #投影矩阵和图片数据
        return pcs,gcs,im_geotrans,im_data
 
    #写文件，以写成tif为例
    def write_img(self,filename,im_proj,im_geotrans,im_data):
        #gdal数据类型包括
        #gdal.GDT_Byte, 
        #gdal .GDT_UInt16, gdal.GDT_Int16, gdal.GDT_UInt32, gdal.GDT_Int32,
        #gdal.GDT_Float32, gdal.GDT_Float64
        #im_proj为dataset.GetProjection()坐标系信息
        #im_geotrans = dataset.GetGeoTransform()仿射矩阵
        #判断栅格数据的数据类型
        if 'int8' in im_data.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in im_data.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32
 
        #判读数组维数

        if len(im_data.shape) == 3:
            im_bands, im_height, im_width = im_data.shape
        else:
            im_bands, (im_height, im_width) = 1, im_data.shape

        #创建文件
        driver = gdal.GetDriverByName("GTiff")            #数据类型必须有，因为要计算需要多大内存空间
        dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)
 
        dataset.SetGeoTransform(im_geotrans)              #写入仿射变换参数
        dataset.SetProjection(im_proj)                    #写入投影坐标系
 
        if im_bands == 1:
            dataset.GetRasterBand(1).WriteArray(im_data)  #写入数组数据
        else:
            for i in range(im_bands):
                dataset.GetRasterBand(i+1).WriteArray(im_data[i])

def cal_lonlat(path,row,col):
    pcs,gcs,geotrans,data= GRID().read_img(path)
    x,y=CoordTransf(row,col,geotrans)
    lon,lat=geo2lonlat(pcs,gcs,x,y)
    return lon,lat
def cal_rowcol(path,lon,lat):
    pcs,gcs,geotrans,data= GRID().read_img(path)
    x,y=lonlat2geo(pcs,gcs,lon,lat)
    col,row=geo2imgxy(x,y,geotrans)
    return row,col
#截取小范围底图1024*1024函数
def smalldom(lon,lat,path,out_path):
    pcs,gcs,geotrans,data= GRID().read_img(path)
    h,w=data.shape
    dataset=gdal.Open(path)
    #获取原图坐标系信息       
    proj=dataset.GetProjection()
    row,col=cal_rowcol(path,lon,lat)
    row_0=int(max(row-512,0))
    col_0=int(max(col-512,0))
    row_1=int(min(h,row+512))
    col_1=int(min(col+512,w))
    small_data=data[row_0:row_1,col_0:col_1]
    #更换子图放射矩阵的左上角坐标值
    x_0,y_0=CoordTransf(row_0,col_0,geotrans)
    geotrans_small=list(geotrans)
    geotrans_small[0]=x_0
    geotrans_small[3]=y_0
    geotrans_small=tuple(geotrans_small)
    GRID().write_img(out_path,proj,geotrans_small,small_data)
    return small_data

#已知经纬度计算高程信息
def cal_demh(dem_path,lon,lat):
    '''
    lon为着陆点经度
    lat为着陆点纬度
    dem_path为DEM路径
    '''
    pcs,gcs,im_geotrans,im_data=GRID().read_img(dem_path)
    row,col=cal_rowcol(dem_path,lon,lat)
    h=im_data[int(row),int(col)]
    if abs(h+99999)<1e-6:
        return 0
    #返回值为0时认为该点处高程信息缺失
    else:
        return h

    

if __name__=='__main__':
    path = r"D:\moonlocate\data\ce6\baseimg\ZX28x6_PMRS_DOM_1m.tif"
    _,_,_,dom_data=GRID().read_img(path)
    #五院dom 

    h_or,w_or=dom_data.shape
    print(h_or,w_or)
    dir=r'D:\moonlocate\data\ce6\baseimg\LRO'
    dir1=['low_resolution','high_resolution']
    dir2=['0','45','90','135','180','225','270','315']
    for m in range(2):
        for n in range(8):
            dir_path=os.path.join(dir,dir1[m],dir2[n])
            if os.path.exists(dir_path):
                dom_list=os.listdir(dir_path)
                if dom_list!=[]:
                    for dom in dom_list:
                        #获取文件名
                        dom_name=os.path.splitext(dom)[0]
                        out_dir=os.path.join(dir_path,dom_name)
                        if not os.path.exists(out_dir):
                            os.makedirs(out_dir)
                        #底图路径
                        tar_path=os.path.join(dir_path,dom)
                        #读取底图进行百分比拉伸预处理
                        pcs,gcs,geotrans,data = GRID().read_img(tar_path)
                        mask = data >= 0
                        print("valid pixels", len(data[mask]), len(data[mask])/np.size(data))
                        print('np.max(data[mask]), np.min(data[mask])', np.max(data[mask]), np.min(data[mask]))
                        data_min = np.percentile(data[mask!=0], 0.01)
                        data_max = np.percentile(data[mask!=0], 99.9)
                        data[data > data_max] = data_max
                        data[data < data_min] = data_min
                        print('data_max, data_min', data_max, data_min)
                        data = (data-data_min) / (data_max - data_min) * 255
                        print('done')
                        data[mask==0] = 0
                        data =data.astype(np.uint8)
                        #方便读取投影信息
                        dataset = gdal.Open(tar_path)
                        GRID().write_img(filename=tar_path.replace('.tif', '_processed.tif'), im_proj=dataset.GetProjection(), im_geotrans=geotrans, im_data=data)
                        tar_path_pro=tar_path.replace('.tif', '_processed.tif')
                        #重新读取
                        pcs,gcs,geotrans,data = GRID().read_img(tar_path_pro)
                        #无重叠分割
                        df=2048
                        overlap=2048
                        height,width = data.shape
                        len_1=height//overlap
                        len_2=width//overlap
                        #裁剪底图
                        for i in range(len_1):
                            for j in range(len_2):
                                #判断是否超出边界
                                h=min(height,(i*overlap+df))
                                w=min(width,(j*overlap+df))                    
                                cur_image = data[i*overlap:h,j*overlap:w]
                                if cur_image is None:
                                    continue
                                else:
                                    #在分割前底图的行号
                                    formatted_row = "{:03d}".format(i)
                                    row=str(formatted_row)
                                    #在分割前底图的列号
                                    formatted_col = "{:03d}".format(j)
                                    col=str(formatted_col)
                                    #判断左上角和右下角是否在经纬度范围内
                                    #左上角行列号
                                    row_0=i*overlap
                                    col_0=j*overlap
                                    #计算左上角经纬度
                                    lon_0,lat_0=cal_lonlat(tar_path,row_0,col_0)
                                    #求解该经纬度在五院DOM中的坐标
                                    row_d0,col_d0=cal_rowcol(path,lon_0,lat_0)
                                    row_d0=int(row_d0)
                                    col_d0=int(col_d0)
                                    #右下角行列号
                                    row_1=h
                                    col_1=w
                                    lon_1,lat_1=cal_lonlat(tar_path,row_1,col_1)
                                    #求解该经纬度在五院DOM中的坐标
                                    row_d1,col_d1=cal_rowcol(path,lon_1,lat_1)
                                    row_d1=int(row_d1)
                                    col_d1=int(col_d1)
                                    #判断左上和右下有一个在范围内
                                    if (0<=row_d0<h_or and 0<=col_d0<w_or) or (0<=row_d1<h_or and 0<=col_d1<w_or):
                                        savepath=os.path.join(out_dir,'split_'+row+'_'+col+'.png')
                                        cv2.imwrite(savepath,cur_image)

    # h_or,w_or=dom_data.shape
    # print(h_or,w_or)
    # out_dir = r"D:\moonlocate\data\ce6\baseimg\LRO\high_resolution\temp"
    # tar_path= r"D:\moonlocate\data\ce6\baseimg\LRO\high_resolution\90\M1232891941LE.tif"
    # #读取底图进行百分比拉伸预处理
    # pcs,gcs,geotrans,data = GRID().read_img(tar_path)
    # mask = data >= 0
    # print("valid pixels", len(data[mask]), len(data[mask])/np.size(data))
    # print('np.max(data[mask]), np.min(data[mask])', np.max(data[mask]), np.min(data[mask]))
    # # data_min = np.percentile(data[mask!=0], 0.01)
    # data_min = 0
    # data_max = np.percentile(data[mask!=0], 99.9)
    # data[data > data_max] = data_max
    # data[data < data_min] = data_min
    # print('data_max, data_min', data_max, data_min)
    # data = (data-data_min) / (data_max - data_min) * 255
    # print('done')
    # data[mask==0] = 0
    # data =data.astype(np.uint8)
    # # #方便读取投影信息
    # # dataset = gdal.Open(tar_path)
    # # GRID().write_img(filename=tar_path.replace('.tif', '_processed.tif'), im_proj=dataset.GetProjection(), im_geotrans=geotrans, im_data=data)
    # # tar_path_pro=tar_path.replace('.tif', '_processed.tif')
    # # #重新读取
    # # pcs,gcs,geotrans,data = GRID().read_img(tar_path_pro)
    # #无重叠分割
    # df=2048
    # overlap=2048
    # height,width = data.shape
    # len_1=height//overlap
    # len_2=width//overlap
    # #裁剪底图
    # print('裁剪底图')
    # for i in range(len_1):
    #     if not 29<=i<=33:
    #         continue
    #     for j in range(len_2):
    #         #判断是否超出边界
    #         if not 0<=j<=3:
    #             continue
    #         h=min(height,(i*overlap+df))
    #         w=min(width,(j*overlap+df))                    
    #         cur_image = data[i*overlap:h,j*overlap:w]
    #         if cur_image is None:
    #             continue
    #         else:
    #             #在分割前底图的行号
    #             formatted_row = "{:03d}".format(i)
    #             row=str(formatted_row)
    #             #在分割前底图的列号
    #             formatted_col = "{:03d}".format(j)
    #             col=str(formatted_col)
    #             #判断左上角和右下角是否在经纬度范围内
    #             #左上角行列号
    #             row_0=i*overlap
    #             col_0=j*overlap
    #             #计算左上角经纬度
    #             lon_0,lat_0=cal_lonlat(tar_path,row_0,col_0)
    #             #求解该经纬度在五院DOM中的坐标
    #             row_d0,col_d0=cal_rowcol(path,lon_0,lat_0)
    #             row_d0=int(row_d0)
    #             col_d0=int(col_d0)
    #             #右下角行列号
    #             row_1=h
    #             col_1=w
    #             lon_1,lat_1=cal_lonlat(tar_path,row_1,col_1)
    #             #求解该经纬度在五院DOM中的坐标
    #             row_d1,col_d1=cal_rowcol(path,lon_1,lat_1)
    #             row_d1=int(row_d1)
    #             col_d1=int(col_d1)
    #             #判断左上和右下有一个在范围内
    #             if (0<=row_d0<h_or and 0<=col_d0<w_or) or (0<=row_d1<h_or and 0<=col_d1<w_or):
    #                 savepath=os.path.join(out_dir,'split_'+row+'_'+col+'.png')
    #                 cv2.imwrite(savepath,cur_image)
    path = r"D:\moonlocate\data\ce6\baseimg\ZX28x6_PMRS_DOM_1m.tif"
    lon,lat=cal_lonlat(path,0,0)
    lon_1,lat_1=cal_lonlat(path,17761,48599)
    print(lon,lat,lon_1,lat_1)





