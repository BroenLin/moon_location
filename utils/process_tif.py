# -*- coding: utf-8 -*-
import numpy as np
import cv2
from osgeo import gdal
from osgeo import osr
import matplotlib.pyplot as plt
from PIL import Image


class GRID():
    def read_img(self, filename):
        dataset = gdal.Open(filename)
        pcs = osr.SpatialReference()
        pcs.ImportFromWkt(dataset.GetProjection())
        gcs = pcs.CloneGeogCS()
        im_width = dataset.RasterXSize  # դ����������
        im_height = dataset.RasterYSize  # դ����������
        rand = dataset.RasterCount
        # ����������ͶӰ����ķ������
        im_geotrans = dataset.GetGeoTransform()  # �������
        im_data = dataset.ReadAsArray(0, 0, im_width, im_height)  # ������д�����飬��Ӧդ�����
        del dataset

        return pcs, gcs, im_geotrans, im_data

    # д�ļ�����д��tifΪ��
    def write_img(self, filename, im_proj, im_geotrans, im_data):
        # gdal�������Ͱ���
        # gdal.GDT_Byte,
        # gdal .GDT_UInt16, gdal.GDT_Int16, gdal.GDT_UInt32, gdal.GDT_Int32,
        # gdal.GDT_Float32, gdal.GDT_Float64

        # �ж�դ�����ݵ���������
        if 'int8' in im_data.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in im_data.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32

        # �ж�����ά��

        if len(im_data.shape) == 3:
            im_bands, im_height, im_width = im_data.shape
        else:
            im_bands, (im_height, im_width) = 1, im_data.shape

        # �����ļ�
        driver = gdal.GetDriverByName("GTiff")  # �������ͱ����У���ΪҪ������Ҫ����ڴ�ռ�
        dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)

        dataset.SetGeoTransform(im_geotrans)  # д�����任����
        dataset.SetProjection(im_proj)  # д��ͶӰ

        if im_bands == 1:
            dataset.GetRasterBand(1).WriteArray(im_data)  # д����������
        else:
            for i in range(im_bands):
                dataset.GetRasterBand(i + 1).WriteArray(im_data[i])


if __name__ == "__main__":
    # path = r'E:\0llq\2023moon_location\M1374203631RC_pyr.tif'
    # pcs, gcs, geotrans, data = GRID().read_img(path)
    # print(pcs)
    # print(geotrans)
    # print(data.shape)
    # print(np.max(data), np.min(data))
    # refimg = data
    
    # normalization
    # img_name = 'M1273954657RE'
    '''path = f"/hdd/llq/2023moon_location/{img_name}/{img_name}.tif"
    pcs, gcs, geotrans, data = GRID().read_img(path)
    print(pcs)
    print(geotrans)
    print(data.shape)
    mask = data >= 0
    print("valid pixels", len(data[mask]), len(data[mask])/np.size(data))
    print('np.max(data[mask]), np.min(data[mask])', np.max(data[mask]), np.min(data[mask]))
    data_min = np.percentile(data[mask!=0], 0.01)
    data_max = np.percentile(data[mask!=0], 99.9)
    data[data > data_max] = data_max
    data[data < data_min] = data_min
    print('data_max, data_min', data_max, data_min)
    data = (data-data_min) / (data_max - data_min) * 255
    plt.hist(data[mask].flatten(), bins=256)
    plt.savefig('hist_temp.png')
    plt.close()
    print('done1')
    data[mask==0] = 0
    data =data.astype(np.uint8)
    cv2.imwrite('temp.png', data)'''

    # rewrite tif
    path = r'data/ce6/baseimg/BX28x6_PMRS_DOM_1m.tif'
    pcs, gcs, geotrans, data = GRID().read_img(path)
    print(pcs)
    print(geotrans)
    print(data.shape)
    mask = data >= 0
    print("valid pixels", len(data[mask]), len(data[mask])/np.size(data))
    print('np.max(data[mask]), np.min(data[mask])', np.max(data[mask]), np.min(data[mask]))
    data_min = np.percentile(data[mask!=0], 0.01)
    # data_min = np.percentile(data[mask!=0], 0.01)
    data_max = np.percentile(data[mask!=0], 99.9)
    data[data > data_max] = data_max
    data[data < data_min] = data_min
    print('data_max, data_min', data_max, data_min)
    data = (data-data_min) / (data_max - data_min) * 255
    data[mask==0] = 0
    data =data.astype(np.uint8)
    # Image.MAX_IMAGE_PIXELS = None
    # mat = Image.open('temp.png')
    # data = np.array(mat)
    # print('data.shape', data.shape)
    dataset = gdal.Open(path)
    GRID().write_img(filename=path.replace('.tif', '_processed.tif'), im_proj=dataset.GetProjection(), im_geotrans=geotrans, im_data=data)