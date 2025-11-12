"""从blockAT中获取各个图片的K，R，T，控制点的position（三维点），算出其在畸变校正后的底图的坐标
利用单应变换将坐标投影到原底图上"""
import sys
import os
import shutil
import time
from datetime import datetime
import ccmasterkernel
import numpy as np
import cv2
from numpy import linalg as la

projectDir = r'D:\0school\202306_moon_location\code\20231024'  # 项目的输出路径主目录
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
projectDirPath = os.path.join(projectDir, current_time)  # 项目的输出路径
blockPath = os.path.join(projectDirPath, "block_llq.xml")  # block的输出路径
blockATPath = os.path.join(projectDirPath, "blockAT.xml")  # blockAT的输出路径
# importBlock = r"D:\0school\202306_moon_location\code\20230804\2023-08-11_08-48-11\blockAT.xml"  # ce5
# importBlock = r'D:\0school\202306_moon_location\code\20231024\2023-10-17_17-07-12\blockAT.xml' # ce4
# importBlock = r'D:\0school\202306_moon_location\code\20231024\20231026ce5_out\2023-10-26_11-55-33\blockAT.xml'  # ce5
importBlock = r'D:\0school\202306_moon_location\code\20231024\20231026ce4_out\2023-10-26_16-33-13\blockAT.xml'  # ce4


# importBlock = r'D:\0school\202306_moon_location\code\20231024\20231027ce5\20231027-1433\Block_1 - AT - export.xml'  # ce5


def Matrix3_np(matrix3: ccmasterkernel.Matrix3):
    """
    将ccmasterkernel中的Matrix3转为numpy的形式
    """
    np_list = []
    for i in range(3):
        temp = []
        for j in range(3):
            temp.append(np.longdouble(matrix3.getElement(i, j)))
        np_list.append(temp)
    return np.array(np_list).astype(np.longdouble)


def get_intersection_svd(A: np.array, b: np.array):
    # 利用svd求线性方程组的最小二乘解 https://zhuanlan.zhihu.com/p/131097680
    A = A.astype(np.float)
    u, sigma, vt = la.svd(A)
    v = vt.T
    sigma2 = np.zeros(shape=A.shape, dtype=A.dtype)
    for i in range(len(sigma)):
        sigma2[i][i] = 1 / sigma[i]
    sigma2 = sigma2.T
    ut = u.T
    x = np.dot(v, np.dot(sigma2, np.dot(ut, b)))
    return x


def draw_lines(lines: list, image: np.array, color):
    """
    :para lines:列表，元素为[a,b,c]，表示直线ax+by+c=0
    :para image:在该图像上绘制直线
    :para color直线的颜色
    """
    r, c = image.shape[0], image.shape[1]
    for line_list in lines:
        line_points = []
        for i in range(0, c, 10):
            x, y = map(int, [i, -(line_list[2] + line_list[0] * i) / line_list[1]])
            if 0 < y < r:
                line_points.append([x, y])
        for i in range(len(line_points) - 1):
            cv2.line(image, line_points[i], line_points[i + 1], color, 1)


def get_pt_from_reprojection(undistorted_bottom_photo_path, block_input, num_pts_Homography=100) -> np.array:
    """
    Args:
    :param undistorted_bottom_photo_path: 去畸变后的底图路径
    :param block_input: 输入的ccmasterkernel
    :param num_pts_Homography: 计算单应矩阵所用的点对数量
    :return: pt_bottom_list 计算出来的底图上的降落点坐标列表
             np.average(pt_bottom_list, axis=0) 底图上的降落点坐标列表的均值
    """
    # 获取底图对应的K和R，T
    photogroups = block_input.getPhotogroups()
    photogroup_descent = photogroups.getPhotogroup(0)
    photos_descent = photogroup_descent.getPhotoArray()
    photogroup_bottom = photogroups.getPhotogroup(1)
    # print('principalPoint', photogroup_bottom.principalPoint.x, photogroup_bottom.principalPoint.y)
    # print('hasValidFocalLengthData', photogroup_bottom.hasValidFocalLengthData())
    # print('FocalLength', photogroup_bottom.getFocalLength_px())
    k_camera_bottom = np.matrix([[photogroup_bottom.getFocalLength_px(), 0, photogroup_bottom.principalPoint.x],
                                 [0, photogroup_bottom.getFocalLength_px(), photogroup_bottom.principalPoint.y],
                                 [0, 0, 1]])
    # 底图的旋转矩阵和平移向量
    photo_bottom = photogroup_bottom.getPhotoArray()[0]
    R_bottom = Matrix3_np(photo_bottom.pose.rotation)
    T_bottom = np.matrix([[photo_bottom.pose.center.x],
                          [photo_bottom.pose.center.y],
                          [photo_bottom.pose.center.z]])
    # print('photo_bottom', photo_bottom.imageFilePath)

    # 获取最后一帧降落图的连接点
    useful_tiepoints = []
    id_last_descent = len(photos_descent) - 1
    for i in range(block_input.getNumTiePoints()):
        tie_point = block_input.getTiePoint(i)
        id_temp, photoId_temp, x_temp, y_temp = [], [], [], []
        for num_measurement in range(tie_point.getNumMeasurements()):
            measurement = tie_point.getMeasurement(num_measurement)
            photoId_temp.append(measurement.photoId)
        # print('photoId_temp', photoId_temp)
        if id_last_descent in photoId_temp:
            useful_tiepoints.append(tie_point)
    # print('len(useful_tiepoints)', len(useful_tiepoints))

    # 将三维点投影到二维底图
    # https://communities.bentley.com/products/3d_imaging_and_point_cloud_software/f/contextcapture-descartes-pointools-forum/240839/ccmasterkernel-u20-photogroup-functions-pixeltoray-point3dtopixel/786601?focus=true
    pt_bottom_undistort_list = []
    for tie_point in useful_tiepoints:
        # 世界坐标系下的三维点
        Xw = np.array([tie_point.position.x, tie_point.position.y, tie_point.position.z]).reshape(3, 1)
        # 底图相机坐标系下的三维点
        Xb = np.dot(R_bottom, (Xw - T_bottom))
        # 底图上的二维点
        xb = np.dot(k_camera_bottom, Xb).reshape(3)
        xb = xb / xb[0, 2]
        pt_bottom_undistort_list.append(xb)

    # 计算去畸变图与原图之间的单应矩阵
    img_ori = cv2.imread(photo_bottom.imageFilePath, 0)
    img_undistort = cv2.imread(undistorted_bottom_photo_path, 0)
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img_undistort, None)
    kp2, des2 = sift.detectAndCompute(img_ori, None)
    bf = cv2.BFMatcher()
    matches = bf.match(des1, des2)
    # 按照匹配度排序
    distance_list = [m.distance for m in matches]
    arg_sort_distance = np.argsort(distance_list)
    good = []
    for i in range(num_pts_Homography):
        good.append(matches[arg_sort_distance[i]])
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(srcPoints=src_pts, dstPoints=dst_pts)

    # 将去畸变图像上的降落点投影到原图上
    pt_bottom_ori_list = []
    for pt_undistort in pt_bottom_undistort_list:
        src_pt = np.array(pt_undistort).reshape((3, 1))
        dst_pt = np.dot(H, src_pt)
        dst_pt = dst_pt / dst_pt[-1]
        pt_bottom_ori_list.append(dst_pt)
    print('降落点在底图上的二维坐标', np.average(pt_bottom_ori_list, axis=0))
    return pt_bottom_ori_list, np.average(pt_bottom_ori_list, axis=0)


def main():
    if not ccmasterkernel.isLicenseValid():
        print("License error: ", ccmasterkernel.lastLicenseErrorMsg())
        sys.exit(0)

    # --------------------------------------------------------------------
    # create project
    # --------------------------------------------------------------------
    # os.path.basename返回path最后的文件名，
    projectName = os.path.basename(projectDirPath)
    project = ccmasterkernel.Project()
    project.setName(projectName)
    project.setDescription('Automatically generated from python script')
    project.setProjectFilePath(os.path.join(projectDirPath, projectName))
    err = project.writeToFile()
    if not err.isNone():
        print(err.message)
        sys.exit(0)  # 退出
    print('Project %s successfully created.' % projectName)
    print(f"saved in :{projectDirPath}")
    print('')

    # --------------------------------------------------------------------
    # import block
    # --------------------------------------------------------------------
    project.importBlocks(importBlock)
    block = project.getBlock(0)

    # 获取photoId对应的Photo名称，和R，T
    photogroups = block.getPhotogroups()
    photogroup_descent = photogroups.getPhotogroup(0)
    print('principalPoint', photogroup_descent.principalPoint.x, photogroup_descent.principalPoint.y)
    print('hasValidFocalLengthData', photogroup_descent.hasValidFocalLengthData())
    print('FocalLength', photogroup_descent.getFocalLength_px())
    k_camera_descent = np.matrix([[photogroup_descent.getFocalLength_px(), 0, photogroup_descent.principalPoint.x],
                                  [0, photogroup_descent.getFocalLength_px(), photogroup_descent.principalPoint.y],
                                  [0, 0, 1]])
    photogroup_bottom = photogroups.getPhotogroup(1)
    print('principalPoint', photogroup_bottom.principalPoint.x, photogroup_bottom.principalPoint.y)
    print('hasValidFocalLengthData', photogroup_bottom.hasValidFocalLengthData())
    print('FocalLength', photogroup_bottom.getFocalLength_px())
    k_camera_bottom = np.matrix([[photogroup_bottom.getFocalLength_px(), 0, photogroup_bottom.principalPoint.x],
                                 [0, photogroup_bottom.getFocalLength_px(), photogroup_bottom.principalPoint.y],
                                 [0, 0, 1]])
    # 底图的旋转矩阵和平移向量
    photo_bottom = photogroup_bottom.getPhotoArray()[0]
    R_bottom = Matrix3_np(photo_bottom.pose.rotation)
    T_bottom = np.matrix([[photo_bottom.pose.center.x],
                          [photo_bottom.pose.center.y],
                          [photo_bottom.pose.center.z]])
    photos_descent = photogroup_descent.getPhotoArray()
    print('photo_bottom', photo_bottom.imageFilePath)
    print('photo_descent')
    for idx, photo in enumerate(photos_descent):
        print(idx, photo.imageFilePath)

    # 获取底图的畸变参数
    k1, k2, p1, p2, k3 = photogroup_bottom.distortion.k1, \
                         photogroup_bottom.distortion.k2, \
                         photogroup_bottom.distortion.p1, \
                         photogroup_bottom.distortion.p2, \
                         photogroup_bottom.distortion.k3
    print('k1, k2, p1, p2, k3', k1, k2, p1, p2, k3)

    # 获取最后一帧降落图的连接点
    useful_tiepoints = []
    id_last_descent = len(photos_descent) - 1
    for i in range(block.getNumTiePoints()):
        tie_point = block.getTiePoint(i)
        id_temp, photoId_temp, x_temp, y_temp = [], [], [], []
        for num_measurement in range(tie_point.getNumMeasurements()):
            measurement = tie_point.getMeasurement(num_measurement)
            photoId_temp.append(measurement.photoId)
        # print('photoId_temp', photoId_temp)
        if id_last_descent in photoId_temp:
            useful_tiepoints.append(tie_point)
    print('len(useful_tiepoints)', len(useful_tiepoints))

    # 根据三维点直接画投影
    # https://communities.bentley.com/products/3d_imaging_and_point_cloud_software/f/contextcapture-descartes-pointools-forum/240839/ccmasterkernel-u20-photogroup-functions-pixeltoray-point3dtopixel/786601?focus=true
    # img = cv2.imread(photo_bottom.imageFilePath)
    img = cv2.imread('split_002_004_undistort.jpg')
    for tie_point in useful_tiepoints:
        # 世界坐标系下的三维点
        Xw = np.array([tie_point.position.x, tie_point.position.y, tie_point.position.z]).reshape(3, 1)
        # 底图相机坐标系下的三维点
        Xb = np.dot(R_bottom, (Xw - T_bottom))
        # 底图上的二维点
        xb = np.dot(k_camera_bottom, Xb).reshape(3)
        print(type(xb))
        xb = xb / xb[0, 2]
        print(xb)
        cv2.circle(img, center=(int(xb[0, 0]), int(xb[0, 1])), thickness=2, radius=2, color=(0, 0, 255))
    cv2.imwrite('temp_3d_undistort.png', img)

    # 求tie_point在底图上的极线
    '''intersection_list = []
    for idx_temp, tie_point in enumerate(useful_tiepoints[:20]):
        print('idx of useful_tiepoints', idx_temp)
        lines_list = []  # 存储用户连接点在底图上的极线
        print('photoId:', end='')
        for num_measurement in range(tie_point.getNumMeasurements()):
            measurement = tie_point.getMeasurement(num_measurement)
            photo_id = measurement.photoId
            print(photo_id, end=' ')
            R_descent = Matrix3_np(photos_descent[photo_id].pose.rotation)
            T_descent = np.matrix([[photos_descent[photo_id].pose.center.x],
                                   [photos_descent[photo_id].pose.center.y],
                                   [photos_descent[photo_id].pose.center.z]])
            R_descent = R_descent.T
            T_descent = -np.dot(R_descent, T_descent)
            pt_descent = np.array([
                [measurement.imagePosition.x, measurement.imagePosition.y, 1]
            ])  # 降落图像上的连接点
            R_b2d = np.dot(R_descent.T, R_bottom)
            T_b2d = np.dot(R_descent.T, T_bottom) - np.dot(R_descent.T, T_descent)
            print(T_b2d)
            T_b2d_x = np.matrix([[0, -T_b2d[2], T_b2d[1]],
                                 [T_b2d[2], 0, -T_b2d[0]],
                                 [-T_b2d[1], T_b2d[0], 0]])
            print(T_b2d_x)
            exit()
            E = np.dot(T_b2d_x, R_b2d)
            line = np.dot(pt_descent, np.dot(k_camera1.I.T, np.dot(E, k_camera2.I)))
            line_temp = [line[0, 0], line[0, 1], line[0, 2]]
            line = np.reshape(line_temp, (3, 1)).astype(np.longdouble)
            lines_list.append(line)
        print('极线数量', len(lines_list))
        lines_np = np.array(lines_list).reshape(-1, 3)
        A_lines = lines_np[:, :2]
        print(A_lines.shape)
        b_lines = -lines_np[:, 2].reshape(-1, 1)
        intersection = get_intersection_svd(A_lines, b_lines)
        # print(intersection)
        intersection_list.append(intersection)
        print(intersection.reshape(2).astype(int))'''

    '''intersection_list = []
    lines_list = []  # 存储用户连接点在底图上的极线
    # measures_list = [{'PhotoId': 51, 'x': 512.7, 'y': 538.4},
    #                  {'PhotoId': 51, 'x': 513.2, 'y': 539.7},
    #                  {'PhotoId': 51, 'x': 514.1, 'y': 541.1}]
    # measures_list = [
    #                  {'PhotoId': 72, 'x': 512.7, 'y': 538.4},
    #                  {'PhotoId': 71, 'x': 513.2, 'y': 539.7},
    #                  {'PhotoId': 70, 'x': 514.1, 'y': 541.1}
    #                  ]
    measures_list = []
    for i in range(int(2352 / 2 - 3), int(2352 / 2 + 4)):
        for j in range(int(1728 / 2 - 3), int(1728 / 2 + 4)):
            dict_temp = {'PhotoId': id_last_descent, 'x': i, 'y': j}
            measures_list.append(dict_temp)
    print(measures_list)
    R_bottom = R_bottom.T
    for measure in measures_list:
        photo_id = measure['PhotoId']
        R_descent = Matrix3_np(photos_descent[photo_id].pose.rotation)
        T_descent = np.matrix([[photos_descent[photo_id].pose.center.x],
                               [photos_descent[photo_id].pose.center.y],
                               [photos_descent[photo_id].pose.center.z]])
        R_descent = R_descent.T
        print(photo_id)
        print(R_descent)
        print(T_descent)
        pt_descent = np.array([
            [measure['x'], measure['y'], 1]
        ])  # 降落图像上的连接点
        R_b2d = np.dot(R_descent.T, R_bottom)
        T_b2d = np.dot(R_descent.T, T_bottom) - np.dot(R_descent.T, T_descent)
        T_b2d_x = np.matrix([[0, -T_b2d[2], T_b2d[1]],
                             [T_b2d[2], 0, -T_b2d[0]],
                             [-T_b2d[1], T_b2d[0], 0]])
        E = np.dot(T_b2d_x, R_b2d)
        line = np.dot(pt_descent, np.dot(k_camera_descent.I.T, np.dot(E, k_camera_bottom.I)))
        line_temp = [line[0, 0], line[0, 1], line[0, 2]]
        line = np.reshape(line_temp, (3, 1)).astype(np.longdouble)
        lines_list.append(line)
    print('极线数量', len(lines_list))
    lines_np = np.array(lines_list).reshape(-1, 3)
    A_lines = lines_np[:, :2]
    print(A_lines.shape)
    b_lines = -lines_np[:, 2].reshape(-1, 1)
    intersection = get_intersection_svd(A_lines, b_lines)
    # print(intersection)
    intersection_list.append(intersection)
    print(intersection.reshape(2).astype(int))

    # 绘制极线与交点
    # img = cv2.imread(photo_bottom.imageFilePath)
    img = cv2.imread('split_028_002_undistort.jpg')
    draw_lines(lines_list, img, color=(0, 255, 255))
    for intersection in intersection_list:
        cv2.circle(img, center=intersection.reshape(2).astype(int), thickness=1, radius=1, color=(0, 0, 255))
    cv2.imwrite(f'temp_epilines_undistort.png', img)'''

    '''intersection_temp = intersection.reshape((1,1,2))
    # 上述求的极线交点是cc进行相机畸变校正后的底图上的点，需进行反投影
    # 求去畸变后的投影矩阵，获取在原图上的点
    # https://blog.csdn.net/joyopirate/article/details/131900272
    k_camera_bottom_undistort, _ = cv2.getOptimalNewCameraMatrix(cameraMatrix=k_camera_bottom, distCoeffs=np.array([k1, k2, p1, p2, k3]),alpha=1, imageSize=(img.shape[1], img.shape[0]))
    print(k_camera_bottom_undistort)
    map1, map2 = cv2.initUndistortRectifyMap(cameraMatrix=k_camera_bottom, distCoeffs=np.array([k1, k2, p1, p2, k3]),
                                             R=np.eye(3), m1type=cv2.CV_32FC1,
                                             newCameraMatrix=k_camera_bottom,
                                             size=(img.shape[1], img.shape[0]))
    print(map1[1210, 1895], map2[1210, 1895], map1[1895, 1210], map2[1895, 1210])
    cv2.circle(img, center=(int(map1[1210, 1895]), int(map2[1210, 1895])), color=(0, 255, 255), radius=3, thickness=3)
    cv2.circle(img, center=(int(map2[1210, 1895]), int(map1[1210, 1895])), color=(255, 255, 0), radius=3, thickness=3)
    cv2.circle(img, center=(int(map1[1895, 1210]), int(map2[1895, 1210])), color=(0, 255, 0), radius=3, thickness=3)
    cv2.circle(img, center=(int(map2[1895, 1210]), int(map1[1895, 1210])), color=(0, 0, 255), radius=3, thickness=3)
    cv2.imwrite(f'temp2.png', img)
    img2 = cv2.imread('split_028_002.png')
    img2 = cv2.remap(img2, map1, map2, cv2.INTER_CUBIC)  # 重映射
    cv2.imwrite('temp2_remap.png', img2)
    img2 = cv2.imread('split_028_002_undistort.jpg')
    img2 = cv2.remap(img2, map1, map2, cv2.INTER_CUBIC)  # 重映射
    cv2.imwrite('split_028_002_undistort_remap.png', img2)'''

    # --------------------------------------------------------------------
    # 输出为xml文件
    # --------------------------------------------------------------------
    myBlockExportOptions = ccmasterkernel.BlockExportOptions()
    myBlockExportOptions.includeAutomaticTiePoints = True
    myBlockExportOptions.tiePointsExternalFile = True
    myBlockExportOptions.exportUndistortedPhotos = True
    block.export(blockPath, myBlockExportOptions)


if __name__ == '__main__':
    starttime = time.time()
    main()
    endtime = time.time()
    print("Alll time:{:.2f}s".format(endtime - starttime))
    # shutil.rmtree(projectDirPath)
