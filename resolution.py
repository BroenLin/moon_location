import numpy as np
import os
import argparse
import math
import pdb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d


# 计算空间分辨率R_g和视野范围R_l
def resoution_compute(h, w=1728, a_fov=45):
    a_fov = a_fov * math.pi / 180
    R_l = 2 * h * math.tan(a_fov / 2)
    R_g = R_l / w
    return R_g, R_l


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Visualize camera extrinsic parameters')
    parser.add_argument('--folder', default="./output/cams",
                        help='Path to the folder containing txt files')
    parser.add_argument('--resoution', default="./output",
                        help='Path to the folder containing resoution_matrix')
    args = parser.parse_args()
    folder_path = args.folder
    resoution_root = args.resoution
    txt_files = [os.path.join(folder_path, file) for file in os.listdir(
        folder_path) if file.endswith('.txt')]
    # 获取最后一帧的外参矩阵
    last_file = txt_files[-1]
    with open(last_file, 'r') as file:
        lines = file.readlines()

    # ssh改
    x = np.zeros(len(txt_files))
    y = np.zeros(len(txt_files))
    z = np.zeros(len(txt_files))

    # 轨迹重建 by ssh
    # print(lines)
    R = np.zeros((3, 3))
    T = np.zeros((3, 1))
    R[0] = np.fromstring(lines[1], sep=' ')[:3]
    R[1] = np.fromstring(lines[2], sep=' ')[:3]
    R[2] = np.fromstring(lines[3], sep=' ')[:3]

    T[0] = np.fromstring(lines[1], sep=' ')[3]
    T[1] = np.fromstring(lines[2], sep=' ')[3]
    T[2] = np.fromstring(lines[3], sep=' ')[3]

    t = np.zeros((3, 1))
    t = -np.dot(np.transpose(R), T)

    # 绘图
    x[-1] = t[0][0]
    y[-1] = t[1][0]
    z[-1] = t[2][0]

    origin0 = t[2][0]

    origin = np.fromstring(lines[3], sep=' ')[3]

    # origin为最后一帧的平移向量的Z
    # 输出高度，分辨率和视场范围的矩阵
    # pdb.set_trace()
    h_matrix = np.zeros((len(txt_files), 3))
    filenum = 0
    for file in txt_files:
        with open(file, 'r') as f:
            lines = f.readlines()

        # ssh改
        # print(lines)
        R1 = np.zeros((3, 3))
        T1 = np.zeros((3, 1))
        R1[0] = np.fromstring(lines[1], sep=' ')[:3]
        R1[1] = np.fromstring(lines[2], sep=' ')[:3]
        R1[2] = np.fromstring(lines[3], sep=' ')[:3]

        T1[0] = np.fromstring(lines[1], sep=' ')[3]
        T1[1] = np.fromstring(lines[2], sep=' ')[3]
        T1[2] = np.fromstring(lines[3], sep=' ')[3]

        t = np.zeros((3, 1))
        # t = -np.dot(np.transpose(R1), T1)
        # t = np.dot(np.transpose(R), t) - np.dot(np.transpose(R), T)
        t = np.dot(np.transpose(R), np.dot((- np.transpose(R1)), T1) - T)
        t1 = t[2][0]

        # 绘图
        if filenum >= 0:
            x[filenum] = t[0][0] - x[-1]
            y[filenum] = t[1][0] - y[-1]
            z[filenum] = t[2][0] - z[-1]
            # 原先的错误结果
            # x[filenum] = T1[0] - T[0]
            # y[filenum] = T1[1] - T[1]
            # z[filenum] = T1[2] - T[2]
            # print(z[filenum])
        t_z = t1 - origin0
        # t_z=t1

        # t_z=np.fromstring(lines[3], sep=' ')[3]-origin
        if filenum == 0:
            k = t_z
        # print(filenum, k)
        # h_matrix[filenum,0],h_matrix[filenum,1],h_matrix[filenum,2]=resoution_compute(t_z/k,15000)
        filenum += 1
        # print(h_matrix)
    # print(x,y,z)

    resoution_path = os.path.join(resoution_root, 'resoution_matrix1.txt')
    index1_path = os.path.join(resoution_root, 'resoution1.2m_index1.txt')
    index2_path = os.path.join(resoution_root, 'resoution5m_index1.txt')

    # #轨迹重建绘图
    plt.figure("3D Scatter", facecolor="lightgray")
    ax3d = plt.gca(projection="3d")  # 创建三维坐标

    plt.title('3D Scatter', fontsize=20)
    ax3d.set_xlabel('x', fontsize=14)
    ax3d.set_ylabel('y', fontsize=14)
    ax3d.set_zlabel('z', fontsize=14)
    plt.tick_params(labelsize=10)
    d = x**2 + y**2 + z**2
    ax3d.scatter(x, y, z, s=20, c=d, cmap="jet", marker="o")

    # plt.show()
    # #保存计算出的相机高度和降落图像分辨率结果
    # np.savetxt(resoution_path,h_matrix)
    # np.savetxt(index1_path,index_1,fmt="%d")
    # np.savetxt(index2_path,index_2,fmt="%d")

    # 高度计算
    vec = np.zeros((3, 1))
    vec[0] = x[-3] - x[-1]
    vec[1] = y[-3] - y[-1]
    vec[2] = z[-3] - z[-1]
    vec_length = np.linalg.norm(vec)
    ratio = 0
    hight = []
    for i in range(len(txt_files) - 1):
        vec_cur = np.zeros(3)
        vec_cur[0] = x[i] - x[-1]
        vec_cur[1] = y[i] - y[-1]
        vec_cur[2] = z[i] - z[-1]
        # 计算向量A在向量B方向上的投影向量长度
        high = np.dot(vec_cur, vec) / (vec_length ** 2) * vec
        # 计算投影向量C的长度
        high_length = np.linalg.norm(high)
        if i == 0:
            ratio = 15000.0 / high_length
        high_length = high_length * ratio
        hight.append(high_length)
        # print(txt_files[i], high_length)

    # #高度估计绘图
    # x = np.arange(1, len(txt_files), 1)
    # #pdb.set_trace()
    # plt.scatter(x,hight,s=50,c='red',edgecolors='blue',alpha=0.9)
    # plt.show()

    # 分辨率计算
    reso = np.zeros((len(txt_files), 2))
    for i in range(len(hight)):
        reso[i, 0], reso[i, 1] = resoution_compute(hight[i])
        print(i, hight[i], reso[i][0])
    # index_1=np.where((reso[:,0]<=30)&(reso[:,0]>=15))
    # index_2=np.where((reso[:,0]<=6) & (reso[:,0]>=3))
    # print('20米分辨率区间:')
    # print(index_1)
    # print('5米分辨率区间:')
    # for i in range(len(index_2)):
    #     print(index_2[i], txt_files[i].split('_')[0][-4:])
    # print(index_2)
