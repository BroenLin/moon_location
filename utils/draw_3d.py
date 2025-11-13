import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np


def draw_3d(folder_path):
    """针对的是colmap的外参输出，因此有R和T的转化 https://blog.csdn.net/weixin_44120025/article/details/124604229"""
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
    R = np.zeros((3, 3))
    T = np.zeros((3, 1))
    R[0] = np.fromstring(lines[1], sep=' ')[:3]
    R[1] = np.fromstring(lines[2], sep=' ')[:3]
    R[2] = np.fromstring(lines[3], sep=' ')[:3]

    T[0] = np.fromstring(lines[1], sep=' ')[3]
    T[1] = np.fromstring(lines[2], sep=' ')[3]
    T[2] = np.fromstring(lines[3], sep=' ')[3]

    t = np.zeros((3, 1))
    t = -np.dot(np.transpose(R), T)  # 针对colamp

    # 绘图
    x[-1] = t[0][0]
    y[-1] = t[1][0]
    z[-1] = t[2][0]

    filenum = 0
    for file in txt_files:
        with open(file, 'r') as f:
            lines = f.readlines()
        R1 = np.zeros((3, 3))
        T1 = np.zeros((3, 1))
        R1[0] = np.fromstring(lines[1], sep=' ')[:3]
        R1[1] = np.fromstring(lines[2], sep=' ')[:3]
        R1[2] = np.fromstring(lines[3], sep=' ')[:3]

        T1[0] = np.fromstring(lines[1], sep=' ')[3]
        T1[1] = np.fromstring(lines[2], sep=' ')[3]
        T1[2] = np.fromstring(lines[3], sep=' ')[3]

        t = np.zeros((3, 1))
        t = np.dot(np.transpose(R), np.dot((- np.transpose(R1)), T1) - T)  # 针对colamp

        # 绘图
        if filenum >= 0:
            x[filenum] = t[0][0] - x[-1]
            y[filenum] = t[1][0] - y[-1]
            z[filenum] = t[2][0] - z[-1]
        filenum += 1
    return x,y,z


    '''# #轨迹重建绘图
    plt.figure("3D Scatter", facecolor="lightgray")
    ax3d = plt.axes(projection='3d')  # 创建三维坐标
    plt.title('3D Scatter', fontsize=20)
    ax3d.set_xlabel('x', fontsize=14)
    ax3d.set_ylabel('y', fontsize=14)
    ax3d.set_zlabel('z', fontsize=14)
    plt.tick_params(labelsize=10)
    d = x**2 + y**2 + z**2
    print(x)
    ax3d.scatter(x, y, z, s=20, c=d, cmap="jet", marker="o")
    plt.show()'''



def draw_3d_npy(npy_path):
    KRT_dict = np.load(npy_path, allow_pickle=True).item()
    img_names_list = list(KRT_dict.keys())
    img_names_list.remove('K')
    img_names_list = sorted(img_names_list)

    # ssh改
    x = np.zeros(len(img_names_list))
    y = np.zeros(len(img_names_list))
    z = np.zeros(len(img_names_list))

    # 轨迹重建 by ssh
    # 获取最后一帧的外参矩阵
    R = KRT_dict[img_names_list[-1]]['R']
    T = KRT_dict[img_names_list[-1]]['T']
    t = T

    # 绘图
    x[-1] = t[0][0]
    y[-1] = t[1][0]
    z[-1] = t[2][0]

    filenum = 0
    for img_name in img_names_list:
        R1 = KRT_dict[img_name]['R']
        T1 = KRT_dict[img_name]['T']
        t = np.zeros((3, 1))
        t = np.dot(R, T1 - T)

        # 绘图
        if filenum >= 0:
            x[filenum] = t[0][0] - x[-1]
            y[filenum] = t[1][0] - y[-1]
            z[filenum] = t[2][0] - z[-1]
        filenum += 1
    return x,y,z


# txt_root = r"D:\moonlocate\output\test\colmap\temp_cams\1"
# draw_3d(txt_root)



