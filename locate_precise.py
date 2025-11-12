import os
from utils.automaster import bundleAdjustment
import re
def txt2tie_points_list(txt_file:str):
    with open(txt_file, 'r') as f:
        str_f = f.read()[1:-1]  # 去掉第一层的'['和']'
        str_f_list = str_f.split('],')
        str_f_list = [i.replace('[', '').replace(']', '') for i in str_f_list]
        tie_points_list = []
        for tie_point_str in str_f_list:
            tie_point_list = []
            str_measures = tie_point_str.split('},')
            str_measures = [i.replace('{', '').replace('}', '') for i in str_measures]
            print('str_measures')
            for str_measure in str_measures:
                print(str_measure)
                num =re.findall(r"\d+\.?\d*", str_measure)
                x, y, photoId = float(num[0]), float(num[1]), int(num[2])
                tie_point_list.append({'x':x, 'y':y, 'PhotoId':photoId})
            tie_points_list.append(tie_point_list)
        return tie_points_list
#输出着陆点的精定位经纬度坐标
if __name__=='__main__':
    workspacedir=r'D:\moonlocate'   
    poseTxt=os.path.join(workspacedir, "output/test_precise/cc/txt_cam.txt")# 位姿输入
    #----------------------getUseTiePointsforLandingSite获取连接点--------------------#
    #用户连接点输入文件
    contrltxtpath=os.path.join(workspacedir, "output/test_precise/cc/Suvery.txt")
    contrlpoints=txt2tie_points_list(contrltxtpath)
    #---------------------------------------------集束调整-----------------------------------------#
    photosDirPath = os.path.join(workspacedir, "output/test_precise/ccinput_images")
    projectDir =os.path.join(workspacedir, "output/test_precise/cc")# 项目的输出路径主目录
    #降落点在旋转后底图上的坐标
    pt_bottom=bundleAdjustment(photosDirPath=photosDirPath,poseTxt=poseTxt,projectDir=projectDir,contrlpoints=contrlpoints)
    # print("from locate_precise.py 降落点在底图的二维坐标", pt_bottom.reshape(3)[:2])
    #把坐标保存为txt，在子进程中读入
    with open(f"{projectDir}/precise_xy.txt", 'w') as file:
        file.write(str(pt_bottom[0][0])+" "+str(pt_bottom[1][0]))
