import os
import argparse
import numpy as np
from typing import Dict, List
import math
import os
import re
import argparse
import cv2
import numpy as np
import os
import shutil
import struct
from typing import Dict, List, NamedTuple, Tuple


# ============================ read_model.py ============================#
class CameraModel(NamedTuple):
    model_id: int
    model_name: str
    num_params: int


class Camera(NamedTuple):
    id: int
    model: str
    width: int
    height: int
    params: List[float]


class Image(NamedTuple):
    id: int
    qvec: List[float]
    tvec: List[float]
    camera_id: int
    name: str
    point3d_ids: List[int] = []


class Point3D(NamedTuple):
    id: int
    xyz: List[float]
    rgb: List[int]
    error: float
    image_ids: List[int]
    point2d_ids: List[int]


CAMERA_MODELS = {
    CameraModel(0, "SIMPLE_PINHOLE", 3),
    CameraModel(1, "PINHOLE", 4),
    CameraModel(2, "SIMPLE_RADIAL", 4),
    CameraModel(3, "RADIAL", 5),
    CameraModel(4, "OPENCV", 8),
    CameraModel(5, "OPENCV_FISHEYE", 8),
    CameraModel(6, "FULL_OPENCV", 12),
    CameraModel(7, "FOV", 5),
    CameraModel(8, "SIMPLE_RADIAL_FISHEYE", 4),
    CameraModel(9, "RADIAL_FISHEYE", 5),
    CameraModel(10, "THIN_PRISM_FISHEYE", 12)
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model) for camera_model in CAMERA_MODELS])


def read_next_bytes(fid, num_bytes: int, format_char_sequence: str) -> Tuple:
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack("<" + format_char_sequence, data)


def read_cameras_text(path: str) -> Dict[int, Camera]:
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasText(const std::string& path)
        void Reconstruction::ReadCamerasText(const std::string& path)
    """
    model_cameras: Dict[int, Camera] = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elements = line.split()
                cam_id = int(elements[0])
                model = elements[1]
                width = int(elements[2])
                height = int(elements[3])
                params = list(map(float, elements[4:]))
                model_cameras[cam_id] = Camera(cam_id, model, width, height, params)
    return model_cameras


def read_cameras_binary(path: str) -> Dict[int, Camera]:
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    model_cameras: Dict[int, Camera] = {}
    with open(path, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        print("num of cameras")
        print(num_cameras)
        for camera_line_index in range(num_cameras):
            camera_properties = read_next_bytes(fid, 24, "iiQQ")
            cam_id = camera_properties[0]
            print("camera id")
            print(cam_id)

            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = list(read_next_bytes(fid, 8 * num_params, "d" * num_params))
            model_cameras[cam_id] = Camera(cam_id, model_name, width, height, params)
        assert len(model_cameras) == num_cameras
    return model_cameras


def read_images_text(path: str) -> List[Image]:
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesText(const std::string& path)
        void Reconstruction::WriteImagesText(const std::string& path)
    """
    model_images: List[Image] = []
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elements = line.split()
                im_id = int(elements[0])
                qvec = list(map(float, elements[1:5]))
                tvec = list(map(float, elements[5:8]))
                cam_id = int(elements[8])
                image_name = elements[9]
                elements = fid.readline().split()
                point3d_ids = list(map(int, elements[2::3]))
                model_images.append(Image(im_id, qvec, tvec, cam_id, image_name, point3d_ids))
    return model_images


def read_images_binary(path: str) -> List[Image]:
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    model_images: List[Image] = []
    with open(path, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for image_index in range(num_reg_images):
            binary_image_properties = read_next_bytes(fid, 64, "idddddddi")
            im_id = binary_image_properties[0]
            qvec = binary_image_properties[1:5]
            tvec = binary_image_properties[5:8]
            cam_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":  # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points_2d = read_next_bytes(fid, 8, "Q")[0]
            x_y_id_s = read_next_bytes(fid, 24 * num_points_2d, "ddq" * num_points_2d)
            point3d_ids = list(map(int, x_y_id_s[2::3]))
            model_images.append(Image(im_id, qvec, tvec, cam_id, image_name, point3d_ids))
    return model_images


def read_points_3d_text(path: str) -> Dict[int, Point3D]:
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """
    model_points3d: Dict[int, Point3D] = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elements = line.split()
                point_id = int(elements[0])
                xyz = list(map(float, elements[1:4]))
                rgb = list(map(int, elements[4:7]))
                error = float(elements[7])
                image_ids = list(map(int, elements[8::2]))
                point2d_ids = list(map(int, elements[9::2]))
                model_points3d[point_id] = Point3D(point_id, xyz, rgb, error, image_ids, point2d_ids)
    return model_points3d


def read_points3d_binary(path: str) -> Dict[int, Point3D]:
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """
    model_points3d: Dict[int, Point3D] = {}
    with open(path, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for point_line_index in range(num_points):
            binary_point_line_properties = read_next_bytes(fid, 43, "QdddBBBd")
            point_id = binary_point_line_properties[0]
            xyz = list(binary_point_line_properties[1:4])
            rgb = list(binary_point_line_properties[4:7])
            error = binary_point_line_properties[7]
            track_length = read_next_bytes(fid, 8, "Q")[0]
            track_elements = read_next_bytes(fid, 8 * track_length, "ii" * track_length)
            image_ids = list(map(int, track_elements[0::2]))
            point2d_ids = list(map(int, track_elements[1::2]))
            model_points3d[point_id] = Point3D(point_id, xyz, rgb, error, image_ids, point2d_ids)
    return model_points3d


def read_model(path: str, ext: str) -> Tuple[Dict[int, Camera], List[Image], Dict[int, Point3D]]:
    if ext == ".txt":
        model_cameras = read_cameras_text(os.path.join(path, "cameras" + ext))
        model_images = read_images_text(os.path.join(path, "images" + ext))
        model_points_3d = read_points_3d_text(os.path.join(path, "points3D") + ext)
    else:
        model_cameras = read_cameras_binary(os.path.join(path, "cameras" + ext))
        model_images = read_images_binary(os.path.join(path, "images" + ext))
        model_points_3d = read_points3d_binary(os.path.join(path, "points3D") + ext)
    return model_cameras, model_images, model_points_3d


def quaternion_to_rotation_matrix(qvec: List[float]) -> np.ndarray:
    return np.array([
        [1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2]])


def convert_colmap_results(input_folder: str, output_folder: str, num_src_images: int=-1, theta0: float=5, sigma1: float=1,
                           sigma2: float=10, convert_format: bool=False):
    """
    转换COLMAP结果为PatchmatchNet输入

    参数：
    - input_folder (str): 输入文件夹路径
    - output_folder (str): 输出文件夹路径
    - num_src_images (int): 相关图像数量 -1
    - theta0 (float): theta0 参数 5
    - sigma1 (float): sigma1 参数1
    - sigma2 (float): sigma2 参数 10
    - convert_format (bool): 是否转换图像到jpg格式 False
    """

    # 输入输出文件夹有效性检查
    if input_folder is None or not os.path.isdir(input_folder):
        raise Exception("Invalid input folder")

    if output_folder is None or not os.path.isdir(output_folder):
        raise Exception("Invalid output folder")

    image_dir = os.path.join(input_folder, "images")
    model_dir = os.path.join(input_folder, "sparse")
    cam_dir = os.path.join(output_folder, "cams")

    cameras, images, points3d = read_model(model_dir, ".bin")  # 这里假定有一个 `read_model` 函数
    num_images = len(images)

    # 定义相机模型和其对应参数
    param_type = {
        "SIMPLE_PINHOLE": ["f", "cx", "cy"],
        "PINHOLE": ["fx", "fy", "cx", "cy"],
        "SIMPLE_RADIAL": ["f", "cx", "cy", "k"],
        "SIMPLE_RADIAL_FISHEYE": ["f", "cx", "cy", "k"],
        "RADIAL": ["f", "cx", "cy", "k1", "k2"],
        "RADIAL_FISHEYE": ["f", "cx", "cy", "k1", "k2"],
        "OPENCV": ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2"],
        "OPENCV_FISHEYE": ["fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4"],
        "FULL_OPENCV": ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"],
        "FOV": ["fx", "fy", "cx", "cy", "omega"],
        "THIN_PRISM_FISHEYE": ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3", "k4", "sx1", "sy1"]
    }

    # intrinsic
    intrinsic = {}
    for camera_id, cam in cameras.items():
        params_dict = {key: value for key, value in zip(param_type[cam.model], cam.params)}
        if "f" in param_type[cam.model]:
            params_dict["fx"] = params_dict["f"]
            params_dict["fy"] = params_dict["f"]
        i = np.array([
            [params_dict["fx"], 0, params_dict["cx"]],
            [0, params_dict["fy"], params_dict["cy"]],
            [0, 0, 1]
        ])
        intrinsic[camera_id] = i

    # extrinsic
    extrinsic = []
    for i in range(num_images):
        e = np.zeros((4, 4))
        e[:3, :3] = quaternion_to_rotation_matrix(images[i].qvec)
        e[:3, 3] = images[i].tvec
        e[3, 3] = 1
        extrinsic.append(e)

    if num_src_images < 0:
        num_src_images = num_images

    os.makedirs(cam_dir, exist_ok=True)

    for i in range(num_images):
        with open(os.path.join(cam_dir, f"{images[i].name}_cam.txt"), "w") as f:
            f.write("extrinsic\n")
            for j in range(4):
                for k in range(4):
                    f.write(str(extrinsic[i][j, k]) + " ")
                f.write("\n")
            f.write("\nintrinsic\n")
            for j in range(3):
                for k in range(3):
                    f.write(str(intrinsic[images[i].camera_id][j, k]) + " ")
                f.write("\n")
            f.write("\n")

def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))

    sy = math.sqrt(R[2, 1] * R[2, 1] + R[2, 2] * R[2, 2])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def get_r_from_txt(txt_path):
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            if line == 'extrinsic\n':
                nums_temp = re.findall(r'-?\d*\.?\d*e?-?\d+', lines[idx + 1] + lines[idx + 2] + lines[idx + 3])
                assert len(nums_temp) == 12, len(nums_temp)
                assert lines[idx + 4] == '0.0 0.0 0.0 1.0 \n'
                nums_temp = [float(num) for num in nums_temp]
        return np.array(nums_temp).reshape((3, 4))

def convert_txt_and_extract_images(cam_dir: str, txt_output_path: str, images_dir: str):
    """
    转换外参到需要的格式，并提取图片信息

    参数：
    - cam_dir (str): 存储 cam 文件的文件夹路径
    - txt_output_path (str): 输出 txt 文件的路径
    - images_dir (str): 存储所有图片的文件夹路径
    """

    txt_list = os.listdir(cam_dir)
    img_list = os.listdir(images_dir)

    # 清空或创建输出文件
    with open(txt_output_path, 'w') as f:
        f.write('')

    for idx_txt, txt_file in enumerate(txt_list):
        txt_file_path = os.path.join(cam_dir, txt_file)
        print(txt_file_path)

        mat_tempt = get_r_from_txt(txt_file_path)
        R, T = mat_tempt[:3, :3], mat_tempt[:3, 3]
        # colmap所需转换
        R = R.T
        T = -R @ T

        Easting, Northing, Height = T[0], T[1], -T[2]
        Omega, Phi, Kappa = rotationMatrixToEulerAngles(R)

        # 以下为图片复制的示例代码，如果需要，可以取消注释
        # img_num = int(re.findall('\d+', txt_file)[0])
        # for img_name in img_list:
        #     if '_%04d.jpg' % img_num in img_name:
        #         photo_file = img_name
        #         img_old = os.path.join(images_dir, img_name)
        #         img_new = os.path.join('images2', img_name)
        #         shutil.copy(img_old, img_new)

        # 写入输出文件
        with open(txt_output_path, 'a') as f:
            photo_file = os.path.join(images_dir, img_list[idx_txt])
            str_temp = ''
            for i in [Easting, Northing, Height, Omega, Phi, Kappa]:
                str_temp += ' ' + str(i)
            f.write(photo_file + str_temp + '\n')

def convert_txt_and_extract_images_from_npy(KRT_path: str, txt_output_path: str, images_dir: str):
    """
    转换外参到需要的格式，并提取图片信息

    参数：
    - cam_dir (str): 存储 cam 文件的文件夹路径
    - txt_output_path (str): 输出 txt 文件的路径
    - images_dir (str): 存储所有图片的文件夹路径
    """
    img_list = os.listdir(images_dir)
    KRT_dict = np.load(KRT_path, allow_pickle=True).item()
    # 清空或创建输出文件
    with open(txt_output_path, 'w') as f:
        f.write('')
    for img_name in img_list:
        R, T = KRT_dict[img_name]['R'], KRT_dict[img_name]['T']
        T = np.array(T).reshape(-1)
        Easting, Northing, Height = T[0], T[1], -T[2]
        Omega, Phi, Kappa = rotationMatrixToEulerAngles(R)
        # 写入输出文件
        with open(txt_output_path, 'a') as f:
            photo_file = os.path.join(images_dir, img_name)
            str_temp = ''
            for i in [Easting, Northing, Height, Omega, Phi, Kappa]:
                str_temp += ' ' + str(i)
            f.write(photo_file + str_temp + '\n')

# 主函数
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert colmap results into input for PatchmatchNet")
    parser.add_argument("--input_folder", type=str, default="../output/test/colmap/all/dense/0", help="Project input dir.")
    parser.add_argument("--output_folder", type=str, default="../output/test/colmap/all/dense/0", help="Project output dir.")
    parser.add_argument("--num_src_images", type=int, default=-1, help="Related images")
    parser.add_argument("--theta0", type=float, default=5)
    parser.add_argument("--sigma1", type=float, default=1)
    parser.add_argument("--sigma2", type=float, default=10)
    parser.add_argument("--convert_format", action="store_true", default=False,
                        help="If set, convert image to jpg format.")

    args = parser.parse_args()

    # convert_colmap_results(args.input_folder, args.output_folder, args.num_src_images, args.theta0, args.sigma1,
    #                        args.sigma2, args.convert_format)


    convert_colmap_results(args.input_folder, args.output_folder, args.num_src_images, args.theta0, args.sigma1,
                           args.sigma2, args.convert_format)
    convert_txt_and_extract_images('../output/test/colmap/all/dense/0/cams', '../output/test/colmap/all/dense/0/txt_cam.txt', '../output/test/colmap/all/dense/0/images_all')
