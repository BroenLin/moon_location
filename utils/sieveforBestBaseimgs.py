from pathlib import Path
import cv2
import matplotlib
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from skimage.metrics import structural_similarity as s_sim
import cv2
import os

from imgclass.descentImageClass import DescentImage, load_images_from_folder
#
# # Print the current working directory before changing it
# print(f"Current working directory before change: {os.getcwd()}")
# # Change the current working directory to the parent diectory
# os.chdir("../")
# # Print the current working directory after changing it
#
# print(f"Current working directory after change: {os.getcwd()}")
from models.matching import Matching


# from models.utils import read_image

def process_resize(w, h, resize):
    """
       处理图像的尺寸调整。

       参数:
       w (int): 原始图像的宽度。
       h (int): 原始图像的高度。
       resize (list): 调整尺寸的目标尺寸。可能有两种格式：[目标尺寸] 或 [目标宽度, 目标高度]。
       """
    assert (len(resize) > 0 and len(resize) <= 2)
    if len(resize) == 1 and resize[0] > -1:
        scale = resize[0] / max(h, w)
        w_new, h_new = int(round(w * scale)), int(round(h * scale))
    elif len(resize) == 1 and resize[0] == -1:
        w_new, h_new = w, h
    else:  # len(resize) == 2:
        w_new, h_new = resize[0], resize[1]

    # Issue warning if resolution is too small or too large.
    if max(w_new, h_new) < 160:
        print('Warning: input resolution is very small, results may vary')
    elif max(w_new, h_new) > 2000:
        print('Warning: input resolution is very large, results may vary')

    return w_new, h_new


def frame2tensor(frame, device):
    """
        将图像帧转换为张量。

        参数:
        frame (numpy.ndarray): 图像帧。
        device (torch.device): 目标设备（CPU或GPU）。
        """
    return torch.from_numpy(frame / 255.).float()[None, None].to(device)


def read_image(path, device, resize, rotation=0, resize_float=False):
    """
        从路径读取并处理图像。

        参数:
        path (str): 图像文件的路径。
        device (torch.device): 目标设备（CPU或GPU）。
        resize (list): 调整尺寸的目标尺寸。
        rotation (int): 图像旋转的角度（度数）。
        resize_float (bool): 是否在调整大小时使用浮点数。
        """
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None, None, None
    w, h = image.shape[1], image.shape[0]
    w_new, h_new = process_resize(w, h, resize)
    scales = (float(w) / float(w_new), float(h) / float(h_new))

    if resize_float:
        image = cv2.resize(image.astype('float32'), (w_new, h_new))
    else:
        image = cv2.resize(image, (w_new, h_new)).astype('float32')

    if rotation != 0:
        image = np.rot90(image, k=rotation)
        if rotation % 2:
            scales = scales[::-1]

    inp = frame2tensor(image, device)
    return image, inp, scales


def read_image_forDescentImage(imgclass: DescentImage, device, resize, rotation=0, resize_float=False):
    """
        从DescentImage类的图像处理图像。

        参数:
        imgclass (DescentImage): DescentImage类的实例。
        device (torch.device): 目标设备（CPU或GPU）。
        resize (list): 调整尺寸的目标尺寸。
        rotation (int): 图像旋转的角度（度数）。
        resize_float (bool): 是否在调整大小时使用浮点数。
        """
    image = imgclass.data
    if image is None:
        return None, None, None
    w, h = image.shape[1], image.shape[0]
    w_new, h_new = process_resize(w, h, resize)
    scales = (float(w) / float(w_new), float(h) / float(h_new))

    if resize_float:
        image = cv2.resize(image.astype('float32'), (w_new, h_new))
    else:
        image = cv2.resize(image, (w_new, h_new)).astype('float32')

    if rotation != 0:
        image = np.rot90(image, k=rotation)
        if rotation % 2:
            scales = scales[::-1]

    inp = frame2tensor(image, device)
    return image, inp, scales


def image_pair_matching(image_path1, image_path2, match_node='outdoor', resize=[640, 480], device='cuda',
                        match_threshold=0.01, save_path=None, useviz=True):
    """
    匹配一对图片并返回关键点匹配。注意匹配点都是resize后的图上坐标

    参数:
        image_path1 (str): 第一张图片的路径。
        image_path2 (str): 第二张图片的路径。
        match_node (str): match_node, 'indoor' 或 'outdoor'。默认值是 'indoor'。
        resize (list): 用于调整图片大小的两个整数的列表 [宽度, 高度]。默认是 [640, 480]。
        device (str): 'cuda' 或 'cpu'，取决于你想在GPU还是CPU上运行模型。默认是 'cuda'。
        match_threshold (float): 关键点匹配的阈值。更低的值将导致更多的匹配。默认是 0.01。

    返回:
        mkpts0 (np.array): 在第一张图片中有匹配的关键点。
        mkpts1 (np.array): 在第二张图片中与 mkpts0 对应的关键点。
        mconf (np.array): mkpts0 和 mkpts1 关键点的匹配置信度分数。
    """
    if resize is None:
        resize = [640, 480]
    config = {
        'extraction': {
            'nms_radius': 4,
            'keypoint_threshold': 0.005,
            'max_keypoints': 1024,
        },
        'match': {
            'weights': match_node,
            'sinkhorn_iterations': 20,
            'match_threshold': match_threshold,
        }
    }

    matching = Matching(config).eval().to(device)

    image0, inp0, scales0 = read_image(Path(image_path1), device, resize)
    image1, inp1, scales1 = read_image(Path(image_path2), device, resize)

    if image0 is None or image1 is None:
        print('Problem reading image pair.')
        return None, None, None

    pred = matching({'image0': inp0, 'image1': inp1})
    pred = {k: v[0].cpu().detach().numpy() for k, v in pred.items()}

    kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
    matches, conf = pred['matches0'], pred['matching_scores0']

    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]
    mconf = conf[valid]

    if useviz:
        visualize_matches(image0, image1, mkpts0, mkpts1, mconf, save_path=save_path)

    return mkpts0, mkpts1, mconf


def image_pair_matching_forImgClass(imageClass1, imageClass2, match_node='outdoor', resize=[640, 480], device='cuda',
                                    match_threshold=0.01, save_path=None, useviz=True):
    """
        匹配一对图片并返回关键点匹配。注意匹配点都是resize后的图上坐标

        参数:
            image_path1 (str): 第一张图片的路径。
            image_path2 (str): 第二张图片的路径。
            match_node (str): 可选 'indoor' 或 'outdoor'。默认值是 'indoor'。
            resize (list): 用于调整图片大小的两个整数的列表 [宽度, 高度]。默认是 [640, 480]。
            device (str): 'cuda' 或 'cpu'，取决于你想在GPU还是CPU上运行模型。默认是 'cuda'。
            match_threshold (float): 关键点匹配的阈值。更低的值将导致更多的匹配。默认是 0.01。

        返回:
            mkpts0 (np.array): 在第一张图片中有匹配的关键点。
            mkpts1 (np.array): 在第二张图片中与 mkpts0 对应的关键点。
            mconf (np.array): mkpts0 和 mkpts1 关键点的匹配置信度分数。
        """
    if resize is None:
        resize = [640, 480]
    config = {
        'extraction': {
            'nms_radius': 4,
            'keypoint_threshold': 0.005,
            'max_keypoints': 1024,
        },
        'match_mode': {
            'weights': match_node,
            'sinkhorn_iterations': 20,
            'match_threshold': match_threshold,
        }
    }

    matching = Matching(config).eval().to(device)

    image0, inp0, scales0 = read_image_forDescentImage(imageClass1, device, resize)
    image1, inp1, scales1 = read_image_forDescentImage(imageClass2, device, resize)

    if image0 is None or image1 is None:
        print('Problem reading image pair.')
        return None, None, None

    pred = matching({'image0': inp0, 'image1': inp1})
    pred = {k: v[0].cpu().detach().numpy() for k, v in pred.items()}

    kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
    matches, conf = pred['matches0'], pred['matching_scores0']

    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]
    mconf = conf[valid]

    if useviz:
        visualize_matches(image0, image1, mkpts0, mkpts1, mconf, save_path=save_path)

    return mkpts0, mkpts1, mconf


def plot_image_pair(imgs, dpi=100, size=6, pad=.5):
    n = len(imgs)
    assert n == 2, 'number of images must be two'
    figsize = (size * n, size * 3 / 4) if size is not None else None
    _, ax = plt.subplots(1, n, figsize=figsize, dpi=dpi)
    for i in range(n):
        ax[i].imshow(imgs[i], cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
        ax[i].get_yaxis().set_ticks([])
        ax[i].get_xaxis().set_ticks([])
        for spine in ax[i].spines.values():  # remove frame
            spine.set_visible(False)
    plt.tight_layout(pad=pad)


def plot_keypoints(kpts0, kpts1, color='w', ps=2):
    ax = plt.gcf().axes
    ax[0].scatter(kpts0[:, 0], kpts0[:, 1], c=color, s=ps)
    ax[1].scatter(kpts1[:, 0], kpts1[:, 1], c=color, s=ps)


def plot_matches(kpts0, kpts1, color, lw=1.5, ps=4):
    fig = plt.gcf()
    ax = fig.axes
    fig.canvas.draw()

    transFigure = fig.transFigure.inverted()
    fkpts0 = transFigure.transform(ax[0].transData.transform(kpts0))
    fkpts1 = transFigure.transform(ax[1].transData.transform(kpts1))

    fig.lines = [matplotlib.lines.Line2D(
        (fkpts0[i, 0], fkpts1[i, 0]), (fkpts0[i, 1], fkpts1[i, 1]), zorder=1,
        transform=fig.transFigure, c=color[i], linewidth=lw)
        for i in range(len(kpts0))]
    ax[0].scatter(kpts0[:, 0], kpts0[:, 1], c=color, s=ps)
    ax[1].scatter(kpts1[:, 0], kpts1[:, 1], c=color, s=ps)


def make_matching_plot(image0, image1, kpts0, kpts1, mkpts0, mkpts1,
                       color, text, path, show_keypoints=False,
                       fast_viz=False, opencv_display=False,
                       opencv_title='matches', small_text=[]):
    plot_image_pair([image0, image1])
    if show_keypoints:
        plot_keypoints(kpts0, kpts1, color='k', ps=4)
        plot_keypoints(kpts0, kpts1, color='w', ps=2)
    plot_matches(mkpts0, mkpts1, color)

    fig = plt.gcf()
    txt_color = 'k' if image0[:100, :150].mean() > 200 else 'w'
    fig.text(
        0.01, 0.99, '\n'.join(text), transform=fig.axes[0].transAxes,
        fontsize=15, va='top', ha='left', color=txt_color)

    txt_color = 'k' if image0[-100:, :150].mean() > 200 else 'w'
    fig.text(
        0.01, 0.01, '\n'.join(small_text), transform=fig.axes[0].transAxes,
        fontsize=5, va='bottom', ha='left', color=txt_color)

    plt.savefig(str(path), bbox_inches='tight', pad_inches=0)
    plt.close()


def visualize_matches(image0, image1, mkpts0, mkpts1, mconf, save_path=None):
    """
    Visualize the matching keypoints between two images.

    Parameters:
        img0 (np.array): The first image array in grayscale.
        img1 (np.array): The second image array in grayscale.
        mkpts0 (np.array): Keypoints in the first image that have a match.
        mkpts1 (np.array): Keypoints in the second image that correspond to mkpts0.
        mconf (np.array): Matching confidence scores for the keypoints in mkpts0 and mkpts1.
        save_path (str, optional): If provided, saves the visualization to this file path. Otherwise, shows the plot.
    """

    # Visualize the matches.
    color = cm.jet(mconf)
    text = [
        'Match',
        'Keypoints: {}:{}'.format(len(mkpts0), len(mkpts1)),
        'Matches: {}'.format(len(mkpts0)),
    ]
    # if rot0 != 0 or rot1 != 0:
    #     text.append('Rotation: {}:{}'.format(rot0, rot1))

    # print(image0.shape,image1.shape)
    make_matching_plot(
        image0, image1, mkpts0, mkpts1, mkpts0, mkpts1, color,
        text, save_path)


def match_and_rank_top5_images(bestdescentimg, baseimgs, outputdir, match_node='outdoor', resize=[640, 480],
                               device='cuda', match_threshold=0.01, useviz=True):
    """
        对（bestdescentimg）和一组（baseimgs）进行匹配，并返回匹配得分最高的前5张图像。

        Parameters:
        bestdescentimg (object): 最佳着陆图像，是一个特定的图像类实例。
        baseimgs (list): 底图图像列表，这些图像将与最佳下行图像进行比较。
        outputdir (str): 输出目录的路径，用于保存匹配结果和分数。
        match_node (str): 可选值有'indoor'、'outdoor'等，默认为'indoor'。
        resize (list): 用于重新调整图像尺寸的两个元素列表，默认为[640, 480]。
        device (str): 运行匹配算法的设备，一般为'cuda'或'cpu'，默认为'cuda'。
        match_threshold (float): 用于匹配的阈值，默认为0.01。
        useviz (bool): 是否进行可视化，默认为True。

        Returns:
        sorted_scores: 返回一个包含最高分数的前5个匹配结果的列表。
        """

    scores = []
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    for idx, baseimg in enumerate(baseimgs):
        # Match the images
        save_path = os.path.join(outputdir, f"{bestdescentimg.name}_{baseimg.name}_match.jpg")
        # print("")
        # baseimg.print_attributes()
        mkpts0, mkpts1, mconf = image_pair_matching_forImgClass(bestdescentimg, baseimg, save_path=save_path,
                                                                match_node=match_node, resize=resize, device=device,
                                                                match_threshold=match_threshold, useviz=useviz)

        if len(mkpts0) > 7:  # Thresholding by number of matches

            # Score storing, can include more metrics as needed
            data = {"BestDescentImg": bestdescentimg.name, "BaseImg": baseimg.name, "Baseimgidx": idx,
                    "MatchNum": len(mkpts0), "AvgConfidence": mconf.mean()}  # , "SSIM": ssim}
            scores.append(data)

    # Sort the scores based on the metric of interest, in this case average confidence of match
    sorted_scores = sorted(scores, key=lambda x: x["AvgConfidence"], reverse=True)[:5]

    # Save these scores to a file
    with open(f"{outputdir}/sorted_scores.txt", 'w') as file:
        for data in sorted_scores:
            # Convert each dictionary to a string line
            line = " ".join([f"{key}:{val}" for key, val in data.items()])
            file.write(line + "\n")

    return sorted_scores


def rotate(image, angle):
    # 顺时针旋转
    if angle == 0:
        return image
    # grab the dimensions of the image and then determine the center
    (h, w) = image.shape[:2]
    (cX, cY) = (w / 2, h / 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH),borderValue=(255,255,255))


def match_and_rank_best_images(bestdescentimg, baseimgs, outputdir, match_mode='outdoor', resize=[640, 480],
                               device='cuda', match_threshold=0.01, useviz=True):
    """
        对最佳着陆图像（bestdescentimg）和一组底图（baseimgs）进行匹配，并返回匹配得分最高的图像。

        Parameters:
        bestdescentimg (BestDescentImg): 最佳着陆图像，是一个特定的图像类实例。
        baseimgs (list): 底图列表，这些图像将与最佳下行图像进行比较。
        outputdir (str): 输出目录的路径，用于保存匹配结果和分数。
        match_mode (str): 可选值有'indoor'、'outdoor'等，默认为'indoor'。
        resize (list): 用于重新调整图像尺寸的两个元素列表，默认为[640, 480]。
        device (str): 运行匹配算法的设备，一般为'cuda'或'cpu'，默认为'cuda'。
        match_threshold (float): 用于匹配的阈值，默认为0.01。
        useviz (bool): 是否进行可视化，默认为True。

        Returns:
        bestbaseimg（DescentImage）: 返回一个匹配得分最高的基础图像对象。
        sorted_scores: 返回一个包含匹配结果的列表。
        """
    scores = []
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    for idx, baseimg in enumerate(baseimgs):
        baseimg_origin_data = baseimg.data
        print('正在进行第{}张分割后的底图与最佳下降图像匹配'.format(idx+1))
        # rotate the image in a clockwise direction
        for angle in [45, 90, 135, 180, 235, 270, 315, 0]:
            baseimg.data = rotate(baseimg_origin_data, angle)
            # Match the images
            save_path = os.path.join(outputdir, f"{bestdescentimg.name}_{baseimg.name}_{angle}_match.jpg")
            # print("")
            # baseimg.print_attributes()
            mkpts0, mkpts1, mconf = image_pair_matching_forImgClass(bestdescentimg, baseimg, save_path=save_path,
                                                                    match_mode=match_mode, resize=resize, device=device,
                                                                    match_threshold=match_threshold, useviz=useviz)

            if len(mkpts0) > 7:  # Thresholding by number of matches

                # Score storing, can include more metrics as needed
                data = {"BestDescentImg": bestdescentimg.name, "BaseImg": baseimg.name, "Baseimgidx": idx,
                        "MatchNum": len(mkpts0), "AvgConfidence": mconf.mean(), 'Angle': angle}  # , "SSIM": ssim}
                scores.append(data)

    # Sort the scores based on the metric of interest, in this case average confidence of match
    sorted_scores = sorted(scores, key=lambda x: x["AvgConfidence"], reverse=True)[:5]

    # Save these scores to a file
    with open(f"{outputdir}/sorted_scores.txt", 'w') as file:
        for data in sorted_scores:
            # Convert each dictionary to a string line
            line = " ".join([f"{key}:{val}" for key, val in data.items()])
            file.write(line + "\n")
    bestid = sorted_scores[0].get("Baseimgidx")
    bestbaseimg = baseimgs[bestid].copy()
    angle_best = sorted_scores[0].get("Angle")
    bestbaseimg_origin_data = bestbaseimg.data
    bestbaseimg.data = rotate(bestbaseimg_origin_data, angle_best)
    return bestbaseimg, sorted_scores, angle_best


if __name__ == "__main__":

    # 循环匹配
    match_mode = 'outdoor'
    resize = [2048,2048]
    device = 'cuda'
    match_threshold = 0.01
    useviz = True

    descent_image1 = './data/ce4/ce4_tiny_02040.jpg'
    img1 = cv2.imread(descent_image1, 0)
    descent_image1 = DescentImage(img1)
    descent_image1.set_attributes(path=descent_image1, originName="ce4_tiny_02040", name="ce4_tiny_02040")
    bestdescentimg = descent_image1.copy()

    baseimgspath = "data/ce4/baseimgs/NAC_DTM_CHANGE4_M1303619844_140CM_split2048_rotate"
    baseimgs = load_images_from_folder(baseimgspath, imgGray=True)
    outputdir = "./output/test/sp/loop_rotate/"

    bestbaseimg, top5_scores = match_and_rank_best_images(bestdescentimg, baseimgs, outputdir,
                                                          match_mode=match_mode, resize=resize, device=device,
                                                          match_threshold=match_threshold, useviz=useviz)
    print("Top 5 Matches:", top5_scores)
