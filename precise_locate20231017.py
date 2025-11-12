'''# 先进行colmap重建
colmap feature_extractor --database_path D:\moonlocate\colmap\llq\test20231017_ce5\database.db --image_path  D:\moonlocate\colmap\llq\ce5_llq_52_20231017
colmap exhaustive_matcher --database_path D:\moonlocate\colmap\llq\test20231017_ce5\database.db
colmap mapper --database_path D:\moonlocate\colmap\llq\test20231017_ce5\database.db --image_path D:\moonlocate\colmap\llq\ce5_llq_52_20231017 --output_path D:\moonlocate\colmap\llq\test20231017_ce5
python utils/colmap_input_moon_sparse.py --input_folder D:\moonlocate\colmap\llq\test20231017_ce5\0  --output_folder "D:\moonlocate\colmap\llq\test20231017_ce5\temp_cams\0"

colmap feature_extractor --database_path D:\moonlocate\colmap\llq\test20231017_ce5\database.db --image_path D:\moonlocate\colmap\llq\ce5_llq_52_20231017 --image_list_path "D:\moonlocate\colmap\llq\test20231017_ce5\missing_img.txt"
colmap vocab_tree_matcher --database_path D:\moonlocate\colmap\llq\test20231017_ce5\database.db --VocabTreeMatching.vocab_tree_path "D:\moonlocate\colmap\vocab_tree_flickr100K_words32K.bin" --VocabTreeMatching.match_list_path "D:\moonlocate\colmap\llq\test20231017_ce5\all_img.txt"
colmap image_registrator --database_path D:\moonlocate\colmap\llq\test20231017_ce5\database.db --input_path "D:\moonlocate\colmap\llq\test20231017_ce5\0" --output_path "D:\moonlocate\colmap\llq\test20231017_ce5\0"
colmap bundle_adjuster --input_path D:\moonlocate\colmap\llq\test20231017_ce5\0 --output_path D:\moonlocate\colmap\llq\test20231017_ce5\0
python utils/colmap_input_moon_sparse.py --input_folder D:\moonlocate\colmap\llq\test20231017_ce5\0  --output_folder "D:\moonlocate\colmap\llq\test20231017_ce5\temp_cams\0"
'''

# from locate_precise import *
from utils.reconstructforDescentImagesandBestBaseImage import convert_txt_and_extract_images
import cv2

if __name__=='__main__':
    '''cc_input_txt = r"D:\moonlocate\colmap\llq\test20231017_ce5\txt_cam.txt"
    convert_txt_and_extract_images(cam_dir=r"D:\moonlocate\colmap\llq\test20231017_ce5\temp_cams\0",
                                   txt_output_path=cc_input_txt,
                                   images_dir=r"D:\moonlocate\colmap\llq\ce5_llq_52_20231017")'''
    
    # 运行D:\moonlocate\test_check_ccblock.py

    img= cv2.imread(r"D:\moonlocate\colmap\llq\ce5_llq_52_20231017\split_028_002.jpg")
    cv2.circle(img, center=(1870, 1223),
            color=(0, 0, 255), thickness=1, radius=1)
    cv2.imwrite(r'D:\moonlocate\colmap\llq\test20231017_ce5\result_cc.png', img)


