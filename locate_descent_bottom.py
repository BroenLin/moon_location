import sys
import os
import time
import ccmasterkernel
import numpy as np
import time
import argparse

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

def create_cc(bottom_image_path:str,blockAT_descent:str, projectDir:str):
    year, month, day, hour, minute, second, _, _, _ = time.localtime( time.time() )
    current_time = "%s-%s-%s_%s-%s-%s"%(year, month, day, hour, minute, second)
    projectDirPath = os.path.join(projectDir, current_time+'_descent_bottom')  # 项目的输出路径
    blockATPath = os.path.join(projectDirPath, "blockAT.xml")  # blockAT的输出路径
    

    assert not os.path.exists(projectDirPath), "项目已存在"
    print('MasterKernel version %s' % ccmasterkernel.version())
    print('')

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
    project.importBlocks(blockAT_descent)
    block = project.getBlock(0)

    block.setName('block #1')
    block.setDescription('input block')
    assert block.getPhotoDownsamplingRate() is None  # 不进行降采样
    assert os.path.isfile(bottom_image_path)  # 确保底图存在
    lastPhoto = block.getPhotogroups().addPhotoInAutoMode(bottom_image_path)
    if not lastPhoto.pose.center is None:
        block.setPositioningLevel(ccmasterkernel.PositioningLevel.PositioningLevel_georeferenced)


    print('')

    # check block
    print('%s photo(s) added in %s photogroup(s):' % (photogroups.getNumPhotos(), photogroups.getNumPhotogroups()))

    photogroups = project.getBlock(0).getPhotogroups()

    if not block.isReadyForAT():
        if block.reachedLicenseLimit():
            print('Error: Block size exceeds license capabilities.')
        if block.getPhotogroups().getNumPhotos() < 3:
            print('Error: Insufficient number of photos.')
        else:
            print('Error: Missing focal lengths and sensor sizes.')
        sys.exit(0)

    # --------------------------------------------------------------------
    # AT
    # --------------------------------------------------------------------
    blockAT = ccmasterkernel.Block(project)
    project.addBlock(blockAT)
    blockAT.setBlockTemplate(ccmasterkernel.BlockTemplate.Template_adjusted, block)

    # 设置：
    blockAT_ = blockAT.getAT()
    ATSeetings = blockAT_.getSettings()
    MyAdjustmentAndPositioning = ccmasterkernel.bindings.AdjustmentAndPositioning(
        ccmasterkernel.bindings.AdjustmentAndPositioning.Positioning_PositionMetadata)
    ATSeetings.adjustmentConstraints = MyAdjustmentAndPositioning
    # 轨迹重建时，特征点密度选择normal，否则在ce5(20240118ce5_1)上会失败
    MyKeyPointsDensity = ccmasterkernel.bindings.KeyPointsDensity(
        ccmasterkernel.bindings.KeyPointsDensity.KeyPointsDensity_normal)
    ATSeetings.keyPointsDensity = MyKeyPointsDensity
    blockAT.getAT().setSettings(ATSeetings)

    err = project.writeToFile()
    if not err.isNone():
        print(err.message)
        sys.exit(0)

    atSubmitError = blockAT.getAT().submitProcessing()

    if not atSubmitError.isNone():
        print('Error: Failed to submit aerotriangulation.')
        print(atSubmitError.message)
        sys.exit(0)

    print('The aerotriangulation job has been submitted and is waiting to be processed...')

    iPreviousProgress = 0
    iProgress = 0
    previousJobStatus = ccmasterkernel.JobStatus.Job_unknown
    jobStatus = ccmasterkernel.JobStatus.Job_unknown

    while 1:
        jobStatus = blockAT.getAT().getJobStatus()

        if jobStatus != previousJobStatus:
            print(ccmasterkernel.jobStatusAsString(jobStatus))

        if jobStatus == ccmasterkernel.JobStatus.Job_failed or jobStatus == ccmasterkernel.JobStatus.Job_cancelled or jobStatus == ccmasterkernel.JobStatus.Job_completed:
            break

        if iProgress != iPreviousProgress:
            print('%s%% - %s' % (iProgress, blockAT.getAT().getJobMessage()))

        iPreviousProgress = iProgress
        iProgress = blockAT.getAT().getJobProgress()
        time.sleep(1)
        blockAT.getAT().updateJobStatus()
        previousJobStatus = jobStatus

    if jobStatus != ccmasterkernel.JobStatus.Job_completed:
        print('"Error: Incomplete aerotriangulation.')

        if blockAT.getAT().getJobMessage() != '':
            print(blockAT.getAT().getJobMessage())

    print('Aerotriangulation completed.')

    # --------------------------------------------------------------------
    # 输出为xml文件
    # --------------------------------------------------------------------
    myBlockExportOptions = ccmasterkernel.BlockExportOptions()
    # myBlockExportOptions.includeAutomaticTiePoints = True
    # myBlockExportOptions.tiePointsExternalFile = True
    # myBlockExportOptions.exportUndistortedPhotos = True
    blockAT.export(blockATPath, myBlockExportOptions)


    # 将内外参输出为npy文件，并保存为文本
    KRT_path = os.path.join(projectDirPath, 'KRT_dict.npy')
    KRT_dict = {}
    # 以图片名称为key，{'R':, 'T':}为value；以K为key,内参为value
    photogroups = blockAT.getPhotogroups()
    photogroup_descent = photogroups.getPhotogroup(0)
    k_camera = np.matrix([[photogroup_descent.getFocalLength_px(), 0, photogroup_descent.principalPoint.x],
                                [0, photogroup_descent.getFocalLength_px(), photogroup_descent.principalPoint.y],
                                [0, 0, 1]])
    KRT_dict['K'] = k_camera
    for photo_descent in photogroup_descent.getPhotoArray():
        R_descent = Matrix3_np(photo_descent.pose.rotation)
        T_descent = np.matrix([[photo_descent.pose.center.x],
                            [photo_descent.pose.center.y],
                            [photo_descent.pose.center.z]])
        photo_name = os.path.basename(photo_descent.imageFilePath)
        KRT_dict[photo_name] = {'R':R_descent, 'T':T_descent}
    photogroup_bottom = photogroups.getPhotogroup(1)
    photo_bottom = photogroup_bottom.getPhotoArray()[0]
    R_bottom = Matrix3_np(photo_bottom.pose.rotation)
    T_bottom = np.matrix([[photo_bottom.pose.center.x],
                        [photo_bottom.pose.center.y],
                        [photo_bottom.pose.center.z]])
    photo_name = os.path.basename(photo_bottom.imageFilePath)
    KRT_dict[photo_name] = {'R':R_bottom, 'T':T_bottom}
    np.save(KRT_path, KRT_dict)
    with open(KRT_path.replace('.npy', '.txt'), 'w') as f:
        for k in KRT_dict:
            f.write(str(k)+':'+str(KRT_dict[k])+'\n')
    return KRT_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='输入底图路径、下降图像blockAT、项目保存路径、外参保存路径')
    parser.add_argument("bottom_image_path", type=str)
    parser.add_argument("blockAT_descent", type=str)
    parser.add_argument("projectDir", type=str)
    parser.add_argument("KRT_path", type=str)
    args = parser.parse_args()

    bottom_image_path = args.bottom_image_path
    blockAT_descent = args.blockAT_descent
    projectDir = args.projectDir
    KRT_path = args.KRT_path
    os.makedirs(projectDir, exist_ok=True)
    os.makedirs(os.path.dirname(KRT_path), exist_ok=True)
    KRT_dict = create_cc(bottom_image_path=bottom_image_path,blockAT_descent=blockAT_descent, projectDir=projectDir)
    np.save(KRT_path, KRT_dict)
