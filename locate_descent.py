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

def create_cc(photosDirPath:str,projectDir:str, bottom_added: bool):
    year, month, day, hour, minute, second, _, _, _ = time.localtime( time.time() )
    current_time = "%s-%s-%s_%s-%s-%s"%(year, month, day, hour, minute, second)
    if bottom_added:
        projectDirPath = os.path.join(projectDir, current_time+'_descent_bottom')  # 项目的输出路径
    else:
        projectDirPath = os.path.join(projectDir, current_time+'_descent')  # 项目的输出路径
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
    # create block
    # --------------------------------------------------------------------
    block = ccmasterkernel.Block(project)
    project.addBlock(block)

    block.setName('block #1')
    block.setDescription('input block')
    assert block.getPhotoDownsamplingRate() is None  # 不进行降采样
    photogroups = block.getPhotogroups()
    files = os.listdir(photosDirPath)


    for file in files:
        file = os.path.join(photosDirPath, file)
        # add photo, create a new photogroup if needed
        lastPhoto = photogroups.addPhotoInAutoMode(file)
        if lastPhoto is None:
            print('Could not add photo %s.' % file)
            continue
        # upgrade block positioningLevel if a photo with position is found (GPS tag)
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
    # str=ATSeetings.getComponentConstructionModeAsString()

    # print("ATSettings:")
    # # 获取所有属性和对应的值，并打印出来
    # for attr_name in dir(ATSeetings):
    #     if not attr_name.startswith("__"):  # 排除系统内置属性
    #         attr_value = getattr(ATSeetings, attr_name)
    #         print(f"{attr_name}: {attr_value}")
    # print(" ")

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
    if bottom_added:
        photogroup_bottom = photogroups.getPhotogroup(1)
        for photo_bottom in photogroup_bottom.getPhotoArray():
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
    parser = argparse.ArgumentParser(description='输入图片路径、重建项目保存路径、外参保存路径')
    parser.add_argument("photosDirPath", type=str)
    parser.add_argument("projectDir", type=str)
    parser.add_argument("KRT_path", type=str)
    parser.add_argument("--bottom", action='store_true')
    args = parser.parse_args()

    photosDirPath = args.photosDirPath
    projectDir = args.projectDir
    KRT_path = args.KRT_path
    bottom_added = False
    if args.bottom:
        bottom_added = True
    os.makedirs(projectDir, exist_ok=True)
    os.makedirs(os.path.dirname(KRT_path), exist_ok=True)
    KRT_dict = create_cc(photosDirPath=photosDirPath, projectDir=projectDir, bottom_added=bottom_added)
    np.save(KRT_path, KRT_dict)
