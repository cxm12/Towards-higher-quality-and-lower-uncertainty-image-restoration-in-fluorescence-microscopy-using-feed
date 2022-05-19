#!/usr/bin/env python
# coding: utf-8
# DropOut

from __future__ import print_function, unicode_literals, absolute_import, division
import sys
from tifffile import imread, imsave
import os
import glob
import imageio
sys.path.append('../../')

from csbdeep.func_mcx import *
from csbdeep.utils.tf import IS_TF_1, keras
from csbdeep.models import IsotropicFeedbackDropOut, IsotropicCARE, IsotropicCAREDropOut
from csbdeep.utils import normalize


def pearson_distance(vector1, vector2):
    """
    系数的取值总是在-1.0到1.0之间，接近0的变量被成为无相关性，接近1或者-1被称为具有强相关性。
    Calculate distance between two vectors using pearson method
    See more : http://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient
    """
    sum1 = sum(vector1)
    sum2 = sum(vector2)
    
    sum1Sq = sum([pow(v, 2) for v in vector1])
    sum2Sq = sum([pow(v, 2) for v in vector2])
    
    pSum = sum([vector1[i] * vector2[i] for i in range(len(vector1))])
    
    num = pSum - (sum1 * sum2 / len(vector1))
    den = np.sqrt((sum1Sq - pow(sum1, 2) / len(vector1)) * (sum2Sq - pow(sum2, 2) / len(vector1)))
    
    if den.all() == 0: return 0.0
    return 1.0 - num / den


def Variance():
    # a = [[54, 756, 505]]
    testnum = 2  # 30  #
    meanvar = 0
    patchsize = 108  # 256  #
    meanvarlst = []
    
    if testset == 'Isotropic_Retina':
        axes = 'ZCYX'
    elif testset == 'Isotropic_Drosophila':
        axes = 'ZYX'
    
    # # CARE model
    # 将模式转为训练模式，在测试时才能使用dropout!!!
    keras.backend.set_learning_phase(1)
    testset1 = testset  # 'example'  #
    if ('_FB' in modeltype) or ('_srcnn' in modeltype):
        model = IsotropicFeedbackDropOut(config=None, name='epoch200/my_model%s/%s' % (modeltype, testset1),
                                         basedir='models', modeltype=modeltype)
    elif '_dropout' in modeltype:
        model = IsotropicCAREDropOut(config=None, name='epoch200/my_model%s/%s' % (modeltype, testset),
                                     basedir='models', modeltype=modeltype, finetune=True)

    if testset == 'Isotropic_Drosophila':
        subsample = 1  # 0
    elif testset == 'Isotropic_Retina':
        subsample = 10.2
    savepathzyx = './models/epoch200/my_model%s/' % (modeltype) + '/' + testset + '/' + 'result/AllT%d/S%d/' % (testnum, subsample)

    os.makedirs(savepathzyx, exist_ok=True)
    filename = os.listdir(testdatapath)
    imnum = len(filename)
    for i in range(0, imnum):
        name = filename[i][:-4]
        print(name)
        x = imread(testdatapath + name + '.tif')
        # example[35,2,768,768]   # retina [35, 2, 1024, 1024]
        ## Drop: (108, 1352, 532)
        factor = np.ones(x.ndim)
        factor[0] = subsample
    
        print('image axes         =', axes)
        print('image shape        =', x.shape)
        print('Z subsample factor =', subsample)
    
        # ## Apply CARE network to raw image
        # Predict the restored image (image will be successively split into smaller tiles if there are memory issues).
        resultlst = []
        resultlstv = []
        if testset == 'Isotropic_Retina':
            x = x[:, :, :, 200:400]  # x[:, :, 200:400, 200:400]  #
        if testset == 'Isotropic_Drosophila':
            x = x[:, 200:1000, 100:500]
            
        for ti in range(testnum):
            if modeltype == '_dropout':
                restored, bic = model.predict(x, axes, subsample)
                # print('bic.shape = ', bic.shape)  # (360, 800, 800, 2)
                print('restored.shape = ', restored.shape)  # (357, 2, 800, 800)
            else:
                restored = model.predict(x, axes, subsample, n_tiles=(1, 1, 4, 4))
            resultlst.append(restored)
            restoredn = normalize(restored, 0.1, 99.9, clip=True)
            resultlstv.append(restoredn)
            print(f'x = {x.mean()}, restored = {restored.mean()}, restorednorm = {restoredn.mean()}')

        Var0 = np.var(np.array(resultlstv), axis=0)
        imsave(savepathzyx + name + '-noNormVar.tif', Var0)
        v = np.mean(Var0)
        vmax = np.max(Var0)
        meanvarlst.append(v)
        print(f'Image - {name}- STD/STDMax{v, vmax}')
        
        if testset == 'Isotropic_Retina':
            z, c, h, w = restored.shape
        elif testset == 'Isotropic_Drosophila':
            z, h, w = restored.shape
        Mean = np.mean(np.array(resultlst), axis=0)
        Var1 = np.var(np.array(resultlst), axis=0)  # * 255 * 100 Compute the variance along the specified axis.
        NormVar1 = normalize(Var1, 0, 100)  # 0.1, 99.9, clip=True) * 255
        
        if testset == 'Isotropic_Drosophila':
            a = [[505, 168], [574, 154], [792, 9], [244, 19], [496, 237], [56, 37], [650, 234],
                 [407, 115], [1346-400, 205], [935-400, 33]]
            patchnum = 10
        if testset == 'Isotropic_Retina':
            patchnum = 1
            randz = 18  # np.random.randint(0, z - patchsize)  # a[i][0]  #
            randh = 53  # np.random.randint(0, h)  # a[i][1]  #
            randw = 10  # np.random.randint(0, w - patchsize)  # a[i][2]  #
            print('randz, randh, randw = ', randz, randh, randw)
            # randz, randh, randw =  18 53 10
            
        for j in range(patchnum):
            # ## 在整张图向上画框，指出patch的位置
            # if modeltype == '_dropout':
            #     biccrop = drawbox(bic[:, 0, randh, :], x=randz, y=randw, patchsize=patchsize)
            #     savecolorim(savepathzyx + name + '-BiccropCallz%d.png' % j, biccrop)
            # Meancrop = drawbox(Mean[:, 0, randh, :], x=randz, y=randw, patchsize=patchsize)
            # savecolorim(savepathzyx + name + '-MeancropCallz%d.png' % j, Meancrop)
            patchlst = []
            patchlstv = []
            if testset == 'Isotropic_Drosophila':
                randh = a[j][0]  # np.random.randint(0, h)  #
                randw = a[j][1]  # np.random.randint(0, w - patchsize)  #
                print('randh, randw = ', randh, randw)
                # randh, randw =  [1298, 240],[309, 138], [488, 156],594 121,1340 88,600 165,807 193,505 168,574 154,792 9
            for im in resultlst:
                if testset == 'Isotropic_Retina':
                    arr = im[randz:randz + patchsize, :, randh, randw:randw + patchsize]
                if testset == 'Isotropic_Drosophila':
                    arr = im[:, randh, randw:randw + patchsize]  # (1080, 1352, 532)
                patchlst.append(arr)
            for im in resultlstv:
                if testset == 'Isotropic_Retina':
                    arrv = im[randz:randz + patchsize, :, randh, randw:randw + patchsize]
                if testset == 'Isotropic_Drosophila':
                    arrv = im[:, randh, randw:randw + patchsize]  # (1080, 1352, 532)
                patchlstv.append(arrv)
                
            if testset == 'Isotropic_Drosophila':
                savecolorim(savepathzyx + name + '-MeanCallz%d.png' % j, np.squeeze(Mean[:, randh, :]))
                savecolorim1(savepathzyx + name + '-NormVarCallz%d.png' % j, np.squeeze(NormVar1[:, randh, :]))
            if testset == 'Isotropic_Retina':
                savecolorim(savepathzyx + name + '-MeanCallz%d.png' % j, np.squeeze(Mean[:, 0, randh, :]))
                savecolorim1(savepathzyx + name + '-NormVarCallz%d.png' % j, np.squeeze(NormVar1[:, 0, randh, :]))

            if modeltype == '_dropout':
                if testset == 'Isotropic_Drosophila':
                    savecolorim(savepathzyx + name + '-BicC_num%d.png' % j, np.squeeze(bic[:, randh, randw:randw + patchsize]))
                if testset == 'Isotropic_Retina':
                    # print('bic[randz:randz + patchsize, 0, randh, randw:randw + patchsize].shape = ', bic[randz:randz + patchsize, randh, randw:randw + patchsize].shape)
                    savecolorim(savepathzyx + name + '-BicC_num%d.png' % j, bic[randz:randz + patchsize, randh, randw:randw + patchsize, 0])

            Var = np.var(np.array(patchlstv), axis=0)  # * 255 * 100 Compute the variance along the specified axis.
            NormVar = normalize(Var, 0, 100)  # 0.1, 99.9, clip=True) * 255
            Meanpatch = np.mean(np.array(patchlst), axis=0)
            
            # scio.savemat(savepathzyx + name + '-Var%d.mat'%j, {'Var': Var[:, :]})
            # imsave(savepathzyx + name + '-Var.tif', np.expand_dims(Var[:, :], -1))
            # cv2.imwrite(savepathzyx + name + '_NormVar%d.png'%j, np.expand_dims(NormVar[:, :], -1))  # 四舍五入， 有saturation的过程。0~255
            if testset == 'Isotropic_Drosophila':  # Meanpatch.shape =  (1080, 108)
                savecolorim(savepathzyx + name + '-MeanC_num%d.png' % j, Meanpatch)
                savecolorim1(savepathzyx + name + '-NormVarC_num%d.png' % j, NormVar)
            if testset == 'Isotropic_Retina':  # Meanpatch.shape = (108, 2, 108)
                savecolorim(savepathzyx + name + '-MeanC_num%d.png' % j, Meanpatch[:, 0, :])
                savecolorim1(savepathzyx + name + '-NormVarC_num%d.png' % j, NormVar[:, 0, :])
            mean = np.mean(Var)
            print('variance = %.6f' % mean)
            meanvar += mean
        del resultlst, patchlst

    print('%d image, Mean D of Testset %s is %8f' % (imnum, testset, meanvar / imnum))
    return meanvar / imnum


def Variance_Liver():
    testnum = 30  #
    patchsize = 256
    axes = 'ZYX'
    PCCLst = []
    psnrlst = []
    ssimlst = []
    mselst = []
    meanvarlst = []

    if 'dropout' in modeltype:
        keras.backend.set_learning_phase(1)
    model = IsotropicFeedbackDropOut(config=None, name='epoch200/my_model%s/%s' % (modeltype, testset),
                                         basedir='models', modeltype=modeltype)
         
    subsample = 1
    savepathzyx = './models/epoch200/my_model%s/' % (modeltype) + '/' + testset + '/' + 'result/AllT%d/S%d/' % (
        testnum, subsample)
    os.makedirs(savepathzyx, exist_ok=True)
    name = 'input_subsample_8'
    
    y = imread(testdatapath + 'input_subsample_1_groundtruth.tif')  # [301, 752, 752]
    x = imread(testdatapath + 'input_subsample_8.tif')  # [301, 752, 752]
    factor = np.ones(x.ndim)
    factor[0] = subsample
    
    print('image axes         =', axes)
    print('image shape        =', x.shape)
    print('Z subsample factor =', subsample)
    
    # ## Apply CARE network to raw image
    resultlst = []
    resultlstv = []
    for ti in range(testnum):
        if modeltype == '_dropout':
            restored, bic = model.predict(x, axes, subsample)
            print('restored.shape = ', restored.shape)  # (357, 2, 800, 800)
        else:
            restored = model.predict(x, axes, subsample)
        resultlst.append(restored[:, :, :])
        restoredn = normalize(restored, 0.1, 99.9, clip=True)
        resultlstv.append(restoredn[:, :, :])
        print(f'x = {x.mean()}, restored = {restored.mean()}, y = {y.mean()}, restorednorm = {restoredn.mean()}')

    mean = np.mean(np.array(resultlst), axis=0)
    psm, ssmm = compute_psnr_and_ssim(normalize(mean, 0.1, 99.9, clip=True) * 255,
                                      normalize(y, 0.1, 99.9, clip=True) * 255)
    rmse = np.mean(np.square(mean - y), dtype=np.float64)
    Var0 = np.var(np.array(resultlstv), axis=0)
    pcc0 = pearson_distance(Var0.ravel(), mean.ravel())
    imsave(savepathzyx + name + '-noNormVar.tif', Var0)
    v = np.mean(Var0)
    vmax = np.max(Var0)
    meanvarlst.append(v)
    psnrlst.append(psm)
    ssimlst.append(ssmm)
    mselst.append(rmse)
    print(f'Image - {name}- PSNR/SSIM{psm, ssmm} /PCC/MSE {pcc0, rmse} STD/STDMax{v, vmax}')

    # z, h, w = restored.shape
    patchnum = 10  # 1  #
    meanvar = 0
    meandf = 0
    meanKLdiv= 0
    a = [[18, 240, 309], [10, 488, 399], [28, 594, 121], [44, 465, 105], [24, 375, 330], [24, 387, 165],
         [39, 466, 193], [40, 574, 154], [18, 521, 434], [32, 244, 19]]
    Mdfmaplst = []
    Varmaplst = []
   
    for j in range(patchnum):
        randz = a[j][0]
        randh = a[j][1]
        randw = a[j][2]

        patchlst = []
        patchlstv = []
        patchdflst = []
        for im in resultlst:
            arr = im[randz:randz + patchsize, randh, randw:randw + patchsize]
            patchlst.append(arr)
            arrdf = np.abs(
                normalize(im, 0.1, 99.9, clip=True)[randz:randz + patchsize, randh, randw:randw + patchsize]
                - normalize(y, 0.1, 99.9, clip=True)[randz:randz + patchsize, randh, randw:randw + patchsize])
            patchdflst.append(arrdf)
        for im in resultlstv:
            arr = im[randz:randz + patchsize, randh, randw:randw + patchsize]
            patchlstv.append(arr)

        if modeltype == '_dropout':
            savecolorim(savepathzyx + name + '-LRC_num%d.png' % j, np.squeeze(x[randz:randz + patchsize, randh, randw:randw + patchsize]))
            savecolorim(savepathzyx + name + '-GTC_num%d.png' % j, np.squeeze(y[randz:randz + patchsize, randh, randw:randw + patchsize]))
            
        Meanp = np.mean(np.array(patchlst), axis=0)
        Varp = np.var(np.array(patchlstv), axis=0)
        vp = np.mean(Varp)
        Meandfp = np.mean(np.array(patchdflst), axis=0)  # mean of difference
        pccp = pearson_distance(Varp.ravel(), Meandfp.ravel())
        psp, ssp = compute_psnr_and_ssim(normalize(Meanp, 0.1, 99.9, clip=True) * 255,
             normalize(y[randz:randz + patchsize, randh, randw:randw + patchsize], 0.1, 99.9, clip=True) * 255)
        rmsep = np.mean(np.square(Meanp - y[randz:randz + patchsize, randh, randw:randw + patchsize]), dtype=np.float64)
        print('patch%d  randz%d - PSNR/SSIM/MSE/PCC/Var of mean test = %f/%f/%f/%f/%f' % (j, randz, psp, ssp, rmsep, pccp, vp))

        PCCLst.append(pccp)
        meanvarlst.append(vp)
        mselst.append(rmsep)
        ssimlst.append(ssp)
        psnrlst.append(psp)
        meanvar += vp
        meandf += np.mean(Meandfp)
        meanKLdiv += 0
        
        print(Meandfp.shape)
        savecolorim(savepathzyx + name + '-MeandfnoNormC_num%d.png' % j,
                    np.clip(Meandfp * 255, 0, 255), norm=False)  # [200:-200, 200:-200]
        savecolorim(savepathzyx + name + '-MeanC_num%d.png' % j, Meanp)

        del patchdflst, patchlstv, patchlst
        Mdfmaplst.append(Meandfp)
        Varmaplst.append(Var)
    del resultlst, resultlstv
    print('%d image, Mean Var/ Mean DF of Testset %s is %8f/ %8f' % (patchnum, testset, meanvar/patchnum, meandf/patchnum))
    print(f'meanssim{np.mean(np.array(ssimlst))} psnr{np.mean(np.array(psnrlst))} pcc{np.mean(np.array(PCCLst))}'
          f'mse{np.mean(np.array(mselst))} STD{np.mean(np.array(meanvarlst))}')
    return meanvar/patchnum, meandf/patchnum, Mdfmaplst, Varmaplst, rmse, mselst, ssimlst, psnrlst  # Meandf, Var



if __name__ == '__main__':
    testsetlst = ['Isotropic_Drosophila']  # ['Isotropic_Retina']  # ['Isotropic_Liver']  # , []  # ,
    modellst = ['_FBdropout']
    
    my_seed = 34573529
    
    resultVarlst = []

    if IS_TF_1:  # tensorflow1.0
        tf.set_random_seed(my_seed)
    else:
        tf.random.set_seed(my_seed)
    DF = []
    Var = []
    for testset in testsetlst:
        for modeltype in modellst:
            print("####------------ Method %s------------###" % modeltype)
            np.random.seed(my_seed)
            testdatapath = '../../DataSet/Isotropic/' + testset + '/test_data/'
            
            if testset == 'Isotropic_Liver':
                var, mdf, dfmlst, vmlst, rmse, mselst, ssimlst, psnrlst = Variance_Liver()
                print('ps, ssm, rmse = ', np.mean(np.array(psnrlst)), np.mean(np.array(ssimlst)), rmse)
                print('List = ', mselst, ssimlst, psnrlst)
                resultVarlst.append('Model%s TestSET %s, ps, ssm, rmse = %.9f/ %.9f/ %.9f'
                          % (modeltype, testset, np.mean(np.array(psnrlst)), np.mean(np.array(ssimlst)), rmse))
                DF.extend(dfmlst)
                Var.extend(vmlst)
            else:
                var = Variance()
            resultVarlst.append('Model%s TestSET %s, mean Var= %.9f' % (modeltype, testset, var))
            print('************* Model%s TestSET %s, mean Variance = %.9f **********' % (modeltype, testset, var))
            print(resultVarlst)
