#!/usr/bin/env python
# coding: utf-8
# DropOut

from __future__ import print_function, unicode_literals, absolute_import, division
from tifffile import imread, imsave
import os
import random
from csbdeep.utils.tf import keras
from csbdeep.utils import normalize
from csbdeep.models import Config, CAREDropOut, CARE, CAREDropOutDis, FeedbackDropOut
from csbdeep.func_mcx import *


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


def test():
    testnum = 1  #
    PCCLst = []
    psnrlst = []
    mselst = []
    ssimlst = []
    meanvarlst = []
    patchsize = 100  # 1024  #
    
    # 将模式转为训练模式，在测试时才能使用dropout!!!  必须在模型定义之前
    keras.backend.set_learning_phase(1)
    axes = 'ZYX'
    if 'dropout' in modeltype:
        keras.backend.set_learning_phase(1)
        
    # model = FeedbackDropOut(config=None, name='epoch200/my_model%s/%s' % (modeltype, testset), basedir='models',
    #                                 modeltype=modeltype, step=3)
    model = FeedbackDropOut(config=None, name='epoch200/my_model%s/%s' % (modeltype, testset),
                            basedir='E:/file\python_project\Medical\CSBDeep-master\examples\denoising2D/models',
                            modeltype=modeltype, step=3)
    
    savepathzyx = './models/epoch200/my_model%s/' % modeltype + '/' + testset + '/' + 'result/%s/AllT%d' % (
        level, testnum) + '/patch/'
    savepathvar = './models/epoch200/my_model%s/' % modeltype + '/' + testset + '/' + 'result/%s/AllT%d' % (
        level, testnum) + '/Var/'
    savepathdf = './models/epoch200/my_model%s/' % modeltype + '/' + testset + '/' + 'result/%s/AllT%d' % (
        level, testnum) + '/Res/'
    savepathgt = './models/epoch200/my_model%s/' % modeltype + '/' + testset + '/' + 'result/%s/AllT%d' % (
        level, testnum) + '/GT/'
    savepathlr = './models/epoch200/my_model%s/' % modeltype + '/' + testset + '/' + 'result/%s/AllT%d' % (
        level, testnum) + '/LR/'
    
    os.makedirs(savepathzyx, exist_ok=True)
    os.makedirs(savepathvar, exist_ok=True)
    os.makedirs(savepathgt, exist_ok=True)
    os.makedirs(savepathlr, exist_ok=True)
    os.makedirs(savepathdf, exist_ok=True)
    
    filename = os.listdir(testdatapath)
    imnum = 1  # len(filename)
    NameLst = []
    namelst = ['EXP278_Smed_fixed_RedDot1_sub_5_N7_m0004']  # 'EXP278_Smed_fixed_RedDot1_sub_5_N7_m0013', 'EXP278_Smed_fixed_RedDot1_sub_5_N7_m0010',
               # 'EXP278_Smed_fixed_RedDot1_sub_5_N7_m0007',
    for i in range(imnum):
        name = namelst[i]
        NameLst.append(name)
        PCCLst.append(name)
        psnrlst.append(name)
        ssimlst.append(name)
        # print(name)
        x = imread(testdatapath + name + '.tif')[:, 200:-200, 200:-200]
        y = imread(testGTpath + name + '.tif')[:, 200:-200, 200:-200]
        
        # ## Apply CARE network to raw image
        resultlst = []
        resultlstv = []
        
        for ti in range(testnum):
            restored = model.predict(x, axes, n_tiles=ntile)  # [256, 256] -0.402~58.15
            resultlst.append(restored[:, :, :])
            restorednorm = normalize(restored, 0.1, 99.9, clip=True)
            resultlstv.append(restorednorm)
            print(f'x = {x.mean()}, restored = {restored.mean()}, y = {y.mean()}, restorednorm = {restorednorm.mean()}')
        
        # PSNR/SSIM
        mean = np.mean(np.array(resultlst), axis=0)
        psm, ssmm = compute_psnr_and_ssim(normalize(mean, 0.1, 99.9, clip=True) * 255, normalize(y, 0.1, 99.9, clip=True) * 255)
        rmse = np.mean(np.square(mean - y), dtype=np.float64)
        psnrlst.append(psm)
        ssimlst.append(ssmm)
        mselst.append(rmse)
        Var0 = np.var(np.array(resultlstv), axis=0)
        Varnonorm = np.var(np.array(resultlst), axis=0)
        pcc0 = pearson_distance(Var0.ravel(), mean.ravel())
        imsave(savepathzyx + name + '-noNormVar.tif', Var0)
        v = np.mean(Var0)
        vnonorm = np.mean(Varnonorm)
        vmax = np.max(Var0)
        meanvarlst.append(v)
        print(f'Image - {name}- PSNR/SSIM{psm, ssmm} /PCC/MSE {pcc0, rmse} STD/STDMax{v, vnonorm, vmax}')
        # Feedback v, vnonorm = 0.0045230254, 39.730705
        # CARE v, vnonorm = 0.0002043391, 0.79694086
        
        for randc in range(89, 90):
            randh = 0
            randw = 0
            
            patchdflst = []
            patchlst = []
            for im in resultlst:
                patchlst.append(im[randc, randh:randh + patchsize, randw:randw + patchsize])
                arrdf = np.abs(
                    normalize(im, 0.1, 99.9, clip=True)[randc, randh:randh + patchsize, randw:randw + patchsize]
                    - normalize(y, 0.1, 99.9, clip=True)[randc, randh:randh + patchsize, randw:randw + patchsize])
                patchdflst.append(arrdf)
            
            ##  PSNR/SSIM
            psm, ssmm = compute_psnr_and_ssim(mean[randc, randh:randh + patchsize, randw:randw + patchsize],
                                              y[randc, randh:randh + patchsize, randw:randw + patchsize])
            
            Var = np.var(np.array(patchdflst), axis=0)
            pv = np.mean(Var)
            Meandf = np.mean(np.array(patchdflst), axis=0)  # mean of difference
            pcc = pearson_distance(Var.ravel(), Meandf.ravel())
            PCCLst.append(pcc)
            print('Image%s - Patch_Depth%d - PSNR/SSIM/Var of mean test = %f%f%f' % (name, randc, psm, ssmm, pv),
                  'PCC:', pcc)

            Mean = np.mean(np.array(patchlst), axis=0)
            Mean = np.squeeze(Mean)
            Meandf = np.squeeze(Meandf)
            NormVar = normalize(Var, 0.1, 99.9, clip=True)
            NormVar = np.squeeze(NormVar)
            
            savecolorim(savepathdf + name + '-MeandfnoNormC_z%d.png' % randc,
                        np.clip(Meandf * 255, 0, 255), norm=False)
            savecolorim(savepathzyx + name + '-MeanC_z%d.png' % randc, Mean)
            savecolorim1(savepathvar + name + '-NormVarC_z%d.png' % randc, NormVar)
            savecolorim(savepathlr + name + '-MeanCLR_z%d.png' % randc, np.clip(
                normalize(x[randc, randh:randh + patchsize, randw:randw + patchsize], 0.1, 99.9, clip=True) * 255, 0,
                255))
            savecolorim(savepathgt + name + '-MeanCGT_z%d.png' % randc, np.clip(
                normalize(y[randc, randh:randh + patchsize, randw:randw + patchsize], 0.1, 99.9, clip=True) * 255, 0,
                255))
            del patchdflst, patchlst
        del resultlst
    # 将数组写入文件
    file = open(savepathzyx + "Result%d.txt" % imnum, 'w')
    file.write(str(NameLst)+'\n PSNR \n'+str(psnrlst)+'\n SSIM \n'+str(ssimlst)+'\n MSE \n'+str(mselst)+
    '\n PCC \n'+str(PCCLst) + '\n STD \n' + str(meanvarlst))
    file.close()
    
    print('%d image, Mean MSE/ Mean PSNR/ Mean SSIM of Testset %s is %8f/ %8f/ %8f'
          % (imnum, testset, np.float(np.mean(np.array(mselst))),
             np.float(np.mean(np.array(psnrlst))), np.float(np.mean(np.array(ssimlst)))), np.mean(np.array(meanvarlst)))
    
    return np.float(np.mean(np.array(psnrlst))), np.float(np.mean(np.array(ssimlst))), np.mean(np.array(meanvarlst)), \
           np.float(np.mean(np.array(mselst))), psnrlst, ssimlst, PCCLst, mselst, meanvarlst


if __name__ == '__main__':
    testset = 'Denoising_Planaria'

    # ['_FB1dropout', '_FB2dropout']   #
    modellst = ['_FBdropout']  # ['_dropout', '_srcnndropout', '_edsrdropout']
    ntile = (1, 4, 8)
    resultVarlst = []
    my_seed0 = 34573529
    
    DF = []
    Var = []
    for modeltype in modellst:
        if modeltype == '_FBdropout':
            ntile = (1, 8, 8)
        for lv in range(0, 1):
            level = 'condition_%d' % (lv + 1)
            # traindatapath = '../../DataSet/' + testset + '/train_data/data_label.npz'
            # testdatapath = '../../DataSet/' + testset + '/test_data/' + level + '/'
            # testGTpath = '../../DataSet/' + testset + '/test_data/GT/'

            traindatapath = 'E:/file/python_project/Medical/CSBDeep-master/DataSet/' + testset + '/train_data/data_label.npz'
            testdatapath = 'E:/file/python_project/Medical/CSBDeep-master/DataSet/' + testset + '/test_data/' + level + '/'
            testGTpath = 'E:/file/python_project/Medical/CSBDeep-master/DataSet/Denoising_Planaria/test_data/GT/'

            np.random.seed(my_seed0)
            np.random.seed(my_seed0)
            random.seed(my_seed0)

            if IS_TF_1:  # tensorflow1.0
                tf.set_random_seed(my_seed0)
            else:
                tf.set_random_seed(my_seed0)

            ps, ss, var, mse, pslst, sslst, pcclst, mselst, varlst = test()
            print('PSNR list: ', pslst, '\n SSIM: ', sslst, '\n MSE: ', mselst)
            
    print(resultVarlst)
