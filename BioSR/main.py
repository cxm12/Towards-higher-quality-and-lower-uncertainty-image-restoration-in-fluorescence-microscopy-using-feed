#!/usr/bin/env python
# coding: utf-8
# DropOut

from __future__ import print_function, unicode_literals, absolute_import, division
import random
import os
from tifffile import imsave, imread
from scipy import misc
import glob

import sys
sys.path.append('../')
from csbdeep.utils import normalize
from csbdeep.utils.tf import IS_TF_1
from csbdeep.utils import axes_dict
from csbdeep.io import load_training_data
from csbdeep.models import Config, CAREDropOut, CARE, FeedbackDropOut, SRCNN
from csbdeep.func_mcx import *
from csbdeep.utils.tf import keras


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


def loadData(batchsize=16):
    if testset == 'example':
        axes = None
    else:
        axes = 'SCYX'  #
    (X, Y), (X_val, Y_val), axes = load_training_data(traindatapath, validation_split=0.05, axes=axes, verbose=True)
    # X/Y []
    print(X.shape, Y.shape)
    # exit()
    
    c = axes_dict(axes)['C']
    n_channel_in, n_channel_out = X.shape[c], Y.shape[c]
    
    # plt.figure(figsize=(12, 5))
    # plot_some(X_val[:5], Y_val[:5])
    # plt.suptitle('5 example validation patches (top row: source, bottom row: target)')
    
    config = Config(axes, n_channel_in, n_channel_out, unet_kern_size=3, train_batch_size=batchsize)
    # config = Config(axes, n_channel_in, n_channel_out, unet_kern_size=3, train_batch_size=8, train_steps_per_epoch=40)
    print(config)
    vars(config)
    
    return config, X, Y, X_val, Y_val


def train(config, X, Y, X_val, Y_val):
    if ('_FB' in modeltype) or ('_srcnn' in modeltype):
        model = FeedbackDropOut(config, name='epoch200/my_model%s/%s' % (modeltype, testset), basedir='models', modeltype=modeltype, scale=2)
    elif '_dropout' in modeltype:
        model = CAREDropOut(config, name='epoch200/my_model%s/%s' % (modeltype, testset), basedir='models', modeltype=modeltype, finetune=True, scale=2)
    elif modeltype == '':
        model = CARE(config, name='epoch200/my_model/%s' % testset, basedir='models', scale=2)

    pre_ep = 200
    history = model.train(X, Y, validation_data=(X_val, Y_val), epochs=pre_ep)
    model.keras_model.save_weights('./models/epoch200/my_model%s/%s/weights_ep%d.h5' % (modeltype, testset, pre_ep))
   
    model.export_TF(fname='./models/epoch200/my_model%s/' % (modeltype) + testset + '/finalTF_SavedModel' + modeltype + '.zip')
        
    print('Train End !!! \n Max model')
    return model


def Variance():
    meanvar = 0
    testnum = 10
    meandf = 0
    
    # 将模式转为训练模式，在测试时才能使用dropout!!!
    keras.backend.set_learning_phase(1)
    testdatapath = '../../DataSet/BioSR_WF_to_SIM/DL-SR-main/dataset/train/%s/LR/*.tif' % testset
    savepath = './models/epoch200/my_model%s/%s/result/AllT%d/SR/' % (modeltype, testset, testnum)
    savepathres = './models/epoch200/my_model%s/%s/result/AllT%d/Res/' % (modeltype, testset, testnum)
    savepathvar = './models/epoch200/my_model%s/%s/result/AllT%d/Var/' % (modeltype, testset, testnum)
    savepathgt = './models/epoch200/my_model%s/%s/result/AllT%d/GT/' % (modeltype, testset, testnum)
    savepathlr = './models/epoch200/my_model%s/%s/result/AllT%d/LR/' % (modeltype, testset, testnum)
    
    os.makedirs(savepath, exist_ok=True)
    model = FeedbackDropOut(config=None, name='epoch200/my_model%s/%s' % (modeltype, testset), basedir='models',
                                    modeltype=modeltype, scale=2)
        
    meanvarlst = []
    Mdfmaplst = []
    Varmaplst = []
    PCCLst = []
    psnrlst = []
    ssimlst = []
    mselst = []
    num = 100
    for randim in range(num):
        name = 'im%d_LR.png' % (randim+1)
        print('randim = ', randim, name, testdatapath[:-5] + name[:-4] + '.tif')
        x = imread(testdatapath[:-5] + name[:-4] + '.tif')  # [128, 128]
        y = imread((testdatapath[:-5] + name[:-4]).replace('LR', 'GT') + '.tif')  # [256, 256]
        savecolorim(savepathgt + name + '-GT.png', np.clip(y * 255, 0, 255), norm=False)
        savecolorim(savepathlr + name + '-LR.png', np.clip(x * 255, 0, 255), norm=False)
        
        x = np.expand_dims(x, -1)
        axes = 'YXC'
        
        # ## Apply CARE network to raw image
        resultlst = []
        for ti in range(testnum):
            restored = model.predict(x, axes)
            resultlst.append(restored)
        Mean = np.squeeze(np.mean(np.array(resultlst), 0))
        Meandf = np.abs(Mean - y)
        Var = np.var(np.array(resultlst), axis=0)  # * 255 * 100 Compute the variance along the specified axis.
        Mdfmaplst.append(Meandf)
        Varmaplst.append(Var)
        
        savecolorim(savepathres + name + '-MeandfnoNormC.png', np.clip(Meandf * 255, 0, 255), norm=False)
        savecolorim1(savepathvar + name + '-NormVarC.png', NormVar)
        savecolorim(savepath + name + '-MeanC.png', Mean)
        
        pcc = pearson_distance(Var.ravel(), Meandf.ravel())
        PCCLst.append(pcc)
        psp, ssp = compute_psnr_and_ssim(normalize(Mean, 0.1, 99.9, clip=True) * 255,
                                          normalize(y, 0.1, 99.9, clip=True) * 255)
        rmsep = np.mean(np.square(Mean - y), dtype=np.float64)
        mselst.append(rmsep)
        ssimlst.append(ssp)
        psnrlst.append(psp)
        meanvar += np.mean(Var)
        meandf += np.mean(Meandf)
        v = np.mean(Var)
        vmax = np.max(Var)
        meanvarlst.append(v)
        print(f'Image - {name}- PSNR/SSIM{psp, ssp} /PCC/MSE {pcc, rmsep} STD/STDMax{v, vmax}')
        del resultlst
    file = open(savepath + "Psnrssimpccmse100.txt", 'w')
    file.write('\n PSNR \n' + str(psnrlst) + '\n SSIM \n' + str(ssimlst) + '\n MSE \n' + str(
        mselst) + '\n PCC \n' + str(PCCLst) + '\n STD \n' + str(meanvarlst))
    file.close()
    print('%d image, Mean Var/ Mean DF of Testset %s is %8f/ %8f' % (num, testset, meanvar / num, meandf / num))
    print(np.mean(np.array(psnrlst)), np.mean(np.array(ssimlst)), np.mean(np.array(mselst)))
    return meanvar / num, meandf / num, Mdfmaplst, Varmaplst, psnrlst, ssimlst, mselst, PCCLst


if __name__ == '__main__':
    istrain = True  # False
    save = True
    my_seed0 = 34573529
    server = 0
    
    if istrain:
        testsetlst = ['Microtubules']   #  ['CCPs']  # ['F-actin']  #
        modellst = ['_FBdropout']  #
        for testset in testsetlst:
            for modeltype in modellst:
                traindatapath = '../../DataSet/BioSR_WF_to_SIM/DL-SR-main/dataset/train/%s/my_training_data.npz' % testset
                np.random.seed(my_seed0)
                random.seed(my_seed0)
            
                if IS_TF_1:
                    tf.set_random_seed(my_seed0)
                else:
                    import tensorflow as tf0
    
                    tf0.random.set_seed(my_seed0)

                print('******************************************************************')
                print('*** Train on %s, Model %s ***' % (testset, modeltype))
                print('******************************************************************')
                epoch = 200
                
                config, X, Y, X_val, Y_val = loadData(batchsize=16)
                model = train(config, X, Y, X_val, Y_val)
    else:
        testsetlst = ['F-actin']  # ['CCPs']  # ['Microtubules']  # ['Microtubules',, ,
        modellst = ['_FBdropout']
        resultVarlst = []
        DF = []
        Var = []
        for testset in testsetlst:
            for modeltype in modellst:
                np.random.seed(my_seed0)
                print('******************************************************************')
                print('*** Test on %s, Model %s ***' % (testset, modeltype))
                print('******************************************************************')
                var, mdf, dfmlst, vmlst, psnrlst, ssimlst, mselst, PCCLst = Variance()
                DF.extend(dfmlst)
                Var.extend(vmlst)
                resultVarlst.append('Model%s TestSET %s, mean Var/MeanDf= %.9f/%.9f' % (modeltype, testset, var, mdf))
                print('************* Model%s TestSET %s, mean Variance/MeanDf = %.9f/%.9f **********' % (modeltype, testset, var, mdf))
                print(resultVarlst)
                print('psnrlst, ssimlst, mselst, PCCLst = \n', psnrlst, '\n', ssimlst, '\n', mselst, '\n', PCCLst)
