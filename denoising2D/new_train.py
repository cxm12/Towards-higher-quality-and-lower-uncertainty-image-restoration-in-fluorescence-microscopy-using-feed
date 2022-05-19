#!/usr/bin/env python
# coding: utf-8
# DropOut

from __future__ import print_function, unicode_literals, absolute_import, division
import sys
import random
sys.path.append('../../')
from csbdeep.utils import normalize, axes_dict, plot_some
from csbdeep.io import load_training_data
from csbdeep.models import Config, CAREDropOut, CARE, CAREDropOutDis, FeedbackDropOut
from csbdeep.func_mcx import *


def loadData(batchsize=16):
    if testset == 'example':
        axes = None
    else:
        axes = 'SCZYX'  #
    (X, Y), (X_val, Y_val), axes = load_training_data(traindatapath, validation_split=0.05, axes=axes, verbose=True)
    # Tribolium: X/Y [14725, 16, 64, 64, 1]
    # Planaria: X/Y [17005, 16, 64, 64, 1]
    
    c = axes_dict(axes)['C']
    n_channel_in, n_channel_out = X.shape[c], Y.shape[c]
    # print('++++++++++++++++++++++++', axes)

    # plt.figure(figsize=(12, 5))
    # plot_some(X_val[:5], Y_val[:5])
    # plt.suptitle('5 example validation patches (top row: source, bottom row: target)')
    
    config = Config(axes, n_channel_in, n_channel_out, unet_kern_size=3, train_batch_size=batchsize)
    # config = Config(axes, n_channel_in, n_channel_out, unet_kern_size=3, train_batch_size=8, train_steps_per_epoch=40)
    print(config)
    vars(config)
    
    return config, X, Y, X_val, Y_val


def train(config, X, Y, X_val, Y_val):
    
    model = FeedbackDropOut(config, name='epoch200/my_model%s/%s' % (modeltype, testset), basedir='models', modeltype=modeltype)
    
    pre_ep = 200
    history = model.train(X, Y, validation_data=(X_val, Y_val), epochs=pre_ep)
    model.keras_model.save_weights('./models/epoch200/my_model%s/%s/weights_ep%d.h5' % (modeltype, testset, pre_ep))
    
    model.export_TF(
            fname='./models/epoch200/my_model%s/' % (modeltype) + testset + '/finalTF_SavedModel' + modeltype + '.zip')
    print('Train End !!!')
    return model


if __name__ == '__main__':
    testset = 'Denoising_Planaria'
    modeltypelst = ['_FBdropout']
    ntile = (1, 4, 8)
    result = []

    for modeltype in modeltypelst:
        for lv in range(1, 2):
            level = 'condition_%d' % (lv+1)
            traindatapath = '../../DataSet/' + testset + '/train_data/data_label.npz'

            config, X, Y, X_val, Y_val = loadData(batchsize=16)

            my_seed0 = 34573529
            maxpsnr = 0
            maxseed = 0
            for sd in range(1):
                my_seed = my_seed0 + sd
                np.random.seed(my_seed)
                random.seed(my_seed)
    
                if IS_TF_1:  # tensorflow1.0
                    tf.set_random_seed(my_seed)
                else:
                    tf.random.set_seed(my_seed)
    
                epoch = 200
                config, X, Y, X_val, Y_val = loadData(batchsize=16)
                model = train(config, X, Y, X_val, Y_val)
