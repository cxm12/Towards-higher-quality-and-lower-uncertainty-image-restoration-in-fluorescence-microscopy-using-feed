#!/usr/bin/env python
# coding: utf-8
# DropOut

from __future__ import print_function, unicode_literals, absolute_import, division
import sys
import random
sys.path.append('../../')
from csbdeep.utils import axes_dict
from csbdeep.io import load_training_data
from csbdeep.models import Config, CAREDropOutDis, IsotropicFeedbackDropOut, IsotropicCARE, IsotropicCAREDropOut
from csbdeep.func_mcx import *


def loadData(batchsize=16):
    # # Training data
    axes = 'SCYX'
    datapath = '../../DataSet/Isotropic/%s/train_data/data_label.npz' % testset
    
    (X, Y), (X_val, Y_val), axes = load_training_data(datapath, validation_split=0.1, axes=axes, verbose=True)
    c = axes_dict(axes)['C']
    print(X.shape, Y.shape)
    # SYXC
    n_channel_in, n_channel_out = X.shape[c], Y.shape[c]
    
    config = Config(axes, n_channel_in, n_channel_out, train_steps_per_epoch=X.shape[0]//batchsize, train_batch_size=batchsize)
    vars(config)
    return config, X, Y, X_val, Y_val


def train(config, X, Y, X_val, Y_val, ep=200):
    # We now create an isotropic CARE model with the chosen configuration:
    model = IsotropicFeedbackDropOut(config, name='epoch200/my_model%s/%s' % (modeltype, testset), basedir='models', modeltype=modeltype)
    
    # # Training
    history = model.train(X, Y, validation_data=(X_val, Y_val), epochs=ep)
    model.keras_model.save_weights('./models/epoch200/my_model%s/%s/weights_ep%d.h5' % (modeltype, testset, ep))

    # # Evaluation
    # Example results for validation images.
    _P = model.keras_model.predict(X_val[:5])
    
    # # Export model to be used with CSBDeep **Fiji** plugins and **KNIME** workflows
    model.export_TF(
        fname='./models/epoch200/my_model%s/%s/TF_SavedModel' % (modeltype, testset) + '.zip')

   
if __name__ == '__main__':
    testsetlst = ['Isotropic_Liver', 'Isotropic_Drosophila', 'Isotropic_Retina']  #
    modellst = ['_FBdropout']

    ntile = (1, 4, 4)
    for modeltype in modellst:
        for testset in testsetlst:
            my_seed = 34573529
            np.random.seed(my_seed)
            random.seed(my_seed)
    
            if IS_TF_1:  # tensorflow1.0
                tf.set_random_seed(my_seed)
            else:
                tf.random.set_seed(my_seed)

            epoch = 200
            config, X, Y, X_val, Y_val = loadData(batchsize=16)  # 64!!!
            model = train(config, X, Y, X_val, Y_val, epoch)
