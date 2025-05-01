# coding=utf-8
import keras
import numpy as np
from keras import backend as K
from keras.layers import Conv2D, merge, BatchNormalization
from keras.layers import Conv3D, MaxPooling3D, UpSampling3D, Activation, add
from keras.layers import Conv3DTranspose
from keras.layers import Dropout
from keras.layers import Input, Lambda
from keras.layers import MaxPool2D
from keras.layers import UpSampling2D
from keras.layers import concatenate
from keras.layers.convolutional import MaxPooling2D, Convolution2D, UpSampling2D, Deconvolution2D
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.models import Model

from all_params import *


def res_dilated_block(x, nb_filters, dilated_rate):
    res_path = TimeDistributed(
        Conv2D(filters=nb_filters, kernel_size=(int(3), int(3)), padding='same'))(x)
    res_path = BatchNormalization(axis=-1)(res_path)
    res_path = Activation('relu')(res_path)
    res_path = TimeDistributed(
        Conv2D(filters=nb_filters, kernel_size=(int(3), int(3)), padding='same', dilation_rate=(dilated_rate, dilated_rate)))(res_path)
    res_path = BatchNormalization(axis=-1)(res_path)
    res_path = Activation('relu')(res_path)
    res_path = TimeDistributed(
        Conv2D(filters=nb_filters, kernel_size=(int(3), int(3)), padding='same', dilation_rate=(dilated_rate + 1, dilated_rate + 1)))(res_path)
    res_path = BatchNormalization(axis=-1)(res_path)
    res_path = Activation('relu')(res_path)
    shortcut = TimeDistributed(Conv2D(nb_filters, kernel_size=(int(1), int(1)), padding='same'))(
        x)  # ,activation='relu'
    shortcut = BatchNormalization(axis=-1)(shortcut)
    shortcut = Activation('relu')(shortcut)
    # res_path = merge([shortcut, res_path], mode='sum')
    res_path = add([shortcut, res_path])
    res_path = Activation('relu')(res_path)
    # res_path = add([shortcut, res_path])
    return res_path


def res_block(x, nb_filters, strides):
    res_path = TimeDistributed(Conv2D(filters=nb_filters[0], kernel_size=(int(3), int(3)), padding='same', strides=strides[0]))(x)
    res_path = BatchNormalization(axis=-1)(res_path)
    res_path = Activation('relu')(res_path)
    res_path = TimeDistributed(Conv2D(filters=nb_filters[1], kernel_size=(int(3), int(3)), padding='same', strides=strides[1]))(res_path)
    res_path = BatchNormalization(axis=-1)(res_path)
    res_path = Activation('relu')(res_path)
    shortcut = TimeDistributed(Conv2D(nb_filters[1], kernel_size=(int(1), int(1)), padding='same', strides=strides[0]))(x)  # ,activation='relu'
    shortcut = BatchNormalization(axis=-1)(shortcut)
    shortcut = Activation('relu')(shortcut)
    # res_path = merge([shortcut, res_path], mode='sum')
    res_path = add([shortcut, res_path])
    res_path = Activation('relu')(res_path)
    # res_path = add([shortcut, res_path])
    return res_path


def conv_block1(x, nb_filters, strides):
    res_path = TimeDistributed(Conv2D(filters=nb_filters[0], kernel_size=(int(3), int(3)), padding='same', strides=strides[0]))(x)
    res_path = BatchNormalization(axis=-1)(res_path)
    res_path = Activation('relu')(res_path)
    res_path = TimeDistributed(Conv2D(filters=nb_filters[1], kernel_size=(int(3), int(3)), padding='same', strides=strides[1]))(res_path)
    res_path = BatchNormalization(axis=-1)(res_path)
    res_path = Activation('relu')(res_path)
    # shortcut = TimeDistributed(Conv2D(nb_filters[1], kernel_size=(int(1),int(1)),padding='same',activation='relu', strides=strides[0]))(x)
    # shortcut = BatchNormalization(axis=-1)(shortcut)
    # res_path = merge([shortcut, res_path], mode='sum')
    # res_path = add([shortcut, res_path])
    # res_path = Activation('relu')(res_path)
    # res_path = add([shortcut, res_path])
    return res_path


def res_block2(x, nb_filters, strides):
    res_path = Conv2D(filters=nb_filters[0], kernel_size=(int(3), int(3)), padding='same', strides=strides[0])(x)
    res_path = BatchNormalization(axis=-1)(res_path)
    res_path = Activation('relu')(res_path)
    res_path = Conv2D(filters=nb_filters[1], kernel_size=(int(3), int(3)), padding='same', strides=strides[1])(res_path)
    res_path = BatchNormalization(axis=-1)(res_path)
    res_path = Activation('relu')(res_path)

    shortcut = Conv2D(nb_filters[1], kernel_size=(int(1), int(1)), padding='same', activation='relu', strides=strides[0])(x)
    shortcut = BatchNormalization(axis=-1)(shortcut)
    shortcut = Activation('relu')(shortcut)
    res_path = add([shortcut, res_path])

    # shortcut = TimeDistributed(Conv2D(nb_filters[1], kernel_size=(int(1),int(1)),padding='same',activation='relu', strides=strides[0]))(x)
    # shortcut = BatchNormalization(axis=-1)(shortcut)
    # res_path = merge([shortcut, res_path], mode='sum')
    # res_path = add([shortcut, res_path])
    # res_path = Activation('relu')(res_path)
    # res_path = add([shortcut, res_path])
    return res_path


def model_unet_lstm(train_flag=True):
    if (train_flag == True):
        inputs = (IMG_Z, IMG_ROWS, IMG_COLS, 1)
    else:
        inputs = (IMG_Z, TESTIMG_ROWS, TESTIMG_COLS, 1)
    inputs = Input(inputs)
    res1 = conv_block1(inputs, [int(32), int(32)], [(int(1), int(1)), (int(1), int(1))])
    pool1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(res1)
    # print res1.shape
    res2 = conv_block1(pool1, [int(64), int(64)], [(int(1), int(1)), (int(1), int(1))])
    pool2 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(res2)

    res3 = conv_block1(pool2, [int(128), int(128)], [(int(1), int(1)), (int(1), int(1))])
    pool3 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(res3)

    res4 = conv_block1(pool3, [int(256), int(256)], [(int(1), int(1)), (int(1), int(1))])
    conv4 = two_lstm(res4, int(256), int(3), int(3))

    # ------------------xiugai--------------------//
    # temp = TimeDistributed(
    #     Conv2D(filters=256, kernel_size=(int(3), int(3)), padding='same', strides=strides[0]))(pool3)
    # temp = BatchNormalization(axis=-1)(temp)
    # temp = Activation('relu')(temp)

    # lstm = two_lstm(temp, int(256), int(3), int(3))
    #
    # temp = TimeDistributed(
    #     Conv2D(filters=256, kernel_size=(int(3), int(3)), padding='same', strides=strides[0]))(lstm)
    # temp = BatchNormalization(axis=-1)(temp)
    # conv4 = Activation('relu')(temp)
    # res_path = TimeDistributed(
    #     Conv2D(filters=nb_filters[1], kernel_size=(int(3), int(3)), padding='same', strides=strides[1]))(res_path)
    # res_path = BatchNormalization(axis=-1)(res_path)
    # res_path = Activation('relu')(res_path)
    # shortcut = TimeDistributed(
    #     Conv2D(nb_filters[1], kernel_size=(int(1), int(1)), padding='same', activation='relu', strides=strides[0]))(x)
    # shortcut = BatchNormalization(axis=-1)(shortcut)
    # res_path = merge([shortcut, res_path], mode='sum')
    # res_path = add([shortcut, res_path])
    # res_path = Activation('relu')(res_path)

    '''
    conv5 = TimeDistributed(Convolution2D(512, 3, 3, activation='relu', border_mode='same'))(pool4)
    conv5 = BatchNormalization(axis=-1, name='bnconv51')(conv5)
    conv5 = time_ConvGRU_bottleNeck_block(conv5, 512, 3, 3)
    conv5 = TimeDistributed(Convolution2D(512, 3, 3, activation='relu', border_mode='same'))(conv5)
    conv5 = BatchNormalization(axis=-1, name='bnconv52')(conv5)
    '''
    '''
    up6 = merge([TimeDistributed(UpSampling2D(size=(2, 2)))(conv5), conv4], mode='concat', concat_axis=-1)
    conv6 = TimeDistributed(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))(up6)
    conv6 = BatchNormalization(axis=-1, name='bnconv61')(conv6)
    conv6 = TimeDistributed(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))(conv6)
    conv6 = BatchNormalization(axis=-1, name='bnconv62')(conv6)
    '''

    up5 = concatenate([TimeDistributed(Deconvolution2D(int(128), kernel_size=(1, 1), strides=(2, 2)))(conv4), res3], axis=-1)
    up5 = BatchNormalization(axis=-1)(up5)
    up5 = Activation('relu')(up5)
    # up7 = concatenate([TimeDistributed(UpSampling2D(size=(int(2), int(2))))(conv4), res3], axis=-1)
    res5 = conv_block1(up5, [int(128), int(128)], [(int(1), int(1)), (int(1), int(1))])
    # conv101 = TimeDistributed(Conv2D(int(4), (int(1), int(1)), activation='relu'))(res5)
    # conv101=TimeDistributed(UpSampling2D(size=(int(2), int(2)),name='upp1'),name="1")(conv101)

    up6 = concatenate([TimeDistributed(Deconvolution2D(int(64), kernel_size=(1, 1), strides=(2, 2)))(res5), res2], axis=-1)
    up6 = BatchNormalization(axis=-1)(up6)
    up6 = Activation('relu')(up6)
    res6 = conv_block1(up6, [int(64), int(64)], [(int(1), int(1)), (int(1), int(1))])
    # conv102 = TimeDistributed(Conv2D(int(4), (int(1), int(1)), activation='relu'))(res6)
    # conv102 = add([conv101, conv102],name='addd1')
    # conv102 = TimeDistributed(UpSampling2D(size=(int(2), int(2)),name='upp2'),name='2')(conv102)

    up7 = concatenate([TimeDistributed(Deconvolution2D(int(32), kernel_size=(1, 1), strides=(2, 2), ))(res6), res1], axis=-1)
    up7 = BatchNormalization(axis=-1)(up7)
    up7 = Activation('relu')(up7)
    res7 = conv_block1(up7, [int(32), int(32)], [(int(1), int(1)), (int(1), int(1))])
    # up10 = TimeDistributed(UpSampling2D(size=(2, 2)))(res7)
    conv8 = TimeDistributed(Conv2D(int(4), (int(1), int(1)), activation='relu'))(res7)
    # conv10 = add([conv8, conv102],name='addd2')
    # conv10 = BatchNormalization(axis=-1, name='bnconv10')(conv10)
    out = Activation('softmax')(conv8)
    # layers['outputs'] = core.Activation('softmax')(layers['outputs'])
    # out = Lambda(time_dist_softmax, output_shape=time_dist_softmax_out_shape)(conv10)
    model = Model(inputs=inputs, outputs=out)
    return model


def model_unet_map_fusion_lstm(train_flag=True):
    if (train_flag == True):
        inputs = (IMG_Z, IMG_ROWS, IMG_COLS, 1)
    else:
        inputs = (IMG_Z, TESTIMG_ROWS, TESTIMG_COLS, 1)
    inputs = Input(inputs)
    res1 = conv_block1(inputs, [int(32), int(32)], [(int(1), int(1)), (int(1), int(1))])
    pool1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(res1)
    # print res1.shape
    res2 = conv_block1(pool1, [int(64), int(64)], [(int(1), int(1)), (int(1), int(1))])
    pool2 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(res2)

    res3 = conv_block1(pool2, [int(128), int(128)], [(int(1), int(1)), (int(1), int(1))])
    pool3 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(res3)

    res4 = conv_block1(pool3, [int(256), int(256)], [(int(1), int(1)), (int(1), int(1))])
    conv4 = two_lstm(res4, int(256), int(3), int(3))
    # ------------------xiugai--------------------//
    # temp = TimeDistributed(
    #     Conv2D(filters=256, kernel_size=(int(3), int(3)), padding='same', strides=strides[0]))(pool3)
    # temp = BatchNormalization(axis=-1)(temp)
    # temp = Activation('relu')(temp)

    # lstm = two_lstm(temp, int(256), int(3), int(3))
    #
    # temp = TimeDistributed(
    #     Conv2D(filters=256, kernel_size=(int(3), int(3)), padding='same', strides=strides[0]))(lstm)
    # temp = BatchNormalization(axis=-1)(temp)
    # conv4 = Activation('relu')(temp)
    # res_path = TimeDistributed(
    #     Conv2D(filters=nb_filters[1], kernel_size=(int(3), int(3)), padding='same', strides=strides[1]))(res_path)
    # res_path = BatchNormalization(axis=-1)(res_path)
    # res_path = Activation('relu')(res_path)
    # shortcut = TimeDistributed(
    #     Conv2D(nb_filters[1], kernel_size=(int(1), int(1)), padding='same', activation='relu', strides=strides[0]))(x)
    # shortcut = BatchNormalization(axis=-1)(shortcut)
    # res_path = merge([shortcut, res_path], mode='sum')
    # res_path = add([shortcut, res_path])
    # res_path = Activation('relu')(res_path)

    up5 = concatenate([TimeDistributed(Deconvolution2D(int(128), kernel_size=(1, 1), strides=(2, 2)))(conv4), res3], axis=-1)
    up5 = BatchNormalization(axis=-1)(up5)
    up5 = Activation('relu')(up5)
    # up7 = concatenate([TimeDistributed(UpSampling2D(size=(int(2), int(2))))(conv4), res3], axis=-1)
    res5 = conv_block1(up5, [int(128), int(128)], [(int(1), int(1)), (int(1), int(1))])
    conv101 = TimeDistributed(Conv2D(int(4), (int(1), int(1)), activation='relu'))(res5)
    conv101 = TimeDistributed(UpSampling2D(size=(int(2), int(2)), name='upp1'), name="1")(conv101)

    up6 = concatenate([TimeDistributed(Deconvolution2D(int(64), kernel_size=(1, 1), strides=(2, 2)))(res5), res2], axis=-1)
    up6 = BatchNormalization(axis=-1)(up6)
    up6 = Activation('relu')(up6)
    res6 = conv_block1(up6, [int(64), int(64)], [(int(1), int(1)), (int(1), int(1))])
    conv102 = TimeDistributed(Conv2D(int(4), (int(1), int(1)), activation='relu'))(res6)
    conv102 = add([conv101, conv102], name='addd1')
    conv102 = TimeDistributed(UpSampling2D(size=(int(2), int(2)), name='upp2'), name='2')(conv102)

    up7 = concatenate([TimeDistributed(Deconvolution2D(int(32), kernel_size=(1, 1), strides=(2, 2), ))(res6), res1], axis=-1)
    up7 = BatchNormalization(axis=-1)(up7)
    up7 = Activation('relu')(up7)
    res7 = conv_block1(up7, [int(32), int(32)], [(int(1), int(1)), (int(1), int(1))])
    # up10 = TimeDistributed(UpSampling2D(size=(2, 2)))(res7)
    conv8 = TimeDistributed(Conv2D(int(4), (int(1), int(1)), activation='relu'))(res7)
    conv10 = add([conv8, conv102], name='addd2')
    # conv10 = BatchNormalization(axis=-1, name='bnconv10')(conv10)
    out = Activation('softmax')(conv10)
    # layers['outputs'] = core.Activation('softmax')(layers['outputs'])
    # out = Lambda(time_dist_softmax, output_shape=time_dist_softmax_out_shape)(conv10)
    model = Model(inputs=inputs, outputs=out)
    return model


def slice_weight(x, weight):
    return K.concatenate([x[:, :, :, :, 0:1], x[:, :, :, :, 2:2] * weight, x[:, :, :, :, 3:3]], axis=-1)


def slice_weight_2d(x, weight):
    return K.concatenate([x[:, :, :, 0:1], x[:, :, :, 2:2] * weight, x[:, :, :, 3:3]], axis=-1)


def slice_weight_output_shape(input_shape):
    return input_shape


def model_3dunet_res_lstm_sep(train_flag=True):
    if (train_flag == True):
        inputs = (IMG_Z, IMG_ROWS, IMG_COLS, 1)
    else:
        inputs = (IMG_Z, TESTIMG_ROWS, TESTIMG_COLS, 1)
    inputs = Input(inputs)
    res1 = res_block(inputs, [int(32), int(32)], [(int(1), int(1)), (int(1), int(1))])
    # pool1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(res1)
    # print res1.shape
    res2 = res_block(res1, [int(64), int(64)], [(int(2), int(2)), (int(1), int(1))])
    # pool2 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(res2)

    res3 = res_block(res2, [int(128), int(128)], [(int(2), int(2)), (int(1), int(1))])
    #  pool3 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(res3)

    res4 = res_block(res3, [int(256), int(256)], [(int(2), int(2)), (int(1), int(1))])
    conv4 = two_lstm(res4, int(256), int(3), int(3))

    up7 = concatenate([TimeDistributed(UpSampling2D(size=(2, 2)))(conv4), res3], axis=-1)

    res5 = res_block(up7, [int(128), int(128)], [(int(1), int(1)), (int(1), int(1))])
    conv101 = TimeDistributed(Conv2D(int(4), (int(1), int(1)), activation='relu'))(res5)
    conv101 = TimeDistributed(UpSampling2D(size=(int(2), int(2)), name='upp1'), name="1")(conv101)

    up8 = concatenate([TimeDistributed(UpSampling2D(size=(int(2), int(2))))(res5), res2], axis=-1)
    res6 = res_block(up8, [int(64), int(64)], [(int(1), int(1)), (int(1), int(1))])
    conv102 = TimeDistributed(Conv2D(int(4), (int(1), int(1)), activation='relu'))(res6)
    conv102 = add([conv101, conv102], name='addd1')
    conv102 = TimeDistributed(UpSampling2D(size=(int(2), int(2)), name='upp2'), name='2')(conv102)

    up9 = concatenate([TimeDistributed(UpSampling2D(size=(int(2), int(2))))(res6), res1], axis=-1)
    res7 = res_block(up9, [int(32), int(32)], [(int(1), int(1)), (int(1), int(1))])

    # detect=Flatten(res7)
    # detect=TimeDistributed(Dense( 1,input_shape=),)
    # up10 = TimeDistributed(UpSampling2D(size=(2, 2)))(res7)
    conv10 = TimeDistributed(Conv2D(int(4), (int(1), int(1)), activation='relu', name='conv10'))(res7)

    conv102 = Lambda(slice_weight, output_shape=slice_weight_output_shape, arguments={'weight': 0.5})(conv102)
    # conv102 = TimeDistributed(Lambda(lambda x : np.concatenate((x[:,:,:,0:1],x[:,:,:,2:2]*0.5,x[:,:,:,3:3]))) )(conv102)
    conv10 = add([conv10, conv102], name='addd2')

    # lambda （lambda conv10: conv10*1.1）
    # conv10 = BatchNormalization(axis=-1, name='bnconv10')(conv10)
    out = Activation('softmax')(conv10)
    # layers['outputs'] = core.Activation('softmax')(layers['outputs'])
    # out = Lambda(time_dist_softmax, output_shape=time_dist_softmax_out_shape)(conv10)
    model = Model(inputs=inputs, outputs=out)
    return model


def model_3dunet_res_lstm_sweat(train_flag=True):
    if (train_flag == True):
        inputs = (5, 480, 128, 1)
    else:
        inputs = (1, 480, 1792, 1)
    inputs = Input(inputs)
    res1 = res_block(inputs, [int(32), int(32)], [(int(1), int(1)), (int(1), int(1))])
    # pool1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(res1)
    # print res1.shape
    res2 = res_block(res1, [int(64), int(64)], [(int(2), int(2)), (int(1), int(1))])
    # pool2 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(res2)

    res3 = res_block(res2, [int(128), int(128)], [(int(2), int(2)), (int(1), int(1))])
    #  pool3 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(res3)

    res4 = res_block(res3, [int(256), int(256)], [(int(2), int(2)), (int(1), int(1))])
    conv4 = two_lstm(res4, int(256), int(3), int(3))

    up7 = concatenate([TimeDistributed(UpSampling2D(size=(2, 2)))(conv4), res3], axis=-1)

    res5 = res_block(up7, [int(128), int(128)], [(int(1), int(1)), (int(1), int(1))])
    conv101 = TimeDistributed(Conv2D(int(3), (int(1), int(1)), activation='relu'))(res5)
    conv101 = TimeDistributed(UpSampling2D(size=(int(2), int(2)), name='upp1'), name="1")(conv101)

    up8 = concatenate([TimeDistributed(UpSampling2D(size=(int(2), int(2))))(res5), res2], axis=-1)
    res6 = res_block(up8, [int(64), int(64)], [(int(1), int(1)), (int(1), int(1))])
    conv102 = TimeDistributed(Conv2D(int(3), (int(1), int(1)), activation='relu'))(res6)
    conv102 = add([conv101, conv102], name='addd1')
    conv102 = TimeDistributed(UpSampling2D(size=(int(2), int(2)), name='upp2'), name='2')(conv102)

    up9 = concatenate([TimeDistributed(UpSampling2D(size=(int(2), int(2))))(res6), res1], axis=-1)
    res7 = res_block(up9, [int(32), int(32)], [(int(1), int(1)), (int(1), int(1))])

    # detect=Flatten(res7)
    # detect=TimeDistributed(Dense( 1,input_shape=),)
    # up10 = TimeDistributed(UpSampling2D(size=(2, 2)))(res7)
    conv10 = TimeDistributed(Conv2D(int(3), (int(1), int(1)), activation='relu', name='conv10'))(res7)

    conv102 = Lambda(slice_weight, output_shape=slice_weight_output_shape, arguments={'weight': 0.5})(conv102)
    # conv102 = TimeDistributed(Lambda(lambda x : np.concatenate((x[:,:,:,0:1],x[:,:,:,2:2]*0.5,x[:,:,:,3:3]))) )(conv102)
    # conv10 = Lambda(slice_weight, output_shape=slice_weight_output_shape, arguments={'weight': 2})(conv10)
    # conv102 = Lambda(slice_weight, output_shape=slice_weight_output_shape, arguments={'weight': 1.5})(conv102)
    conv10 = add([conv10, conv102], name='addd2')
    print("-----------conv10-------------------", conv10.shape)
    # conv10 = Lambda(slice_weight, output_shape=slice_weight_output_shape, arguments={'weight': 1.5})(conv10)
    # lambda （lambda conv10: conv10*1.1）
    # conv10 = BatchNormalization(axis=-1, name='bnconv10')(conv10)
    out = Activation('softmax')(conv10)
    print("-----------cout-------------------", out.shape)
    # layers['outputs'] = core.Activation('softmax')(layers['outputs'])
    # out = Lambda(time_dist_softmax, output_shape=time_dist_softmax_out_shape)(conv10)
    model = Model(inputs=inputs, outputs=out)
    return model


def model_3dunet_res_lstm(train_flag=True):
    if (train_flag == True):
        inputs = (IMG_Z, IMG_ROWS, IMG_COLS, 1)
    else:
        inputs = (IMG_Z, TESTIMG_ROWS, TESTIMG_COLS, 1)
    inputs = Input(inputs)
    res1 = res_block(inputs, [int(32), int(32)], [(int(1), int(1)), (int(1), int(1))])
    # pool1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(res1)
    # print res1.shape
    res2 = res_block(res1, [int(64), int(64)], [(int(2), int(2)), (int(1), int(1))])
    # pool2 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(res2)

    res3 = res_block(res2, [int(128), int(128)], [(int(2), int(2)), (int(1), int(1))])
    #  pool3 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(res3)

    res4 = res_block(res3, [int(256), int(256)], [(int(2), int(2)), (int(1), int(1))])
    conv4 = two_lstm(res4, int(256), int(3), int(3))
    # ------------------xiugai--------------------//

    '''
    conv5 = TimeDistributed(Convolution2D(512, 3, 3, activation='relu', border_mode='same'))(pool4)
    conv5 = BatchNormalization(axis=-1, name='bnconv51')(conv5)
    conv5 = time_ConvGRU_bottleNeck_block(conv5, 512, 3, 3)
    conv5 = TimeDistributed(Convolution2D(512, 3, 3, activation='relu', border_mode='same'))(conv5)
    conv5 = BatchNormalization(axis=-1, name='bnconv52')(conv5)
    '''
    '''
    up6 = merge([TimeDistributed(UpSampling2D(size=(2, 2)))(conv5), conv4], mode='concat', concat_axis=-1)
    conv6 = TimeDistributed(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))(up6)
    conv6 = BatchNormalization(axis=-1, name='bnconv61')(conv6)
    conv6 = TimeDistributed(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))(conv6)
    conv6 = BatchNormalization(axis=-1, name='bnconv62')(conv6)
    '''
    up7 = concatenate([TimeDistributed(UpSampling2D(size=(2, 2)))(conv4), res3], axis=-1)
    # up7 = concatenate([TimeDistributed(UpSampling2D(size=(int(2), int(2))))(conv4), res3], axis=-1)
    res5 = res_block(up7, [int(128), int(128)], [(int(1), int(1)), (int(1), int(1))])
    # conv101 = TimeDistributed(Conv2D(int(4), (int(1), int(1)), activation='relu'))(res5)
    # conv101 = TimeDistributed(UpSampling2D(size=(int(2), int(2)), name='upp1'), name="1")(conv101)

    up8 = concatenate([TimeDistributed(UpSampling2D(size=(int(2), int(2))))(res5), res2], axis=-1)
    res6 = res_block(up8, [int(64), int(64)], [(int(1), int(1)), (int(1), int(1))])
    # conv102 = TimeDistributed(Conv2D(int(4), (int(1), int(1)), activation='relu'))(res6)
    # conv102 = add([conv101, conv102], name='addd1')
    # conv102 = TimeDistributed(UpSampling2D(size=(int(2), int(2)), name='upp2'), name='2')(conv102)

    up9 = concatenate([TimeDistributed(UpSampling2D(size=(int(2), int(2))))(res6), res1], axis=-1)
    res7 = res_block(up9, [int(32), int(32)], [(int(1), int(1)), (int(1), int(1))])
    # up10 = TimeDistributed(UpSampling2D(size=(2, 2)))(res7)
    conv10 = TimeDistributed(Conv2D(int(4), (int(1), int(1)), activation='relu'))(res7)
    # conv10 = add([conv10, conv102], name='addd2')
    # conv10 = BatchNormalization(axis=-1, name='bnconv10')(conv10)
    out = Activation('softmax')(conv10)
    # layers['outputs'] = core.Activation('softmax')(layers['outputs'])
    # out = Lambda(time_dist_softmax, output_shape=time_dist_softmax_out_shape)(conv10)
    model = Model(inputs=inputs, outputs=out)
    return model


def model_unet_res_no_sep(train_flag=True):
    if (train_flag == True):
        inputs = (IMG_ROWS, IMG_COLS, 1)
    else:
        inputs = (TESTIMG_ROWS, TESTIMG_COLS, 1)
    inputs = Input(inputs)
    res1 = res_block2(inputs, [int(32), int(32)], [(int(1), int(1)), (int(1), int(1))])
    # pool1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(res1)
    # print res1.shape
    res2 = res_block2(res1, [int(64), int(64)], [(int(2), int(2)), (int(1), int(1))])
    # pool2 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(res2)

    res3 = res_block2(res2, [int(128), int(128)], [(int(2), int(2)), (int(1), int(1))])
    #  pool3 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(res3)

    res4 = res_block2(res3, [int(256), int(256)], [(int(2), int(2)), (int(1), int(1))])
    # conv4 = two_lstm(res4, int(256), int(3), int(3))
    # ------------------xiugai--------------------//
    up7 = concatenate([UpSampling2D(size=(2, 2))(res4), res3], axis=-1)
    # up7 = concatenate([TimeDistributed(UpSampling2D(size=(int(2), int(2))))(conv4), res3], axis=-1)
    res5 = res_block2(up7, [int(128), int(128)], [(int(1), int(1)), (int(1), int(1))])
    # conv101 = TimeDistributed(Conv2D(int(4), (int(1), int(1)), activation='relu'))(res5)
    # conv101 = TimeDistributed(UpSampling2D(size=(int(2), int(2)), name='upp1'), name="1")(conv101)

    up8 = concatenate([UpSampling2D(size=(int(2), int(2)))(res5), res2], axis=-1)
    res6 = res_block2(up8, [int(64), int(64)], [(int(1), int(1)), (int(1), int(1))])
    # conv102 = TimeDistributed(Conv2D(int(4), (int(1), int(1)), activation='relu'))(res6)
    # conv102 = add([conv101, conv102], name='addd1')
    # conv102 = TimeDistributed(UpSampling2D(size=(int(2), int(2)), name='upp2'), name='2')(conv102)

    up9 = concatenate([UpSampling2D(size=(int(2), int(2)))(res6), res1], axis=-1)
    res7 = res_block2(up9, [int(32), int(32)], [(int(1), int(1)), (int(1), int(1))])
    # up10 = TimeDistributed(UpSampling2D(size=(2, 2)))(res7)
    conv10 = Conv2D(int(4), (int(1), int(1)), activation='relu')(res7)
    # conv10 = add([conv10, conv102], name='addd2')
    # conv10 = BatchNormalization(axis=-1, name='bnconv10')(conv10)
    out = Activation('softmax')(conv10)
    # layers['outputs'] = core.Activation('softmax')(layers['outputs'])
    # out = Lambda(time_dist_softmax, output_shape=time_dist_softmax_out_shape)(conv10)
    model = Model(inputs=inputs, outputs=out)
    return model


def model_3dunet_res_lstm_no_sep(train_flag=True):
    if (train_flag == True):
        inputs = (IMG_Z, IMG_ROWS, IMG_COLS, 1)
    else:
        inputs = (IMG_Z, TESTIMG_ROWS, TESTIMG_COLS, 1)
    inputs = Input(inputs)
    res1 = res_block(inputs, [int(32), int(32)], [(int(1), int(1)), (int(1), int(1))])
    # pool1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(res1)
    # print res1.shape
    res2 = res_block(res1, [int(64), int(64)], [(int(2), int(2)), (int(1), int(1))])
    # pool2 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(res2)

    res3 = res_block(res2, [int(128), int(128)], [(int(2), int(2)), (int(1), int(1))])
    #  pool3 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(res3)

    res4 = res_block(res3, [int(256), int(256)], [(int(2), int(2)), (int(1), int(1))])
    conv4 = two_lstm(res4, int(256), int(3), int(3))
    # ------------------xiugai--------------------//

    '''
    conv5 = TimeDistributed(Convolution2D(512, 3, 3, activation='relu', border_mode='same'))(pool4)
    conv5 = BatchNormalization(axis=-1, name='bnconv51')(conv5)
    conv5 = time_ConvGRU_bottleNeck_block(conv5, 512, 3, 3)
    conv5 = TimeDistributed(Convolution2D(512, 3, 3, activation='relu', border_mode='same'))(conv5)
    conv5 = BatchNormalization(axis=-1, name='bnconv52')(conv5)
    '''
    '''
    up6 = merge([TimeDistributed(UpSampling2D(size=(2, 2)))(conv5), conv4], mode='concat', concat_axis=-1)
    conv6 = TimeDistributed(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))(up6)
    conv6 = BatchNormalization(axis=-1, name='bnconv61')(conv6)
    conv6 = TimeDistributed(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))(conv6)
    conv6 = BatchNormalization(axis=-1, name='bnconv62')(conv6)
    '''
    up7 = concatenate([TimeDistributed(UpSampling2D(size=(2, 2)))(conv4), res3], axis=-1)
    # up7 = concatenate([TimeDistributed(UpSampling2D(size=(int(2), int(2))))(conv4), res3], axis=-1)
    res5 = res_block(up7, [int(128), int(128)], [(int(1), int(1)), (int(1), int(1))])
    # conv101 = TimeDistributed(Conv2D(int(4), (int(1), int(1)), activation='relu'))(res5)
    # conv101 = TimeDistributed(UpSampling2D(size=(int(2), int(2)), name='upp1'), name="1")(conv101)

    up8 = concatenate([TimeDistributed(UpSampling2D(size=(int(2), int(2))))(res5), res2], axis=-1)
    res6 = res_block(up8, [int(64), int(64)], [(int(1), int(1)), (int(1), int(1))])
    # conv102 = TimeDistributed(Conv2D(int(4), (int(1), int(1)), activation='relu'))(res6)
    # conv102 = add([conv101, conv102], name='addd1')
    # conv102 = TimeDistributed(UpSampling2D(size=(int(2), int(2)), name='upp2'), name='2')(conv102)

    up9 = concatenate([TimeDistributed(UpSampling2D(size=(int(2), int(2))))(res6), res1], axis=-1)
    res7 = res_block(up9, [int(32), int(32)], [(int(1), int(1)), (int(1), int(1))])
    # up10 = TimeDistributed(UpSampling2D(size=(2, 2)))(res7)
    conv10 = TimeDistributed(Conv2D(int(4), (int(1), int(1)), activation='relu'))(res7)
    # conv10 = add([conv10, conv102], name='addd2')
    # conv10 = BatchNormalization(axis=-1, name='bnconv10')(conv10)
    out = Activation('softmax')(conv10)
    # layers['outputs'] = core.Activation('softmax')(layers['outputs'])
    # out = Lambda(time_dist_softmax, output_shape=time_dist_softmax_out_shape)(conv10)
    model = Model(inputs=inputs, outputs=out)
    return model


def model_dilated_res_lstm(train_flag=True, d_width=None):
    if (train_flag == True):
        inputs = (IMG_Z, IMG_ROWS, IMG_COLS, 1)
    else:
        inputs = (IMG_Z, TESTIMG_ROWS, TESTIMG_COLS, 1)  # (IMG_Z, 500, d_width, 1)
    inputs = Input(inputs)
    res1 = res_block(inputs, [int(32), int(32)], [(int(1), int(1)), (int(1), int(1))])
    # pool1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(res1)
    # print res1.shape
    res2 = res_block(res1, [int(64), int(64)], [(int(2), int(2)), (int(1), int(1))])
    # pool2 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(res2)
    res3 = res_block(res2, [int(128), int(128)], [(int(2), int(2)), (int(1), int(1))])
    # res3 = res_block(res2, [int(128), int(128)], [(int(2), int(2)), (int(1), int(1))])
    #  pool3 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(res3)
    res4 = res_dilated_block(res3, int(256), int(2))
    # res4 = res_block(res3, [int(256), int(256)], [(int(2), int(2)), (int(1), int(1))])
    conv4 = two_lstm(res4, int(256), int(3), int(3))
    # ------------------xiugai--------------------//
    up7 = conv4
    up7 = concatenate([conv4, res3], axis=-1)
    # up7 = concatenate([TimeDistributed(UpSampling2D(size=(int(2), int(2))))(conv4), res3], axis=-1)
    res5 = res_block(up7, [int(128), int(128)], [(int(1), int(1)), (int(1), int(1))])
    conv101 = TimeDistributed(Conv2D(int(4), (int(1), int(1)), activation='relu'))(res5)
    conv101 = TimeDistributed(UpSampling2D(size=(int(2), int(2)), name='upp1'), name="1")(conv101)

    # up8=res5
    up8 = concatenate([TimeDistributed(UpSampling2D(size=(int(2), int(2))))(res5), res2], axis=-1)
    res6 = res_block(up8, [int(64), int(64)], [(int(1), int(1)), (int(1), int(1))])
    conv102 = TimeDistributed(Conv2D(int(4), (int(1), int(1)), activation='relu'))(res6)
    conv102 = add([conv101, conv102], name='addd1')
    conv102 = TimeDistributed(UpSampling2D(size=(int(2), int(2)), name='upp2'), name='2')(conv102)

    up9 = concatenate([TimeDistributed(UpSampling2D(size=(int(2), int(2))))(res6), res1], axis=-1)
    res7 = res_block(up9, [int(32), int(32)], [(int(1), int(1)), (int(1), int(1))])
    # up10 = TimeDistributed(UpSampling2D(size=(2, 2)))(res7)
    conv10 = TimeDistributed(Conv2D(int(4), (int(1), int(1)), activation='relu'))(res7)
    conv10 = add([conv10, conv102], name='addd2')
    # conv10 = BatchNormalization(axis=-1, name='bnconv10')(conv10)
    out = Activation('softmax')(conv10)
    # layers['outputs'] = core.Activation('softmax')(layers['outputs'])
    # out = Lambda(time_dist_softmax, output_shape=time_dist_softmax_out_shape)(conv10)
    model = Model(inputs=inputs, outputs=out)
    return model
    # if (train_flag == True):
    #     inputs = (IMG_Z, IMG_ROWS, IMG_COLS, 1)
    # else:
    #     inputs = (IMG_Z, TESTIMG_ROWS, TESTIMG_COLS, 1)
    # inputs = Input(inputs)
    #
    # x=inputs
    # nb_filters=[int(32),int(32)]
    # strides=[(int(1), int(1)), (int(2), int(2))]
    # res_path = TimeDistributed(
    #     Conv2D(filters=nb_filters[0], kernel_size=(int(3), int(3)), padding='same', strides=strides[0]))(x)
    # res_path = BatchNormalization(axis=-1)(res_path)
    # res_path = Activation('relu')(res_path)
    # res_path = TimeDistributed(
    #     Conv2D(filters=nb_filters[1], kernel_size=(int(3), int(3)), padding='same', strides=strides[1]))(res_path)
    # res_path = BatchNormalization(axis=-1)(res_path)
    # res_path = Activation('relu')(res_path)
    # shortcut = TimeDistributed(Conv2D(nb_filters[1], kernel_size=(int(1), int(1)), padding='same', strides=strides[1]))(
    #     x)  # ,activation='relu'
    # shortcut = BatchNormalization(axis=-1)(shortcut)
    # shortcut = Activation('relu')(shortcut)
    # # res_path = merge([shortcut, res_path], mode='sum')
    # res_path = add([shortcut, res_path])
    # res1 = Activation('relu')(res_path)
    #
    # # res1 = res_block(inputs, [int(32), int(32)], )
    # # pool1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(res1)
    # # print res1.shape
    # # res2 = res_block(res1, [int(64), int(64)], [(int(1), int(1)), (int(1), int(1))])
    # # pool2 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(res2)
    #
    # # res3 = res_block(res2, [int(128), int(128)], [(int(1), int(1)), (int(1), int(1))])
    # #  pool3 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(res3)
    # # res1 = res_block(inputs, [int(32), int(32)], [(int(1), int(1)), (int(2), int(2))])
    # # res1 = res_dilated_block(inputs, int(32), int(2))
    # res2 = res_dilated_block(res1, int(64), int(2))
    # # res4 = res_dilated_block(res2, int(128), int(2)
    # res4 = res_dilated_block(res2, int(128) , int(2))
    # res4 = Bidirectional(
    #     keras.layers.ConvLSTM2D(int(128/2), kernel_size=(int(3), int(3)), data_format='channels_last',
    #                             padding='same', return_sequences=True))(res4)
    # res4 = Bidirectional(
    #     keras.layers.ConvLSTM2D(int(128 / 2), kernel_size=(int(3), int(3)), data_format='channels_last',
    #                             padding='same', return_sequences=True))(res4)
    # res4 = BatchNormalization(axis=-1)(res4)
    # #------------sweat gland stream------------------
    # # sweat_stream = Bidirectional(
    # #     keras.layers.ConvLSTM2D(int(128/2), kernel_size=(int(3), int(3)), data_format='channels_last',
    # #                             padding='same', return_sequences=True))(res4)
    # # sweat_stream = BatchNormalization(axis=-1)(sweat_stream)
    # # sweat_stream=res_block(sweat_stream, [int(128), int(128)], [(int(1), int(1)), (int(1), int(1))])
    # res4=TimeDistributed(UpSampling2D(size=(2, 2)))(res4)
    # sweat_stream = res_block(res4, [int(64), int(64)], [(int(1), int(1)), (int(1), int(1))])
    # sweat_stream = res_block(sweat_stream, [int(32), int(32)], [(int(1), int(1)), (int(1), int(1))])
    # sweat_stream = res_block(sweat_stream, [int(32), int(32)], [(int(1), int(1)), (int(1), int(1))])
    # sweat_stream = TimeDistributed(Conv2D(int(4), (int(1), int(1)), activation='relu'))(sweat_stream)
    # # ------------epidermis and dermis  stream------------------
    # # epi_dermis_stream = Bidirectional(
    # #     keras.layers.ConvLSTM2D(int(128/2), kernel_size=(int(3), int(3)), data_format='channels_last',
    # #                             padding='same', return_sequences=True))(res4)
    # # epi_dermis_stream = BatchNormalization(axis=-1)(epi_dermis_stream)
    # # epi_dermis_stream = res_dilated_block(epi_dermis_stream, int(64), int(2))
    # # epi_dermis_stream = res_dilated_block(epi_dermis_stream, int(32), int(2))
    # # # epi_dermis_stream = res_block(epi_dermis_stream, [int(32), int(32)], [(int(1), int(1)), (int(1), int(1))])
    # #
    # # epi_dermis_stream = TimeDistributed(Conv2D(int(2), (int(1), int(1)), activation='relu'))(epi_dermis_stream)
    #
    # #--------------concatenate -two steam--------------------------------
    # # final_prediction = concatenate([epi_dermis_stream, sweat_stream], axis=-1)
    # final_prediction=sweat_stream
    # # conv10 = BatchNormalization(axis=-1, name='bnconv10')(conv10)
    # out = Activation('softmax')(final_prediction)
    # # layers['outputs'] = core.Activation('softmax')(layers['outputs'])
    # # out = Lambda(time_dist_softmax, output_shape=time_dist_softmax_out_shape)(conv10)
    # model = Model(inputs=inputs, outputs=out)
    # return model


def model_3dunet_dilated_res_lstm_sep(train_flag=True, TESTIMG_ROW=img_height, TESTIMG_COL=img_width):  #
    if (train_flag == True):
        inputs = (IMG_Z, IMG_ROWS, IMG_COLS, 1)
    else:
        inputs = (IMG_Z, TESTIMG_ROW, TESTIMG_COL, 1)  # (IMG_Z, 500, d_width, 1)
    inputs = Input(inputs)
    res1 = res_block(inputs, [int(32), int(32)], [(int(1), int(1)), (int(1), int(1))])
    # pool1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(res1)
    # print res1.shape
    res2 = res_block(res1, [int(64), int(64)], [(int(2), int(2)), (int(1), int(1))])
    # pool2 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(res2)
    res3 = res_block(res2, [int(128), int(128)], [(int(2), int(2)), (int(1), int(1))])
    # res3 = res_block(res2, [int(128), int(128)], [(int(2), int(2)), (int(1), int(1))])
    #  pool3 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(res3)
    res4 = res_dilated_block(res3, int(256), int(2))
    # res4 = res_block(res3, [int(256), int(256)], [(int(2), int(2)), (int(1), int(1))])
    conv4 = two_lstm(res4, int(256), int(3), int(3))
    # ------------------xiugai--------------------//
    # up7=conv4
    up7 = concatenate([conv4, res3], axis=-1)
    # up7 = concatenate([TimeDistributed(UpSampling2D(size=(int(2), int(2))))(conv4), res3], axis=-1)
    res5 = res_block(up7, [int(128), int(128)], [(int(1), int(1)), (int(1), int(1))])
    conv101 = TimeDistributed(Conv2D(int(4), (int(1), int(1)), activation='relu'))(res5)
    conv101 = TimeDistributed(UpSampling2D(size=(int(2), int(2)), name='upp1'), name="1")(conv101)

    # up8=res5
    up8 = concatenate([TimeDistributed(UpSampling2D(size=(int(2), int(2))))(res5), res2], axis=-1)
    res6 = res_block(up8, [int(64), int(64)], [(int(1), int(1)), (int(1), int(1))])
    conv102 = TimeDistributed(Conv2D(int(4), (int(1), int(1)), activation='relu'))(res6)
    conv102 = add([conv101, conv102], name='addd1')
    conv102 = TimeDistributed(UpSampling2D(size=(int(2), int(2)), name='upp2'), name='2')(conv102)

    up9 = concatenate([TimeDistributed(UpSampling2D(size=(int(2), int(2))))(res6), res1], axis=-1)
    res7 = res_block(up9, [int(32), int(32)], [(int(1), int(1)), (int(1), int(1))])
    # up10 = TimeDistributed(UpSampling2D(size=(2, 2)))(res7)
    conv10 = TimeDistributed(Conv2D(int(4), (int(1), int(1)), activation='relu'))(res7)
    conv102 = Lambda(slice_weight, output_shape=slice_weight_output_shape, arguments={'weight': 0.5})(conv102)
    conv10 = add([conv10, conv102], name='addd2')
    # conv10 = BatchNormalization(axis=-1, name='bnconv10')(conv10)
    out = Activation('softmax')(conv10)
    # layers['outputs'] = core.Activation('softmax')(layers['outputs'])
    # out = Lambda(time_dist_softmax, output_shape=time_dist_softmax_out_shape)(conv10)
    model = Model(inputs=inputs, outputs=out)
    return model


def model_dilated_res_lstm_no_sep(train_flag=True, d_width=None):
    if (train_flag == True):
        inputs = (IMG_Z, IMG_ROWS, IMG_COLS, 1)
    else:
        inputs = (IMG_Z, TESTIMG_ROWS, TESTIMG_COLS, 1)  # (IMG_Z, 500, d_width, 1)
    inputs = Input(inputs)
    res1 = res_block(inputs, [int(32), int(32)], [(int(1), int(1)), (int(1), int(1))])
    # pool1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(res1)
    # print res1.shape
    res2 = res_block(res1, [int(64), int(64)], [(int(2), int(2)), (int(1), int(1))])
    # pool2 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(res2)
    res3 = res_block(res2, [int(128), int(128)], [(int(2), int(2)), (int(1), int(1))])
    # res3 = res_block(res2, [int(128), int(128)], [(int(2), int(2)), (int(1), int(1))])
    #  pool3 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(res3)
    res4 = res_dilated_block(res3, int(256), int(2))
    # res4 = res_block(res3, [int(256), int(256)], [(int(2), int(2)), (int(1), int(1))])
    conv4 = two_lstm(res4, int(256), int(3), int(3))
    # ------------------xiugai--------------------//
    # up7=conv4
    up7 = concatenate([conv4, res3], axis=-1)
    # up7 = concatenate([TimeDistributed(UpSampling2D(size=(int(2), int(2))))(conv4), res3], axis=-1)
    res5 = res_block(up7, [int(128), int(128)], [(int(1), int(1)), (int(1), int(1))])
    # conv101 = TimeDistributed(Conv2D(int(4), (int(1), int(1)), activation='relu'))(res5)
    # conv101 = TimeDistributed(UpSampling2D(size=(int(2), int(2)), name='upp1'), name="1")(conv101)

    # up8=res5
    up8 = concatenate([TimeDistributed(UpSampling2D(size=(int(2), int(2))))(res5), res2], axis=-1)
    res6 = res_block(up8, [int(64), int(64)], [(int(1), int(1)), (int(1), int(1))])
    # conv102 = TimeDistributed(Conv2D(int(4), (int(1), int(1)), activation='relu'))(res6)
    # conv102 = add([conv101, conv102], name='addd1')
    # conv102 = TimeDistributed(UpSampling2D(size=(int(2), int(2)), name='upp2'), name='2')(conv102)

    up9 = concatenate([TimeDistributed(UpSampling2D(size=(int(2), int(2))))(res6), res1], axis=-1)
    res7 = res_block(up9, [int(32), int(32)], [(int(1), int(1)), (int(1), int(1))])
    # up10 = TimeDistributed(UpSampling2D(size=(2, 2)))(res7)
    conv10 = TimeDistributed(Conv2D(int(4), (int(1), int(1)), activation='relu'))(res7)
    # conv10 = add([conv10, conv102], name='addd2')
    # conv10 = BatchNormalization(axis=-1, name='bnconv10')(conv10)
    out = Activation('softmax')(conv10)
    # layers['outputs'] = core.Activation('softmax')(layers['outputs'])
    # out = Lambda(time_dist_softmax, output_shape=time_dist_softmax_out_shape)(conv10)
    model = Model(inputs=inputs, outputs=out)
    return model


def model_unet_res_no_fusion(train_flag=True):
    if (train_flag == True):
        inputs = (IMG_ROWS, IMG_COLS, 1)
    else:
        inputs = (TESTIMG_ROWS, TESTIMG_COLS, 1)
    inputs = Input(inputs)
    res1 = res_block2(inputs, [int(32), int(32)], [(int(1), int(1)), (int(1), int(1))])
    # pool1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(res1)
    # print res1.shape
    res2 = res_block2(res1, [int(64), int(64)], [(int(2), int(2)), (int(1), int(1))])
    # pool2 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(res2)

    res3 = res_block2(res2, [int(128), int(128)], [(int(2), int(2)), (int(1), int(1))])
    #  pool3 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(res3)

    res4 = res_block2(res3, [int(256), int(256)], [(int(2), int(2)), (int(1), int(1))])
    conv4 = two_lstm(res4, int(256), int(3), int(3))

    # up7 = concatenate([UpSampling2D(size=(2, 2))(res4), res3], axis=-1)
    up7 = concatenate([TimeDistributed(UpSampling2D(size=(int(2), int(2))))(conv4), res3], axis=-1)
    res5 = res_block2(up7, [int(128), int(128)], [(int(1), int(1)), (int(1), int(1))])
    conv101 = TimeDistributed(Conv2D(int(4), (int(1), int(1)), activation='relu'))(res5)
    # conv101 = TimeDistributed(UpSampling2D(size=(int(2), int(2)), name='upp1'), name="1")(conv101)

    up8 = concatenate([UpSampling2D(size=(int(2), int(2)))(res5), res2], axis=-1)
    res6 = res_block2(up8, [int(64), int(64)], [(int(1), int(1)), (int(1), int(1))])
    # conv102 = TimeDistributed(Conv2D(int(4), (int(1), int(1)), activation='relu'))(res6)
    # conv102 = add([conv101, conv102], name='addd1')
    # conv102 = TimeDistributed(UpSampling2D(size=(int(2), int(2)), name='upp2'), name='2')(conv102)

    up9 = concatenate([UpSampling2D(size=(int(2), int(2)))(res6), res1], axis=-1)
    res7 = res_block2(up9, [int(32), int(32)], [(int(1), int(1)), (int(1), int(1))])
    # up10 = TimeDistributed(UpSampling2D(size=(2, 2)))(res7)
    conv10 = Conv2D(int(4), (int(1), int(1)), activation='relu')(res7)
    # conv10 = add([conv10, conv102], name='addd2')
    # conv10 = BatchNormalization(axis=-1, name='bnconv10')(conv10)
    out = Activation('softmax')(conv10)
    # layers['outputs'] = core.Activation('softmax')(layers['outputs'])
    # out = Lambda(time_dist_softmax, output_shape=time_dist_softmax_out_shape)(conv10)
    model = Model(inputs=inputs, outputs=out)
    return model


def model_unet_seperate_predict(train_flag=True):
    if (train_flag):
        inputs = (IMG_ROWS, IMG_COLS, 1)
    else:
        inputs = (TESTIMG_ROWS, TESTIMG_COLS, 1)
    inputs = Input(inputs)
    conv1 = Conv2D(int(32), (int(3), int(3)), padding="same", activation="relu")(inputs)
    conv1 = BatchNormalization(axis=-1, name='bnconv11')(conv1)
    conv1 = Conv2D(int(32), (int(3), int(3)), padding="same", activation="relu")(conv1)
    conv1 = BatchNormalization(axis=-1, name='bnconv12')(conv1)
    pool1 = MaxPooling2D(pool_size=(int(2), int(2)))(conv1)

    conv2 = Conv2D(int(64), (int(3), int(3)), padding="same", activation="relu")(pool1)
    conv2 = BatchNormalization(axis=-1, name='bnconv21')(conv2)
    conv2 = Conv2D(int(64), (int(3), int(3)), padding="same", activation="relu")(conv2)
    conv2 = BatchNormalization(axis=-1, name='bnconv22')(conv2)
    pool2 = MaxPooling2D(pool_size=(int(2), int(2)))(conv2)

    conv3 = Conv2D(int(128), (int(3), int(3)), padding="same", activation="relu")(pool2)
    conv3 = BatchNormalization(axis=-1, name='bnconv31')(conv3)
    conv3 = Conv2D(int(128), (int(3), int(3)), padding="same", activation="relu")(conv3)
    conv3 = BatchNormalization(axis=-1, name='bnconv32')(conv3)
    pool3 = MaxPooling2D(pool_size=(int(2), int(2)))(conv3)

    conv4 = Conv2D(int(256), (int(3), int(3)), padding="same", activation="relu")(pool3)
    conv4 = BatchNormalization(axis=-1, name='bnconv41')(conv4)
    conv4 = Conv2D(int(256), (int(3), int(3)), padding="same", activation="relu")(conv4)
    # conv4 = time_ConvGRU_bottleNeck_block(conv4, 256, 3, 3)
    conv4 = BatchNormalization(axis=-1, name='bnconv42')(conv4)
    # pool4 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv4)

    Deconv1 = Deconvolution2D(int(128), kernel_size=(int(3), int(3)), strides=(int(2), int(2)), activation='relu',
                              padding='same')(conv4)
    Deconv1 = BatchNormalization(axis=-1)(Deconv1)
    up7 = concatenate([Deconv1, conv3], axis=-1)

    conv7 = Conv2D(int(128), (int(3), int(3)), padding="same", activation="relu")(up7)
    conv7 = BatchNormalization(axis=-1, name='bnconv71')(conv7)
    conv7 = Conv2D(int(128), (int(3), int(3)), padding="same", activation="relu")(conv7)
    conv7 = BatchNormalization(axis=-1, name='bnconv72')(conv7)
    conv101 = Conv2D(int(4), (int(1), int(1)), activation='relu')(conv7)
    conv101 = UpSampling2D(size=(int(2), int(2)), name='upp1')(conv101)

    Deconv2 = Deconvolution2D(int(64), kernel_size=(int(3), int(3)), strides=(int(2), int(2)), activation='relu',
                              padding='same')(conv7)
    Deconv2 = BatchNormalization(axis=-1)(Deconv2)
    up8 = concatenate([Deconv2, conv2], axis=-1)
    conv8 = Conv2D(int(64), (int(3), int(3)), padding="same", activation="relu")(up8)
    conv8 = BatchNormalization(axis=-1, name='bnconv81')(conv8)
    conv8 = Conv2D(int(64), (int(3), int(3)), padding="same", activation="relu")(conv8)
    conv8 = BatchNormalization(axis=-1, name='bnconv82')(conv8)
    conv102 = Conv2D(int(4), (int(1), int(1)), activation='relu')(conv8)
    conv102 = add([conv101, conv102], name='add1')
    conv102 = UpSampling2D(size=(int(2), int(2)), name='upp2')(conv102)

    Deconv3 = Deconvolution2D(int(32), kernel_size=(int(3), int(3)), strides=(int(2), int(2)), activation='relu',
                              padding='same')(conv8)
    Deconv3 = BatchNormalization(axis=-1)(Deconv3)
    up9 = concatenate([Deconv3, conv1], axis=-1)
    conv9 = Conv2D(int(32), (int(3), int(3)), padding="same", activation="relu")(up9)
    conv9 = BatchNormalization(axis=-1, name='bnconv91')(conv9)
    conv9 = Conv2D(int(32), (int(3), int(3)), padding="same", activation="relu")(conv9)
    conv9 = BatchNormalization(axis=-1, name='bnconv92')(conv9)
    conv10 = Conv2D(int(4), (int(3), int(3)), padding="same", activation="relu")(conv9)
    conv10 = BatchNormalization(axis=-1, name='bnconv10')(conv10)
    conv102 = Lambda(slice_weight_2d, output_shape=slice_weight_output_shape, arguments={'weight': 0.0})(conv102)
    conv10 = add([conv10, conv102], name='add2')
    out = Activation('softmax')(conv10)
    # layers['outputs'] = core.Activation('softmax')(layers['outputs'])
    # out = Lambda(time_dist_softmax, output_shape=time_dist_softmax_out_shape)(conv10)
    model = Model(inputs=inputs, outputs=out)
    return model


def model_unet(train_flag=True):
    if (train_flag):
        inputs = (IMG_ROWS, IMG_COLS, 1)
    else:
        inputs = (TESTIMG_ROWS, TESTIMG_COLS, 1)
    inputs = Input(inputs)
    conv1 = Conv2D(int(32), (int(3), int(3)), padding="same", activation="relu")(inputs)
    conv1 = BatchNormalization(axis=-1, name='bnconv11')(conv1)
    conv1 = Conv2D(int(32), (int(3), int(3)), padding="same", activation="relu")(conv1)
    conv1 = BatchNormalization(axis=-1, name='bnconv12')(conv1)
    pool1 = MaxPooling2D(pool_size=(int(2), int(2)))(conv1)

    conv2 = Conv2D(int(64), (int(3), int(3)), padding="same", activation="relu")(pool1)
    conv2 = BatchNormalization(axis=-1, name='bnconv21')(conv2)
    conv2 = Conv2D(int(64), (int(3), int(3)), padding="same", activation="relu")(conv2)
    conv2 = BatchNormalization(axis=-1, name='bnconv22')(conv2)
    pool2 = MaxPooling2D(pool_size=(int(2), int(2)))(conv2)

    conv3 = Conv2D(int(128), (int(3), int(3)), padding="same", activation="relu")(pool2)
    conv3 = BatchNormalization(axis=-1, name='bnconv31')(conv3)
    conv3 = Conv2D(int(128), (int(3), int(3)), padding="same", activation="relu")(conv3)
    conv3 = BatchNormalization(axis=-1, name='bnconv32')(conv3)
    pool3 = MaxPooling2D(pool_size=(int(2), int(2)))(conv3)

    conv4 = Conv2D(int(256), (int(3), int(3)), padding="same", activation="relu")(pool3)
    conv4 = BatchNormalization(axis=-1, name='bnconv41')(conv4)
    conv4 = Conv2D(int(256), (int(3), int(3)), padding="same", activation="relu")(conv4)
    # conv4 = time_ConvGRU_bottleNeck_block(conv4, 256, 3, 3)
    conv4 = BatchNormalization(axis=-1, name='bnconv42')(conv4)
    # pool4 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv4)
    '''
    conv5 = TimeDistributed(Convolution2D(512, 3, 3, activation='relu', border_mode='same'))(pool4)
    conv5 = BatchNormalization(axis=-1, name='bnconv51')(conv5)
    conv5 = time_ConvGRU_bottleNeck_block(conv5, 512, 3, 3)
    conv5 = TimeDistributed(Convolution2D(512, 3, 3, activation='relu', border_mode='same'))(conv5)
    conv5 = BatchNormalization(axis=-1, name='bnconv52')(conv5)
    '''
    '''
    up6 = merge([TimeDistributed(UpSampling2D(size=(2, 2)))(conv5), conv4], mode='concat', concat_axis=-1)
    conv6 = TimeDistributed(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))(up6)
    conv6 = BatchNormalization(axis=-1, name='bnconv61')(conv6)
    conv6 = TimeDistributed(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))(conv6)
    conv6 = BatchNormalization(axis=-1, name='bnconv62')(conv6)
    '''
    Deconv1 = Deconvolution2D(int(128), kernel_size=(int(3), int(3)), strides=(int(2), int(2)), activation='relu', padding='same')(conv4)
    Deconv1 = BatchNormalization(axis=-1)(Deconv1)
    up7 = concatenate([Deconv1, conv3], axis=-1)

    conv7 = Conv2D(int(128), (int(3), int(3)), padding="same", activation="relu")(up7)
    conv7 = BatchNormalization(axis=-1, name='bnconv71')(conv7)
    conv7 = Conv2D(int(128), (int(3), int(3)), padding="same", activation="relu")(conv7)
    conv7 = BatchNormalization(axis=-1, name='bnconv72')(conv7)

    Deconv2 = Deconvolution2D(int(64), kernel_size=(int(3), int(3)), strides=(int(2), int(2)), activation='relu', padding='same')(conv7)
    Deconv2 = BatchNormalization(axis=-1)(Deconv2)
    up8 = concatenate([Deconv2, conv2], axis=-1)
    conv8 = Conv2D(int(64), (int(3), int(3)), padding="same", activation="relu")(up8)
    conv8 = BatchNormalization(axis=-1, name='bnconv81')(conv8)
    conv8 = Conv2D(int(64), (int(3), int(3)), padding="same", activation="relu")(conv8)
    conv8 = BatchNormalization(axis=-1, name='bnconv82')(conv8)

    Deconv3 = Deconvolution2D(int(32), kernel_size=(int(3), int(3)), strides=(int(2), int(2)), activation='relu', padding='same')(conv8)
    Deconv3 = BatchNormalization(axis=-1)(Deconv3)
    up9 = concatenate([Deconv3, conv1], axis=-1)
    conv9 = Conv2D(int(32), (int(3), int(3)), padding="same", activation="relu")(up9)
    conv9 = BatchNormalization(axis=-1, name='bnconv91')(conv9)
    conv9 = Conv2D(int(32), (int(3), int(3)), padding="same", activation="relu")(conv9)
    conv9 = BatchNormalization(axis=-1, name='bnconv92')(conv9)
    conv10 = Conv2D(int(4), (int(3), int(3)), padding="same", activation="relu")(conv9)
    conv10 = BatchNormalization(axis=-1, name='bnconv10')(conv10)

    out = Activation('softmax')(conv10)
    # layers['outputs'] = core.Activation('softmax')(layers['outputs'])
    # out = Lambda(time_dist_softmax, output_shape=time_dist_softmax_out_shape)(conv10)
    model = Model(inputs=inputs, outputs=out)
    return model


def model_unet_dilation(train_flag=True):
    if (train_flag):
        inputs = (IMG_ROWS, IMG_COLS, 1)
    else:
        inputs = (TESTIMG_ROWS, TESTIMG_COLS, 1)
    inputs = Input(inputs)
    conv1 = Conv2D(int(32), (int(3), int(3)), padding="same")(inputs)
    conv1 = BatchNormalization(axis=-1, name='bnconv11')(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv2D(int(32), (int(3), int(3)), padding="same")(conv1)
    conv1 = BatchNormalization(axis=-1, name='bnconv12')(conv1)
    conv1 = Activation('relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(int(2), int(2)))(conv1)

    conv2 = Conv2D(int(64), (int(3), int(3)), padding="same")(pool1)
    conv2 = BatchNormalization(axis=-1, name='bnconv21')(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = Conv2D(int(64), (int(3), int(3)), padding="same")(conv2)
    conv2 = BatchNormalization(axis=-1, name='bnconv22')(conv2)
    conv2 = Activation('relu')(conv2)
    pool2 = MaxPooling2D(pool_size=(int(2), int(2)))(conv2)

    conv3 = Conv2D(int(128), (int(3), int(3)), padding="same")(pool2)
    conv3 = BatchNormalization(axis=-1, name='bnconv31')(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = Conv2D(int(128), (int(3), int(3)), padding="same", dilation_rate=2)(conv3)
    conv3 = BatchNormalization(axis=-1, name='bnconv32')(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = Conv2D(int(128), (int(3), int(3)), padding="same", dilation_rate=3)(conv3)
    conv3 = BatchNormalization(axis=-1, name='bnconv33')(conv3)
    conv3 = Activation('relu')(conv3)
    # pool3 = MaxPooling2D(pool_size=(int(2), int(2)))(conv3)

    conv4 = Conv2D(int(256), (int(3), int(3)), padding="same")(conv3)
    conv4 = BatchNormalization(axis=-1, name='bnconv41')(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = Conv2D(int(256), (int(3), int(3)), padding="same")(conv4)
    # conv4 = time_ConvGRU_bottleNeck_block(conv4, 256, 3, 3)
    conv4 = BatchNormalization(axis=-1, name='bnconv42')(conv4)
    conv4 = Activation('relu')(conv4)
    # pool4 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv4)
    '''
    conv5 = TimeDistributed(Convolution2D(512, 3, 3, activation='relu', border_mode='same'))(pool4)
    conv5 = BatchNormalization(axis=-1, name='bnconv51')(conv5)
    conv5 = time_ConvGRU_bottleNeck_block(conv5, 512, 3, 3)
    conv5 = TimeDistributed(Convolution2D(512, 3, 3, activation='relu', border_mode='same'))(conv5)
    conv5 = BatchNormalization(axis=-1, name='bnconv52')(conv5)
    '''
    '''
    up6 = merge([TimeDistributed(UpSampling2D(size=(2, 2)))(conv5), conv4], mode='concat', concat_axis=-1)
    conv6 = TimeDistributed(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))(up6)
    conv6 = BatchNormalization(axis=-1, name='bnconv61')(conv6)
    conv6 = TimeDistributed(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))(conv6)
    conv6 = BatchNormalization(axis=-1, name='bnconv62')(conv6)
    '''
    # Deconv1=Deconvolution2D(int(128), kernel_size=(int(3), int(3)), strides=(int(2), int(2)), activation='relu',padding='same')(conv4)
    # Deconv1=BatchNormalization(axis=-1)(Deconv1)
    up7 = concatenate([conv4, conv3], axis=-1)

    conv7 = Conv2D(int(128), (int(3), int(3)), padding="same")(up7)
    conv7 = BatchNormalization(axis=-1, name='bnconv71')(conv7)
    conv7 = Activation('relu')(conv7)
    conv7 = Conv2D(int(128), (int(3), int(3)), padding="same")(conv7)
    conv7 = BatchNormalization(axis=-1, name='bnconv72')(conv7)
    conv7 = Activation('relu')(conv7)

    Deconv2 = Deconvolution2D(int(64), kernel_size=(int(3), int(3)), strides=(int(2), int(2)), padding='same')(conv7)
    Deconv2 = BatchNormalization(axis=-1)(Deconv2)
    Deconv2 = Activation('relu')(Deconv2)

    up8 = concatenate([Deconv2, conv2], axis=-1)
    conv8 = Conv2D(int(64), (int(3), int(3)), padding="same")(up8)
    conv8 = BatchNormalization(axis=-1, name='bnconv81')(conv8)
    conv8 = Activation('relu')(conv8)
    conv8 = Conv2D(int(64), (int(3), int(3)), padding="same")(conv8)
    conv8 = BatchNormalization(axis=-1, name='bnconv82')(conv8)
    conv8 = Activation('relu')(conv8)

    Deconv3 = Deconvolution2D(int(32), kernel_size=(int(3), int(3)), strides=(int(2), int(2)), padding='same')(conv8)
    Deconv3 = BatchNormalization(axis=-1)(Deconv3)
    Deconv3 = Activation('relu')(Deconv3)
    up9 = concatenate([Deconv3, conv1], axis=-1)
    conv9 = Conv2D(int(32), (int(3), int(3)), padding="same")(up9)
    conv9 = BatchNormalization(axis=-1, name='bnconv91')(conv9)
    conv9 = Activation('relu')(conv9)
    conv9 = Conv2D(int(32), (int(3), int(3)), padding="same")(conv9)
    conv9 = BatchNormalization(axis=-1, name='bnconv92')(conv9)
    conv9 = Activation('relu')(conv9)
    conv10 = Conv2D(int(4), (int(3), int(3)), padding="same")(conv9)
    conv10 = BatchNormalization(axis=-1, name='bnconv10')(conv10)

    out = Activation('softmax')(conv10)
    # layers['outputs'] = core.Activation('softmax')(layers['outputs'])
    # out = Lambda(time_dist_softmax, output_shape=time_dist_softmax_out_shape)(conv10)
    model = Model(inputs=inputs, outputs=out)
    return model


def unet_model_3dunet(train_flag=True, downsize_filters_factor=1, pool_size=(2, 2, 2), n_labels=1,
                      initial_learning_rate=0.00001, deconvolution=True):
    """
    Builds the 3D UNet Keras model.
    :param input_shape: Shape of the input data (n_chanels, x_size, y_size, z_size).
    :param downsize_filters_factor: Factor to which to reduce the number of filters. Making this value larger will
    reduce the amount of memory the model will need during training.
    :param pool_size: Pool size for the max pooling operations.
    :param n_labels: Number of binary labels that the model is learning.
    :param initial_learning_rate: Initial learning rate for the model. This will be decayed during training.
    :param deconvolution: If set to True, will use transpose convolution(deconvolution) instead of upsamping. This
    increases the amount memory required during training.
    :return: Untrained 3D UNet Model
    """
    if (train_flag):
        input_shape = (IMG_Z, IMG_ROWS, IMG_COLS, 1)
    else:
        input_shape = (IMG_Z, TESTIMG_ROWS, TESTIMG_COLS, 1)
    inputs = Input(input_shape)
    conv1 = Conv3D(int(32 / downsize_filters_factor), (3, 3, 3), activation='relu', padding='same')(inputs)
    conv1 = BatchNormalization(axis=-1, name='bnconv11')(conv1)
    conv1 = Conv3D(int(32 / downsize_filters_factor), (3, 3, 3), activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization(axis=-1, name='bnconv12')(conv1)
    pool1 = MaxPooling3D(pool_size=pool_size)(conv1)

    conv2 = Conv3D(int(64 / downsize_filters_factor), (3, 3, 3), activation='relu', padding='same')(pool1)
    conv2 = BatchNormalization(axis=-1, name='bnconv21')(conv2)
    conv2 = Conv3D(int(64 / downsize_filters_factor), (3, 3, 3), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization(axis=-1, name='bnconv22')(conv2)
    pool2 = MaxPooling3D(pool_size=pool_size)(conv2)

    conv3 = Conv3D(int(128 / downsize_filters_factor), (3, 3, 3), activation='relu', padding='same')(pool2)
    conv3 = BatchNormalization(axis=-1, name='bnconv31')(conv3)
    conv3 = Conv3D(int(128 / downsize_filters_factor), (3, 3, 3), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization(axis=-1, name='bnconv32')(conv3)
    pool3 = MaxPooling3D(pool_size=pool_size)(conv3)

    conv4 = Conv3D(int(256 / downsize_filters_factor), (3, 3, 3), activation='relu', padding='same')(pool3)
    conv4 = BatchNormalization(axis=-1, name='bnconv41')(conv4)
    conv4 = Conv3D(int(256 / downsize_filters_factor), (3, 3, 3), activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization(axis=-1, name='bnconv42')(conv4)

    up5 = Conv3DTranspose(128, kernel_size=(3, 3, 3), strides=(2, 2, 2), activation='relu', padding='same')(conv4)
    up5 = BatchNormalization(axis=-1, name='bnup5')(up5)
    # up5 = get_upconv(128, deconvolution=deconvolution, depth=2,
    #                  nb_filters=int(512 / downsize_filters_factor), image_shape=input_shape[-3:])(conv4)
    up5 = concatenate([up5, conv3], axis=4)
    conv5 = Conv3D(int(128 / downsize_filters_factor), (3, 3, 3), activation='relu', padding='same')(up5)
    conv5 = BatchNormalization(axis=-1, name='bnconv51')(conv5)
    conv5 = Conv3D(int(128 / downsize_filters_factor), (3, 3, 3), activation='relu', padding='same')(conv5)
    conv5 = BatchNormalization(axis=-1, name='bnconv52')(conv5)

    up6 = Conv3DTranspose(64, kernel_size=(3, 3, 3), strides=(2, 2, 2), activation='relu', padding='same')(conv5)
    up6 = BatchNormalization(axis=-1, name='bnup6')(up6)
    # up6 = get_upconv(pool_size=pool_size, deconvolution=deconvolution, depth=1,
    #                  nb_filters=int(256 / downsize_filters_factor), image_shape=input_shape[-3:])(conv5)
    up6 = concatenate([up6, conv2], axis=4)
    conv6 = Conv3D(int(64 / downsize_filters_factor), (3, 3, 3), activation='relu', padding='same')(up6)
    conv6 = BatchNormalization(axis=-1, name='bnconv61')(conv6)
    conv6 = Conv3D(int(64 / downsize_filters_factor), (3, 3, 3), activation='relu', padding='same')(conv6)
    conv6 = BatchNormalization(axis=-1, name='bnconv62')(conv6)

    up7 = Conv3DTranspose(32, kernel_size=(3, 3, 3), strides=(2, 2, 2), activation='relu', padding='same')(conv6)
    up7 = BatchNormalization(axis=-1, name='bnup7')(up7)
    # up7 = get_upconv(pool_size=pool_size, deconvolution=deconvolution, depth=0,
    #                  nb_filters=int(128 / downsize_filters_factor), image_shape=input_shape[-3:])(conv6)
    up7 = concatenate([up7, conv1], axis=4)
    conv7 = Conv3D(int(32 / downsize_filters_factor), (3, 3, 3), activation='relu', padding='same')(up7)
    conv7 = BatchNormalization(axis=-1, name='bnconv71')(conv7)
    conv7 = Conv3D(int(32 / downsize_filters_factor), (3, 3, 3), activation='relu', padding='same')(conv7)
    conv7 = BatchNormalization(axis=-1, name='bnconv72')(conv7)

    conv8 = Conv3D(4, (1, 1, 1))(conv7)
    # act = Activation('sigmoid')(conv8)
    act = Activation('softmax')(conv8)
    model = Model(inputs=inputs, outputs=act)

    return model


def unet_model_3dvnet(train_flag=True, downsize_filters_factor=1, pool_size=(2, 2, 2), n_labels=1,
                      initial_learning_rate=0.00001, deconvolution=True):
    """
    V-Net
    """
    if (train_flag):
        input_shape = (IMG_Z, IMG_ROWS, IMG_COLS, 1)
    else:
        input_shape = (IMG_Z, TESTIMG_ROWS, TESTIMG_COLS, 1)
    inputs = Input(input_shape)

    conv1 = Conv3D(int(32 / downsize_filters_factor), (5, 5, 5), padding='same')(inputs)
    conv1 = Activation('relu')(conv1)
    conv1 = BatchNormalization(axis=-1)(conv1)

    # conv1 = Conv3D(int(32 / downsize_filters_factor), (3, 3, 3), padding='same')(inputs)
    # conv1 = BatchNormalization(axis=-1)(conv1)
    # conv1 = Activation('relu')(conv1)
    # conv1 = Conv3D(int(32 / downsize_filters_factor), (3, 3, 3), padding='same')(conv1)
    # conv1 = BatchNormalization(axis=-1)(conv1)
    # conv1 = Activation('relu')(conv1)
    res1 = Conv3D(int(32 / downsize_filters_factor), (1, 1, 1))(inputs)
    res1 = Activation('relu')(res1)
    res1 = BatchNormalization(axis=-1)(res1)
    conv1 = add([conv1, res1])  # for concate

    down2 = Conv3D(int(64 / downsize_filters_factor), (3, 3, 3), strides=(2, 2, 2), padding='same')(conv1)
    down2 = Activation('relu')(down2)
    down2 = BatchNormalization(axis=-1)(down2)

    # conv2 = Conv3D(int(64 / downsize_filters_factor), (3, 3, 3), padding='same')(down2)
    # conv2 = BatchNormalization(axis=-1)(conv2)
    # conv2 = Activation('relu')(conv2)
    # conv2 = Conv3D(int(64 / downsize_filters_factor), (3, 3, 3), padding='same')(conv2)
    # conv2 = BatchNormalization(axis=-1)(conv2)
    # conv2 = Activation('relu')(conv2)
    # conv2 = Conv3D(int(64 / downsize_filters_factor), (3, 3, 3), padding='same')(conv2)
    # conv2 = BatchNormalization(axis=-1)(conv2)
    # conv2 = Activation('relu')(conv2)
    # conv2 = Conv3D(int(64 / downsize_filters_factor), (3, 3, 3), padding='same')(conv2)
    # conv2 = BatchNormalization(axis=-1)(conv2)
    # conv2 = Activation('relu')(conv2)
    conv2 = Conv3D(int(64 / downsize_filters_factor), (5, 5, 5), padding='same')(down2)
    conv2 = Activation('relu')(conv2)
    conv2 = BatchNormalization(axis=-1)(conv2)
    conv2 = Conv3D(int(64 / downsize_filters_factor), (5, 5, 5), padding='same')(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = BatchNormalization(axis=-1)(conv2)
    conv2 = add([conv2, down2])  # for concatenate

    down3 = Conv3D(int(128 / downsize_filters_factor), (2, 2, 2), strides=(2, 2, 2), padding='same')(conv2)
    down3 = Activation('relu')(down3)
    down3 = BatchNormalization(axis=-1)(down3)
    conv3 = Conv3D(int(128 / downsize_filters_factor), (5, 5, 5), padding='same')(down3)
    conv3 = Activation('relu')(conv3)
    conv3 = BatchNormalization(axis=-1)(conv3)
    conv3 = Conv3D(int(128 / downsize_filters_factor), (5, 5, 5), padding='same')(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = BatchNormalization(axis=-1)(conv3)

    # conv3 = Conv3D(int(128 / downsize_filters_factor), (3, 3, 3), padding='same')(down3)
    # conv3 = BatchNormalization(axis=-1)(conv3)
    # conv3 = Activation('relu')(conv3)
    # conv3 = Conv3D(int(128 / downsize_filters_factor), (3, 3, 3), padding='same')(conv3)
    # conv3 = BatchNormalization(axis=-1)(conv3)
    # conv3 = Activation('relu')(conv3)
    # conv3 = Conv3D(int(128 / downsize_filters_factor), (3, 3, 3), padding='same')(conv3)
    # conv3 = BatchNormalization(axis=-1)(conv3)
    # conv3 = Activation('relu')(conv3)
    # conv3 = Conv3D(int(128 / downsize_filters_factor), (3, 3, 3), padding='same')(conv3)
    # conv3 = BatchNormalization(axis=-1)(conv3)
    # conv3 = Activation('relu')(conv3)
    conv3 = add([conv3, down3])  # for concatenate

    down4 = Conv3D(int(256 / downsize_filters_factor), (2, 2, 2), strides=(2, 2, 2), padding='same')(conv3)
    down4 = Activation('relu')(down4)
    down4 = BatchNormalization(axis=-1)(down4)

    # conv4 = Conv3D(int(256 / downsize_filters_factor), (3, 3, 3), padding='same')(down4)
    # conv4 = BatchNormalization(axis=-1)(conv4)
    # conv4 = Activation('relu')(conv4)
    # conv4 = Conv3D(int(256 / downsize_filters_factor), (3, 3, 3), padding='same')(conv4)
    # conv4 = BatchNormalization(axis=-1)(conv4)
    # conv4 = Activation('relu')(conv4)
    # conv4 = Conv3D(int(256 / downsize_filters_factor), (3, 3, 3), padding='same')(conv4)
    # conv4 = BatchNormalization(axis=-1)(conv4)
    # conv4 = Activation('relu')(conv4)
    # conv4 = Conv3D(int(256 / downsize_filters_factor), (3, 3, 3), padding='same')(conv4)
    # conv4 = BatchNormalization(axis=-1)(conv4)
    # conv4 = Activation('relu')(conv4)
    conv4 = Conv3D(int(256 / downsize_filters_factor), (5, 5, 5), padding='same')(down4)
    conv4 = Activation('relu')(conv4)
    conv4 = BatchNormalization(axis=-1)(conv4)
    conv4 = Conv3D(int(256 / downsize_filters_factor), (5, 5, 5), padding='same')(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = BatchNormalization(axis=-1)(conv4)

    conv4 = add([conv4, down4])

    up5 = Conv3DTranspose(int(128 / downsize_filters_factor), (2, 2, 2), strides=(2, 2, 2), padding='same')(conv4)
    up5 = Activation('relu')(up5)
    up5 = BatchNormalization(axis=-1)(up5)
    conv5 = concatenate([up5, conv3], axis=4)  # concatenate
    # conv5 = Conv3D(int(128 / downsize_filters_factor), (3, 3, 3), padding='same')(conv5)
    # conv5 = BatchNormalization(axis=-1)(conv5)
    # conv5 = Activation('relu')(conv5)
    # conv5 = Conv3D(int(128 / downsize_filters_factor), (3, 3, 3), padding='same')(conv5)
    # conv5 = BatchNormalization(axis=-1)(conv5)
    # conv5 = Activation('relu')(conv5)
    # conv5 = Conv3D(int(128 / downsize_filters_factor), (3, 3, 3), padding='same')(conv5)
    # conv5 = BatchNormalization(axis=-1)(conv5)
    # conv5 = Activation('relu')(conv5)
    # conv5 = Conv3D(int(128 / downsize_filters_factor), (3, 3, 3), padding='same')(conv5)
    # conv5 = BatchNormalization(axis=-1)(conv5)
    # conv5 = Activation('relu')(conv5)
    conv5 = Conv3D(int(128 / downsize_filters_factor), (5, 5, 5), padding='same')(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = BatchNormalization(axis=-1)(conv5)
    conv5 = Conv3D(int(128 / downsize_filters_factor), (5, 5, 5), padding='same')(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = BatchNormalization(axis=-1)(conv5)
    conv5 = add([conv5, up5])

    up6 = Conv3DTranspose(int(64 / downsize_filters_factor), (2, 2, 2), strides=(2, 2, 2), padding='same')(conv5)
    up6 = Activation('relu')(up6)
    up6 = BatchNormalization(axis=-1)(up6)
    conv6 = concatenate([up6, conv2], axis=4)  # concatenate
    # conv6 = Conv3D(int(64 / downsize_filters_factor), (3, 3, 3), padding='same')(conv6)
    # conv6 = BatchNormalization(axis=-1)(conv6)
    # conv6 = Activation('relu')(conv6)
    # conv6 = Conv3D(int(64 / downsize_filters_factor), (3, 3, 3), padding='same')(conv6)
    # conv6 = BatchNormalization(axis=-1)(conv6)
    # conv6 = Activation('relu')(conv6)
    # conv6 = Conv3D(int(64 / downsize_filters_factor), (3, 3, 3), padding='same')(conv6)
    # conv6 = BatchNormalization(axis=-1)(conv6)
    # conv6 = Activation('relu')(conv6)
    # conv6 = Conv3D(int(64 / downsize_filters_factor), (3, 3, 3), padding='same')(conv6)
    # conv6 = BatchNormalization(axis=-1)(conv6)
    # conv6 = Activation('relu')(conv6)
    conv6 = Conv3D(int(64 / downsize_filters_factor), (5, 5, 5), padding='same')(conv6)
    conv6 = Activation('relu')(conv6)
    conv6 = BatchNormalization(axis=-1)(conv6)
    conv6 = Conv3D(int(64 / downsize_filters_factor), (5, 5, 5), padding='same')(conv6)
    conv6 = Activation('relu')(conv6)
    conv6 = BatchNormalization(axis=-1)(conv6)
    conv6 = add([conv6, up6])

    up7 = Conv3DTranspose(int(32 / downsize_filters_factor), (2, 2, 2), strides=(2, 2, 2), padding='same')(conv6)
    up7 = Activation('relu')(up7)
    up7 = BatchNormalization(axis=-1)(up7)
    conv7 = concatenate([up7, conv1], axis=4)  # concatenate
    conv7 = Conv3D(int(32 / downsize_filters_factor), (5, 5, 5), padding='same')(conv7)
    conv7 = Activation('relu')(conv7)
    conv7 = BatchNormalization(axis=-1)(conv7)

    # conv7 = Conv3D(int(32 / downsize_filters_factor), (3, 3, 3), padding='same')(conv7)
    # conv7 = BatchNormalization(axis=-1)(conv7)
    # conv7 = Activation('relu')(conv7)
    # conv7 = Conv3D(int(32 / downsize_filters_factor), (3, 3, 3), padding='same')(conv7)
    # conv7 = BatchNormalization(axis=-1)(conv7)
    # conv7 = Activation('relu')(conv7)
    conv7 = add([conv7, up7])

    conv8 = Conv3D(4, (1, 1, 1))(conv7)
    # conv8 = BatchNormalization(axis=-1)(conv8)
    conv8 = Activation('relu')(conv8)
    # act = Activation('sigmoid')(conv8)
    act = Activation('softmax')(conv8)
    model = Model(inputs=inputs, outputs=act)

    return model


def model_3dunet_res(train_flag=True, downsize_filters_factor=1, pool_size=(2, 2, 2), n_labels=1,
                     initial_learning_rate=0.00001, deconvolution=False):
    """
    Builds the 3D UNet Keras model.
    :param input_shape: Shape of the input data (n_chanels, x_size, y_size, z_size).
    :param downsize_filters_factor: Factor to which to reduce the number of filters. Making this value larger will
    reduce the amount of memory the model will need during training.
    :param pool_size: Pool size for the max pooling operations.
    :param n_labels: Number of binary labels that the model is learning.
    :param initial_learning_rate: Initial learning rate for the model. This will be decayed during training.
    :param deconvolution: If set to True, will use transpose convolution(deconvolution) instead of upsamping. This
    increases the amount memory required during training.
    :return: Untrained 3D UNet Model
    """
    if (train_flag):
        input_shape = (IMG_Z, IMG_ROWS, IMG_COLS, 1)
    else:
        input_shape = (IMG_Z, TESTIMG_ROWS, TESTIMG_COLS, 1)
    inputs = Input(input_shape)
    conv1 = Conv3D(int(32 / downsize_filters_factor), (3, 3, 3), activation='relu', padding='same')(inputs)
    conv1 = BatchNormalization(axis=-1, name='bnconv11')(conv1)
    conv1 = Conv3D(int(32 / downsize_filters_factor), (3, 3, 3), activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization(axis=-1, name='bnconv12')(conv1)
    res1 = Conv3D(int(32 / downsize_filters_factor), (1, 1, 1), activation='relu', padding='same')(inputs)
    res1 = BatchNormalization(axis=-1, name='res1')(res1)
    add1 = add([conv1, res1])
    pool1 = MaxPooling3D(pool_size=pool_size)(add1)

    conv2 = Conv3D(int(64 / downsize_filters_factor), (3, 3, 3), activation='relu', padding='same')(pool1)
    conv2 = BatchNormalization(axis=-1, name='bnconv21')(conv2)
    conv2 = Conv3D(int(64 / downsize_filters_factor), (3, 3, 3), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization(axis=-1, name='bnconv22')(conv2)
    res2 = Conv3D(int(64 / downsize_filters_factor), (1, 1, 1), activation='relu', padding='same')(pool1)
    res2 = BatchNormalization(axis=-1, name='res2')(res2)
    add2 = add([conv2, res2])
    pool2 = MaxPooling3D(pool_size=pool_size)(add2)

    conv3 = Conv3D(int(128 / downsize_filters_factor), (3, 3, 3), activation='relu', padding='same')(pool2)
    conv3 = BatchNormalization(axis=-1, name='bnconv31')(conv3)
    conv3 = Conv3D(int(128 / downsize_filters_factor), (3, 3, 3), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization(axis=-1, name='bnconv32')(conv3)
    res3 = Conv3D(int(128 / downsize_filters_factor), (1, 1, 1), activation='relu', padding='same')(pool2)
    res3 = BatchNormalization(axis=-1, name='res3')(res3)
    add3 = add([conv3, res3])
    pool3 = MaxPooling3D(pool_size=pool_size)(add3)

    conv4 = Conv3D(int(256 / downsize_filters_factor), (3, 3, 3), activation='relu', padding='same')(pool3)
    conv4 = BatchNormalization(axis=-1, name='bnconv41')(conv4)
    conv4 = Conv3D(int(256 / downsize_filters_factor), (3, 3, 3), activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization(axis=-1, name='bnconv42')(conv4)
    res4 = Conv3D(int(256 / downsize_filters_factor), (1, 1, 1), activation='relu', padding='same')(pool3)
    res4 = BatchNormalization(axis=-1, name='res4')(res4)
    add4 = add([conv4, res4])

    # up5 = get_upconv(pool_size=pool_size, deconvolution=deconvolution, depth=2,
    #                  nb_filters=int(512 / downsize_filters_factor), image_shape=input_shape[-3:])(add4)
    up5 = Conv3DTranspose(128, kernel_size=(3, 3, 3), strides=(2, 2, 2), activation='relu', padding='same')(add4)
    up5 = BatchNormalization(axis=-1, name='bnup5')(up5)
    up5 = concatenate([up5, add3], axis=4)
    conv5 = Conv3D(int(128 / downsize_filters_factor), (3, 3, 3), activation='relu', padding='same')(up5)
    conv5 = BatchNormalization(axis=-1, name='bnconv51')(conv5)
    conv5 = Conv3D(int(128 / downsize_filters_factor), (3, 3, 3), activation='relu', padding='same')(conv5)
    conv5 = BatchNormalization(axis=-1, name='bnconv52')(conv5)
    res5 = Conv3D(int(128 / downsize_filters_factor), (1, 1, 1), activation='relu', padding='same')(up5)
    res5 = BatchNormalization(axis=-1, name='res5')(res5)
    add5 = add([conv5, res5])

    # up6 = get_upconv(pool_size=pool_size, deconvolution=deconvolution, depth=1,
    #                  nb_filters=int(256 / downsize_filters_factor), image_shape=input_shape[-3:])(conv5)
    up6 = Conv3DTranspose(64, kernel_size=(3, 3, 3), strides=(2, 2, 2), activation='relu', padding='same')(add5)
    up6 = BatchNormalization(axis=-1, name='bnup6')(up6)
    up6 = concatenate([up6, add2], axis=4)
    conv6 = Conv3D(int(64 / downsize_filters_factor), (3, 3, 3), activation='relu', padding='same')(up6)
    conv6 = BatchNormalization(axis=-1, name='bnconv61')(conv6)
    conv6 = Conv3D(int(64 / downsize_filters_factor), (3, 3, 3), activation='relu', padding='same')(conv6)
    conv6 = BatchNormalization(axis=-1, name='bnconv62')(conv6)
    res6 = Conv3D(int(64 / downsize_filters_factor), (1, 1, 1), activation='relu', padding='same')(up6)
    res6 = BatchNormalization(axis=-1, name='res6')(res6)
    add6 = add([conv6, res6])

    # up7 = get_upconv(pool_size=pool_size, deconvolution=deconvolution, depth=0,
    #                  nb_filters=int(128 / downsize_filters_factor), image_shape=input_shape[-3:])(conv6)
    up7 = Conv3DTranspose(32, kernel_size=(3, 3, 3), strides=(2, 2, 2), activation='relu', padding='same')(add6)
    up7 = BatchNormalization(axis=-1, name='bnup7')(up7)
    up7 = concatenate([up7, add1], axis=4)
    conv7 = Conv3D(int(32 / downsize_filters_factor), (3, 3, 3), activation='relu', padding='same')(up7)
    conv7 = BatchNormalization(axis=-1, name='bnconv71')(conv7)
    conv7 = Conv3D(int(32 / downsize_filters_factor), (3, 3, 3), activation='relu', padding='same')(conv7)
    conv7 = BatchNormalization(axis=-1, name='bnconv72')(conv7)
    res7 = Conv3D(int(32 / downsize_filters_factor), (1, 1, 1), activation='relu', padding='same')(up7)
    res7 = BatchNormalization(axis=-1, name='res7')(res7)
    add7 = add([conv7, res7])

    conv8 = Conv3D(4, (1, 1, 1))(add7)
    # act = Activation('sigmoid')(conv8)
    act = Activation('softmax')(conv8)
    model = Model(inputs=inputs, outputs=act)

    return model


def model_unet_map_fusion(train_flag=True, downsize_filters_factor=1, pool_size=(2, 2), n_labels=1,
                          initial_learning_rate=0.00001, deconvolution=False):
    """#model_unet_map_fusion
    Builds the 3D UNet Keras model.
    :param input_shape: Shape of the input data (n_chanels, x_size, y_size, z_size).
    :param downsize_filters_factor: Factor to which to reduce the number of filters. Making this value larger will
    reduce the amount of memory the model will need during training.
    :param pool_size: Pool size for the max pooling operations.
    :param n_labels: Number of binary labels that the model is learning.
    :param initial_learning_rate: Initial learning rate for the model. This will be decayed during training.
    :param deconvolution: If set to True, will use transpose convolution(deconvolution) instead of upsamping. This
    increases the amount memory required during training.
    :return: Untrained 3D UNet Model
    """
    if (train_flag):
        input_shape = (IMG_ROWS, IMG_COLS, 1)
    else:
        input_shape = (TESTIMG_ROWS, TESTIMG_COLS, 1)
    inputs = Input(input_shape)
    conv1 = Conv2D(int(32 / downsize_filters_factor), (3, 3), activation='relu', padding='same')(inputs)
    conv1 = BatchNormalization(axis=-1, name='bnconv11')(conv1)
    conv1 = Conv2D(int(32 / downsize_filters_factor), (3, 3), activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization(axis=-1, name='bnconv12')(conv1)
    # res1=Conv2D(int(32 / downsize_filters_factor), (1, 1), activation='relu', padding='same')(inputs)
    # res1=BatchNormalization(axis=-1, name='res1')(res1)
    # add1=add([conv1,conv1])
    pool1 = MaxPooling2D(pool_size=pool_size)(conv1)

    conv2 = Conv2D(int(64 / downsize_filters_factor), (3, 3), strides=(2, 2), activation='relu', padding='same')(pool1)
    conv2 = BatchNormalization(axis=-1, name='bnconv21')(conv2)
    conv2 = Conv2D(int(64 / downsize_filters_factor), (3, 3), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization(axis=-1, name='bnconv22')(conv2)
    # res2 = Conv2D(int(64 / downsize_filters_factor), ( 1, 1), activation='relu', padding='same')(pool1)
    # res2 = BatchNormalization(axis=-1, name='res2')(res2)
    # add2 = add([conv2, conv2])
    pool2 = MaxPooling2D(pool_size=pool_size)(conv2)

    conv3 = Conv2D(int(128 / downsize_filters_factor), (3, 3), strides=(2, 2), activation='relu', padding='same')(pool2)
    conv3 = BatchNormalization(axis=-1, name='bnconv31')(conv3)
    conv3 = Conv2D(int(128 / downsize_filters_factor), (3, 3), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization(axis=-1, name='bnconv32')(conv3)
    # res3 = Conv2D(int(128 / downsize_filters_factor), ( 1, 1), activation='relu', padding='same')(pool2)
    # res3 = BatchNormalization(axis=-1, name='res3')(res3)
    # add3 = add([conv3, conv3])
    pool3 = MaxPooling2D(pool_size=pool_size)(conv3)

    conv4 = Conv2D(int(256 / downsize_filters_factor), (3, 3), strides=(2, 2), activation='relu', padding='same')(pool3)
    conv4 = BatchNormalization(axis=-1, name='bnconv41')(conv4)
    conv4 = Conv2D(int(256 / downsize_filters_factor), (3, 3), activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization(axis=-1, name='bnconv42')(conv4)
    # res4 = Conv2D(int(256 / downsize_filters_factor), ( 1, 1), activation='relu', padding='same')(pool3)
    # res4 = BatchNormalization(axis=-1, name='res4')(res4)
    # add4 = add([conv4, con4])

    # up5 = get_upconv(pool_size=pool_size, deconvolution=deconvolution, depth=2,
    #                  nb_filters=int(512 / downsize_filters_factor), image_shape=input_shape[-3:])(add4)
    up5 = Deconvolution2D(128, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(conv4)
    up5 = BatchNormalization(axis=-1, name='bnup5')(up5)
    up5 = concatenate([up5, conv3], axis=-1)
    conv5 = Conv2D(int(128 / downsize_filters_factor), (3, 3), activation='relu', padding='same')(up5)
    conv5 = BatchNormalization(axis=-1, name='bnconv51')(conv5)
    conv5 = Conv2D(int(128 / downsize_filters_factor), (3, 3), activation='relu', padding='same')(conv5)
    conv5 = BatchNormalization(axis=-1, name='bnconv52')(conv5)
    # res5 = Conv2D(int(128 / downsize_filters_factor), ( 1, 1), activation='relu', padding='same')(up5)
    # res5 = BatchNormalization(axis=-1, name='res5')(res5)
    # add5 = add([conv5, res5])

    conv101 = Conv2D(int(4), (int(1), int(1)), activation='relu')(conv5)
    conv101 = UpSampling2D(size=(int(2), int(2)), name='upp1')(conv101)

    # up6 = get_upconv(pool_size=pool_size, deconvolution=deconvolution, depth=1,
    #                  nb_filters=int(256 / downsize_filters_factor), image_shape=input_shape[-3:])(conv5)
    up6 = Deconvolution2D(64, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(conv5)
    up6 = BatchNormalization(axis=-1, name='bnup6')(up6)
    up6 = concatenate([up6, conv2], axis=-1)
    conv6 = Conv2D(int(64 / downsize_filters_factor), (3, 3), activation='relu', padding='same')(up6)
    conv6 = BatchNormalization(axis=-1, name='bnconv61')(conv6)
    conv6 = Conv2D(int(64 / downsize_filters_factor), (3, 3), activation='relu', padding='same')(conv6)
    conv6 = BatchNormalization(axis=-1, name='bnconv62')(conv6)
    # res6 = Conv2D(int(64 / downsize_filters_factor),  (1, 1), activation='relu', padding='same')(up6)
    # res6 = BatchNormalization(axis=-1, name='res6')(res6)
    # add6 = add([conv6, res6])

    conv102 = Conv2D(int(4), (int(1), int(1)), activation='relu')(conv6)
    conv102 = add([conv101, conv102], name='addd1')
    conv102 = UpSampling2D(size=(int(2), int(2)), name='upp2')(conv102)

    # up7 = get_upconv(pool_size=pool_size, deconvolution=deconvolution, depth=0,
    #                  nb_filters=int(128 / downsize_filters_factor), image_shape=input_shape[-3:])(conv6)
    up7 = Deconvolution2D(32, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(conv6)
    up7 = BatchNormalization(axis=-1, name='bnup7')(up7)
    up7 = concatenate([up7, conv1], axis=-1)
    conv7 = Conv2D(int(32 / downsize_filters_factor), (3, 3), activation='relu', padding='same')(up7)
    conv7 = BatchNormalization(axis=-1, name='bnconv71')(conv7)
    conv7 = Conv2D(int(32 / downsize_filters_factor), (3, 3), activation='relu', padding='same')(conv7)
    conv7 = BatchNormalization(axis=-1, name='bnconv72')(conv7)
    # res7 = Conv2D(int(32 / downsize_filters_factor), ( 1, 1), activation='relu', padding='same')(up7)
    # res7 = BatchNormalization(axis=-1, name='res7')(res7)
    # add7 = add([conv7, res7])

    conv8 = Conv2D(4, (1, 1))(conv7)

    conv8 = add([conv8, conv102])
    # act = Activation('sigmoid')(conv8)
    act = Activation('softmax')(conv8)
    model = Model(inputs=inputs, outputs=act)

    return model


def time_GRU_unet_1_level_bn_resresult(train_flag=True):
    if (train_flag == True):
        inputs = (IMG_Z, IMG_ROWS, IMG_COLS, 1)
    else:
        inputs = (IMG_Z, TESTIMG_ROWS, TESTIMG_COLS, 1)
    inputs = Input(inputs)
    res1 = res_block(inputs, [int(32), int(32)], [(int(1), int(1)), (int(1), int(1))])
    # pool1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(res1)
    # print res1.shape
    res2 = res_block(res1, [int(64), int(64)], [(int(2), int(2)), (int(1), int(1))])
    # pool2 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(res2)

    res3 = res_block(res2, [int(128), int(128)], [(int(2), int(2)), (int(1), int(1))])
    #  pool3 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(res3)

    res4 = res_block(res3, [int(256), int(256)], [(int(2), int(2)), (int(1), int(1))])
    conv4 = two_lstm(res4, int(256), int(3), int(3))
    # ------------------xiugai--------------------//

    '''
    conv5 = TimeDistributed(Convolution2D(512, 3, 3, activation='relu', border_mode='same'))(pool4)
    conv5 = BatchNormalization(axis=-1, name='bnconv51')(conv5)
    conv5 = time_ConvGRU_bottleNeck_block(conv5, 512, 3, 3)
    conv5 = TimeDistributed(Convolution2D(512, 3, 3, activation='relu', border_mode='same'))(conv5)
    conv5 = BatchNormalization(axis=-1, name='bnconv52')(conv5)
    '''
    '''
    up6 = merge([TimeDistributed(UpSampling2D(size=(2, 2)))(conv5), conv4], mode='concat', concat_axis=-1)
    conv6 = TimeDistributed(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))(up6)
    conv6 = BatchNormalization(axis=-1, name='bnconv61')(conv6)
    conv6 = TimeDistributed(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))(conv6)
    conv6 = BatchNormalization(axis=-1, name='bnconv62')(conv6)
    '''
    up7 = concatenate([TimeDistributed(UpSampling2D(size=(2, 2)))(conv4), res3], axis=-1)
    # up7 = concatenate([TimeDistributed(UpSampling2D(size=(int(2), int(2))))(conv4), res3], axis=-1)
    res5 = res_block(up7, [int(128), int(128)], [(int(1), int(1)), (int(1), int(1))])
    conv101 = TimeDistributed(Conv2D(int(4), (int(1), int(1)), activation='relu'))(res5)
    conv101 = TimeDistributed(UpSampling2D(size=(int(2), int(2)), name='upp1'), name="1")(conv101)

    up8 = concatenate([TimeDistributed(UpSampling2D(size=(int(2), int(2))))(res5), res2], axis=-1)
    res6 = res_block(up8, [int(64), int(64)], [(int(1), int(1)), (int(1), int(1))])
    conv102 = TimeDistributed(Conv2D(int(4), (int(1), int(1)), activation='relu'))(res6)
    conv102 = add([conv101, conv102], name='addd1')
    conv102 = TimeDistributed(UpSampling2D(size=(int(2), int(2)), name='upp2'), name='2')(conv102)

    up9 = concatenate([TimeDistributed(UpSampling2D(size=(int(2), int(2))))(res6), res1], axis=-1)
    res7 = res_block(up9, [int(32), int(32)], [(int(1), int(1)), (int(1), int(1))])
    # up10 = TimeDistributed(UpSampling2D(size=(2, 2)))(res7)
    conv10 = TimeDistributed(Conv2D(int(4), (int(1), int(1)), activation='relu'))(res7)
    conv10 = add([conv10, conv102], name='addd2')
    # conv10 = BatchNormalization(axis=-1, name='bnconv10')(conv10)
    out = Activation('softmax')(conv10)
    # layers['outputs'] = core.Activation('softmax')(layers['outputs'])
    # out = Lambda(time_dist_softmax, output_shape=time_dist_softmax_out_shape)(conv10)
    model = Model(inputs=inputs, outputs=out)
    return model


def time_GRU_unet_1_level_bn_reswulstm(inputs=(IMG_Z, IMG_ROWS, IMG_COLS, 1)):
    inputs = Input(inputs)
    res1 = res_block(inputs, [32, 32], [(1, 1), (1, 1)])
    # pool1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(res1)
    # print res1.shape
    res2 = res_block(res1, [64, 64], [(2, 2), (1, 1)])
    # pool2 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(res2)

    res3 = res_block(res2, [128, 128], [(2, 2), (1, 1)])
    #  pool3 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(res3)

    res4 = res_block(res3, [256, 256], [(2, 2), (1, 1)])
    conv4 = two_lstmwulstm(res4, 256, 3, 3)

    '''
    conv5 = TimeDistributed(Convolution2D(512, 3, 3, activation='relu', border_mode='same'))(pool4)
    conv5 = BatchNormalization(axis=-1, name='bnconv51')(conv5)
    conv5 = time_ConvGRU_bottleNeck_block(conv5, 512, 3, 3)
    conv5 = TimeDistributed(Convolution2D(512, 3, 3, activation='relu', border_mode='same'))(conv5)
    conv5 = BatchNormalization(axis=-1, name='bnconv52')(conv5)
    '''
    '''
    up6 = merge([TimeDistributed(UpSampling2D(size=(2, 2)))(conv5), conv4], mode='concat', concat_axis=-1)
    conv6 = TimeDistributed(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))(up6)
    conv6 = BatchNormalization(axis=-1, name='bnconv61')(conv6)
    conv6 = TimeDistributed(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))(conv6)
    conv6 = BatchNormalization(axis=-1, name='bnconv62')(conv6)
    '''
    up7 = merge([TimeDistributed(UpSampling2D(size=(2, 2)))(conv4), res3], mode='concat', concat_axis=-1)
    res5 = res_block(up7, [128, 128], [(1, 1), (1, 1)])
    conv101 = TimeDistributed(Convolution2D(5, 1, 1, activation='relu'))(res5)
    conv101 = TimeDistributed(UpSampling2D(size=(2, 2), name='upp1'), name="1")(conv101)

    up8 = merge([TimeDistributed(UpSampling2D(size=(2, 2)))(res5), res2], mode='concat', concat_axis=-1)
    res6 = res_block(up8, [64, 64], [(1, 1), (1, 1)])
    conv102 = TimeDistributed(Convolution2D(5, 1, 1, activation='relu'))(res6)
    conv102 = add([conv101, conv102], name='addd1')
    conv102 = TimeDistributed(UpSampling2D(size=(2, 2), name='upp2'), name='2')(conv102)

    up9 = merge([TimeDistributed(UpSampling2D(size=(2, 2)))(res6), res1], mode='concat', concat_axis=-1)
    res7 = res_block(up9, [32, 32], [(1, 1), (1, 1)])
    # up10 = TimeDistributed(UpSampling2D(size=(2, 2)))(res7)
    conv10 = TimeDistributed(Convolution2D(5, 1, 1, activation='relu'))(res7)
    conv10 = add([conv10, conv102], name='addd2')
    # conv10 = BatchNormalization(axis=-1, name='bnconv10')(conv10)
    out = Activation('softmax')(conv10)
    # layers['outputs'] = core.Activation('softmax')(layers['outputs'])
    # out = Lambda(time_dist_softmax, output_shape=time_dist_softmax_out_shape)(conv10)
    model = Model(inputs=inputs, outputs=out)
    return model


def time_GRU_unet_1_level_bn_reswulstmxxx(inputs=(IMG_Z, IMG_ROWS, IMG_COLS, 1)):
    inputs = Input(inputs)
    res1 = res_block(inputs, [32, 32], [(1, 1), (1, 1)])
    # pool1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(res1)
    # print res1.shape
    res2 = res_block(res1, [64, 64], [(2, 2), (1, 1)])
    # pool2 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(res2)

    res3 = res_block(res2, [128, 128], [(2, 2), (1, 1)])
    #  pool3 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(res3)

    res4 = res_block(res3, [256, 256], [(2, 2), (1, 1)])
    conv4 = one_lstm(res4, 256, 3, 3)

    '''
    conv5 = TimeDistributed(Convolution2D(512, 3, 3, activation='relu', border_mode='same'))(pool4)
    conv5 = BatchNormalization(axis=-1, name='bnconv51')(conv5)
    conv5 = time_ConvGRU_bottleNeck_block(conv5, 512, 3, 3)
    conv5 = TimeDistributed(Convolution2D(512, 3, 3, activation='relu', border_mode='same'))(conv5)
    conv5 = BatchNormalization(axis=-1, name='bnconv52')(conv5)
    '''
    '''
    up6 = merge([TimeDistributed(UpSampling2D(size=(2, 2)))(conv5), conv4], mode='concat', concat_axis=-1)
    conv6 = TimeDistributed(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))(up6)
    conv6 = BatchNormalization(axis=-1, name='bnconv61')(conv6)
    conv6 = TimeDistributed(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))(conv6)
    conv6 = BatchNormalization(axis=-1, name='bnconv62')(conv6)
    '''
    up7 = merge([TimeDistributed(UpSampling2D(size=(2, 2)))(conv4), res3], mode='concat', concat_axis=-1)
    res5 = res_block(up7, [128, 128], [(1, 1), (1, 1)])
    conv101 = TimeDistributed(Convolution2D(5, 1, 1, activation='relu'))(res5)
    conv101 = TimeDistributed(UpSampling2D(size=(2, 2), name='upp1'), name="1")(conv101)

    up8 = merge([TimeDistributed(UpSampling2D(size=(2, 2)))(res5), res2], mode='concat', concat_axis=-1)
    res6 = res_block(up8, [64, 64], [(1, 1), (1, 1)])
    conv102 = TimeDistributed(Convolution2D(5, 1, 1, activation='relu'))(res6)
    conv102 = add([conv101, conv102], name='addd1')
    conv102 = TimeDistributed(UpSampling2D(size=(2, 2), name='upp2'), name='2')(conv102)

    up9 = merge([TimeDistributed(UpSampling2D(size=(2, 2)))(res6), res1], mode='concat', concat_axis=-1)
    res7 = res_block(up9, [32, 32], [(1, 1), (1, 1)])
    # up10 = TimeDistributed(UpSampling2D(size=(2, 2)))(res7)
    conv10 = TimeDistributed(Convolution2D(5, 1, 1, activation='relu'))(res7)
    conv10 = add([conv10, conv102], name='addd2')
    # conv10 = BatchNormalization(axis=-1, name='bnconv10')(conv10)
    out = Activation('softmax')(conv10)
    # layers['outputs'] = core.Activation('softmax')(layers['outputs'])
    # out = Lambda(time_dist_softmax, output_shape=time_dist_softmax_out_shape)(conv10)
    model = Model(inputs=inputs, outputs=out)
    return model


def time_ConvGRU_bottleNeck_block(x, filters, row, col):
    reduced_filters = filters
    if filters >= 8:
        reduced_filters = int(round(filters / 1))
    x = TimeDistributed(Convolution2D(nb_filter=reduced_filters, nb_row=1, nb_col=1, activation='relu', border_mode='same', name="qian_conv"))(x)
    x = BatchNormalization(axis=-1, name='bnconvx1')(x)
    # x = Bidirectional(ConvGRU2D(nb_filter=reduced_filters, nb_row=row, nb_col=col,init='glorot_uniform', dim_ordering='tf', border_mode='same',return_sequences=True),merge_mode='sum')(x)
    x = Bidirectional(keras.layers.ConvLSTM2D(reduced_filters, kernel_size=(3, 3), data_format='channels_last', padding='same', return_sequences=True, name="conv_lstm"))(x)

    # keras.layers.ConvLSTM2D(filters, kernel_size, strides=(1, 1), padding='same')
    x = TimeDistributed(Convolution2D(nb_filter=reduced_filters, nb_row=1, nb_col=1, activation='relu', border_mode='same', name="hou_conv"))(x)
    x = BatchNormalization(axis=-1, name='bnconvx2')(x)
    return x


def one_lstm(x, filters, row, col):
    reduced_filters = filters
    if filters >= 8:
        reduced_filters = int(round(filters / 1))
    x = TimeDistributed(Convolution2D(nb_filter=reduced_filters, nb_row=1, nb_col=1, activation='relu', border_mode='same', name="qian_conv"))(x)
    x = BatchNormalization(axis=-1, name='bnconvx1')(x)
    # x = Bidirectional(ConvGRU2D(nb_filter=reduced_filters, nb_row=row, nb_col=col,init='glorot_uniform', dim_ordering='tf', border_mode='same',return_sequences=True),merge_mode='sum')(x)
    x = Bidirectional(keras.layers.ConvLSTM2D(reduced_filters, kernel_size=(3, 3), data_format='channels_last', padding='same', return_sequences=True, name="conv_lstm"))(x)

    # keras.layers.ConvLSTM2D(filters, kernel_size, strides=(1, 1), padding='same')
    x = TimeDistributed(Convolution2D(nb_filter=reduced_filters, nb_row=1, nb_col=1, activation='relu', border_mode='same', name="hou_conv"))(x)
    x = BatchNormalization(axis=-1, name='bnconvx2')(x)
    return x


def two_lstm(x, filters, row, col):
    reduced_filters = filters
    # if filters >=8:
    #     reduced_filters =int(round(filters/1))
    # x = TimeDistributed(Conv2D(nb_filter=reduced_filters, nb_row=1, nb_col=1, activation='relu',border_mode='same',name="qian_conv"))(x)
    x = TimeDistributed(Conv2D(name="qian_conv", activation="relu", padding="same", filters=reduced_filters, kernel_size=(1, 1)))(x)

    x = BatchNormalization(axis=-1, name='bnconvx1')(x)
    # x = Bidirectional(ConvGRU2D(nb_filter=reduced_filters, nb_row=row, nb_col=col,init='glorot_uniform', dim_ordering='tf', border_mode='same',return_sequences=True),merge_mode='sum')(x)
    x = Bidirectional(keras.layers.ConvLSTM2D(int(reduced_filters / 2), kernel_size=(int(3), int(3)), data_format='channels_last', padding='same', return_sequences=True, name="conv_lstm"))(x)
    x = BatchNormalization(axis=-1, name='bnconvx2')(x)
    # x = TimeDistributed(Conv2D(nb_filter=reduced_filters, nb_row=1, nb_col=1, activation='relu',border_mode='same',name="zhong_conv"))(x)
    x = TimeDistributed(Conv2D(name="zhong_conv", activation="relu", padding="same", filters=reduced_filters, kernel_size=(1, 1)))(x)

    x = BatchNormalization(axis=-1, name='bnconvx3')(x)
    x = Bidirectional(
        keras.layers.ConvLSTM2D(int(reduced_filters / 2), kernel_size=(int(3), int(3)), data_format='channels_last', padding='same',
                                return_sequences=True, name="conv_lstm1"))(x)
    x = BatchNormalization(axis=-1, name='bnconvx4')(x)
    # keras.layers.ConvLSTM2D(filters, kernel_size, strides=(1, 1), padding='same')
    # x = TimeDistributed(Conv2D(nb_filter=reduced_filters, nb_row=1, nb_col=1, activation='relu',border_mode='same',name="hou_conv"))(x)
    x = TimeDistributed(Conv2D(name="hou_conv", activation="relu", padding="same", filters=reduced_filters, kernel_size=(1, 1)))(x)

    x = BatchNormalization(axis=-1, name='bnconvx5')(x)
    return x


def two_lstmwulstm(x, filters, row, col):
    reduced_filters = filters
    if filters >= 8:
        reduced_filters = int(round(filters / 1))
    x = TimeDistributed(Convolution2D(nb_filter=reduced_filters, nb_row=1, nb_col=1, activation='relu', border_mode='same', name="qian_conv"))(x)
    x = BatchNormalization(axis=-1, name='bnconvx1')(x)
    x = TimeDistributed(Convolution2D(nb_filter=reduced_filters, nb_row=1, nb_col=1, activation='relu', border_mode='same', name="qian_conv"))(x)
    x = BatchNormalization(axis=-1, name='bnconvx6')(x)
    # x = Bidirectional(ConvGRU2D(nb_filter=reduced_filters, nb_row=row, nb_col=col,init='glorot_uniform', dim_ordering='tf', border_mode='same',return_sequences=True),merge_mode='sum')(x)
    #  x = Bidirectional(keras.layers.ConvLSTM2D(reduced_filters/2, kernel_size=(3,3), data_format='channels_last', padding='same',return_sequences=True,name="conv_lstm"))(x)
    #   x = BatchNormalization(axis=-1, name='bnconvx2')(x)
    x = TimeDistributed(Convolution2D(nb_filter=reduced_filters, nb_row=1, nb_col=1, activation='relu', border_mode='same', name="zhong_conv"))(x)
    x = BatchNormalization(axis=-1, name='bnconvx3')(x)
    x = TimeDistributed(Convolution2D(nb_filter=reduced_filters, nb_row=1, nb_col=1, activation='relu', border_mode='same', name="qian_conv"))(x)
    x = BatchNormalization(axis=-1, name='bnconvx7')(x)
    #  x = Bidirectional(
    #      keras.layers.ConvLSTM2D(reduced_filters / 2, kernel_size=(3, 3), data_format='channels_last', padding='same',
    #                               return_sequences=True, name="conv_lstm1"))(x)
    x = BatchNormalization(axis=-1, name='bnconvx4')(x)
    # keras.layers.ConvLSTM2D(filters, kernel_size, strides=(1, 1), padding='same')
    x = TimeDistributed(Convolution2D(nb_filter=reduced_filters, nb_row=1, nb_col=1, activation='relu', border_mode='same', name="hou_conv"))(x)
    x = BatchNormalization(axis=-1, name='bnconvx5')(x)
    return x


def time_dist_softmax(x):
    assert K.ndim(x) == 5
    # e = K.exp(x - K.max(x, axis=2, keepdims=True))
    e = K.exp(x)
    s = K.sum(e, axis=2, keepdims=True)
    return e / s


def time_dist_softmax_out_shape(input_shape):
    shape = list(input_shape)
    return tuple(shape)


def time_GRU_unet_1_level_bnxiugai(inputs=(IMG_Z, IMG_ROWS, IMG_COLS, 1)):
    inputs = Input(inputs)
    conv1 = TimeDistributed(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))(inputs)
    conv1 = BatchNormalization(axis=-1, name='bnconv11')(conv1)
    conv1 = TimeDistributed(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))(conv1)
    conv1 = BatchNormalization(axis=-1, name='bnconv12')(conv1)
    pool1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv1)

    conv2 = TimeDistributed(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))(pool1)
    conv2 = BatchNormalization(axis=-1, name='bnconv21')(conv2)
    conv2 = TimeDistributed(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))(conv2)
    conv2 = BatchNormalization(axis=-1, name='bnconv22')(conv2)
    pool2 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv2)

    conv3 = TimeDistributed(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))(pool2)
    conv3 = BatchNormalization(axis=-1, name='bnconv31')(conv3)
    conv3 = TimeDistributed(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))(conv3)
    conv3 = BatchNormalization(axis=-1, name='bnconv32')(conv3)
    pool3 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv3)

    conv4 = TimeDistributed(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))(pool3)
    conv4 = BatchNormalization(axis=-1, name='bnconv41')(conv4)
    conv4 = TimeDistributed(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))(conv4)
    conv4 = time_ConvGRU_bottleNeck_block(conv4, 256, 3, 3)
    conv4 = BatchNormalization(axis=-1, name='bnconv42')(conv4)
    pool4 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv4)
    '''
    conv5 = TimeDistributed(Convolution2D(512, 3, 3, activation='relu', border_mode='same'))(pool4)
    conv5 = BatchNormalization(axis=-1, name='bnconv51')(conv5)
    conv5 = time_ConvGRU_bottleNeck_block(conv5, 512, 3, 3)
    conv5 = TimeDistributed(Convolution2D(512, 3, 3, activation='relu', border_mode='same'))(conv5)
    conv5 = BatchNormalization(axis=-1, name='bnconv52')(conv5)
    '''
    '''
    up6 = merge([TimeDistributed(UpSampling2D(size=(2, 2)))(conv5), conv4], mode='concat', concat_axis=-1)
    conv6 = TimeDistributed(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))(up6)
    conv6 = BatchNormalization(axis=-1, name='bnconv61')(conv6)
    conv6 = TimeDistributed(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))(conv6)
    conv6 = BatchNormalization(axis=-1, name='bnconv62')(conv6)
    '''
    up7 = merge([TimeDistributed(UpSampling2D(size=(2, 2)))(conv4), conv3], mode='concat', concat_axis=-1)
    conv7 = TimeDistributed(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))(up7)
    conv7 = BatchNormalization(axis=-1, name='bnconv71')(conv7)
    conv7 = TimeDistributed(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))(conv7)
    conv7 = BatchNormalization(axis=-1, name='bnconv72')(conv7)

    up8 = merge([TimeDistributed(UpSampling2D(size=(2, 2)))(conv7), conv2], mode='concat', concat_axis=-1)
    conv8 = TimeDistributed(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))(up8)
    conv8 = BatchNormalization(axis=-1, name='bnconv81')(conv8)
    conv8 = TimeDistributed(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))(conv8)
    conv8 = BatchNormalization(axis=-1, name='bnconv82')(conv8)

    up9 = merge([TimeDistributed(UpSampling2D(size=(2, 2)))(conv8), conv1], mode='concat', concat_axis=-1)
    conv9 = TimeDistributed(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))(up9)
    conv9 = BatchNormalization(axis=-1, name='bnconv91')(conv9)
    conv9 = TimeDistributed(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))(conv9)
    conv9 = BatchNormalization(axis=-1, name='bnconv92')(conv9)
    conv10 = TimeDistributed(Convolution2D(5, 1, 1, activation='relu'))(conv9)
    conv10 = BatchNormalization(axis=-1, name='bnconv10')(conv10)
    out = Activation('softmax')(conv10)
    # layers['outputs'] = core.Activation('softmax')(layers['outputs'])
    # out = Lambda(time_dist_softmax, output_shape=time_dist_softmax_out_shape)(conv10)
    model = Model(inputs=inputs, outputs=out)
    return model


def time_GRU_unet_1_level_bn(inputs=(IMG_Z, IMG_ROWS, IMG_COLS, 1)):
    inputs = Input(inputs)
    conv1 = TimeDistributed(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))(inputs)
    conv1 = BatchNormalization(axis=-1, name='bnconv11')(conv1)
    conv1 = TimeDistributed(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))(conv1)
    conv1 = BatchNormalization(axis=-1, name='bnconv12')(conv1)
    pool1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv1)

    conv2 = TimeDistributed(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))(pool1)
    conv2 = BatchNormalization(axis=-1, name='bnconv21')(conv2)
    conv2 = TimeDistributed(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))(conv2)
    conv2 = BatchNormalization(axis=-1, name='bnconv22')(conv2)
    pool2 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv2)

    conv3 = TimeDistributed(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))(pool2)
    conv3 = BatchNormalization(axis=-1, name='bnconv31')(conv3)
    conv3 = TimeDistributed(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))(conv3)
    conv3 = BatchNormalization(axis=-1, name='bnconv32')(conv3)
    pool3 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv3)

    conv4 = TimeDistributed(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))(pool3)
    conv4 = BatchNormalization(axis=-1, name='bnconv41')(conv4)
    conv4 = TimeDistributed(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))(conv4)
    conv4 = two_lstm(conv4, 256, 3, 3)
    conv4 = BatchNormalization(axis=-1, name='bnconv42')(conv4)
    pool4 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv4)
    '''
    conv5 = TimeDistributed(Convolution2D(512, 3, 3, activation='relu', border_mode='same'))(pool4)
    conv5 = BatchNormalization(axis=-1, name='bnconv51')(conv5)
    conv5 = time_ConvGRU_bottleNeck_block(conv5, 512, 3, 3)
    conv5 = TimeDistributed(Convolution2D(512, 3, 3, activation='relu', border_mode='same'))(conv5)
    conv5 = BatchNormalization(axis=-1, name='bnconv52')(conv5)
    '''
    '''
    up6 = merge([TimeDistributed(UpSampling2D(size=(2, 2)))(conv5), conv4], mode='concat', concat_axis=-1)
    conv6 = TimeDistributed(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))(up6)
    conv6 = BatchNormalization(axis=-1, name='bnconv61')(conv6)
    conv6 = TimeDistributed(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))(conv6)
    conv6 = BatchNormalization(axis=-1, name='bnconv62')(conv6)
    '''
    up7 = merge([TimeDistributed(UpSampling2D(size=(2, 2)))(conv4), conv3], mode='concat', concat_axis=-1)
    conv7 = TimeDistributed(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))(up7)
    conv7 = BatchNormalization(axis=-1, name='bnconv71')(conv7)
    conv7 = TimeDistributed(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))(conv7)
    conv7 = BatchNormalization(axis=-1, name='bnconv72')(conv7)

    up8 = merge([TimeDistributed(UpSampling2D(size=(2, 2)))(conv7), conv2], mode='concat', concat_axis=-1)
    conv8 = TimeDistributed(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))(up8)
    conv8 = BatchNormalization(axis=-1, name='bnconv81')(conv8)
    conv8 = TimeDistributed(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))(conv8)
    conv8 = BatchNormalization(axis=-1, name='bnconv82')(conv8)

    up9 = merge([TimeDistributed(UpSampling2D(size=(2, 2)))(conv8), conv1], mode='concat', concat_axis=-1)
    conv9 = TimeDistributed(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))(up9)
    conv9 = BatchNormalization(axis=-1, name='bnconv91')(conv9)
    conv9 = TimeDistributed(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))(conv9)
    conv9 = BatchNormalization(axis=-1, name='bnconv92')(conv9)
    conv10 = TimeDistributed(Convolution2D(5, 1, 1, activation='relu'))(conv9)
    conv10 = BatchNormalization(axis=-1, name='bnconv10')(conv10)
    out = Activation('softmax')(conv10)
    # layers['outputs'] = core.Activation('softmax')(layers['outputs'])
    # out = Lambda(time_dist_softmax, output_shape=time_dist_softmax_out_shape)(conv10)
    model = Model(inputs=inputs, outputs=out)
    return model


def unet_lstm(inputs=(IMG_Z, IMG_ROWS, IMG_COLS)):
    inputs = Input(inputs)
    conv1 = TimeDistributed(Conv2D(32, (3, 3), padding="same", activation="relu"))(inputs)
    conv1 = BatchNormalization(axis=-1, name='bnconv11')(conv1)
    conv1 = TimeDistributed(Conv2D(32, (3, 3), padding="same", activation="relu"))(conv1)
    conv1 = BatchNormalization(axis=-1, name='bnconv12')(conv1)
    pool1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv1)

    conv2 = TimeDistributed(Conv2D(64, (3, 3), padding="same", activation="relu"))(pool1)
    conv2 = BatchNormalization(axis=-1, name='bnconv21')(conv2)
    conv2 = TimeDistributed(Conv2D(64, (3, 3), padding="same", activation="relu"))(conv2)
    conv2 = BatchNormalization(axis=-1, name='bnconv22')(conv2)
    pool2 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv2)

    conv3 = TimeDistributed(Conv2D(128, (3, 3), padding="same", activation="relu"))(pool2)
    conv3 = BatchNormalization(axis=-1, name='bnconv31')(conv3)
    conv3 = TimeDistributed(Conv2D(128, (3, 3), padding="same", activation="relu"))(conv3)
    conv3 = BatchNormalization(axis=-1, name='bnconv32')(conv3)
    pool3 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv3)

    conv4 = TimeDistributed(Conv2D(256, (3, 3), padding="same", activation="relu"))(pool3)
    conv4 = BatchNormalization(axis=-1, name='bnconv41')(conv4)
    conv4 = TimeDistributed(Conv2D(256, (3, 3), padding="same", activation="relu"))(conv4)
    # conv4 = time_ConvGRU_bottleNeck_block(conv4, 256, 3, 3)
    conv4 = BatchNormalization(axis=-1, name='bnconv42')(conv4)
    # pool4 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv4)
    '''
    conv5 = TimeDistributed(Convolution2D(512, 3, 3, activation='relu', border_mode='same'))(pool4)
    conv5 = BatchNormalization(axis=-1, name='bnconv51')(conv5)
    conv5 = time_ConvGRU_bottleNeck_block(conv5, 512, 3, 3)
    conv5 = TimeDistributed(Convolution2D(512, 3, 3, activation='relu', border_mode='same'))(conv5)
    conv5 = BatchNormalization(axis=-1, name='bnconv52')(conv5)
    '''
    '''
    up6 = merge([TimeDistributed(UpSampling2D(size=(2, 2)))(conv5), conv4], mode='concat', concat_axis=-1)
    conv6 = TimeDistributed(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))(up6)
    conv6 = BatchNormalization(axis=-1, name='bnconv61')(conv6)
    conv6 = TimeDistributed(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))(conv6)
    conv6 = BatchNormalization(axis=-1, name='bnconv62')(conv6)
    '''
    up7 = concatenate([TimeDistributed(UpSampling2D(size=(2, 2)))(conv4), conv3], axis=-1)
    conv7 = TimeDistributed(Conv2D(128, (3, 3), padding="same", activation="relu"))(up7)
    conv7 = BatchNormalization(axis=-1, name='bnconv71')(conv7)
    conv7 = TimeDistributed(Conv2D(128, (3, 3), padding="same", activation="relu"))(conv7)
    conv7 = BatchNormalization(axis=-1, name='bnconv72')(conv7)

    up8 = concatenate([TimeDistributed(UpSampling2D(size=(2, 2)))(conv7), conv2], axis=-1)
    conv8 = TimeDistributed(Conv2D(64, (3, 3), padding="same", activation="relu"))(up8)
    conv8 = BatchNormalization(axis=-1, name='bnconv81')(conv8)
    conv8 = TimeDistributed(Conv2D(64, (3, 3), padding="same", activation="relu"))(conv8)
    conv8 = BatchNormalization(axis=-1, name='bnconv82')(conv8)

    up9 = concatenate([TimeDistributed(UpSampling2D(size=(2, 2)))(conv8), conv1], axis=-1)
    conv9 = TimeDistributed(Conv2D(32, (3, 3), padding="same", activation="relu"))(up9)
    conv9 = BatchNormalization(axis=-1, name='bnconv91')(conv9)
    conv9 = TimeDistributed(Conv2D(32, (3, 3), padding="same", activation="relu"))(conv9)
    conv9 = BatchNormalization(axis=-1, name='bnconv92')(conv9)
    conv10 = TimeDistributed(Conv2D(4, (3, 3), padding="same", activation="relu"))(conv9)
    conv10 = BatchNormalization(axis=-1, name='bnconv10')(conv10)
    out = Activation('softmax')(conv10)
    # layers['outputs'] = core.Activation('softmax')(layers['outputs'])
    # out = Lambda(time_dist_softmax, output_shape=time_dist_softmax_out_shape)(conv10)
    model = Model(inputs=inputs, outputs=out)
    return model


def time_GRU_unet_1_level(inputs=(IMG_Z, IMG_ROWS, IMG_COLS, 1)):
    inputs = Input(inputs)
    conv1 = TimeDistributed(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))(inputs)
    conv1 = TimeDistributed(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))(conv1)
    pool1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv1)

    conv2 = TimeDistributed(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))(pool1)
    conv2 = TimeDistributed(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))(conv2)
    pool2 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv2)

    conv3 = TimeDistributed(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))(pool2)
    conv3 = TimeDistributed(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))(conv3)
    pool3 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv3)

    conv4 = TimeDistributed(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))(pool3)
    conv4 = TimeDistributed(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))(conv4)
    pool4 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv4)

    conv5 = TimeDistributed(Convolution2D(512, 3, 3, activation='relu', border_mode='same'))(pool4)
    conv5 = time_ConvGRU_bottleNeck_block(conv5, 512, 3, 3)
    conv5 = TimeDistributed(Convolution2D(512, 3, 3, activation='relu', border_mode='same'))(conv5)

    up6 = merge([TimeDistributed(UpSampling2D(size=(2, 2)))(conv5), conv4], mode='concat', concat_axis=-1)
    conv6 = TimeDistributed(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))(up6)
    conv6 = TimeDistributed(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))(conv6)

    up7 = merge([TimeDistributed(UpSampling2D(size=(2, 2)))(conv6), conv3], mode='concat', concat_axis=-1)
    conv7 = TimeDistributed(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))(up7)
    conv7 = TimeDistributed(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))(conv7)

    up8 = merge([TimeDistributed(UpSampling2D(size=(2, 2)))(conv7), conv2], mode='concat', concat_axis=-1)
    conv8 = TimeDistributed(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))(up8)
    conv8 = TimeDistributed(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))(conv8)

    up9 = merge([TimeDistributed(UpSampling2D(size=(2, 2)))(conv8), conv1], mode='concat', concat_axis=-1)
    conv9 = TimeDistributed(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))(up9)
    conv9 = TimeDistributed(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))(conv9)
    conv10 = TimeDistributed(Convolution2D(1, 1, 1, activation='relu'))(conv9)
    out = Activation('sigmoid')(conv10)
    # out = Lambda(time_dist_softmax, output_shape=time_dist_softmax_out_shape)(conv10)
    model = Model(inputs=inputs, outputs=out)
    return model


def compute_level_output_shape(filters, depth, pool_size, image_shape):
    """
    Each level has a particular output shape based on the number of filters used in that level and the depth or number 
    of max pooling operations that have been done on the data at that point.
    :param image_shape: shape of the 3d image.
    :param pool_size: the pool_size parameter used in the max pooling operation.
    :param filters: Number of filters used by the last node in a given level.
    :param depth: The number of levels down in the U-shaped model a given node is.
    :return: 5D vector of the shape of the output node 
    """
    if depth != 0:
        output_image_shape = np.divide(image_shape, np.multiply(pool_size, depth)).tolist()
    else:
        output_image_shape = image_shape
    return tuple([None, filters] + [int(x) for x in output_image_shape])


def get_upconv(depth, nb_filters, pool_size, image_shape, kernel_size=(2, 2, 2), strides=(2, 2, 2),
               deconvolution=False):
    if deconvolution:
        try:
            from keras_contrib.layers import Deconvolution3D
        except ImportError:
            raise ImportError("Install keras_contrib in order to use deconvolution. Otherwise set deconvolution=False.")

        return Deconvolution3D(filters=nb_filters, kernel_size=kernel_size,
                               output_shape=compute_level_output_shape(filters=nb_filters, depth=depth,
                                                                       pool_size=pool_size, image_shape=image_shape),
                               strides=strides, input_shape=compute_level_output_shape(filters=nb_filters,
                                                                                       depth=depth + 1,
                                                                                       pool_size=pool_size,
                                                                                       image_shape=image_shape))
    else:
        return UpSampling3D(size=pool_size)


def get_model(input_shape=(IMG_ROWS, IMG_COLS, 1), train=True):
    layers = {}
    layers['inputs'] = Input(shape=input_shape, name='inputs')

    layers['conv1_1'] = Conv2D(32, (3, 3), padding='same', activation='relu', name='conv1_1')(layers['inputs'])
    layers['conv1_2'] = Conv2D(32, (3, 3), padding='same', activation='relu', name='conv1_2')(layers['conv1_1'])
    layers['pool_1'] = MaxPool2D(pool_size=(2, 2), name='pool_1')(layers['conv1_2'])
    if train == True:
        layers['dropout_1'] = Dropout(0.25, name='dropout_1')(layers['pool_1'])
        layers['conv2_1'] = Conv2D(64, (3, 3), padding='same', activation='relu', name='conv2_1')(layers['dropout_1'])
    else:
        layers['conv2_1'] = Conv2D(64, (3, 3), padding='same', activation='relu', name='conv2_1')(layers['pool_1'])
    layers['conv2_2'] = Conv2D(64, (3, 3), padding='same', activation='relu', name='conv2_2')(layers['conv2_1'])
    layers['pool_2'] = MaxPool2D(pool_size=(2, 2), name='pool_2')(layers['conv2_2'])
    if train == True:
        layers['dropout_2'] = Dropout(0.25, name='dropout_2')(layers['pool_2'])
        layers['conv3_1'] = Conv2D(128, (3, 3), padding='same', activation='relu', name='conv3_1')(layers['dropout_2'])
    else:
        layers['conv3_1'] = Conv2D(128, (3, 3), padding='same', activation='relu', name='conv3_1')(layers['pool_2'])
    layers['conv3_2'] = Conv2D(128, (3, 3), padding='same', activation='relu', name='conv3_2')(layers['conv3_1'])
    layers['pool_3'] = MaxPool2D(pool_size=(2, 2), name='pool_3')(layers['conv3_2'])
    if train == True:
        layers['dropout_3'] = Dropout(0.25, name='dropout_3')(layers['pool_3'])
        layers['conv4_1'] = Conv2D(256, (3, 3), padding='same', activation='relu', name='conv4_1')(layers['dropout_3'])
    else:
        layers['conv4_1'] = Conv2D(256, (3, 3), padding='same', activation='relu', name='conv4_1')(layers['pool_3'])
    layers['conv4_2'] = Conv2D(256, (3, 3), padding='same', activation='relu', name='conv4_2')(layers['conv4_1'])
    layers['pool_4'] = MaxPool2D(pool_size=(2, 2), name='pool_4')(layers['conv4_2'])
    if train == True:
        layers['dropout_4'] = Dropout(0.25, name='dropout_4')(layers['pool_4'])
        layers['conv5_1'] = Conv2D(512, (3, 3), padding='same', activation='relu', name='conv5_1')(layers['dropout_4'])
    else:
        layers['conv5_1'] = Conv2D(512, (3, 3), padding='same', activation='relu', name='conv5_1')(layers['pool_4'])
    layers['conv5_2'] = Conv2D(512, (3, 3), padding='same', activation='relu', name='conv5_2')(layers['conv5_1'])

    layers['upsample_1'] = UpSampling2D(size=(2, 2), name='upsample_1')(layers['conv5_2'])
    layers['concat_1'] = concatenate([layers['upsample_1'], layers['conv4_2']], name='concat_1')
    layers['conv6_1'] = Conv2D(256, (3, 3), padding='same', activation='relu', name='conv6_1')(layers['concat_1'])
    layers['conv6_2'] = Conv2D(256, (3, 3), padding='same', activation='relu', name='conv6_2')(layers['conv6_1'])
    if train == True:
        layers['dropout_6'] = Dropout(0.25, name='dropout_6')(layers['conv6_2'])
        layers['upsample_2'] = UpSampling2D(size=(2, 2), name='upsample_2')(layers['dropout_6'])
    else:
        layers['upsample_2'] = UpSampling2D(size=(2, 2), name='upsample_2')(layers['conv6_2'])
    layers['concat_2'] = concatenate([layers['upsample_2'], layers['conv3_2']], name='concat_2')
    layers['conv7_1'] = Conv2D(128, (3, 3), padding='same', activation='relu', name='conv7_1')(layers['concat_2'])
    layers['conv7_2'] = Conv2D(128, (3, 3), padding='same', activation='relu', name='conv7_2')(layers['conv7_1'])
    if train == True:
        layers['dropout_7'] = Dropout(0.25, name='dropout_7')(layers['conv7_2'])
        layers['upsample_3'] = UpSampling2D(size=(2, 2), name='upsample_3')(layers['dropout_7'])
    else:
        layers['upsample_3'] = UpSampling2D(size=(2, 2), name='upsample_3')(layers['conv7_2'])
    layers['concat_3'] = concatenate([layers['upsample_3'], layers['conv2_2']], name='concat_3')
    layers['conv8_1'] = Conv2D(64, (3, 3), padding='same', activation='relu', name='conv8_1')(layers['concat_3'])
    layers['conv8_2'] = Conv2D(64, (3, 3), padding='same', activation='relu', name='conv8_2')(layers['conv8_1'])
    if train == True:
        layers['dropout_8'] = Dropout(0.25, name='dropout_8')(layers['conv8_2'])
        layers['upsample_4'] = UpSampling2D(size=(2, 2), name='upsample_4')(layers['dropout_8'])
    else:
        layers['upsample_4'] = UpSampling2D(size=(2, 2), name='upsample_4')(layers['conv8_2'])
    layers['concat_4'] = concatenate([layers['upsample_4'], layers['conv1_2']], name='concat_4')
    layers['conv9_1'] = Conv2D(32, (3, 3), padding='same', activation='relu', name='conv9_1')(layers['concat_4'])
    layers['conv9_2'] = Conv2D(32, (3, 3), padding='same', activation='relu', name='conv9_2')(layers['conv9_1'])
    if train == True:
        layers['dropout_9'] = Dropout(0.25, name='dropout_9')(layers['conv9_2'])
        layers['outputs'] = Conv2D(1, (1, 1), activation='sigmoid', name='outputs')(layers['dropout_9'])
    else:
        layers['outputs'] = Conv2D(1, (1, 1), activation='sigmoid', name='outputs')(layers['conv9_2'])

    model = Model(inputs=layers['inputs'], outputs=layers['outputs'])

    return model
