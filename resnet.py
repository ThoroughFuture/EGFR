import numpy as np
from keras import layers, models
from functools import partial
import config

def _after_conv(in_tensor):
    norm = layers.BatchNormalization()(in_tensor)
    return layers.Activation('relu')(norm)


def conv3(in_tensor, filters):
    conv = layers.Conv2D(filters, kernel_size=3, strides=1, padding='same', use_bias=False)(in_tensor)
    return _after_conv(conv)


def conv3_downsample(in_tensor, filters):
    conv = layers.Conv2D(filters, kernel_size=3, strides=2, padding='same', use_bias=False)(in_tensor)
    return _after_conv(conv)


def conv1(in_tensor, filters):
    conv = layers.Conv2D(filters, kernel_size=1, strides=1, padding='same', use_bias=False)(in_tensor)
    return _after_conv(conv)


def conv1_downsample(in_tensor, filters):
    conv = layers.Conv2D(filters, kernel_size=1, strides=2, padding='same', use_bias=False)(in_tensor)
    return _after_conv(conv)


def resnet_block(in_tensor, filters, downsample=False, up_filter=False):
    if downsample:
        conv1_rb = conv3_downsample(in_tensor, filters)
    else:
        conv1_rb = conv3(in_tensor, filters)
    conv2_rb = conv3(conv1_rb, filters)
    
    if downsample:
        in_tensor = conv1_downsample(in_tensor, filters)
    result = layers.Add()([conv2_rb, in_tensor])
    
    return layers.Activation('relu')(result)


def resnet_block_bottlneck(in_tensor, filters, downsample=False, up_filter=False):
    if downsample:
        conv1_rb = conv1_downsample(in_tensor, int(filters/4))
    else:
        conv1_rb = conv1(in_tensor, int(filters/4))
    conv2_rb = conv3(conv1_rb, int(filters/4))
    conv3_rb = conv1(conv2_rb, filters)

    if downsample:
        in_tensor = conv1_downsample(in_tensor, filters)
    elif up_filter:
        in_tensor = conv1(in_tensor, filters)
    result = layers.Add()([conv3_rb, in_tensor])

    return result


def block(in_tensor, filters, n_block, downsample=False, convx=resnet_block):
    res = in_tensor
    for i in range(n_block):
        if i == 0:
            res = convx(res, filters, downsample, not downsample)
        else:
            res = convx(res, filters, False)
    return res
    

def resnet(image_batch, filter_numbers=[256, 512, 1024, 2048], 
           block_numbers=[3,4,6,3], convx=resnet_block_bottlneck):
    conv = layers.Conv2D(64, 7, strides=2, padding='same', use_bias=False)(image_batch)
    conv = _after_conv(conv)
    pool1 = layers.MaxPool2D(3, 2, padding='same')(conv)
    conv1_block = block(pool1, filter_numbers[0], block_numbers[0], False, convx)
    conv2_block = block(conv1_block, filter_numbers[1], block_numbers[1], True, convx)
    conv3_block = block(conv2_block, filter_numbers[2], block_numbers[2], True, convx)
    conv4_block = block(conv3_block, filter_numbers[3], block_numbers[3], True, convx)
    pool2 = layers.GlobalAvgPool2D()(conv4_block)
    _y = layers.Dense(config.CLASS_NUM, activation='softmax')(pool2)
    # ResNet-18 & ResNet-34 [64, 128, 256, 512] 
    # [2, 2, 2, 2] & [3, 4, 6, 3]
    # ResNet-50 & ResNet-101 & ResNet-152 [256, 512, 1024, 2048]
    # [3, 4, 6, 3] & [3, 4, 23, 3] & [3, 8, 36, 3]
    return _y


def resnet_18(image_batch):
    return resnet(image_batch, filter_numbers=[64,128,256,512], 
           block_numbers=[2,2,2,2], convx=resnet_block)


def resnet_34(image_batch):
    return resnet(image_batch, filter_numbers=[64,128,256,512], 
           block_numbers=[3,4,6,3], convx=resnet_block)


def resnet_50(image_batch):
    return resnet(image_batch, filter_numbers=[256,512,1024,2048], 
           block_numbers=[3,4,6,3], convx=resnet_block_bottlneck)