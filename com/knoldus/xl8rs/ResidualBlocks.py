

import pandas as pd
import numpy as np
import os
from os import listdir
pd.set_option("display.max_rows", 101)
import cv2
import json

import seaborn as sns
from collections import Counter
from PIL import Image
import math
import seaborn as sns
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

import tensorflow as tf
from tensorflow import reduce_sum
from tensorflow.keras.backend import pow
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, UpSampling2D, Concatenate, Add, Flatten
from tensorflow.keras.losses import binary_crossentropy
from sklearn.model_selection import train_test_split


class ResidualBlocks:
    def __init__(self, img_h, img_w):
        self.img_h = img_h
        self.img_w = img_w

    def conv_block(self, x, filters, kernel_size=3, padding='same', strides=1):
        conv = tf.keras.layers.Activation('relu')(x)
        conv = Conv2D(filters, kernel_size, padding=padding, strides=strides)(conv)
        return conv

    def residual_block(self, x, filters, kernel_size=3, padding='same', strides=1):
        res = self.conv_block(x, filters, 3, padding, strides)
        res = self.conv_block(res, filters, 3, padding, 1)
        skip = Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
        skip = tf.keras.layers.BatchNormalization()(skip)
        output = Add()([skip, res])
        return output

    # batch normalization layer with an optinal activation layer
    def bn_block(self, x, act=True):
        x = tf.keras.layers.BatchNormalization()(x)
        if act:
            x = tf.keras.layers.Activation('relu')(x)
        return x

    def supportive_bock(self, x, filters, kernel_size=3, padding='same', strides=1):
        conv = Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
        conv = self.conv_block(conv, filters, kernel_size, padding, strides)
        shortcut = Conv2D(filters, kernel_size=1, padding=padding, strides=strides)(x)
        shortcut = self.bn_block(shortcut, act=False)
        output = Add()([conv, shortcut])
        return output

    def upsample_concat_block(self, x, xskip):
        u = UpSampling2D((2, 2))(x)
        c = Concatenate()([u, xskip])
        return c

    def tversky(self, y_true, y_pred, smooth=1e-6):
        y_true_pos = tf.keras.layers.Flatten()(y_true)
        y_pred_pos = tf.keras.layers.Flatten()(y_pred)
        true_pos = tf.reduce_sum(y_true_pos * y_pred_pos)
        false_neg = tf.reduce_sum(y_true_pos * (1 - y_pred_pos))
        false_pos = tf.reduce_sum((1 - y_true_pos) * y_pred_pos)
        alpha = 0.7
        return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)

    def focal_tversky_loss(self, y_true, y_pred):
        pt_1 = self.tversky(y_true, y_pred)
        gamma = 0.75
        return tf.keras.backend.pow((1 - pt_1), gamma)

    def build(self, img_h, img_w):
        f = [16, 32, 64, 128, 256]
        inputs = Input((img_h, img_w, 1))

        ## Encoder
        e0 = inputs
        e1 = self.supportive_bock(e0, f[0])
        e2 = self.residual_block(e1, f[1], strides=2)
        e3 = self.residual_block(e2, f[2], strides=2)
        e4 = self.residual_block(e3, f[3], strides=2)
        e5 = self.residual_block(e4, f[4], strides=2)

        ## Bridge
        b0 = self.conv_block(e5, f[4], strides=1)
        b1 = self.conv_block(b0, f[4], strides=1)

        ## Decoder
        u1 = self.upsample_concat_block(b1, e4)
        d1 = self.residual_block(u1, f[4])

        u2 = self.upsample_concat_block(d1, e3)
        d2 = self.residual_block(u2, f[3])

        u3 = self.upsample_concat_block(d2, e2)
        d3 = self.residual_block(u3, f[2])

        u4 = self.upsample_concat_block(d3, e1)
        d4 = self.residual_block(u4, f[1])

        outputs = tf.keras.layers.Conv2D(4, (1, 1), padding="same", activation="sigmoid")(d4)
        model = tf.keras.models.Model(inputs, outputs)
        return model

    # , img_w=800, img_h=256
    def model(self):
        model = self.build(img_h=self.img_h, img_w=self.img_w)
        adam = tf.keras.optimizers.Adam(lr=0.05, epsilon=0.1)
        model.compile(optimizer=adam, loss=self.focal_tversky_loss, metrics=[self.tversky])




