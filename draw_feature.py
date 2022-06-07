#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import matplotlib as plt
import os
import roslib
roslib.load_manifest('nav_cloning')
import rospy
from nav_cloning_net import *
from skimage.transform import resize
import math
import chainer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def draw_feature(model, image):
    rospy.init_node('draw_feature_node', anonymous = True)
    path = roslib.packages.get_pkg_dir('nav_cloning') + '/data/analysis/'
    # bridge = CvBridge()
    dl = deep_learning(n_action = 1)
    load_path = path + 'model.net'
    model = dl.load(load_path)
    image_path = path + '199.jpg'
    image = Image.open(image_path)

    layer_num = []
    layer_num.append(model.get_layer('conv2d'))
    layer_num.append(model.get_layer('conv2d_1'))
    layer_num.append(model.get_layer('conv2d_2'))
    layer_num.append(model.get_layer('fc'))
    layer_num.append(model.get_layer('fc_1'))

    image = img[None, :, :, :]

    for i in range(len(layer_num)-1):
        feature_map = model.predictor(image)
        feature_map = feature_map[0]
        feature = feature_map.shape[2]
    
        fig = plt.gcf()
        fig.canvas.set_window_title(layer_num[i + 1].name)

        for j in range(feature):
            plt.subplots_adjust(wspace = 0.4, hspace = 0.8)
            plt.subplot(feature / 6 + 1, 6, j + 1)
            plt.xticks([])
            plt.yticks([])
            plt.xlabel(f'filter {j}')
            plt.imshow(feature_map[:, :, j])
        plt.show()

