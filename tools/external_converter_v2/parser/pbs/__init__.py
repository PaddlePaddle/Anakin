#! /usr/bin/env python
# Copyright (c) 2017, Cuichaowen. All rights reserved.
# -*- coding: utf-8 -*-

try:
    from caffe_pb2 import *
except ImportError:
    raise ImportError(' No module named caffe_pb2 . ')

try:
    from caffe_yolo_pb2 import *
except ImportError:
    raise ImportError(' No module named caffe_yolo_pb2 . ')
