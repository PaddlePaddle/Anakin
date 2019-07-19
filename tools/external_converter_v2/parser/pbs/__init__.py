#! /usr/bin/env python
# Copyright (c) 2017, Cuichaowen. All rights reserved.
# -*- coding: utf-8 -*-
import os
from ..logger import verbose
from ..logger import shellcolors
from ..logger import with_color
from ..logger import logger

for module in os.listdir(os.path.dirname(__file__)):
    if module == '__init__.py' or module[-3:] != '.py':
        continue
    m = __import__(module[:-3], locals(), globals())
    try:
        attrlist = m.__all__
    except AttributeError:
        attrlist = dir(m)
    for attr in attrlist:
        globals()[attr] = getattr(m, attr)
    logger(verbose.INFO).feed("Import Module: ", module[:-3])
del module
