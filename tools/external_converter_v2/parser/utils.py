#! /usr/bin/env python
# Copyright (c) 2017, Cuichaowen. All rights reserved.
# -*- coding: utf-8 -*-

import math
import time
from functools import wraps


def elapsTime(target):
    """
    Get elapse time for target function
    """
    @warps(target)
    def warpper(*args, **kwargs):
        """
        warpper args
        """
        start = time.time()
        ret = target(*args, **kwargs)
        end = time.time()
        hours, rem = divmod(end - start, 3600)
        minutes, seconds = divmod(rem, 60)
        print("------- Elapsed time(h:min:s): {:0>2}:{:0>2}:{:05.2f}s "\
                        .format(int(hours), int(minutes), seconds))
        return ret
    return warpper


def dict_has_key(target_dict, key):
    """
    Judge if target_dict has target key
    """
    return key in target_dict


def proto_has_field(param_pkg, key_name):
    """
    Judge if proto message field param_pkg has key key_name
    """
    for field in param_pkg.DESCRIPTOR.fields:
        if field.name == key_name:
            return True
    return False
