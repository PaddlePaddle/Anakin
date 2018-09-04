#! /usr/bin/env python
# Copyright (c) 2017, Cuichaowen. All rights reserved.
# -*- coding: utf-8 -*-
from ..logger import *


class OpsParam(object):
    """
    """
    def __init__(self):
        self.__dict__ = {}

    def set_attr(self, **kwargs):
        self.__dict__ = kwargs
        return self

    def set_parser(self, parser):
        self.parser = parser
        return self

    def feed_node_attr(self, node_io):
        for attr_name in self.__dict__.keys():
            if attr_name != "parser":
                value = self[attr_name]
                node_io.add_attr(attr_name, value, type(value))

    def Dict(self):
        return self.__dict__

    def __str__(self):
        return "Operator attrs ( " + str(self.__dict__.keys()) + " )."

    def __contains__(self, key):
        return key in self.__dict__.keys()

    def __getitem__(self, key):
        if key in self:
            return self.__dict__[key]
        return None

    def __call__(self, *args):
        return self.parser(args)


class OpsRegister(object):
    """
    Operator register
    """
    instance = None

    def __init__(self):
        if OpsRegister.instance is None:
            OpsRegister.instance = dict()

    @staticmethod
    def Register(name):
        if name not in OpsRegister():
            OpsRegister.instance[name] = OpsParam()
        return OpsRegister.instance[name]

    @staticmethod
    def UnRegister(name):
        if name in OpsRegister():
            del OpsRegister.instance[name]

    def __str__(self):
        return "OpsRegister has : " + str(len(self.get_op_name_list())) + " num of ops \n" +\
                        "Ops : " + str(self.get_op_name_list())

    def get_op_name_list(self):
        return OpsRegister.instance.keys()

    def __contains__(self, name):
        """ 
        If the target op name in the instance map
        """
        return name in OpsRegister.instance.keys()

    def __getitem__(self, name):
        """
        Get target op by name
        """
        if name in self:
            return OpsRegister.instance[name]
        return OpsParam()
