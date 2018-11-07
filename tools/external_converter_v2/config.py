#! /usr/bin/env python
# Copyright (c) 2017, Cuichaowen. All rights reserved.
# -*- coding: utf-8 -*-

import os
import subprocess
from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

ConfigFilePath = './config.yaml'


class Configuration:
    """
    Parse the config.yaml file.
    Configuration holds all the params defined in configfile.
    """
    def __init__(self, config_file_path=ConfigFilePath):
        data = load(open(config_file_path, 'r').read())
        # parse Options from config file.
        self.framework = data['OPTIONS']['Framework']
        self.SavePath = data['OPTIONS']['SavePath'] \
                if data['OPTIONS']['SavePath'][-1] == '/' \
                else data['OPTIONS']['SavePath'] + '/'
        self.ResultName = data['OPTIONS']['ResultName']
        self.intermediateModelPath = data['OPTIONS']['SavePath'] \
                + data['OPTIONS']['ResultName'] + "anakin.bin"
        self.LaunchBoard = True if data['OPTIONS']['Config']['LaunchBoard'] else False
        self.ip = data['OPTIONS']['Config']['Server']['ip']
        self.port = data['OPTIONS']['Config']['Server']['port']
        self.hasOptimizedGraph = True if data['OPTIONS']['Config']['OptimizedGraph']['enable'] else False
        self.optimizedGraphPath = data['OPTIONS']['Config']['OptimizedGraph']['path'] \
                if self.hasOptimizedGraph else ""
        self.logger_dict = data['OPTIONS']['LOGGER']
        # parse TARGET info from config file.
        if self.framework == "CAFFE":
            proto_list = data['TARGET'][self.framework]['ProtoPaths']
            self.__generate_pbs(proto_list)
            self.framework_config_dict = data['TARGET'][self.framework]
        elif self.framework == "PADDLE":
            pass
        elif self.framework == "LEGO":
            proto_list = data['TARGET'][self.framework]['ProtoPaths']
            self.__generate_pbs(proto_list)
            self.framework_config_dict = data['TARGET'][self.framework]
        elif self.framework == "TENSORFLOW":
            proto_list = data['TARGET'][self.framework]['ProtoPaths']
            self.framework_config_dict = data['TARGET'][self.framework]
        elif self.framework == "MXNET":
            pass
        elif self.framework == "FLUID":
            self.framework_config_dict = data['TARGET'][self.framework]
        else:
            raise NameError('ERROR: Framework not support yet ' % (self.framework))
        try:
            self.generate_pbs_of_anakin()
        except NameError:
            raise

    def generate_pbs_of_anakin(self):
        protoFilesStr = subprocess.check_output(["ls", "parser/proto/"])
        filesList = protoFilesStr.split('\n')
        protoFilesList = []
        for file in filesList:
            suffix = file.split('.')[-1]
            if suffix == "proto":
                protoFilesList.append("parser/proto/" + file)
        if len(protoFilesList) == 0:
            raise NameError('ERROR: There does not have any proto files in proto/ ')
        self.__generate_pbs(protoFilesList, "parser/proto/")

    def __generate_pbs(self, proto_list, default_save_path="parser/pbs/"):
        """
        Genetrate pb files according to proto file list and saved to parser/pbs/ finally.
        parameter:
                proto_list: ['/path/to/proto_0','/path/to/proto_1', ... ]
                default_save_path: default saved to 'parser/pbs/'
        """
        for pFile in proto_list:
            subprocess.check_call(['protoc', '-I', 
                                   os.path.dirname(pFile) + "/",
                                   '--python_out', os.path.dirname(default_save_path) + "/", pFile])
