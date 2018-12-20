#! /usr/bin/env python
# Copyright (c) 2017, Cuichaowen. All rights reserved.
# -*- coding: utf-8 -*-

import os
import sys
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
    def __init__(self, argv, config_file_path=ConfigFilePath):
        data = load(open(config_file_path, 'r').read())
        # parse Options from config file.
        self.framework = data['OPTIONS']['Framework']
        self.SavePath = data['OPTIONS']['SavePath'] \
                if data['OPTIONS']['SavePath'][-1] == '/' \
                else data['OPTIONS']['SavePath'] + '/'
        self.ResultName = data['OPTIONS']['ResultName']
        #self.intermediateModelPath = data['OPTIONS']['SavePath'] \
        #        + data['OPTIONS']['ResultName'] + "anakin.bin"
        self.LaunchBoard = True if data['OPTIONS']['Config']['LaunchBoard'] else False
        self.ip = data['OPTIONS']['Config']['Server']['ip']
        self.port = data['OPTIONS']['Config']['Server']['port']
        self.hasOptimizedGraph = True if data['OPTIONS']['Config']['OptimizedGraph']['enable'] else False
        self.optimizedGraphPath = data['OPTIONS']['Config']['OptimizedGraph']['path'] \
                if self.hasOptimizedGraph else ""
        self.logger_dict = data['OPTIONS']['LOGGER']
        self.framework_config_dict = data['TARGET'][self.framework]
        self.check_protobuf_version()
        if len(argv) > 1:
            self.config_from_cmd(argv)
        if 'ProtoPaths' in data['TARGET'][self.framework].keys():
            proto_list = data['TARGET'][self.framework]['ProtoPaths']
            self.__refresh_pbs(proto_list)
        self.generate_pbs_of_anakin()

    def config_from_cmd(self, argv):
        """
        Read configuration information from the command line.
        """
        cmd = {
            'CAFFE': {
                'proto': ['ProtoPaths', list()],
                'prototxt': ['PrototxtPath', str()],
                'caffemodel': ['ModelPath', str()],
                },
            'FLUID': {
                'modelpath': ['ModelPath', str()],
                'type': ['NetType', str()],
            },
        }
        err_note = '\nUsage1: python ./converter.py ' \
                    + 'CAFFE --proto=/path/to/filename1.proto ' \
                    + '--prototxt=/path/to/filename.prototxt ' \
                    + '--caffemodel=/path/to/filename.caffemodel\n' \
                    + 'Usage2: python ./converter.py ' \
                    + 'FLUID --modelpath=/model/path/ --type=OCR'
        def splitter(arg, key_delim='--', val_delim='='):
            """
            Extract the valid content of the parameter string to form a [key, val] list.
            """
            if (key_delim in arg) and (val_delim in arg):
                element = arg.split(key_delim)[1].split(val_delim)
                return element
            else:
                raise NameError(err_note)
        def filler(arg, dic, val_idx=1):
            """
            Extract the valid content of the parameter string to form a [key, val] list.
            """
            element = splitter(arg)
            key = element[0]
            val = element[1]
            assert key in dic.keys(), \
            "Param %s in cmd is wrong." % (key)
            if type(dic[key][val_idx]) == str: dic[key][val_idx] = val
            elif type(dic[key][val_idx]) == list: dic[key][val_idx].append(val)
        def null_scanner(dic, val_idx=1):
            """
            Make sure the parameters are complete.
            """
            for key in dic:
                assert (bool(dic[key][val_idx])), 'Key [%s] should not be null.' % (key)
        def arg_transmit(dic, target, key_idx=0, val_idx=1):
            """
            Match the command line to yaml.
            """
            if target == 'CAFFE':
                self.ResultName = dic['caffemodel'][val_idx].split("/")[-1].split('.caffemodel')[0]
            elif target == 'FLUID':
                if dic['modelpath'][-1] == '/':
                    self.ResultName = dic['modelpath'][val_idx].split("/")[-2]
                else:
                    self.ResultName = dic['modelpath'][val_idx].split("/")[-1]
            elif self.framework == "ONNX":
                proto_list = dic['TARGET'][self.framework]['ProtoPaths']
                #onnx_file = dic['TARGET'][self.framework]['OnnxPaths']
                self.framework_config_dict = dic['TARGET'][self.framework]
                #print 'self.framework_config_dict', self.framework_config_dict
            else:
                raise NameError(err_note)
            for cmd_key in cmd[target].keys():
                key = dic[cmd_key][key_idx]
                val = dic[cmd_key][val_idx]
                self.framework_config_dict[key] = val
            self.LaunchBoard = False
        target = argv[1]
        assert target in cmd.keys(), "Framework [%s] is not yet supported." % (target)
        for arg in argv[2:]:
            filler(arg, cmd[target])
        null_scanner(cmd[target])
        arg_transmit(cmd[target], target)

    def check_protobuf_version(self):
        """
        Check if the pip-protoc version is equal to sys-protoc version.
        """
        for path in sys.path:
            module_path = os.path.join(path, 'google', 'protobuf', '__init__.py')
            if os.path.exists(module_path):
                with open(module_path) as f:
                    __name__ = '__main__'
                    exec(f.read(), locals())
                break
        try:
            protoc_out = subprocess.check_output(["protoc", "--version"]).split()[1]
        except OSError as exc:
            raise OSError('Can not find Protobuf in system environment.')
        try:
            __version__
        except NameError:
            raise OSError('Can not find Protobuf in python environment.')
        sys_versions = map(int, protoc_out.split('.'))
        pip_versions = map(int, __version__.split('.'))
        assert sys_versions[0] >= 3 and pip_versions[0] >= 3, \
            "Protobuf version must be greater than 3.0. Please refer to the Anakin Docs."
        assert pip_versions[1] >= sys_versions[1], \
            "ERROR: Protobuf must be the same.\nSystem Protobuf %s\nPython Protobuf %s\n" \
            % (protoc_out, __version__) + "Try to execute pip install protobuf==%s" % (protoc_out)

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
        self.__refresh_pbs(protoFilesList, "parser/proto/")

    def pbs_eraser(self, folder_path):
        """
        Delete existing pb2 to avoid conflicts.
        """
        for file_name in os.listdir(os.path.dirname(folder_path)):
            if "pb2." in file_name:
                file_path = os.path.join(folder_path, file_name)
                os.remove(file_path)

    def __refresh_pbs(self, proto_list, default_save_path="parser/pbs/"):
        """
        Genetrate pb files according to proto file list and saved to parser/pbs/ finally.
        parameter:
                proto_list: ['/path/to/proto_0','/path/to/proto_1', ... ]
                default_save_path: default saved to 'parser/pbs/'
        """
        self.pbs_eraser(default_save_path)
        assert type(proto_list) == list, \
        "The ProtoPaths format maybe incorrect, please check if there is any HORIZONTAL LINE."
        for pFile in proto_list:
            assert os.path.exists(pFile), "%s does not exist.\n" % (pFile)
            subprocess.check_call(['protoc', '-I',
                                   os.path.dirname(pFile) + "/",
                                   '--python_out', os.path.dirname(default_save_path) + "/", pFile])
