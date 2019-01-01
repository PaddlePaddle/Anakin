#! /usr/bin/env python
# -*- coding: utf-8 -*-

from parser import NetParser

class Net:

    def __init__(self, config):
        '''
        '''
        self.parser = NetParser(config.DebugConfig)
        self.net_io = self.parser()
