#! /usr/bin/env python
# Copyright (c) 2017, Cuichaowen. All rights reserved.
# -*- coding: utf-8 -*-

import os
import sys
from config import *

def launch(config, graph):
    logger(verbose.WARNING).feed("anakin parser dash board will be launch in site: ")
    graph.run_with_server(config.ip, config.port)

if __name__ == "__main__":
    config = Configuration(sys.argv)
    from parser import *
    logger.init(config.logger_dict)

    if config.DebugConfig is None:
        graph = Graph(config)
        graph.info_table()
        graph.serialization()

        if config.LaunchBoard:
            launch(config, graph)
    else:
        import utils
        net = utils.net.net_parser.Net(config)
        net.storage()
        print net
