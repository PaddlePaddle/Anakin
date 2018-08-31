#! /usr/bin/env python
# Copyright (c) 2017, Cuichaowen. All rights reserved.
# -*- coding: utf-8 -*-

from multiprocessing import Lock
from datetime import datetime
from enum import Enum
from enum import unique
import platform as pf
import atexit
import inspect
import sys
import os


@unique
class verbose(Enum):
    """
    verbose enum
    """
    INFO = 0
    WARNING = 1
    ERROR = 2
    FATAL = 3

    def describe(self):
        """
        Usage: verbose.WARNING.describe()  --> ('WARNING', 1)
        """
        return self.name, self.value

    def __str__(self):
        """
        Usage: str(verbose.WARNING)  --> 'WARNING'
        """
        return 'Target verbose is {0}'.format(self.name)

    @staticmethod
    def default_verbose():
        return verbose.INFO

"""
verbose to msg dict.
"""
VERBOSE_DICT = {'INFO': "INF", 'WARNING': "WAR", 'ERROR': "ERR", 'FATAL': "FTL"}


class shellcolors:
    """
    shell color decorator class
    """
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    ERROR = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    def __init__(self, func):
        self.func = func

    def __call__(self, *args):
        if args[1] == verbose.INFO:
            return self.BOLD + self.func(args[0], args[1]) + self.ENDC
        elif args[1] == verbose.WARNING:
            return self.WARNING + self.func(args[0], args[1]) + self.ENDC
        elif args[1] == verbose.ERROR:
            return self.ERROR + self.func(args[0], args[1]) + self.ENDC
        elif args[1] == verbose.FATAL:
            return self.ERROR + self.BOLD + self.UNDERLINE + self.func(args[0], args[1]) + self.ENDC
        else:
            raise NameError('ERROR: Logger not support verbose: %s' % (str(args[1])))


@shellcolors
def with_color(header, verbose):
    return header


class logger:
    """
    Logger class.
    """
    # config from config.yaml.
    LogToPath = "./"
    FileName = "parser.log"
    WithColor = True
    log_file_plist = []
    # lock mutex for writing log files by multi process.
    lock = Lock()

    Prune = lambda filename: filename.split('/')[-1]

    def __init__(self, verbose=verbose.default_verbose()):
        """
        """
        self.__verbose = verbose
        self.log_head = "" + VERBOSE_DICT[verbose.name] + " | " + str(datetime.now()) + " | "
        # 1 represents line at caller
        callerframerecord = inspect.stack()[1]
        frame = callerframerecord[0]
        info = inspect.getframeinfo(frame)
        Prune = lambda filename: filename.split('/')[-1]
        self.log_head += Prune(info.filename) + ":" + str(info.lineno) + " " + str(info.function) + "() ] "

    def feed(self, *args):
        """
        feed info to log engine.
        """
        msg = ''.join(str(i) for i in args)
        full_msg = ""
        no_color_msg = ""
        try:
            full_msg = (with_color(self.log_head, self.__verbose) + " " + msg) if logger.WithColor else (self.log_head + " " + msg)
            no_color_msg = self.log_head + " " + msg
        except NameError:
            raise
        self.log_to_everywhere(no_color_msg, full_msg)

    @staticmethod
    def init(config_dict):
        """
        load config and initial log resource.
        """
        logger.LogToPath = os.path.dirname(config_dict['LogToPath']) + '/' if config_dict['LogToPath'] else logger.LogToPath
        logger.WithColor = True if config_dict['WithColor'] else False
        machine_msg = pf.node() + "_" + pf.system() + "_" + (str(datetime.now()).replace(' ', '_'))
        logger.FileName = machine_msg + '_' + logger.FileName
        # open log file and create the log output path
        if not os.path.exists(os.path.dirname(logger.LogToPath)):
            os.makedirs(os.path.dirname(logger.LogToPath))
        logger.log_file_plist.append(open(logger.LogToPath + logger.FileName, "w+"))
        # register logger.clean_up function
        atexit.register(logger.clean_up)

    @staticmethod	
    def clean_up():
        """
        clean up all the opened file pointer
        """
        if logger.log_file_plist[0]:
            logger.log_file_plist[0].close()

    def log_to_everywhere(self, no_color_msg, full_msg):
        """
        log to stdout and file
        """
        filename = logger.LogToPath + logger.FileName
        size_in_bytes = os.stat(filename).st_size
        size_in_GB = size_in_bytes * 1.0 / (1024 * 1024 * 1024)
        logger.lock.acquire()
        if size_in_GB > 10:
            logger.log_file_plist[0].truncate()
            logger.log_file_plist[0].seek(0)
        # log to file
        logger.log_file_plist[0].write(no_color_msg + "\n")	
        logger.log_file_plist[0].flush()
        # log to stdout
        print full_msg
        sys.stdout.flush()	
        logger.lock.release()
