import sys
import os


def clip_path(filepath):
    """
    Clip path to get file name
    """
    fileslist = filepath.split('/')
    return fileslist[-1]


def clip_paths(listfilepath):
    """
    Map clip multi-paths
    """
    return map(clip_path, listfilepath)
