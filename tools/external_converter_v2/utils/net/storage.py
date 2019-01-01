#! /usr/bin/env python
# -*- coding: utf-8 -*-

class NetStorage(object):

    def __init__(self, net_proto):
        self.net_proto = net_proto

    def get_node(self, node_name):
        ret = None
        nodes = net_proto.graph_nodes()
        for node in nodes:
            if node.name == node_name:
                ret = node
        return ret

    def update_net(self):
        




