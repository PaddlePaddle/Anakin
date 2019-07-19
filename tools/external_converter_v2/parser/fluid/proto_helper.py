from .. import proto

def add_edge(graph_proto, bottom, top):
    """add_edge in graph_proto
    """
    bottom_target = proto.TargetProto()
    bottom_target.node = top
    graph_proto.edges_out[bottom].target.extend([bottom_target])
    top_target = proto.TargetProto()
    top_target.node = bottom
    graph_proto.edges_in[top].target.extend([top_target])


def drop_nodes(graph_proto, drop_list):
    """drop nodes of graph_proto
    """
    tmp_nodes = filter(lambda node: node.name not in drop_list, graph_proto.nodes)
    del graph_proto.nodes[:]
    graph_proto.nodes.extend(tmp_nodes)

    for drop_node in drop_list:
        if drop_node in graph_proto.edges_in:
            del graph_proto.edges_in[drop_node]
        if drop_node in graph_proto.edges_out:
            del graph_proto.edges_out[drop_node]
        if drop_node in graph_proto.edges_info:
            del graph_proto.edges_info[drop_node]

    for edge_name in graph_proto.edges_in:
        targets = graph_proto.edges_in[edge_name].target
        tmp_targets = filter(lambda target: target.node not in drop_list, targets)
        del targets[:]
        targets.extend(tmp_targets)

    for edge_name in graph_proto.edges_out:
        targets = graph_proto.edges_out[edge_name].target
        tmp_targets = filter(lambda target: target.node not in drop_list, targets)
        del targets[:]
        targets.extend(tmp_targets)
