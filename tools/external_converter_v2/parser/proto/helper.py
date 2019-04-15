"""proto helper
"""

import tensor_pb2

def make_tensor(
        dims, # type: list(int)
        data_type, # type: tensor_pb2.DateTypeProto
        vals, # type: list(float, int...) or bytes
        layout=None, # type: tensor_pb2.LayoutProto
        scale=None, # type: list(float)
):
    """make tensor_pb2.TensorProto
    """
    t = tensor_pb2.TensorProto()

    t.shape.dims.size = len(dims)
    t.shape.dims.value = dims[:]

    # set TensorProto.data
    t.data.type = data_type
    if t.data.type is tensor_pb2.STR:
        t.data.s[:] = vals
    elif t.data.type is tensor_pb2.INT32:
        t.data.i[:] = vals
    elif t.data.type is tensor_pb2.INT8:
        assert type(t.data.c) is bytes
        t.data.c = vals
    elif t.data.type in [tensor_pb2.FLOAT16, tensor_pb2.FLOAT, tensor_pb2.DOUBLE]:
        t.data.f[:] = vals
    elif t.data.type is tensor_pb2.BOOLEN:
        t.data.b[:] = vals
    else:
        raise Exception('unsupported data_type={}'.format(data_type))
    t.data.size = len(vals)

    if layout is not None:
        t.shape.layout = layout
    if scale is not None:
        t.shape.scale.f[:] = scale
        t.shape.scale.type = tensor_pb2.FLOAT
        t.shape.scale.size = len(scale)

    return t


def reverse_cache_data(data):  # type: tensor_pb2.CacheDate -> None
    """tensor_pb2.CacheDate => 1.0 / tensor_pb2.CacheDate
    """
    if data.type is tensor_pb2.INT8:
        data.c[:] = map(lambda x: 1.0 / x, data.c)
    elif data.type is tensor_pb2.INT32:
        data.i[:] = map(lambda x: 1.0 / x, data.i)
    elif data.type in [tensor_pb2.FLOAT, tensor_pb2.FLOAT16, tensor_pb2.DOUBLE]:
        data.f[:] = map(lambda x: 1.0 / x, data.f)
    elif data.type is tensor_pb2.CACHE_LIST:
        for x in data.l:
            reverse_cache_data(x)
    else:
        raise Exception('unsupported data.type={}'.format(data.type))
