from ..operations import OpsParam, OpsRegister
from ..logger import *
from ..pbs import *
from ..graph_io import TensorProtoIO
import numpy as np

def legogru(weights_wx, weights_wh, word_size, hidden_size):
    weights_wx = weights_wx.flatten().reshape(3, hidden_size, word_size)
    weights_i2h = np.concatenate([weights_wx[0].T, weights_wx[1].T, weights_wx[2].T], axis=1)
    weights_wh = weights_wh.flatten().reshape(3, hidden_size, hidden_size)
    weights_h2h = np.concatenate([weights_wh[1].T, weights_wh[2].T], axis=1)
    weights_h2h = np.concatenate([weights_wh[0].T.flatten(), weights_h2h.flatten()]).reshape(3, hidden_size, hidden_size)
    weights = np.concatenate([weights_i2h.flatten(), weights_h2h.flatten()])
    weights = weights.reshape(1, 1, 1, len(weights))
    return weights

def tensor2numpy(tensor_proto = TensorProtoIO()):
    """
    convert tensor to numpy
    """
    return np.array(tensor_proto.get_data())

def ParserFeedDecorator(OpName):
    def warpper(Parser):
        def warpper_args(args):
            Parser(args)
            OpsRegister()[OpName].feed_node_attr(args[0])
            args[3].set_name(OpName)
            args[0].set_op(args[3]())
        return warpper_args
    return warpper

# common 
def NotNeededInInference(args):
    # args is tuple object
    node_io = args[0]
    layer = args[1]
    tensors = args[2]
    #logger(verbose.ERROR).feed("Layer type(", layer.name, " : ", layer.type , ") with ", len(tensors)," tensors  not needed in inference.")



@ParserFeedDecorator("MatchMatrix")
def Parser_batch_match_mat_tensor(args):
    layer = args[1]
    tensors = args[2]
    param = layer.match_matrix_tensor_param
    tensors[0].set_shape([1, param.dim_in, param.dim_t, param.dim_in])
    OpsRegister()["MatchMatrix"].weight_1 = tensors[0]
    OpsRegister()["MatchMatrix"].dim_in = param.dim_in
    OpsRegister()["MatchMatrix"].dim_t = param.dim_t
    OpsRegister()["MatchMatrix"].linear_term = param.linear_term
    OpsRegister()["MatchMatrix"].bias_term = param.bias_term
    OpsRegister()["MatchMatrix"].diag_init = param.diag_init
    OpsRegister()["MatchMatrix"].diag_init_dim_num = param.diag_init_dim_num
    OpsRegister()["MatchMatrix"].init_low = param.init_low
    OpsRegister()["MatchMatrix"].init_up = param.init_up



@ParserFeedDecorator("Convolution")
def Parser_batch_var_size_conv(args):
    layer = args[1]
    tensors = args[2]
    param = layer.var_size_conv_param
    tensors[0].set_shape([param.output_channel, param.input_channel, param.kernel_h, param.kernel_w])
    OpsRegister()["Convolution"].weight_1 = tensors[0]
    OpsRegister()["Convolution"].filter_num = param.output_channel
    OpsRegister()["Convolution"].kernel_size = [param.kernel_h, param.kernel_w]
    OpsRegister()["Convolution"].strides = [param.stride_h, param.stride_w]
    OpsRegister()["Convolution"].padding = [param.kernel_h / 2, param.kernel_w / 2]
    OpsRegister()["Convolution"].dilation_rate = [1, 1]  ########## ??? ##########
    OpsRegister()["Convolution"].group = 1
    OpsRegister()["Convolution"].axis = 0  ########## ??? ##########
    OpsRegister()["Convolution"].bias_term = param.bias_term
    OpsRegister()["Convolution"].input_channel = param.input_channel


@ParserFeedDecorator("Embedding")
def Parser_batch_embed(args):
    layer = args[1]
    tensors = args[2]
    param = layer.embedding_param
    tensors[0].set_shape([1, 1, param.num_voc, param.num_emb])
    OpsRegister()["Embedding"].weight_1 = tensors[0]
    OpsRegister()["Embedding"].word_num = param.num_voc
    OpsRegister()["Embedding"].emb_dim = param.num_emb
    OpsRegister()["Embedding"].padding_idx = -1  ###???
    if param.need_reverse:
        OpsRegister()["Embedding"].num_direct = 2
    else:
        OpsRegister()["Embedding"].num_direct = 1		
    # OpsRegister()["Embedding"].bias_term = 0  ########## ??? ##########
    OpsRegister()["Embedding"].need_reverse = param.need_reverse
    #OpsRegister()["Embedding"].max_seq_len = param.max_seq_len

@ParserFeedDecorator("Dense")
def Parser_batch_full_connect(args):
    layer = args[1]
    tensors = args[2]
    param = layer.full_connect_param
    tensors[0].set_shape([1, 1, param.num_out, param.num_in])
    OpsRegister()["Dense"].bias_term = param.bias
    # set bias term.
    if OpsRegister()["Dense"].bias_term:
        tensors[1].set_shape([1, 1, 1, param.num_out])
        OpsRegister()["Dense"].weight_2 = tensors[1]
    OpsRegister()["Dense"].weight_1 = tensors[0]
    OpsRegister()["Dense"].in_dim = param.num_in
    OpsRegister()["Dense"].out_dim = param.num_out
    OpsRegister()["Dense"].axis = param.axis

@ParserFeedDecorator("Gru")
def Parser_batch_grnn(args):
    layer = args[1]
    tensors = args[2]
    param = layer.grnn_param
    tensors[0].set_shape([1, 3, param.num_input, param.num_hidden])
    # print 'before reshape tensor0 shape:'
    # print tensors[0].get_shape()
    tensors[1].set_shape([1, 3, param.num_hidden, param.num_hidden])
    # compute tensor0's shape
    # tensor_tmp0 = tensors[0].tensor2numpy()
    # tensor_tmp1 = tensors[1].tensor2numpy()
    tensor_tmp0 = tensor2numpy(tensors[0])
    tensor_tmp1 = tensor2numpy(tensors[1])
    tensor_tmp2 = legogru(tensor_tmp0, tensor_tmp1, param.num_input, param.num_hidden)
    tensors[0].set_data(tensor_tmp2.flatten(), "float")
    tensors[0].set_shape(tensor_tmp2.shape)
    # print 'after rehspae tnesor0 shape:'
    # print tensors[0].get_shape()
    # set tensors1
    tensor_tmp3 = np.zeros([1, 1, 1, 3 * param.num_hidden])
    tensors[1].set_data(tensor_tmp3.flatten(), "float")
    tensors[1].set_shape(tensor_tmp3.shape)
    OpsRegister()["Gru"].weight_1 = tensors[0]
    OpsRegister()["Gru"].weight_2 = tensors[1]
    OpsRegister()["Gru"].is_reverse = False
    OpsRegister()["Gru"].gate_activation = "sigmoid"
    OpsRegister()["Gru"].activation = "tanh"
    OpsRegister()["Gru"].gru_formula = "gru_cudnn"

@ParserFeedDecorator("TopKPooling")
def Parser_batch_topk_pooling(args):
    layer = args[1]
    param = layer.top_k_pooling_param
    OpsRegister()["TopKPooling"].top_k = param.top_k
    OpsRegister()["TopKPooling"].feat_map_num = param.feat_map_num


################

@ParserFeedDecorator("Split")
def Parser_split(args):
    split_num = args[1]
    OpsRegister()["Split"].split_num = split_num

@ParserFeedDecorator("ReLU")
def Parser_relu(args):
    pass

@ParserFeedDecorator("Concat")
def Parser_concat(args):
    layers = args[1]
    param = layers.concat_simple_param
    OpsRegister()["Concat"].axis = param.axis


@ParserFeedDecorator("Reverse")
def Parser_batch_reverse_input(args):
    pass

@ParserFeedDecorator("ReverseSequence")
def Parser_batch_reverse_sequence(args):
    pass


@ParserFeedDecorator("BatchConcatD1SeqLayer")
def Parser_batch_concatd1_seq(args):
    pass

@ParserFeedDecorator("TopKAvgPooling")
def Parser_batch_topk_avg_pooling_by_row(args):
    layer = args[1]
    param = layer.top_k_avg_pooling_by_row_param
    OpsRegister()["TopKAvgPooling"].top_ks = list(param.top_ks)
    OpsRegister()["TopKAvgPooling"].feat_map_num = param.feat_map_num
    OpsRegister()["TopKAvgPooling"].is_pooling_by_row = param.is_pooling_by_row
#	pass


@ParserFeedDecorator("SequencePool")
def Parser_batch_extract_last(args):
    OpsRegister()["SequencePool"].pooltype = "LAST"

@ParserFeedDecorator("ConvUnpaddingPadding")
def Paser_unpadding_padding(args):
    pass

@ParserFeedDecorator("Input")
def Parser_input(args):
    layer = args[1]
    OpsRegister()["Input"].input_shape = list([layer.max_len * layer.max_batch, 1, 1, 1])
    OpsRegister()["Input"].max_len = layer.max_len
    OpsRegister()["Input"].max_batch = layer.max_batch


################
'''
BatchReverseInputLayer
BatchEmbeddingLayer
BatchGrnnLayer
BatchReverseSequenceLayer
BatchConcatByColLayer
BatchMatchMatrixTensorLayer   many args***********
RELULayer
BatchVarSizeConvLayer         many args***********
BatchConcatD1SeqLayer           NO
BatchTopKPoolingLayer          
BatchTopKAvgPoolingByRowLayer   NO
BatchExtractLastLayer           NO
BatchConcatLayer
BatchFullConnectLayer         many args***********
'''

LEGO_NODE_FILLER = {
        "BatchVarSizeConvLayer":OpsRegister()["Convolution"].set_parser(Parser_batch_var_size_conv),
        "BatchEmbeddingLayer":OpsRegister()["Embedding"].set_parser(Parser_batch_embed),
        "BatchFullConnectLayer":OpsRegister()["Dense"].set_parser(Parser_batch_full_connect),
        "BatchGrnnLayer":OpsRegister()["Gru"].set_parser(Parser_batch_grnn),
        "BatchMatchMatrixTensorLayer":OpsRegister()["MatchMatrix"].set_parser(Parser_batch_match_mat_tensor),        
        # No blobs
        "BatchSplitLayer":OpsRegister()["Split"].set_parser(Parser_split),
        "RELULayer":OpsRegister()["ReLU"].set_parser(Parser_relu),
        "BatchConcatByColLayer":OpsRegister()["Concat"].set_parser(Parser_concat),
        "BatchConcatLayer":OpsRegister()["Concat"].set_parser(Parser_concat),        
        # No blobs and Not yet realized
        "BatchTopKPoolingLayer":OpsRegister()["TopKPooling"].set_parser(Parser_batch_topk_pooling),
        "BatchReverseInputLayer":OpsRegister()["Reverse"].set_parser(Parser_batch_reverse_input),
        "BatchReverseSequenceLayer":OpsRegister()["ReverseSequence"].set_parser(Parser_batch_reverse_sequence),        
        #new No blobs and Not yet realized
        "BatchConcatD1SeqLayer":OpsRegister()["BatchConcatD1SeqLayer"].set_parser(Parser_batch_concatd1_seq),
        "BatchTopKAvgPoolingByRowLayer":OpsRegister()["TopKAvgPooling"].set_parser(Parser_batch_topk_avg_pooling_by_row),
        "BatchExtractLastLayer":OpsRegister()["SequencePool"].set_parser(Parser_batch_extract_last),
        "UnpaddingPaddingLayer":OpsRegister()["ConvUnpaddingPadding"].set_parser(Paser_unpadding_padding),        
        "Input":OpsRegister()["Input"].set_parser(Parser_input)

}
