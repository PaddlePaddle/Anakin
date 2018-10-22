
# Some blob contains char type data, such as the emb_data of quant_mixed_embedding_layer, but never used.
# Floats are stored in all data structures involved in blob_size_of_layer function.

def blob_size_of_layer(layer):
    if layer.type == "EmbeddingLayer":
        layer  = layer.embedding_param
        return [layer.num_voc*layer.num_emb]  #  if >0 , have blob.
    elif layer.type == "QuantEmbeddingLayer":
        return []
    elif layer.type == "QuantEmbeddingWithVsumLayer":
        return []
    elif layer.type == "QuantEmbeddingWithWeightedVsumLayer":
        return []
    elif layer.type == "FullConnectLayer":
        layer = layer.full_connect_param
        if layer.bias:
            return [layer.num_in*layer.num_out, layer.num_out]
        else:
            return [layer.num_in*layer.num_out, 0]
    elif layer.type == "FullConnectLayerWithid":
        return []
    elif layer.type == "SeqFullConnectLayer":
        return []
    elif layer.type == "CRFDecodeLayer":
        layer = layer.crf_param
        return [layer.num_class*layer.num_class, layer.num_class, layer.num_class]
    elif layer.type == "MultiInputFullConnectLayer":
        return []
    elif layer.type == "VSumLayer":    # return -1
        return []
    elif layer.type == "SoftsignLayer":
        return []
    elif layer.type == "RELULayer":
        return []
    elif layer.type == "ConcatLayer":
        return []
    elif layer.type == "SoftmaxLayer":
        return []
    elif layer.type == "CosineLayer":
        return []
    elif layer.type == "SequenceMatchLayer":
        return []
    elif layer.type == "KMaxPoolingLayer":
        return []
    elif layer.type == "MaxPoolingLayer":
        return []
    elif layer.type == "ReverseInputLayer":
        return []
    elif layer.type == "SplitLayer":
        return []
    elif layer.type == "SeqConcatLayer":
        return []
    elif layer.type == "SigmoidLayer":
        return []
    elif layer.type == "TanhLayer":
        return []
    elif layer.type == "WeightedVSumLayer":
        return []
    elif layer.type == "ExtractLastLayer":
        return []
    elif layer.type == "TemporalBowConvLayer":
        return []
    elif layer.type == "ReshapeLayer":
        return []
    elif layer.type == "FlattenLayer":
        return []
    elif layer.type == "WindowBowEmbeddingLayer":
        layer = layer.win_bow_embedding_param
        return [layer.num_voc*layer.num_emb]
    elif layer.type == "WindowAlignedEmbeddingLayer":
        layer = layer.win_aligned_embedding_param
        return [layer.num_voc*layer.num_emb]
    elif layer.type == "HalfPrecisionEmbeddingLayer":
        return []
    elif layer.type == "WindowPaddingLayer":
        layer = layer.win_padding_param
        return [layer.num_emb]
    elif layer.type == "TemporalConvWithKMaxPoolLayer":
        layer = layer.temporal_conv_with_kmax_pool_param
        if layer.has_bias:
            return [layer.input_size*layer.output_size, layer.output_size]
        else:
            return [layer.input_size*layer.output_size, 0]
    elif layer.type == "TemporalConvWithMaxPoolLayer":
        layer = layer.temporal_conv_with_max_pool_param
        if layer.has_bias:
            return [layer.input_size*layer.output_size, layer.output_size]
        else:
            return [layer.input_size*layer.output_size, 0]
    elif layer.type == "TemporalConvOptLayer":
        layer = layer.temporal_conv_opt_param
        if layer.has_bias:
            return [layer.input_size*layer.output_size, layer.output_size]
        else:
            return [layer.input_size*layer.output_size, 0]
    elif layer.type == "TemporalConvLayer":
        layer = layer.temporal_conv_param
        if layer.has_bias:
            return [layer.emb_size*layer.win_size*layer.output_size, layer.output_size]
        else:
            return [layer.emb_size*layer.win_size*layer.output_size, 0]
    elif layer.type == "GrnnLayer":
        layer = layer.grnn_param
        return [layer.num_hidden*layer.num_input*3, layer.num_hidden*layer.num_hidden*3]
    elif layer.type == "GrnnSingleStepLayer":
        return []
    elif layer.type == "HalfPrecisionEmbeddingWithVsumLayer":
        return []
    elif layer.type == "HalfPrecisionEmbeddingWithWeightedVsumLayer":
        return []
    elif layer.type == "HalfPrecisionEmbeddingGrnnExtractLastLayer":
        return []
    elif layer.type == "SequenceMaskLayer":
        return []
    elif layer.type == "SeqArithmeticLayer":
        return []
    elif layer.type == "ArithmeticLayer":
        return []
    elif layer.type == "ErnnLayer":
        layer = layer.ernn_param
        if layer.use_input_weights:
            return [layer.num_hidden*layer.num_input, layer.num_hidden*layer.num_hidden]
        else:
            return [0, layer.num_hidden*layer.num_hidden]
    elif layer.type == "ErnnSingleStepLayer":
        return []
    elif layer.type == "LSTMRnnLayer":
        return []
    elif layer.type == "SeqClassSoftmaxLayer":
        layer = layer.seq_class_softmax_param
        return [layer.num_hidden*layer.num_voc, layer.num_hidden*layer.num_class]
    elif layer.type == "WindowAlignedPaddingLayer":
        return []
    elif layer.type == "ConvertToLMIDLayer":
        return []
    elif layer.type == "TopNLayer":
        return []
    elif layer.type == "NgramEmbeddingWithFCLayer":
        layer = layer.ngram_embedding_with_fc_param
        if layer.bias_term:
            return [layer.emb_size*layer.voc_size, layer.fc_in*layer.fc_out, layer.fc_out]
        else:
            return [layer.emb_size*layer.voc_size, layer.fc_in*layer.fc_out, 0]
    elif layer.type == "MergeSequenceLayer":
        return []
    elif layer.type == "ReverseSequenceLayer":
        return []
    elif layer.type == "SeqSoftmaxLayer":
        return []
    elif layer.type == "BilinearLayer":
        layer = layer.bilinear_param
        if layer.bias_term:
            return [layer.num_in*layer.num_out, 1]
        else:
            return [layer.num_in*layer.num_out, 0]
    elif layer.type == "MatchMatrixTensorLayer":
        layer = layer.match_matrix_tensor_param
        return [layer.dim_in*layer.dim_t*layer.dim_in]
    elif layer.type == "TopKPoolingLayer":
        return []
    elif layer.type == "ConcatByColLayer":
        return []
    elif layer.type == "VarSizeConvLayer":
        layer = layer.var_size_conv_param
        return [layer.input_channel*layer.output_channel*layer.kernel_h*layer.kernel_w]
    elif layer.type == "BatchMatchMatrixTensorLayer":
        layer = layer.match_matrix_tensor_param
        return [layer.dim_in*layer.dim_t*layer.dim_in]
    elif layer.type == "BatchTopKPoolingLayer":
        return []
    elif layer.type == "BatchConcatByColLayer":
        return []
    elif layer.type == "BatchVarSizeConvLayer":
        layer = layer.var_size_conv_param
        return [layer.input_channel*layer.output_channel*layer.kernel_h*layer.kernel_w]
    elif layer.type == "BatchEmbeddingLayer":
        embedding_param  = layer.embedding_param
        return [embedding_param.num_voc*embedding_param.num_emb]
    elif layer.type == "BatchVSumLayer":
        return []
    elif layer.type == "BatchCosineLayer":
        return []
    elif layer.type == "BatchFullConnectLayer":
        layer = layer.full_connect_param
        if layer.bias:
            return [layer.num_in*layer.num_out, layer.num_out]
        else:
            return [layer.num_in*layer.num_out, 0]
    elif layer.type == "BatchImageConvLayer":
        return []
    elif layer.type == "BatchImagePoolingLayer":
        return []
    elif layer.type == "BatchLRNLayer":
        return []
    elif layer.type == "BatchSoftmaxLayer":
        return []
    elif layer.type == "BatchGrnnLayer":
        grnn_param = layer.grnn_param
        return [grnn_param.num_hidden*grnn_param.num_input*3, grnn_param.num_hidden*grnn_param.num_hidden*3]
    elif layer.type == "BatchSplitLayer":
        return []
    elif layer.type == "BatchExtractLastLayer":
        return []
    elif layer.type == "BatchReverseInputLayer":
        return []
    elif layer.type == "BatchConcatLayer":
        return []
    elif layer.type == "BatchWindowAlignedEmbeddingLayer":
        layer = layer.win_aligned_embedding_param
        return [layer.num_voc*layer.num_emb]
    elif layer.type == "BatchTemporalConvWithMaxPoolLayer":
        layer = layer.temporal_conv_with_max_pool_param
        if layer.has_bias:
            return [layer.input_size*layer.output_size, layer.output_size]
        else:
            return [layer.input_size*layer.output_size, 0]
    elif layer.type == "BatchLSTMRnnLayer":
        return []
    elif layer.type == "DepthConcatLayer":
        return []
    elif layer.type == "StrideWinMaxPoolingLayer":
        return []
    elif layer.type == "DotProductLayer":
        return []
    elif layer.type == "BatchSeqFullConnectLayer":
        return []
    elif layer.type == "BatchSeqSoftmaxLayer":
        return []
    elif layer.type == "BatchRecoverShapeByBoundaryLayer":
        return []
    elif layer.type == "BatchReshapeByBoundaryLayer":
        return []
    elif layer.type == "BatchReverseSequenceLayer":
        return []
    elif layer.type == "BatchMergeSequenceLayer":
        return []
    elif layer.type == "SequenceRescaleLayer":
        return []
    elif layer.type == "BatchGroupInputPaddingLayer":
        return []
    elif layer.type == "BatchTransposeInputLayer":
        return []
    elif layer.type == "BatchOperationLayer":
        return []
    elif layer.type == "VSumByWindowLayer":
        return []
    elif layer.type == "BatchImageBatchNormLayer":
        return []
    elif layer.type == "BatchSeqClassSoftmaxLayer":
        layer = layer.seq_class_softmax_param
        return [layer.num_hidden*layer.num_voc, layer.num_hidden*layer.num_class]
    elif layer.type == "GrnnExtractLastLayer":
        return []
    elif layer.type == "BatchConcatD1SeqLayer":
        return []
    elif layer.type == "BatchTopKAvgPoolingByRowLayer":
        return []
    elif layer.type == "BatchExtractLastLayer":
        return []
    else:
        raise Exception ("ERROR: Unknown layer. type: %s", layer.type)
