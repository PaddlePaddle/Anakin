#include "saber/funcs/impl/cuda/saber_generate_proposals.h"
#include "cuda_fp16.h"
#define TILE_DIM 16
#define NMS_THREADS_PER_BLOCK 64
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

namespace anakin{
namespace saber{
//const float bbox_clip_default = std::log(1000.0 / 16.0);
template<typename Dtype>
__global__ void ker_nchw_to_nhwc(Dtype * out_data,
                                 const int n,
                                 const int c,
                                 const int hw,
                                 const int row_block_num_per_im,
                                 const Dtype* in_data)
{
    __shared__ float tile[TILE_DIM][TILE_DIM];
    int im_id = blockIdx.y / row_block_num_per_im;
    int block_id_y = blockIdx.y % row_block_num_per_im;
    int x_index = blockIdx.x * TILE_DIM + threadIdx.x;
    int y_index = block_id_y * TILE_DIM + threadIdx.y;
    int index_in = im_id * c * hw + x_index + y_index * hw;

    if (x_index < hw && y_index < c) {
        tile[threadIdx.y][threadIdx.x] = in_data[index_in];
    }
    __syncthreads();

    x_index = block_id_y * TILE_DIM + threadIdx.x;
    y_index = blockIdx.x * TILE_DIM + threadIdx.y;
    int index_out = im_id * hw * c + x_index + y_index * c;

    if (x_index < c && y_index < hw) {
        out_data[index_out] = tile[threadIdx.x][threadIdx.y];
    }
}
template<typename Dtype>
void trans(Tensor<NV>* in_tensor, Tensor<NV>* out_tensor, cudaStream_t stream) {
    int n = in_tensor->num();
    int c = in_tensor->channel();
    int hw = in_tensor->height() * in_tensor->width();  
    auto in_data = (const Dtype*)in_tensor->data();
    auto out_data = (Dtype*)out_tensor->mutable_data();
    dim3 block_dim(TILE_DIM, TILE_DIM);
    dim3 grid_dim((hw  + TILE_DIM -1) / TILE_DIM,  n * (c + TILE_DIM -1) / TILE_DIM);
    int row_block_num_per_im = (c + TILE_DIM -1) / TILE_DIM;
    ker_nchw_to_nhwc<Dtype><<<grid_dim, block_dim, 0, stream>>>(out_data,
                                 n,
                                 c,
                                 hw,
                                 row_block_num_per_im,
                                 in_data);
   
}
__global__ void index_init(int* out_data, int h, int w) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = idx; i < h * w; i += blockDim.x * gridDim.x) {
        int w_id = i % w;
        out_data[i] = w_id;
    }
}


template <typename Dtype>
void sort_descending(Tensor<NV>* out_value,
                     Tensor<NV>* out_index,
                     Tensor<NV>* in_value,
                     Tensor<NV>* in_index,
                     const int pre_nms_num,
                     cudaStream_t stream) {
    in_index->reshape(in_value->valid_shape());
    out_value->reshape(Shape({in_value->num(), pre_nms_num, 1, 1}, Layout_NCHW));
    out_index->reshape(Shape({in_value->num(), pre_nms_num, 1, 1}, Layout_NCHW));
    in_index->set_dtype(AK_INT32);
    out_index->set_dtype(AK_INT32);
    int sort_length = in_value->valid_size() / in_value->num();
    index_init<<<CUDA_GET_BLOCKS(in_value->valid_size()),  CUDA_NUM_THREADS, 0,   stream>>>((int*)in_index->mutable_data(), in_value->num(), sort_length);

    Tensor<X86> in_h(in_value->valid_shape());
    Tensor<X86> index_h(in_index->valid_shape());
    cudaMemcpyAsync(in_h.data(), in_value->data(), sizeof(Dtype) * in_value->valid_size(), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(index_h.data(), in_index->data(), sizeof(int) * in_index->valid_size(), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    //cudaDeviceSynchronize();

    auto in_score = (Dtype*)in_h.mutable_data();
    auto out_score = (Dtype*) out_value->mutable_data();
    auto in_index_data = (int*)index_h.mutable_data();
    auto out_index_data = (int *) out_index->mutable_data();
    
    auto compare = [in_score](const int &i, const int &j) {
      return in_score[i] > in_score[j];
    };
    std::vector<Dtype> sorted_scores;
    std::vector<int> sorted_index;
    for (int i = 0; i < in_value->num(); i++) {
        std::partial_sort(in_index_data, in_index_data + pre_nms_num, in_index_data + sort_length, compare);
        for (int j = 0; j < pre_nms_num; j++) {
            sorted_scores.push_back(in_score[in_index_data[j]]);
            sorted_index.push_back(in_index_data[j]);
        }
        in_score += sort_length;
        in_index_data += sort_length;
    }
    cudaMemcpyAsync(out_index_data, &sorted_index[0], sizeof(int)*out_index->valid_size(), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(out_score, &sorted_scores[0], sizeof(Dtype)*out_value->valid_size(), cudaMemcpyHostToDevice, stream);
}
 
//template<typename Dtype>
//void sort_descending(Tensor<NV>* out_value,
//                     Tensor<NV>* out_index,
//                     Tensor<NV>* in_value,
//                     Tensor<NV>* in_index,
//                     cudaStream_t stream) {
//    in_index->set_dtype(AK_INT32);
//    out_index->set_dtype(AK_INT32);
//    in_index->reshape(in_value->valid_shape());
//    out_value->reshape(in_value->valid_shape());
//    out_index->reshape(in_value->valid_shape());
//    auto in_data = (Dtype*)in_value->mutable_data();
//    auto out_data = (Dtype*) out_value->mutable_data();
//    auto in_index_data = (int*)in_index->mutable_data();
//    auto out_index_data = (int *) out_index->mutable_data();
//    int sort_length  = in_value->valid_size()/in_value->num();
//    int count = in_value->valid_size();
//    index_init<<<CUDA_GET_BLOCKS(count),  CUDA_NUM_THREADS, 0,   stream>>>(in_index_data, in_value->num(), sort_length);
//    cudaMemcpyAsync(out_data, in_data, sizeof(Dtype) * in_value->valid_size(), cudaMemcpyDeviceToDevice, stream);
//    cudaStreamSynchronize(stream);
//
//    size_t temp_storage_bytes = 0;
//    void* temp_storage = NULL;
//    cub::DoubleBuffer<Dtype> d_keys(in_data, out_data);
//    cub::DoubleBuffer<int> d_values(in_index_data, out_index_data);
//    cub::DeviceRadixSort::SortPairsDescending<Dtype, int>(
//    temp_storage, temp_storage_bytes, d_keys, d_values, sort_length);
//    cudaMalloc((void**)&temp_storage, temp_storage_bytes);
//    for (int i = 0; i < in_value->num(); i++) {
//        cub::DoubleBuffer<Dtype> d_keys(in_data, out_data);
//        cub::DoubleBuffer<int> d_values(in_index_data, out_index_data);
//        size_t temp_storage_bytes = 0;
//        cub::DeviceRadixSort::SortPairsDescending<Dtype, int>(
//          temp_storage, temp_storage_bytes, d_keys, d_values, sort_length);
//       // thrust::device_vector <Dtype> D(sort_length);
//       // thrust::device_vector <int > Index(sort_length);
//       // thrust::sequence(Index.begin(), Index.end ()); 
//       // thrust::stable_sort_by_key<Dtype, int>(D.begin(), D.end(), Index.begin, thrust::greater<Dtype>());
//        
//        //thrust::stable_sort_by_key<Dtype, int>(out_data, out_data + sort_length, out_index_data, thrust::greater<Dtype>());
//        in_data += sort_length;
//        out_data += sort_length;
//        in_index_data += sort_length;
//        out_index_data += sort_length;
//    }
//}
template <typename T>
__device__ T Min(T a, T b) { return a > b ? b : a; }

template <typename T>
__device__ T Max(T a, T b) { return a > b ? a : b; }

template <typename Dtype>
__global__ void ker_box_decode_and_clip(Dtype* proposals_data,
                                        const Dtype* anchors_data,
                                        const Dtype* deltas_data,
                                        const Dtype* var_data,
                                        const int* index_data, 
                                        const Dtype* im_info_data,
                                        const float bbox_clip_default,
                                        const int img_num,
                                        const int index_length, 
                                        const int anchor_num,
                                        const int count) {
    CUDA_KERNEL_LOOP(tid, count) {
        int im_id = tid / index_length;
        int anchor_id = index_data[tid];
        auto cur_anchor = anchors_data + anchor_id * 4;
        auto cur_delta = deltas_data + anchor_id * 4 + im_id * anchor_num * 4;
        auto cur_proposal = proposals_data + tid * 5;
        auto cur_im_info = im_info_data + im_id * 3;
        Dtype axmin = cur_anchor[0];
        Dtype aymin = cur_anchor[1];
        Dtype axmax = cur_anchor[2];
        Dtype aymax = cur_anchor[3];
        auto w = axmax - axmin + 1.0;
        auto h = aymax - aymin + 1.0;
        auto cx = axmin + 0.5 * w;
        auto cy = aymin + 0.5 * h;
        auto dxmin = cur_delta[0];
        auto dymin = cur_delta[1];
        auto dxmax = cur_delta[2];
        auto dymax = cur_delta[3];
        Dtype d_cx, d_cy, d_w, d_h;
        if (var_data) {
            auto cur_var = var_data + anchor_id * 4;
            d_cx = cx + dxmin * w * cur_var[0];
            d_cy = cy + dymin * h * cur_var[1];
            d_w = exp(Min(dxmax * cur_var[2], bbox_clip_default)) * w;
            d_h = exp(Min(dymax * cur_var[3], bbox_clip_default)) * h;
        } else {
            d_cx = cx + dxmin * w;
            d_cy = cy + dymin * h;
            d_w = exp(Min(dxmax, bbox_clip_default)) * w;
            d_h = exp(Min(dymax, bbox_clip_default)) * h;
        }
        auto oxmin = d_cx - d_w * 0.5;
        auto oymin = d_cy - d_h * 0.5;
        auto oxmax = d_cx + d_w * 0.5 - 1.;
        auto oymax = d_cy + d_h * 0.5 - 1.;
        cur_proposal[0] = im_id;
        cur_proposal[1] = Max(Min(oxmin, cur_im_info[1] - 1.), 0.);
        cur_proposal[2] = Max(Min(oymin, cur_im_info[0] - 1.), 0.);
        cur_proposal[3] = Max(Min(oxmax, cur_im_info[1] - 1.), 0.);
        cur_proposal[4] = Max(Min(oymax, cur_im_info[0] - 1.), 0.);
        //cur_proposal[0] = 8;
        //cur_proposal[1] = 5;
        //cur_proposal[2] = 5;
        //cur_proposal[3] = 5;
        //cur_proposal[4] = 5;
    }
    
}

template<typename Dtype>
void box_decode_and_clip(Tensor<NV>* proposals,
                         const Tensor<NV>* anchors,
                         const Tensor<NV>* deltas,
                         const Tensor<NV>* variances,
                         const Tensor<NV>* index,
                         const Tensor<NV>* im_info,
                         cudaStream_t stream) {
    int img_num = index->num();
    int anchor_num = anchors->valid_size() / 4;
    auto anchors_data = (const Dtype*)anchors->data();
    auto deltas_data = (const Dtype*) deltas->data();
    auto var_data = (const Dtype*) variances->data();
    auto index_data = (const int*) index->data();
    auto im_info_data = (const Dtype*) im_info->data();
    int index_valid_size = index->valid_size();
    int index_length = index->channel();
    proposals->reshape(Shape({img_num * index_length, 5, 1, 1}));
    auto proposals_data = (Dtype*) proposals->mutable_data();
    const float bbox_clip_default =  std::log(1000.0 / 16.0);
    ker_box_decode_and_clip<Dtype><<<CUDA_GET_BLOCKS(index_valid_size), CUDA_NUM_THREADS, 0, stream>>>(
            proposals_data, anchors_data, deltas_data, var_data, index_data,
             im_info_data, bbox_clip_default, img_num, index_length, anchor_num, index->valid_size());
}

template<typename Dtype>
__global__ void ker_filter_bboxes(
                       int *keep,
                       int *keep_num,
                       const Dtype* bboxes, 
                       const Dtype* im_info,
                       const Dtype min_size,
                       const int img_num,
                       const int pre_nms_num) {
    int im_id = blockIdx.x;
    Dtype im_h = im_info[0];
    Dtype im_w = im_info[1];
    Dtype im_scale = im_info[2];

    int cnt = 0;
    __shared__ int keep_index[CUDA_NUM_THREADS];
    for (int tid = threadIdx.x; tid < pre_nms_num; tid += blockDim.x) {
        keep_index[threadIdx.x] = -1;
        __syncthreads();

        auto bboxes_tmp =  bboxes +  (tid + blockIdx.x * pre_nms_num) * 5;
        Dtype xmin = bboxes_tmp[1];
        Dtype ymin = bboxes_tmp[2];
        Dtype xmax = bboxes_tmp[3];
        Dtype ymax = bboxes_tmp[4];

        Dtype w = xmax - xmin + 1.0;
        Dtype h = ymax - ymin + 1.0;
        Dtype cx = xmin + w / 2.;
        Dtype cy = ymin + h / 2.;

        Dtype w_s = (xmax - xmin) / im_scale + 1.;
        Dtype h_s = (ymax - ymin) / im_scale + 1.;

        if (w_s >= min_size && h_s >= min_size && cx <= im_w && cy <= im_h) {
            keep_index[threadIdx.x] = tid;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            int size = (pre_nms_num - tid) < CUDA_NUM_THREADS ? pre_nms_num - tid : CUDA_NUM_THREADS;
            for (int j = 0; j < size; ++j) {
                if (keep_index[j] > -1) {
                    keep[im_id * pre_nms_num + cnt++] = keep_index[j];
                }
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        keep_num[im_id] = cnt;
    }

}

template<typename Dtype>
void filter_bboxes(Tensor<NV>* keep_num, 
                  Tensor<NV>* keep,
                  Tensor<NV>* proposals, 
                  Tensor<NV>* im_info,
                  const Dtype min_size,
                  const int img_num,
                  const int pre_nms_num,
                  cudaStream_t stream) {
    keep_num->reshape(Shape({img_num, 1, 1, 1}, Layout_NCHW));
    keep->reshape(Shape({img_num, pre_nms_num, 1, 1}, Layout_NCHW));
    keep->set_dtype(AK_INT32);
    keep_num->set_dtype(AK_INT32);
    auto proposals_data = (const Dtype*)proposals->data();
    auto im_info_data = (const Dtype*)im_info->data();
    auto keep_num_data = (int*)keep_num->data();
    auto keep_data = (int*)keep->data();
    
    ker_filter_bboxes<Dtype><<<img_num, CUDA_NUM_THREADS, 0, stream>>>(
                      keep_data,
                      keep_num_data,
                      proposals_data,
                      im_info_data,
                      min_size,
                      img_num,
                      pre_nms_num);
}

template <typename Dtype>
 __device__ inline Dtype IoU(const Dtype *a, const Dtype *b) {
  Dtype left = max(a[0], b[0]), right = min(a[2], b[2]);
  Dtype top = max(a[1], b[1]), bottom = min(a[3], b[3]);
  Dtype width = max(right - left + 1, 0.f), height = max(bottom - top + 1, 0.f);
  Dtype inter_s = width * height;
  Dtype s_a = (a[2] - a[0] + 1) * (a[3] - a[1] + 1);
  Dtype s_b = (b[2] - b[0] + 1) * (b[3] - b[1] + 1);
  return inter_s / (s_a + s_b - inter_s);
}


__global__ void NMSKernel(uint64_t *dev_mask,
                          const int n_boxes,
                          const int* keep_index,
                          const float nms_overlap_thresh,
                          const int col_blocks,
                          const float *dev_boxes) {
  const int row_start = blockIdx.y;
  const int col_start = blockIdx.x;

  const int row_size =
      min(n_boxes - row_start * NMS_THREADS_PER_BLOCK, NMS_THREADS_PER_BLOCK);
  const int col_size =
      min(n_boxes - col_start * NMS_THREADS_PER_BLOCK, NMS_THREADS_PER_BLOCK);

  __shared__ float block_boxes[NMS_THREADS_PER_BLOCK * 4];
  if (threadIdx.x < col_size) {
    int box_id = keep_index[NMS_THREADS_PER_BLOCK * col_start + threadIdx.x];
    block_boxes[threadIdx.x * 4 + 0] = dev_boxes[box_id * 5 + 1];
    block_boxes[threadIdx.x * 4 + 1] = dev_boxes[box_id * 5 + 2];
    block_boxes[threadIdx.x * 4 + 2] = dev_boxes[box_id * 5 + 3];
    block_boxes[threadIdx.x * 4 + 3] = dev_boxes[box_id * 5 + 4];
  }
  __syncthreads();

  if (threadIdx.x < row_size) {
    const int cur_box_idx = NMS_THREADS_PER_BLOCK * row_start + threadIdx.x;
    const float *cur_box = dev_boxes + keep_index[cur_box_idx] * 5 + 1;
    int i = 0;
    uint64_t t = 0;
    int start = 0;
    if (row_start == col_start) {
      start = threadIdx.x + 1;
    }
    for (i = start; i < col_size; i++) {
      if (IoU(cur_box, block_boxes + i * 4) > nms_overlap_thresh) {
        t |= 1ULL << i;
      }
    }
    //const int col_blocks = DIVUP(n_boxes, NMS_THREADS_PER_BLOCK);
    dev_mask[cur_box_idx * col_blocks + col_start] = t;
  }
}


template <typename Dtype>
void NMS(Tensor<NV> *keep_out,
         const Tensor<NV> *proposals,
         const int boxes_num,
         const int* keep_index,
         const Dtype nms_threshold,
         const int post_nms_top_n,
         cudaStream_t stream) {
  const int col_blocks = DIVUP(boxes_num, NMS_THREADS_PER_BLOCK);
  dim3 blocks(DIVUP(boxes_num, NMS_THREADS_PER_BLOCK),
              DIVUP(boxes_num, NMS_THREADS_PER_BLOCK));
  dim3 threads(NMS_THREADS_PER_BLOCK);
  keep_out->set_dtype(AK_INT32);

  Tensor<NV> mask(Shape({boxes_num, col_blocks, 1, 1}, Layout_NCHW), AK_UINT64);
  auto boxes_data = (const Dtype*)proposals->data();
  auto mask_data = (uint64_t*) mask.mutable_data();
  NMSKernel<<<blocks, threads, 0, stream>>>(mask_data,
      boxes_num, keep_index, nms_threshold, col_blocks, boxes_data);

  Tensor<X86> mask_h(Shape({boxes_num, col_blocks, 1, 1}, Layout_NCHW), AK_UINT64);
  auto mask_data_h = (uint64_t*) mask_h.mutable_data();
  cudaMemcpyAsync(mask_data_h, mask_data, sizeof(uint64_t) * mask.valid_size(), cudaMemcpyDeviceToHost, stream);
  std::vector<int> keep_index_h(boxes_num);
  cudaMemcpyAsync(keep_index_h.data(), keep_index, sizeof(int)* boxes_num, cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  std::vector<uint64_t> remv(col_blocks);
  memset(&remv[0], 0, sizeof(uint64_t) * col_blocks);

  std::vector<int> keep_vec;
  int num_to_keep = 0;
  for (int i = 0; i < boxes_num; i++) {
      int nblock = i / NMS_THREADS_PER_BLOCK;
      int inblock = i % NMS_THREADS_PER_BLOCK;
      if (num_to_keep >= post_nms_top_n) {
         break;
      }

      if (!(remv[nblock] & (1ULL << inblock))) {
          ++num_to_keep;
          keep_vec.push_back(keep_index_h[i]);
          //keep_vec.push_back(i);
          uint64_t *p = mask_data_h + i * col_blocks;
          for (int j = nblock; j < col_blocks; j++) {
              remv[j] |= p[j];
          }
      }
  }
  keep_out->reshape(Shape({num_to_keep, 1, 1, 1}, Layout_NCHW));
  cudaMemcpyAsync(keep_out->mutable_data(), &keep_vec[0], sizeof(int)*num_to_keep, cudaMemcpyHostToDevice, stream);
}

template <typename Dtype>
__global__ void ker_gather(Dtype* boxes_out,
                           const Dtype* proposals,
                           const int box_num,
                           const int box_dim,
                           const int* keep_index) {
    CUDA_KERNEL_LOOP(tid, box_num * box_dim) {
        int box_id = tid / box_dim;
        int dim_id = tid % box_dim;
        boxes_out[tid] = proposals[keep_index[box_id] * box_dim + dim_id];  
    }
}


template <typename Dtype>
void gather_box(Tensor<NV> *boxes_out,
                   const Tensor<NV>*proposals,
                   const int* index,
                   const int num,
                   cudaStream_t stream) {
   const Dtype* proposals_data = (const Dtype*) proposals->data();
   boxes_out->reshape(std::vector<int>{num, 5, 1, 1});
   Dtype* boxes_out_data = (Dtype*) boxes_out->mutable_data();
   ker_gather<Dtype><<<CUDA_GET_BLOCKS(boxes_out->valid_size()), CUDA_NUM_THREADS, 0, stream>>>(boxes_out_data, proposals_data, num, 5, index);
    
}

template <typename Dtype>
void gather_score(Tensor<NV> *scores_out,
                   const Tensor<NV>*scores,
                   const int* index,
                   const int num,
                   cudaStream_t stream) {
   const Dtype* scores_data = (const Dtype*) scores->data();
   scores_out->reshape(Shape({num, 1, 1, 1}, Layout_NCHW));
   Dtype* scores_out_data = (Dtype*) scores_out->mutable_data();
   ker_gather<Dtype><<<CUDA_GET_BLOCKS(scores_out->valid_size()), CUDA_NUM_THREADS, 0, stream>>>(scores_out_data, scores_data, num, 1, index);

}


template <DataType OpDtype>
SaberStatus SaberGenerateProposals<NV, OpDtype>::dispatch( \
        const std::vector<Tensor<NV>*>& inputs,
        std::vector<Tensor<NV>*>& outputs,
        GenerateProposalsParam<NV>& param) {
    cudaStream_t cuda_stream = this->_ctx->get_compute_stream();
    auto anchors = *inputs[0];
    auto bbox_deltas = *inputs[1];
    auto im_info = *inputs[2];
    auto scores = *inputs[3];
    auto variances = *inputs[4];
    auto rpn_rois = outputs[0];
    auto rpn_roi_probs = outputs[1];
    int pre_nms_top_n = param.pre_nms_top_n;
    int post_nms_top_n = param.post_nms_top_n;
    float nms_threshold = param.nms_thresh;
    float min_size = param.min_size;
    float eta = param.eta;
    CHECK_EQ(eta, 1.0f) << "eta is not equal to 1, now other param has not been supported";
    Shape scores_shape = scores.valid_shape();
    Shape scores_swap_shape({scores_shape[0], scores_shape[2], scores_shape[3] , scores_shape[1]}, Layout_NCHW);
    Shape bbox_deltas_shape = bbox_deltas.valid_shape();
    Shape bbox_deltas_swap_shape({bbox_deltas_shape[0], bbox_deltas_shape[2],
            bbox_deltas_shape[3] , bbox_deltas_shape[1]}, Layout_NCHW);
    _scores_swap.reshape(scores_swap_shape);
    _bbox_deltas_swap.reshape(bbox_deltas_swap_shape);
    /*swap and sort*/
    trans<OpDataType>(&scores, &_scores_swap, cuda_stream);
    trans<OpDataType>(&bbox_deltas, &_bbox_deltas_swap, cuda_stream);
    cudaDeviceSynchronize();
    
    int bbox_num = bbox_deltas.valid_size() / 4;
    rpn_rois->reshape(std::vector<int>{post_nms_top_n, 5, 1, 1});
    rpn_roi_probs->reshape(std::vector<int>{post_nms_top_n, 1, 1, 1});
    int pre_nms_num = (_scores_swap.valid_size() <= 0 || _scores_swap.valid_size() > pre_nms_top_n) ? pre_nms_top_n : _scores_swap.valid_size(); 
    int img_num = _scores_swap.num();
    sort_descending<OpDataType>(&_sorted_scores, &_sorted_index, &_scores_swap, &_scores_index, pre_nms_num, cuda_stream);

    // 2. box decode and clipping
    box_decode_and_clip<OpDataType>(&_proposals,
                        &anchors, &_bbox_deltas_swap,
                        &variances,
                        &_sorted_index,
                        &im_info,
                        cuda_stream);
    // 3. filter bbox
    filter_bboxes<OpDataType>(&_keep_num, &_keep, &_proposals, &im_info,
                  min_size, img_num, pre_nms_num,
                  cuda_stream);
    
    // 4. NMS
    std::vector<int> keep_num_vec;
    keep_num_vec.resize(img_num);
    cudaMemcpyAsync(&keep_num_vec[0], _keep_num.data(), sizeof(int)*img_num, cudaMemcpyDeviceToHost, cuda_stream);
    int total_boxes = 0;
    for (int i = 0; i < img_num; i++) {
        Shape score_slice_shape = _sorted_scores.valid_shape();
        Shape proposals_slice_shape = _proposals.valid_shape();
        proposals_slice_shape[0] = pre_nms_num;
        score_slice_shape[0] = 1;
        Tensor<NV> sorted_scores_slice((void*)((OpDataType*)_sorted_scores.mutable_data() + i * _sorted_scores.get_stride()[0]), NV(), this->_ctx->get_device_id(), score_slice_shape);
        Tensor<NV> proposals_slice((void*)((OpDataType*)_proposals.mutable_data() + i * pre_nms_num * _proposals.get_stride()[0]), NV(), this->_ctx->get_device_id(), proposals_slice_shape);

        auto keep_data = (const int*)_keep.data() + i * pre_nms_num;
        auto keep_num = keep_num_vec[i];
        if (nms_threshold <= 0) {
            gather_box<OpDataType>(&_boxes_out, &proposals_slice, keep_data, keep_num, cuda_stream);
            gather_score<OpDataType>(&_scores_out, &sorted_scores_slice, keep_data, keep_num, cuda_stream);
            total_boxes += keep_num;
        } else {
            NMS<OpDataType>(&_keep_nms, &proposals_slice,  keep_num, keep_data, nms_threshold, post_nms_top_n, cuda_stream);
            auto keep_nms_data  = (const int*)_keep_nms.data();
            auto keep_nms_num  = _keep_nms.valid_size();
            gather_box<OpDataType>(&_boxes_out, &proposals_slice, keep_nms_data, keep_nms_num, cuda_stream);
            gather_score<OpDataType>(&_scores_out, &sorted_scores_slice, keep_nms_data, keep_nms_num, cuda_stream);
        }

        cudaMemcpyAsync((OpDataType*)rpn_rois->mutable_data() + total_boxes * 5,  
                (const OpDataType*)_boxes_out.data(),
               sizeof(OpDataType) * _boxes_out.valid_size(),
               cudaMemcpyDefault,
               cuda_stream);
        cudaMemcpyAsync((OpDataType*)rpn_roi_probs->mutable_data() + total_boxes,
                (const OpDataType*)_scores_out.data(),
                sizeof(OpDataType) * _scores_out.valid_size(),
                cudaMemcpyDefault,
                cuda_stream);
        total_boxes += _keep_nms.valid_size();
    }
    rpn_rois->reshape(std::vector<int>{total_boxes, 5, 1, 1});
    rpn_roi_probs->reshape(std::vector<int>{total_boxes, 1, 1, 1});

    CUDA_POST_KERNEL_CHECK;
    return SaberSuccess;
}

template class SaberGenerateProposals<NV, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberGenerateProposals, GenerateProposalsParam, NV, AK_INT8);
DEFINE_OP_TEMPLATE(SaberGenerateProposals, GenerateProposalsParam, NV, AK_HALF);
}
}
