#include <cfloat>
#include <cmath>
#include <stdexcept>
#include <type_traits>
#include <vector>
#include <cuda_fp16.h>

#include "../tester/utils.h"

namespace {

constexpr int kRmsNormBlockSize = 256;
constexpr int kWarpSize = 32;
constexpr int kAttentionWarpsPerBlock = 8;
constexpr int kAttentionBlockSize =
    kWarpSize * kAttentionWarpsPerBlock;

template <typename T>
__device__ __forceinline__ float toFloat(T value) {
  return static_cast<float>(value);
}

template <>
__device__ __forceinline__ float toFloat<half>(half value) {
  return __half2float(value);
}

template <typename T>
__device__ __forceinline__ T fromFloat(float value) {
  return static_cast<T>(value);
}

template <>
__device__ __forceinline__ half fromFloat<half>(float value) {
  return __float2half_rn(value);
}

__device__ __forceinline__ float warpReduceSum(float value) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

template <typename T, int ValuesPerLane>
__device__ __forceinline__ float attentionDot(
    const T* query, size_t query_offset,
    const float (&query_values)[ValuesPerLane], const T* key,
    size_t key_offset, int head_dim, int lane) {
  if constexpr (std::is_same<T, float>::value) {
    float dot = 0.0f;
    if (lane == 0) {
      for (int dimension = 0; dimension < head_dim; ++dimension) {
        dot = fmaf(query[query_offset + dimension],
                   key[key_offset + dimension], dot);
      }
    }
    return dot;
  } else {
    float dot = 0.0f;
#pragma unroll
    for (int value_index = 0; value_index < ValuesPerLane; ++value_index) {
      const int dimension = lane + value_index * kWarpSize;
      if (dimension < head_dim) {
        dot = fmaf(query_values[value_index],
                   toFloat(key[key_offset + dimension]), dot);
      }
    }
    return warpReduceSum(dot);
  }
}

template <int BlockSize>
__device__ __forceinline__ float blockReduceSum(float value) {
  static_assert(BlockSize % kWarpSize == 0,
                "Block size must contain complete warps");
  __shared__ float warp_sums[BlockSize / kWarpSize];

  const int lane = threadIdx.x % kWarpSize;
  const int warp = threadIdx.x / kWarpSize;
  value = warpReduceSum(value);
  if (lane == 0) {
    warp_sums[warp] = value;
  }
  __syncthreads();

  value = threadIdx.x < BlockSize / kWarpSize ? warp_sums[lane] : 0.0f;
  if (warp == 0) {
    value = warpReduceSum(value);
  }
  return value;
}

template <typename T>
__global__ void rmsNormKernel(const T* input, const T* weight, T* output,
                              size_t rows, size_t hidden_dim, float eps) {
  const size_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  const size_t row_offset = row * hidden_dim;
  float square_sum = 0.0f;
  for (size_t column = threadIdx.x; column < hidden_dim;
       column += blockDim.x) {
    const float value = toFloat(input[row_offset + column]);
    square_sum += value * value;
  }

  square_sum = blockReduceSum<kRmsNormBlockSize>(square_sum);
  __shared__ float inverse_rms;
  if (threadIdx.x == 0) {
    inverse_rms = rsqrtf(square_sum / static_cast<float>(hidden_dim) + eps);
  }
  __syncthreads();

  for (size_t column = threadIdx.x; column < hidden_dim;
       column += blockDim.x) {
    const float normalized =
        toFloat(input[row_offset + column]) * inverse_rms;
    output[row_offset + column] =
        fromFloat<T>(normalized * toFloat(weight[column]));
  }
}

template <typename T, int ValuesPerLane>
__global__ void flashAttentionKernel(
    const T* query, const T* key, const T* value, T* output,
    int batch_size, int target_seq_len, int src_seq_len, int query_heads,
    int kv_heads, int head_dim, bool is_causal) {
  const int lane = threadIdx.x % kWarpSize;
  const int warp_in_block = threadIdx.x / kWarpSize;
  const int query_index =
      blockIdx.x * kAttentionWarpsPerBlock + warp_in_block;
  const int total_queries = batch_size * target_seq_len * query_heads;
  if (query_index >= total_queries) {
    return;
  }

  const int query_head = query_index % query_heads;
  const int target_index = (query_index / query_heads) % target_seq_len;
  const int batch = query_index / (target_seq_len * query_heads);
  const int queries_per_kv_head = query_heads / kv_heads;
  const int kv_head = query_head / queries_per_kv_head;

  const size_t query_offset =
      ((static_cast<size_t>(batch) * target_seq_len + target_index) *
           query_heads +
       query_head) *
      head_dim;

  // Each lane owns dimensions lane, lane + 32, ... . ValuesPerLane is chosen
  // on the host so common small heads do not pay the register cost of a
  // 256-dimensional head.
  float query_values[ValuesPerLane];
  float output_values[ValuesPerLane];
#pragma unroll
  for (int value_index = 0; value_index < ValuesPerLane; ++value_index) {
    const int dimension = lane + value_index * kWarpSize;
    query_values[value_index] =
        dimension < head_dim ? toFloat(query[query_offset + dimension]) : 0.0f;
    output_values[value_index] = 0.0f;
  }

  float row_max = -FLT_MAX;
  float row_sum = 0.0f;
  const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
  const int visible_sources =
      is_causal && target_index + 1 < src_seq_len
          ? target_index + 1
          : src_seq_len;

  // Use the same stable three-pass structure as the mathematical reference:
  // first find the row maximum, then the softmax denominator, and finally the
  // weighted value sum. This avoids repeatedly rescaling a partially
  // accumulated output, whose rounding error is visible for float inputs.
  for (int source_index = 0; source_index < visible_sources;
       ++source_index) {
    const size_t kv_offset =
        ((static_cast<size_t>(batch) * src_seq_len + source_index) *
             kv_heads +
         kv_head) *
        head_dim;

    const float dot = attentionDot<T, ValuesPerLane>(
        query, query_offset, query_values, key, kv_offset, head_dim, lane);
    if (lane == 0) {
      row_max = fmaxf(row_max, dot * scale);
    }
  }
  row_max = __shfl_sync(0xffffffffu, row_max, 0);

  for (int source_index = 0; source_index < visible_sources;
       ++source_index) {
    const size_t kv_offset =
        ((static_cast<size_t>(batch) * src_seq_len + source_index) *
             kv_heads +
         kv_head) *
        head_dim;

    const float dot = attentionDot<T, ValuesPerLane>(
        query, query_offset, query_values, key, kv_offset, head_dim, lane);
    if (lane == 0) {
      row_sum += expf(dot * scale - row_max);
    }
  }

  row_sum = __shfl_sync(0xffffffffu, row_sum, 0);
  const float inverse_sum = row_sum > 0.0f ? 1.0f / row_sum : 0.0f;

  for (int source_index = 0; source_index < visible_sources;
       ++source_index) {
    const size_t kv_offset =
        ((static_cast<size_t>(batch) * src_seq_len + source_index) *
             kv_heads +
         kv_head) *
        head_dim;

    const float dot = attentionDot<T, ValuesPerLane>(
        query, query_offset, query_values, key, kv_offset, head_dim, lane);
    float probability =
        lane == 0 ? expf(dot * scale - row_max) * inverse_sum : 0.0f;
    probability = __shfl_sync(0xffffffffu, probability, 0);

    for (int value_index = 0; value_index < ValuesPerLane; ++value_index) {
      const int dimension = lane + value_index * kWarpSize;
      if (dimension < head_dim) {
        output_values[value_index] =
            fmaf(probability, toFloat(value[kv_offset + dimension]),
                 output_values[value_index]);
      }
    }
  }

#pragma unroll
  for (int value_index = 0; value_index < ValuesPerLane; ++value_index) {
    const int dimension = lane + value_index * kWarpSize;
    if (dimension < head_dim) {
      output[query_offset + dimension] =
          fromFloat<T>(output_values[value_index]);
    }
  }
}

template <typename T>
void allocateDeviceBuffer(T** pointer, size_t element_count) {
  RUNTIME_CHECK(cudaMalloc(reinterpret_cast<void**>(pointer),
                           element_count * sizeof(T)));
}

template <typename T>
void copyToDevice(T* destination, const std::vector<T>& source) {
  RUNTIME_CHECK(cudaMemcpy(destination, source.data(),
                           source.size() * sizeof(T),
                           cudaMemcpyHostToDevice));
}

template <typename T>
void copyToHost(std::vector<T>& destination, const T* source) {
  RUNTIME_CHECK(cudaMemcpy(destination.data(), source,
                           destination.size() * sizeof(T),
                           cudaMemcpyDeviceToHost));
}

}  // namespace

/**
 * @brief Computes RMSNorm over the last dimension of a 2D tensor.
 * @brief 计算二维张量在最后一个维度上的 RMSNorm（均方根归一化）。
 *
 * The input is a row-major matrix with shape [rows, hidden_dim]. For each row
 * i and column j:
 * 输入是一个行优先（row-major）矩阵，形状为 [rows, hidden_dim]。
 * 对于每一行 i 和每一列 j：
 *
 *   output[i, j] = input[i, j] * rsqrt(mean(input[i, :]^2) + eps) * weight[j]
 *
 * The output vector is preallocated with rows * hidden_dim elements.
 * 输出向量已预先分配了 rows * hidden_dim 个元素。
 *
 * @tparam T Data type of input, weight, and output tensors.
 * @tparam T 输入、权重和输出张量的数据类型。
 * @param[in] h_input Flattened input matrix of shape [rows, hidden_dim].
 * @param[in] h_input 展平（flatten）后的输入矩阵，形状为 [rows, hidden_dim]。
 * @param[in] h_weight Per-column scale vector of shape [hidden_dim].
 * @param[in] h_weight 逐列缩放向量，形状为 [hidden_dim]。
 * @param[out] h_output Flattened output matrix of shape [rows, hidden_dim].
 * @param[out] h_output 展平后的输出矩阵，形状为 [rows, hidden_dim]。
 * @param[in] rows Number of rows/tokens.
 * @param[in] rows 行数 / token 数量。
 * @param[in] hidden_dim Size of the normalized dimension.
 * @param[in] hidden_dim 被归一化的维度大小。
 * @param[in] eps Numerical stability epsilon.
 * @param[in] eps 数值稳定性 epsilon（防止除零）。
 */
template <typename T>
void rmsNorm(const std::vector<T>& h_input, const std::vector<T>& h_weight,
              std::vector<T>& h_output, size_t rows, size_t hidden_dim,
              float eps) {
  if (rows == 0 || hidden_dim == 0) {
    h_output.clear();
    return;
  }
  if (h_input.size() != rows * hidden_dim) {
    throw std::invalid_argument("rmsNorm input size does not match its shape");
  }
  if (h_weight.size() != hidden_dim) {
    throw std::invalid_argument("rmsNorm weight size does not match hidden_dim");
  }

  h_output.resize(rows * hidden_dim);
  T* d_input = nullptr;
  T* d_weight = nullptr;
  T* d_output = nullptr;
  allocateDeviceBuffer(&d_input, h_input.size());
  allocateDeviceBuffer(&d_weight, h_weight.size());
  allocateDeviceBuffer(&d_output, h_output.size());

  copyToDevice(d_input, h_input);
  copyToDevice(d_weight, h_weight);
  rmsNormKernel<T><<<static_cast<unsigned int>(rows), kRmsNormBlockSize>>>(
      d_input, d_weight, d_output, rows, hidden_dim, eps);
  RUNTIME_CHECK(cudaGetLastError());
  copyToHost(h_output, d_output);

  RUNTIME_CHECK(cudaFree(d_output));
  RUNTIME_CHECK(cudaFree(d_weight));
  RUNTIME_CHECK(cudaFree(d_input));
}

/**
 * @brief Computes flash attention for given query, key, and value tensors.
 * @brief 根据给定的查询（query）、键（key）和值（value）张量计算 flash attention（闪存注意力）。
 *
 * @tparam T Data type (float) for input/output tensors
 * @tparam T 输入/输出张量的数据类型（float）
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_q 查询张量，形状为 [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_k 键张量，形状为 [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v 值张量，形状为 [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[out] h_o 注意力输出张量，形状为 [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] batch_size 批次（batch）维度大小
 * @param[in] target_seq_len Target sequence length
 * @param[in] target_seq_len 目标序列长度
 * @param[in] src_seq_len Source sequence length
 * @param[in] src_seq_len 源序列长度
 * @param[in] query_heads Number of query attention heads
 * @param[in] query_heads 查询注意力头的数量
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] kv_heads 键/值头的数量（支持分组查询注意力 GQA）
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] head_dim 每个注意力头的维度大小
 * @param[in] is_causal Whether to apply causal masking
 * @param[in] is_causal 是否应用因果掩码（causal masking）
 */
template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len,
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {
  if (batch_size <= 0 || target_seq_len <= 0 || src_seq_len <= 0 ||
      query_heads <= 0 || kv_heads <= 0 || head_dim <= 0) {
    throw std::invalid_argument("flashAttention dimensions must be positive");
  }
  if (query_heads % kv_heads != 0) {
    throw std::invalid_argument(
        "flashAttention query_heads must be divisible by kv_heads");
  }
  if (head_dim > 8 * kWarpSize) {
    throw std::invalid_argument("flashAttention supports head_dim up to 256");
  }

  const size_t query_count =
      static_cast<size_t>(batch_size) * target_seq_len * query_heads *
      head_dim;
  const size_t kv_count =
      static_cast<size_t>(batch_size) * src_seq_len * kv_heads * head_dim;
  if (h_q.size() != query_count || h_k.size() != kv_count ||
      h_v.size() != kv_count) {
    throw std::invalid_argument(
        "flashAttention input size does not match tensor shapes");
  }

  h_o.resize(query_count);
  T* d_query = nullptr;
  T* d_key = nullptr;
  T* d_value = nullptr;
  T* d_output = nullptr;
  allocateDeviceBuffer(&d_query, h_q.size());
  allocateDeviceBuffer(&d_key, h_k.size());
  allocateDeviceBuffer(&d_value, h_v.size());
  allocateDeviceBuffer(&d_output, h_o.size());

  copyToDevice(d_query, h_q);
  copyToDevice(d_key, h_k);
  copyToDevice(d_value, h_v);

  const int total_queries = batch_size * target_seq_len * query_heads;
  const int block_count =
      (total_queries + kAttentionWarpsPerBlock - 1) /
      kAttentionWarpsPerBlock;
  if (head_dim <= 2 * kWarpSize) {
    flashAttentionKernel<T, 2><<<block_count, kAttentionBlockSize>>>(
        d_query, d_key, d_value, d_output, batch_size, target_seq_len,
        src_seq_len, query_heads, kv_heads, head_dim, is_causal);
  } else if (head_dim <= 4 * kWarpSize) {
    flashAttentionKernel<T, 4><<<block_count, kAttentionBlockSize>>>(
        d_query, d_key, d_value, d_output, batch_size, target_seq_len,
        src_seq_len, query_heads, kv_heads, head_dim, is_causal);
  } else {
    flashAttentionKernel<T, 8><<<block_count, kAttentionBlockSize>>>(
        d_query, d_key, d_value, d_output, batch_size, target_seq_len,
        src_seq_len, query_heads, kv_heads, head_dim, is_causal);
  }
  RUNTIME_CHECK(cudaGetLastError());
  copyToHost(h_o, d_output);

  RUNTIME_CHECK(cudaFree(d_output));
  RUNTIME_CHECK(cudaFree(d_value));
  RUNTIME_CHECK(cudaFree(d_key));
  RUNTIME_CHECK(cudaFree(d_query));
}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// 显式模板实例化（与 TESTER.O 链接时所必需）
// DO NOT MODIFY THIS SECTION
// 请勿修改本部分内容
// *********************************************************************
template void rmsNorm<float>(const std::vector<float>&, const std::vector<float>&,
  std::vector<float>&, size_t, size_t, float);
template void rmsNorm<half>(const std::vector<half>&, const std::vector<half>&,
  std::vector<half>&, size_t, size_t, float);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&,
  const std::vector<half>&, std::vector<half>&,
  int, int, int, int, int, int, bool);
