/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
 /*!
 * Copyright (c) 2018 by Contributors
 * \file encoding.cc
 * \brief Encoding Layer
 * \author Hang Zhang
 */

namespace mxnet {
namespace op {

static const unsigned WARP_SIZE = 32;

// The maximum number of threads in a block
static const unsigned MAX_BLOCK_SIZE = 512U;

template<typename In, typename Out>
struct ScalarConvert {
  static __host__ __device__ __forceinline__ Out to(const In v) { return (Out) v; }
};

// Number of threads in a block given an input size up to MAX_BLOCK_SIZE
static unsigned getNumThreads(int nElem, const bool smaller) {
  unsigned threadSizes[5] = {32, 64, 128, 256, MAX_BLOCK_SIZE};
  const int maxi = smaller ? 4 : 5;
  for (int i = 0; i != maxi; ++i) {
    if (static_cast<unsigned>(nElem) <= threadSizes[i]) {
      return threadSizes[i];
    }
  }
  return smaller ? (MAX_BLOCK_SIZE >> 1) : MAX_BLOCK_SIZE;
}

// Returns the index of the most significant 1 bit in `val`.
__device__ __forceinline__ int getMSB(int val) {
  return 31 - __clz(val);
}

#if CUDA_VERSION >= 9000
#define FULLMASK 0xFFFFFFFF
#define __shfl_xor(...) __shfl_xor_sync(FULLMASK, __VA_ARGS__)
#endif

// Sum across all threads within a warp
template<typename T>
static __device__ __forceinline__ T warpSum(T val) {
#if __CUDA_ARCH__ >= 300
for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
    val += __shfl_xor(val, 1 << i, WARP_SIZE);
  }
#else
__shared__ T values[MAX_BLOCK_SIZE];
values[threadIdx.x] = val;
__threadfence_block();
const int base = (threadIdx.x / WARP_SIZE) * WARP_SIZE;
for (int i = 1; i < WARP_SIZE; i++) {
val += values[base + ((i + threadIdx.x) % WARP_SIZE)];
}
#endif
return val;
}

template<typename DType, typename AccReal>
struct Float2 {
  AccReal v1, v2;
  __device__ Float2() {}
  __device__ Float2(DType v1, DType v2)
    : v1(ScalarConvert<DType, AccReal>::to(v1))
      , v2(ScalarConvert<DType, AccReal>::to(v2)) {}
  __device__ Float2(DType v)
    : v1(ScalarConvert<DType, AccReal>::to(v))
      , v2(ScalarConvert<DType, AccReal>::to(v)) {}
  __device__ Float2(int v)
    : v1(ScalarConvert<int, AccReal>::to(v))
      , v2(ScalarConvert<int, AccReal>::to(v)) {}
  __device__ Float2 &operator+=(const Float2 &a) {
    v1 += a.v1;
    v2 += a.v2;
    return *this;
  }
};

template<typename DType, typename AccReal>
static __device__ __forceinline__ Float2<DType, AccReal> warpSum(Float2<DType, AccReal> value) {
  value.v1 = warpSum(value.v1);
  value.v2 = warpSum(value.v2);
  return value;
}

// Sum across (batch, x/y/z) applying Op() pointwise
template<typename T, typename Op>
static __device__ T reduce(Op op, DeviceTensor tensor, int plane) {
  T sum = (T) 0;
  for (int batch = 0; batch < tensor.OuterSize(); ++batch) {
    for (int x = threadIdx.x; x < tensor.InnerSize(); x += blockDim.x) {
      sum += op(batch, plane, x);
    }
  }

  // sum over NumThreads within a warp
  sum = warpSum(sum);

  // 'transpose', and reduce within warp again
  __shared__ T shared[32];
  __syncthreads();
  if (threadIdx.x % WARP_SIZE == 0) {
    shared[threadIdx.x / WARP_SIZE] = sum;
  }
  if (threadIdx.x >= blockDim.x / WARP_SIZE && threadIdx.x < WARP_SIZE) {
    // zero out the other entries in shared
    shared[threadIdx.x] = (T) 0;
  }
  __syncthreads();
  if (threadIdx.x / WARP_SIZE == 0) {
    sum = warpSum(shared[threadIdx.x]);
    if (threadIdx.x == 0) {
      shared[0] = sum;
    }
  }
  __syncthreads();

  // Everyone picks it up, should be broadcast into the whole gradInput
  return shared[0];
}

template<typename xpu, typename real, typename AccReal>
__global__ void Aggregate_Forward_kernel (
    DeviceTensor<real, 3> E,
    DeviceTensor<real, 3> A,
    DeviceTensor<real, 3> X,
    DeviceTensor<real, 2> C) {
    /* declarations of the variables */
    int b, k, d, N;
    /* Get the index and channels */ 
    b = blockIdx.z;
    d = blockIdx.x;
    k = blockIdx.y;
    N = X.getSize(1);

    /* main operation */
    Encoding_(AggOp) g(A,X,C);
    E[b][k][d] = Encoding_(reduce_agg)(g,b,k,d,N);
}


template<typename xpu, typename real, typename AccReal>
__global__ void Aggregate_Backward_kernel (
    DeviceTensor<real, 3> GA,
    DeviceTensor<real, 3> GE,
    DeviceTensor<real, 3> A,
    DeviceTensor<real, 3> X,
    DeviceTensor<real, 2> C) {
    /* declarations of the variables */
    int b, k, i, D;
    /* Get the index and channels */ 
    b = blockIdx.z;
    i = blockIdx.y;
    k = blockIdx.x;
    D = GE.getSize(2);
    /* main operation */
    Encoding_(AggBackOp) g(GE,X,C);
    GA[b][i][k] = Encoding_(reduce_aggback)(g,b,i,k,D);
}

}  // namespace op
}  // namespace mxnet
