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
 * \file bilinear_upsample.cu
 * \brief bilinear upsample operator
 * \author Hang Zhang
 * Adapted from PyTorch
*/
#include <cuda_runtime_api.h>
#include <algorithm>
#include "devicetensor.h"
#include "adaptive_avg_pooling-inl.h"

#define START_IND(a, b, c) static_cast<int>(floor(static_cast<float>(a * c) / b))
#define END_IND(a, b, c) static_cast<int>(ceil(static_cast<float>((a + 1) * c) / b))
#define CUDA_MAX_THREADS 1024   // this is safe, in reality 256 is our limit

namespace mxnet {
namespace op {

template<typename In, typename Out>
struct ScalarConvert {
  static __host__ __device__ __forceinline__ Out to(const In v) { return (Out) v; }
};

/*
 * Description:
 *    this function adaptively average pools an input 4D tensor along dimensions 2 and 3
 *    4D input, 4D output
 */
template <typename T>
__global__ void adaptiveaveragepool(T *input, T *output,
                        int isizeH, int isizeW,
                        int osizeH, int osizeW,
                        int64_t istrideD, int64_t istrideH, int64_t istrideW) {
  // iterators on output pixels
  int oh, ow;

  // select input/output plane based on thread/block ID
  int o_plane = blockIdx.x;
  int i_plane = o_plane;

  output = output + o_plane*osizeH*osizeW;
  input = input + i_plane*istrideD;

  int ostartH = blockDim.y*blockIdx.y + threadIdx.y;
  int oendH = osizeH;
  const int ostepH = blockDim.y*gridDim.y;

  int ostartW = threadIdx.x;
  int oendW = osizeW;
  const int ostepW = blockDim.x;

  // For all output pixels...
  for (oh = ostartH; oh < oendH; oh += ostepH) {
    int istartH = START_IND(oh, osizeH, isizeH);
    int iendH   = END_IND(oh, osizeH, isizeH);
    int kH = iendH - istartH;

    for (ow = ostartW; ow < oendW; ow += ostepW) {
      int istartW = START_IND(ow, osizeW, isizeW);
      int iendW   = END_IND(ow, osizeW, isizeW);
      int kW = iendW - istartW;

      // Compute the average pooling over corresponding input pixels
      T *ptr_input = input + istartH*istrideH + istartW*istrideW;
      T *ptr_output = output + oh*osizeW + ow;
      T sum = ScalarConvert<int, T>::to(0);
      int ih, iw;
      for (ih = 0; ih < kH; ++ih) {
        for (iw = 0; iw < kW; ++iw) {
          T val = ptr_input[iw*istrideW];
          sum += val;
        }
        ptr_input += istrideH;  // next input line
      }
      // Update output
      *ptr_output = sum / kH / kW;
    }
  }
}

/*
 * Description:
 *    this function computes the gradInput from gradOutput
 *    (uses atomic add)
 */
template <typename T>
__global__ void atomicadaptiveaveragegradinput(
  T *gradInput, T *gradOutput,
  int isizeH, int isizeW, int osizeH, int osizeW
) {
  // iterators on output indices
  int oh, ow;

  // select input/output plane based on thread/block ID
  int o_plane = blockIdx.x;
  int i_plane = o_plane;

  gradOutput = gradOutput + o_plane*osizeW*osizeH;
  gradInput = gradInput + i_plane*isizeW*isizeH;

  int ostartH = blockDim.y*blockIdx.y + threadIdx.y;
  int oendH = osizeH;
  int ostepH = blockDim.y*gridDim.y;

  int ostartW = threadIdx.x;
  int oendW = osizeW;
  int ostepW = blockDim.x;

  // For all output pixels...
  for (oh = ostartH; oh < oendH; oh += ostepH) {
    int istartH = START_IND(oh, osizeH, isizeH);
    int iendH   = END_IND(oh, osizeH, isizeH);
    int kH = iendH - istartH;

    for (ow = ostartW; ow < oendW; ow += ostepW) {
      int istartW = START_IND(ow, osizeW, isizeW);
      int iendW   = END_IND(ow, osizeW, isizeW);
      int kW = iendW - istartW;

      // Compute the gradients for over corresponding input pixels
      T *ptr_gradInput = gradInput + istartH*isizeW + istartW;
      T *ptr_gradOutput = gradOutput + oh*osizeW + ow;
      T grad_delta = *ptr_gradOutput / kW / kH;

      int ih, iw;
      for (ih = 0; ih < kH; ++ih) {
        for (iw = 0; iw < kW; ++iw) {
          // atomic add since different threads could update same variable
          atomicAdd(&(ptr_gradInput[iw]), grad_delta);
        }
        ptr_gradInput += isizeW;  // next input line
      }
    }
  }
}


template<typename xpu, typename DType, typename AccReal>
void AdaptiveAvgPoolUpdateOutput(mshadow::Stream<gpu> *s,
                                 const std::vector<TBlob> &input,
                                 const std::vector<TBlob> &output) {
  DeviceTensor<DType, 4> itensor = devicetensor<DType, 4>(input[0]);
  DeviceTensor<DType, 4> otensor = devicetensor<DType, 4>(output[0]);
  DType *input_data = itensor.data_ptr();
  DType *output_data = otensor.data_ptr();

  int64_t sizeB  = itensor.size[0];
  int64_t sizeD  = itensor.size[1];
  int64_t isizeH = itensor.size[2];
  int64_t isizeW = itensor.size[3];

  int64_t istrideD = itensor.stride[1];
  int64_t istrideH = itensor.stride[2];
  int64_t istrideW = itensor.stride[3];

  int64_t osizeH = otensor.size[2];
  int64_t osizeW = otensor.size[3];

  // cuda blocks & threads:
  int blocksH = max(static_cast<int>(16L / sizeD), 1);
  dim3 blocks(sizeB * sizeD, blocksH);
  dim3 threads(32, 8);

  cudaStream_t stream = mshadow::Stream<gpu>::GetStream(s);
  // run averagepool kernel
  adaptiveaveragepool <<<blocks, threads, 0, stream>>> (
    input_data, output_data, isizeH, isizeW, osizeH, osizeW,
    istrideD, istrideH, istrideW);
  MSHADOW_CUDA_POST_KERNEL_CHECK(AdaptiveAvgPoolUpdateOutput);
}

template<typename xpu, typename DType, typename AccReal>
void AdaptiveAvgPoolUpdateGradInput(mshadow::Stream<gpu> *s,
                                    const std::vector<TBlob> &input,
                                    const std::vector<TBlob> &output) {
  DeviceTensor<DType, 4> gradOut = devicetensor<DType, 4>(input[0]);
  DeviceTensor<DType, 4> gradIn = devicetensor<DType, 4>(output[0]);
  DType *gradOutput_data = gradOut.data_ptr();
  DType *gradInput_data = gradIn.data_ptr();

  int64_t sizeB  = gradIn.size[0];
  int64_t sizeD  = gradIn.size[1];
  int64_t isizeH = gradIn.size[2];
  int64_t isizeW = gradIn.size[3];

  int64_t osizeH = gradOut.size[2];
  int64_t osizeW = gradOut.size[3];

  // cuda blocks & threads:
  int blocksH = max(static_cast<int>(16L / sizeD), 1);
  dim3 blocks(sizeB * sizeD, blocksH);
  dim3 threads(32, 8);

  cudaStream_t stream = mshadow::Stream<gpu>::GetStream(s);
  // run updateGradInput kernel, accumulate gradients atomically
  atomicadaptiveaveragegradinput <<<blocks, threads, 0, stream>>> (
    gradInput_data, gradOutput_data, isizeH, isizeW, osizeH, osizeW);
  MSHADOW_CUDA_POST_KERNEL_CHECK(AdaptiveAvgPoolUpdateGradInput);
}

NNVM_REGISTER_OP(AdaptiveAvgPool2D)
.set_attr<FCompute>("FCompute<gpu>", AdaptiveAvgPoolOpForward<gpu>);

NNVM_REGISTER_OP(_backward_AdaptiveAvgPool2D)
.set_attr<FCompute>("FCompute<gpu>", AdaptiveAvgPoolOpBackward<gpu>);

}  // namespace op
}  // namespace mxnet
