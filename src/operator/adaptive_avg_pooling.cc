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
 * \file bilinear_upsample.cc
 * \brief bilinear upsample operator
 * \author Hang Zhang
 * Adapted from PyTorch
*/
#include "devicetensor.h"
#include "adaptive_avg_pooling-inl.h"
#include "elemwise_op_common.h"

#define START_IND(a, b, c) static_cast<int>(floor(static_cast<float>(a * c) / b))
#define END_IND(a, b, c) static_cast<int>(ceil(static_cast<float>((a + 1) * c) / b))

namespace mxnet {
namespace op {

template<typename real>
static void SpatialAdaptiveAveragePooling_updateOutput_frame(
          real *input_p,
          real *output_p,
          int64_t sizeD,
          int64_t isizeH,
          int64_t isizeW,
          int64_t osizeH,
          int64_t osizeW,
          int64_t istrideD,
          int64_t istrideH,
          int64_t istrideW) {
  int64_t d;
#pragma omp parallel for private(d)
  for (d = 0; d < sizeD; d++) {
    /* loop over output */
    int64_t oh, ow;
    for (oh = 0; oh < osizeH; oh++) {
      int istartH = START_IND(oh, osizeH, isizeH);
      int iendH   = END_IND(oh, osizeH, isizeH);
      int kH = iendH - istartH;

      for (ow = 0; ow < osizeW; ow++) {
        int istartW = START_IND(ow, osizeW, isizeW);
        int iendW   = END_IND(ow, osizeW, isizeW);
        int kW = iendW - istartW;

        /* local pointers */
        real *ip = input_p   + d*istrideD + istartH*istrideH + istartW*istrideW;
        real *op = output_p  + d*osizeH*osizeW + oh*osizeW + ow;

        /* compute local average: */
        real sum = 0;
        int ih, iw;
        for (ih = 0; ih < kH; ih++) {
          for (iw = 0; iw < kW; iw++) {
            real val = *(ip + ih*istrideH + iw*istrideW);
            sum += val;
          }
        }

        /* set output to local average */
        *op = sum / kW / kH;
      }
    }
  }
}

template<typename real>
static void SpatialAdaptiveAveragePooling_updateGradInput_frame(
          real *gradInput_p,
          real *gradOutput_p,
          int64_t sizeD,
          int64_t isizeH,
          int64_t isizeW,
          int64_t osizeH,
          int64_t osizeW) {
  int64_t d;
#pragma omp parallel for private(d)
  for (d = 0; d < sizeD; d++) {
    real *gradInput_p_d = gradInput_p + d*isizeW*isizeH;
    real *gradOutput_p_d = gradOutput_p + d*osizeW*osizeH;

    /* calculate average */
    int64_t oh, ow;
    for (oh = 0; oh < osizeH; oh++) {
      int istartH = START_IND(oh, osizeH, isizeH);
      int iendH   = END_IND(oh, osizeH, isizeH);
      int kH = iendH - istartH;

      for (ow = 0; ow < osizeW; ow++) {
        int istartW = START_IND(ow, osizeW, isizeW);
        int iendW   = END_IND(ow, osizeW, isizeW);
        int kW = iendW - istartW;

        real grad_delta = gradOutput_p_d[oh*osizeW +ow] / kH / kW;

        int ih, iw;
        for (ih = istartH; ih < iendH; ih++) {
          for (iw = istartW; iw < iendW; iw++) {
            /* update gradient */
            gradInput_p_d[ih*isizeW + iw] += grad_delta;
          }
        }
      }
    }
  }
}


template<typename xpu, typename DType, typename AccReal>
void AdaptiveAvgPoolUpdateOutput(mshadow::Stream<cpu> *s,
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

  int64_t istrideB = itensor.stride[0];
  int64_t istrideD = itensor.stride[1];
  int64_t istrideH = itensor.stride[2];
  int64_t istrideW = itensor.stride[3];

  int64_t osizeH = otensor.size[2];
  int64_t osizeW = otensor.size[3];

  int64_t b;
#pragma omp parallel for private(b)
  for (b = 0; b < sizeB; b++) {
    SpatialAdaptiveAveragePooling_updateOutput_frame<DType>(
      input_data+b*istrideB, output_data+b*sizeD*osizeH*osizeW,
      sizeD,
      isizeH, isizeW,
      osizeH, osizeW,
      istrideD,
      istrideH, istrideW);
  }
}


template<typename xpu, typename DType, typename AccReal>
void AdaptiveAvgPoolUpdateGradInput(mshadow::Stream<cpu> *s,
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

  int64_t b;
#pragma omp parallel for private(b)
  for (b = 0; b < sizeB; b++) {
    SpatialAdaptiveAveragePooling_updateGradInput_frame<DType>(
      gradInput_data+b*sizeD*isizeH*isizeW, gradOutput_data+b*sizeD*osizeH*osizeW,
      sizeD,
      isizeH, isizeW,
      osizeH, osizeW);
  }
}


DMLC_REGISTER_PARAMETER(AdaptiveAvgPoolParam);

NNVM_REGISTER_OP(AdaptiveAvgPool2D)
.describe(R"code(TODO docs
)code" ADD_FILELINE)
.set_attr_parser(ParamParser<AdaptiveAvgPoolParam>)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::FInferShape>("FInferShape", AdaptiveAvgPoolOpInferShape)
.set_attr<nnvm::FInferType>("FInferType", AdaptiveAvgPoolOpInferType)
.set_attr<FInferStorageType>("FInferStorageType", AdaptiveAvgPoolOpStorageType)
.set_attr<FCompute>("FCompute<cpu>", AdaptiveAvgPoolOpForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_AdaptiveAvgPool2D"})
.add_argument("data", "NDArray-or-Symbol", "Input data");

NNVM_REGISTER_OP(_backward_AdaptiveAvgPool2D)
.set_attr_parser(ParamParser<AdaptiveAvgPoolParam>)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FInferStorageType>("FInferStorageType", AdaptiveAvgPoolOpStorageType)
.set_attr<FCompute>("FCompute<cpu>", AdaptiveAvgPoolOpBackward<cpu>);


}  // namespace op
}  // namespace mxnet
