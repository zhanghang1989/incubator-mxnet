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
 * \file bilinear_upsample-inl.h
 * \brief bilinear upsample operator
 * \author Hang Zhang
 */
#ifndef MXNET_OPERATOR_ADAPTIVE_AVG_POOLING_INL_H_
#define MXNET_OPERATOR_ADAPTIVE_AVG_POOLING_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <mxnet/ndarray.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "../ndarray/ndarray_function.h"
#include "./operator_common.h"
#include "./mxnet_op.h"
#include "./mshadow_op.h"

namespace mxnet {
namespace op {

struct AdaptiveAvgPoolParam : public dmlc::Parameter<AdaptiveAvgPoolParam> {
  int output_size;
  DMLC_DECLARE_PARAMETER(AdaptiveAvgPoolParam) {
    DMLC_DECLARE_FIELD(output_size).set_range(1, 1000)
    .describe("output size");
  }
};

static inline bool IsWriting(const OpReqType ort) {
  return ort == kWriteTo || ort == kWriteInplace;
}

template<typename xpu, typename DType, typename AccReal>
void AdaptiveAvgPoolUpdateOutput(mshadow::Stream<cpu> *s,
                                           const std::vector<TBlob> &input,
                                           const std::vector<TBlob> &output);

template<typename xpu, typename DType, typename AccReal>
void AdaptiveAvgPoolUpdateGradInput(mshadow::Stream<cpu> *s,
                                              const std::vector<TBlob> &input,
                                              const std::vector<TBlob> &output);

#if MXNET_USE_CUDA
template<typename xpu, typename DType, typename AccReal>
void AdaptiveAvgPoolUpdateOutput(mshadow::Stream<gpu> *s,
                                           const std::vector<TBlob> &input,
                                           const std::vector<TBlob> &output);

template<typename xpu, typename DType, typename AccReal>
void AdaptiveAvgPoolUpdateGradInput(mshadow::Stream<gpu> *s,
                                              const std::vector<TBlob> &input,
                                              const std::vector<TBlob> &output);
#endif  // MXNET_USE_CUDA

template <typename xpu>
inline void AdaptiveAvgPoolOpForward(const nnvm::NodeAttrs& attrs,
                                      const OpContext &ctx,
                                      const std::vector<TBlob> &inputs,
                                      const std::vector<OpReqType> &req,
                                      const std::vector<TBlob> &outputs) {
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_REAL_TYPE_SWITCH_EX(inputs[0].type_flag_, DType, AccReal, {
    AdaptiveAvgPoolUpdateOutput<xpu, DType, AccReal>(s, inputs, outputs);
  });
}


template <typename xpu>
inline void AdaptiveAvgPoolOpBackward(const nnvm::NodeAttrs& attrs,
                                     const OpContext &ctx,
                                     const std::vector<TBlob> &inputs,
                                     const std::vector<OpReqType> &req,
                                     const std::vector<TBlob> &outputs) {
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  if (IsWriting(req[0])) {
    // zero grad before backwarding
    size_t out_size = outputs[0].shape_.Size();
    MSHADOW_TYPE_SWITCH(inputs[0].type_flag_, DType, {
      Fill<false>(s, outputs[0], kWriteTo, 0);
    })
  }
  MSHADOW_REAL_TYPE_SWITCH_EX(inputs[0].type_flag_, DType, AccReal, {
    AdaptiveAvgPoolUpdateGradInput<xpu, DType, AccReal>(s, inputs, outputs);
  });
}


static bool AdaptiveAvgPoolOpInferShape(const nnvm::NodeAttrs& attrs,
                                       std::vector<TShape> *in_shape,
                                       std::vector<TShape> *out_shape) {
  using namespace mshadow;
  CHECK_EQ(in_shape->size(), 1U) << "Input:[data]";
  CHECK_EQ(out_shape->size(), 1U) << "Output:[data]";
  const AdaptiveAvgPoolParam& param = nnvm::get<AdaptiveAvgPoolParam>(attrs.parsed);
  TShape dshape(in_shape->at(0));
  if (dshape.ndim() == 0) return false;
  dshape[2] = param.output_size;
  dshape[3] = param.output_size;
  out_shape->clear();
  out_shape->push_back(dshape);
  return true;
}

static bool AdaptiveAvgPoolOpInferType(const nnvm::NodeAttrs& attrs,
                                      std::vector<int> *in_type,
                                      std::vector<int> *out_type) {
  using namespace mshadow;
  CHECK_EQ(in_type->size(), 1U);
  int dtype = (*in_type)[0];
  CHECK_NE(dtype, -1) << "First input must have specified type";
  // For float16 input type beta, gamma, mean, and average are stored in float32.
  // For other input types, these parameters have the same type as input
  // NOTE: This requirement is from cuDNN (v. 4 and 5)
  int dtype_param = 0;
  MSHADOW_REAL_TYPE_SWITCH_EX(dtype, DTypeX, AccRealX, {
      dtype_param = mshadow::DataType<AccRealX>::kFlag; });
  out_type->clear();
  out_type->push_back(dtype_param);
  return true;
}

static inline bool AdaptiveAvgPoolOpStorageType(const nnvm::NodeAttrs &attrs,
                                               const int dev_mask,
                                               DispatchMode *dispatch_mode,
                                               std::vector<int> *in_attrs,
                                               std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 1);
  CHECK_EQ(out_attrs->size(), 1);
  *dispatch_mode = DispatchMode::kFCompute;
  for (int& v : *in_attrs) {
    if (v == - 1) v = kDefaultStorage;
  }
  for (size_t i = 0; i < out_attrs->size(); i++) {
    (*out_attrs)[i] = kDefaultStorage;
  }
  return true;
}


}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_ADAPTIVE_AVG_POOLING_INL_H_
