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
 * \file aggregate-inl.h
 * \brief Aggregate Layer
 * \author Hang Zhang
 */
#ifndef MXNET_OPERATOR_CONTRIB_AGGREGATE_INL_H_
#define MXNET_OPERATOR_CONTRIB_AGGREGATE_INL_H_

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

static inline bool IsWriting(const OpReqType ort) {
  return ort == kWriteTo || ort == kWriteInplace;
}

template<typename xpu, typename DType, typename AccReal>
void AggregateUpdateOutput(mshadow::Stream<cpu> *s,
                          const std::vector<TBlob> &input,
                          const std::vector<TBlob> &output);

template<typename xpu, typename DType, typename AccReal>
void AggregateUpdateGradInput(mshadow::Stream<cpu> *s,
                             const std::vector<TBlob> &input,
                             const std::vector<TBlob> &output);

#if MXNET_USE_CUDA
template<typename xpu, typename DType, typename AccReal>
void AggregateUpdateOutput(mshadow::Stream<gpu> *s,
                          const std::vector<TBlob> &input,
                          const std::vector<TBlob> &output);

template<typename xpu, typename DType, typename AccReal>
void AggregateUpdateGradInput(mshadow::Stream<gpu> *s,
                             const std::vector<TBlob> &input,
                             const std::vector<TBlob> &output);
#endif  // MXNET_USE_CUDA

template <typename xpu>
inline void AggregateOpForward(const nnvm::NodeAttrs& attrs,
                                      const OpContext &ctx,
                                      const std::vector<TBlob> &inputs,
                                      const std::vector<OpReqType> &req,
                                      const std::vector<TBlob> &outputs) {
  CHECK_EQ(inputs.size(), 3U);
  CHECK_EQ(outputs.size(), 1U);
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_REAL_TYPE_SWITCH_EX(inputs[0].type_flag_, DType, AccReal, {
    AggregateUpdateOutput<xpu, DType, AccReal>(s, inputs, outputs);
  });
}


template <typename xpu>
inline void AggregateOpBackward(const nnvm::NodeAttrs& attrs,
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
    AggregateUpdateGradInput<xpu, DType, AccReal>(s, inputs, outputs);
  });
}

static bool AggregateOpInferShape(const nnvm::NodeAttrs& attrs,
                                       std::vector<TShape> *in_shape,
                                       std::vector<TShape> *out_shape) {
  using namespace mshadow;
  // A (b,n,k), X (k,d), C 
  CHECK_EQ(in_shape->size(), 3U) << "Input:[data]";
  CHECK_EQ(out_shape->size(), 1U) << "Output:[data]";
  TShape xshape(in_shape->at(0));
  TShape cshape(in_shape->at(1));
  TShape sshape(in_shape->at(2));
  if (xshape.ndim() == 0) return false;
  if (cshape[0] != sshape[0]) return false;
  xshape[2] = cshape[0];
  out_shape->clear();
  out_shape->push_back(xshape);
  return true;
}

static bool AggregateOpInferType(const nnvm::NodeAttrs& attrs,
                                      std::vector<int> *in_type,
                                      std::vector<int> *out_type) {
  using namespace mshadow;
  CHECK_EQ(in_type->size(), 3U);
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

static inline bool AggregateOpStorageType(const nnvm::NodeAttrs &attrs,
                                               const int dev_mask,
                                               DispatchMode *dispatch_mode,
                                               std::vector<int> *in_attrs,
                                               std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 3);
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


template<typename DType, int Dim>
struct DeviceTensor {
 public:
  MSHADOW_XINLINE DeviceTensor(DType *p, const int *size)
    : dptr_(p) {
    for (int i = 0; i < Dim; ++i) {
      size_[i] = size ? size[i] : 0;
    }
  }

  MSHADOW_XINLINE unsigned getSize(const int i) const {
    assert(i < Dim);
    return size_[i];
  }

  MSHADOW_XINLINE int numElements() const {
    int n = 1;
    for (int i = 0; i < Dim; ++i) {
      n *= size_[i];
    }
    return n;
  }

  MSHADOW_XINLINE DeviceTensor<DType, Dim-1> select(const size_t x) const {
    assert(Dim > 1);
    int offset = x;
    for (int i = 1; i < Dim; ++i) {
      offset *= size_[i];
    }
    DeviceTensor<DType, Dim-1> tensor(dptr_ + offset, nullptr);
    for (int i = 0; i < Dim - 1; ++i) {
      tensor.size_[i] = this->size_[i+1];
    }
    return tensor;
  }

  MSHADOW_XINLINE DeviceTensor<DType, Dim-1> operator[](const size_t x) const {
    assert(Dim > 1);
    int offset = x;
    for (int i = 1; i < Dim; ++i) {
      offset *= size_[i];
    }
    DeviceTensor<DType, Dim-1> tensor(dptr_ + offset, nullptr);
    for (int i = 0; i < Dim - 1; ++i) {
      tensor.size_[i] = this->size_[i+1];
    }
    return tensor;
  }

  MSHADOW_XINLINE size_t InnerSize() const {
    assert(Dim >= 3);
    size_t sz = 1;
    for (size_t i = 2; i < Dim; ++i) {
      sz *= size_[i];
    }
    return sz;
  }

  MSHADOW_XINLINE size_t ChannelCount() const {
    assert(Dim >= 3);
    return size_[1];
  }

  MSHADOW_XINLINE DType* data_ptr() const {
    return dptr_;
  }

  DType *dptr_;
  int size_[Dim];
};

template<typename DType>
struct DeviceTensor<DType, 1> {
  MSHADOW_XINLINE DeviceTensor(DType *p, const int *size)
    : dptr_(p) {
    size_[0] = size ? size[0] : 0;
  }

  MSHADOW_XINLINE unsigned getSize(const int i) const {
    assert(i == 0);
    return size_[0];
  }

  MSHADOW_XINLINE int numElements() const {
    return size_[0];
  }

  MSHADOW_XINLINE DType &operator[](const size_t x) const {
      return *(dptr_ + x);
  }

  MSHADOW_XINLINE DType* data_ptr() const {
    return dptr_;
  }

  DType *dptr_;
  int size_[1];
};

template<typename DType, int Dim>
static DeviceTensor<DType, Dim> devicetensor(const TBlob &blob) {
  DType *data = blob.dptr<DType>();
  const int inDim = blob.shape_.ndim();
  assert(inDim == Dim);
  DeviceTensor<DType, Dim> tensor(data, nullptr);
  for (int i = 0; i < Dim; ++i) {
    tensor.size_[i] = blob.size(i);
  }
  return tensor;
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_CONTRIB_AGGREGATE_INL_H_
