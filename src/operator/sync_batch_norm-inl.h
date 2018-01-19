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
 * \file sync_batch_norm-inl.h
 * \brief
 * \author Hang Zhang
 * Adapted from BatchNormV1
 */
#ifndef MXNET_OPERATOR_SYNC_BATCH_NORM_INL_H_
#define MXNET_OPERATOR_SYNC_BATCH_NORM_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "./operator_common.h"
#include "./mshadow_op.h"

namespace mxnet {
namespace op {

namespace syncbatchnorm {
enum SyncBatchNormOpInputs {kData, kGamma, kBeta, kMean, kStd};
enum SyncBatchNormOpOutputs {kOut};
}  // namespace syncbatchnorm

struct SyncBatchNormParam : public dmlc::Parameter<SyncBatchNormParam> {
  float eps;
  float momentum;
  bool fix_gamma;
  DMLC_DECLARE_PARAMETER(SyncBatchNormParam) {
    DMLC_DECLARE_FIELD(eps).set_default(1e-3f)
    .describe("Epsilon to prevent div 0");
    DMLC_DECLARE_FIELD(momentum).set_default(0.9f)
    .describe("Momentum for moving average");
    DMLC_DECLARE_FIELD(fix_gamma).set_default(true)
    .describe("Fix gamma while training");
  }
};

template<typename xpu>
class SyncBatchNormOp : public Operator {
 public:
  explicit SyncBatchNormOp(SyncBatchNormParam param) {
    this->param_ = param;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 5U);
    CHECK_EQ(out_data.size(), 1U);
    CHECK_EQ(req.size(), 1U);
    if (!ctx.is_train) {
      CHECK_EQ(req[syncbatchnorm::kOut], kWriteTo);
    }
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4> data;
    Tensor<xpu, 4> out;
    if (in_data[syncbatchnorm::kData].ndim() == 2) {
      Shape<4> dshape = Shape4(in_data[syncbatchnorm::kData].shape_[0],
                               in_data[syncbatchnorm::kData].shape_[1], 1, 1);
      data = in_data[syncbatchnorm::kData].get_with_shape<xpu, 4, real_t>(dshape, s);
      out = out_data[syncbatchnorm::kOut].get_with_shape<xpu, 4, real_t>(dshape, s);
    } else {
      data = in_data[syncbatchnorm::kData].get<xpu, 4, real_t>(s);
      out = out_data[syncbatchnorm::kOut].get<xpu, 4, real_t>(s);
    }
    Tensor<xpu, 1> gamma = in_data[syncbatchnorm::kGamma].get<xpu, 1, real_t>(s);
    Tensor<xpu, 1> beta = in_data[syncbatchnorm::kBeta].get<xpu, 1, real_t>(s);
    Tensor<xpu, 1> mean = in_data[syncbatchnorm::kMean].get<xpu, 1, real_t>(s);
    Tensor<xpu, 1> std = in_data[syncbatchnorm::kStd].get<xpu, 1, real_t>(s);

    if (param_.fix_gamma) gamma = 1.f;

    Assign(out, req[syncbatchnorm::kOut], broadcast<1>(gamma / std, data.shape_) * data +
           broadcast<1>(beta - (gamma * mean) / std, data.shape_));
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(out_grad.size(), 1U);
    CHECK_EQ(out_data.size(), 1U);
    CHECK_EQ(in_data.size(), 5U);
    CHECK_EQ(in_grad.size(), 5U);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4> data, grad, grad_in;
    if (in_data[syncbatchnorm::kData].ndim() == 2) {
      Shape<4> dshape = Shape4(out_grad[syncbatchnorm::kOut].shape_[0],
                               out_grad[syncbatchnorm::kOut].shape_[1], 1, 1);
      data = in_data[syncbatchnorm::kData].get_with_shape<xpu, 4, real_t>(dshape, s);
      grad = out_grad[syncbatchnorm::kOut].get_with_shape<xpu, 4, real_t>(dshape, s);
      grad_in = in_grad[syncbatchnorm::kData].get_with_shape<xpu, 4, real_t>(dshape, s);
    } else {
      data = in_data[syncbatchnorm::kData].get<xpu, 4, real_t>(s);
      grad = out_grad[syncbatchnorm::kOut].get<xpu, 4, real_t>(s);
      grad_in = in_grad[syncbatchnorm::kData].get<xpu, 4, real_t>(s);
    }

    Tensor<xpu, 1> mean = out_data[syncbatchnorm::kMean].get<xpu, 1, real_t>(s);
    Tensor<xpu, 1> std = out_data[syncbatchnorm::kStd].get<xpu, 1, real_t>(s);
    Tensor<xpu, 1> gamma = in_data[syncbatchnorm::kGamma].get<xpu, 1, real_t>(s);
    Tensor<xpu, 1> ggamma = in_grad[syncbatchnorm::kGamma].get<xpu, 1, real_t>(s);
    Tensor<xpu, 1> gbeta = in_grad[syncbatchnorm::kBeta].get<xpu, 1, real_t>(s);
    Tensor<xpu, 1> gmean = in_grad[syncbatchnorm::kMean].get<xpu, 1, real_t>(s);
    Tensor<xpu, 1> gstd = in_grad[syncbatchnorm::kStd].get<xpu, 1, real_t>(s);
    // update moving avg

    if (param_.fix_gamma) gamma = 1.f;

    // cal
    gstd = sumall_except_dim<1>((grad * broadcast<1>(gamma, data.shape_)) *
                                (data - broadcast<1>(mean, data.shape_)) *
                                -1.f *
                                F<mshadow_op::power>(broadcast<1>(std, data.shape_),
                                                     -2.f));
    gmean = sumall_except_dim<1>(grad * broadcast<1>(gamma, data.shape_) /
                                 broadcast<1>(std, data.shape_));
    // assign
    if (!param_.fix_gamma) {
      Assign(ggamma, req[syncbatchnorm::kGamma],
             sumall_except_dim<1>(
                 grad * (data - broadcast<1>(mean, data.shape_)) /
                 broadcast<1>(std, data.shape_)));
    } else {
      Assign(ggamma, req[syncbatchnorm::kGamma], 0.0f);
    }
    Assign(grad_in, req[syncbatchnorm::kData],
           (grad * broadcast<1>(gamma, data.shape_)) /
           broadcast<1>(std, data.shape_))

    Assign(gbeta, req[syncbatchnorm::kBeta], sumall_except_dim<1>(grad));
  }

 private:
  SyncBatchNormParam param_;
};  // class SyncBatchNormOp

template<typename xpu>
Operator *CreateOp(SyncBatchNormParam param, int dtype);


#if DMLC_USE_CXX11
class SyncBatchNormProp : public OperatorProperty {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 5U) << "Input:[data, gamma, beta, mean, std]";
    const TShape &dshape = in_shape->at(0);
    if (dshape.ndim() == 0) return false;
    in_shape->at(1) = TShape(Shape1(dshape[1]));
    in_shape->at(2) = TShape(Shape1(dshape[1]));
    in_shape->at(3) = TShape(Shape1(dshape[1]));
    in_shape->at(4) = TShape(Shape1(dshape[1]));
    out_shape->clear();
    out_shape->push_back(dshape);

    return true;
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    using namespace mshadow;
    CHECK_GE(in_type->size(), 1U);
    int dtype = (*in_type)[0];
    CHECK_NE(dtype, -1) << "First input must have specified type";
    // For float16 input type beta, gamma, mean, and average are stored in float32.
    // For other input types, these parameters have the same type as input
    // NOTE: This requirement is from cuDNN (v. 4 and 5)
    int dtype_param = (dtype == kFloat16) ? kFloat32 : dtype;
    for (index_t i = 1; i < in_type->size(); ++i) {
      if ((*in_type)[i] == -1) {
        (*in_type)[i] = dtype_param;
      } else {
        UNIFORM_TYPE_CHECK((*in_type)[i], dtype_param, ListArguments()[i]);
      }
    }
    int n_out = this->ListOutputs().size();
    out_type->clear();
    out_type->push_back(dtype);
    for (int i = 1; i < n_out; ++i ) out_type->push_back(dtype_param);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new SyncBatchNormProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "SyncBatchNorm";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[syncbatchnorm::kOut],
            in_data[syncbatchnorm::kData],
            in_data[syncbatchnorm::kGamma],
            in_data[syncbatchnorm::kBeta],
            in_data[syncbatchnorm::kMean],
            in_data[syncbatchnorm::kStd],
           };
  }

  int NumVisibleOutputs() const override {
    return 1;
  }

  int NumOutputs() const override {
    return 1;
  }

  std::vector<std::string> ListArguments() const override {
    return {"data", "gamma", "beta", "mean", "std"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output"};
  }

  Operator* CreateOperator(Context ctx) const override {
      LOG(FATAL) << "Not Implemented.";
      return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
      std::vector<int> *in_type) const override;

  inline const SyncBatchNormParam& getParam() const {
    return param_;
  }

 private:
  SyncBatchNormParam param_;
};  // class SyncBatchNormProp

#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_SYNC_BATCH_NORM_INL_H_
