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
 * \brief Synchronized BatchNorm modified from BatchNormV1
 * \author Hang Zhang
*/
#ifndef MXNET_OPERATOR_CONTRIB_SYNC_BATCH_NORM_INL_H_
#define MXNET_OPERATOR_CONTRIB_SYNC_BATCH_NORM_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <pthread.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "../operator_common.h"
#include "../mshadow_op.h"

namespace mxnet {
namespace op {

namespace syncbatchnorm {
enum BatchNormOpInputs {kData, kGamma, kBeta};
enum BatchNormOpOutputs {kOut, kMean, kVar};
enum BatchNormOpAuxiliary {kMovingMean, kMovingVar};
enum BatchNormBackResource {kTempSpace};
}  // namespace syncbatchnorm

struct SyncBatchNormParam : public dmlc::Parameter<SyncBatchNormParam> {
  float eps;
  float momentum;
  bool fix_gamma;
  bool use_global_stats;
  bool output_mean_var;
  int ndev;
  std::string key;
  DMLC_DECLARE_PARAMETER(SyncBatchNormParam) {
    DMLC_DECLARE_FIELD(eps).set_default(1e-3f)
    .describe("Epsilon to prevent div 0");
    DMLC_DECLARE_FIELD(momentum).set_default(0.9f)
    .describe("Momentum for moving average");
    DMLC_DECLARE_FIELD(fix_gamma).set_default(true)
    .describe("Fix gamma while training");
    DMLC_DECLARE_FIELD(use_global_stats).set_default(false)
    .describe("Whether use global moving statistics instead of local batch-norm. "
              "This will force change batch-norm into a scale shift operator.");
    DMLC_DECLARE_FIELD(output_mean_var).set_default(false)
    .describe("Output All,normal mean and var");
    DMLC_DECLARE_FIELD(ndev).set_default(1)
      .describe("The count of GPU devices");
    DMLC_DECLARE_FIELD(key)
      .set_default("")
      .describe("Hash key for synchronization");
  }
};

#define MAX_GPU_NUM 16

template<class T>
class SharedND {
 private:
  int nDev = 4;
  bool flag[MAX_GPU_NUM];
  T mean;
  bool meanReady = false;
  bool meanInited = false;

 public:
  T data[MAX_GPU_NUM];
  SharedND(int ndev)
    :nDev(ndev) {
      memset(flag, false, MAX_GPU_NUM * sizeof(bool));
  }

  bool Push(T input, int index) {
    if (flag[index] == false) {
      data[index] = input;
      flag[index] = true;
      return true;
    } else {
      return false;
    }
  }

  T Pop(int index) {
    while(!MeanReady());
    flag[index] = false;
    T tmp = mean;
    ResetMean();
    return tmp;    
  }

  bool MeanReady() {
    if (meanReady) {
      return true;
    }
    for (int i = 0; i < nDev; i++) {
      if (!flag[i]) {
        return false;
      }
    }
    for (int i = 1; i < nDev; i++) {
      data[0] += data[i];
    }
    if (!meanInited) {
      mean = mshadow::NewTensor<cpu, real_t>(data[0].shape_, 0.0f);
      meanInited = true;
    }
    mean = data[0] * 1.0f /  nDev;
    meanReady = true;
    return true;
  }

  void ResetMean() {
    for (int i = 0; i < nDev; i++) {
      if (flag[i]) return;
    }
    meanReady = false;
  }
};

template<class T>
class GlobalSharedND {
 public:
  T* Register(const std::string &key, int ndev) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = registry_.find(key);
    if (it != registry_.end()) return it->second;
    T *newT = new T(ndev);
    registry_[key] = newT;
    return newT;
  }
 private:
  std::mutex mutex_;
  std::map<std::string, T*> registry_;
};

template<class T>
class GlobalSharedRank {
 public:
  T* Register(const std::string &key) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = registry_.find(key);
    if (it != registry_.end()) return it->second;
    T *newT = new T(0);
    registry_[key] = newT;
    return newT;
  }
 private:
  std::mutex mutex_;
  std::map<std::string, T*> registry_;
};

class GlobalSharedBarrier {
 public:
  pthread_barrier_t* Register(const std::string &key, int ndev) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = registry_.find(key);
    if (it != registry_.end()) return it->second;
    pthread_barrier_t *newBarrier = new pthread_barrier_t();
    pthread_barrier_init(newBarrier, NULL, ndev);
    registry_[key] = newBarrier;
    return newBarrier;
  }
 private:
  std::mutex mutex_;
  std::map<std::string, pthread_barrier_t*> registry_;
};

static pthread_mutex_t mm = PTHREAD_MUTEX_INITIALIZER;
static GlobalSharedRank<int> globalSharedRank;
static GlobalSharedBarrier globalSharedBarrier;
static GlobalSharedND<SharedND<mshadow::Tensor<cpu, 1, real_t>>> globalSharedMean;
static GlobalSharedND<SharedND<mshadow::Tensor<cpu, 1, real_t>>> globalSharedVar;
static GlobalSharedND<SharedND<mshadow::Tensor<cpu, 1, real_t>>> globalSharedGrad;
static GlobalSharedND<SharedND<mshadow::Tensor<cpu, 1, real_t>>> globalSharedProd;

template<typename xpu>
class SyncBatchNorm : public Operator {
 public:
  explicit SyncBatchNorm(SyncBatchNormParam param) {
    this->param_ = param;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 3U);
    CHECK_EQ(aux_states.size(), 2U);
    if (ctx.is_train) {
      CHECK_EQ(out_data.size(), 3U);
      CHECK_EQ(req.size(), 3U);
    } else {
      CHECK_GE(out_data.size(), 1U);
      CHECK_GE(req.size(), 1U);
      CHECK_EQ(req[syncbatchnorm::kOut], kWriteTo);
    }

    Stream<xpu> *s = ctx.get_stream<xpu>();
    const real_t scale = static_cast<real_t>(in_data[syncbatchnorm::kData].shape_[1]) /
      static_cast<real_t>(in_data[syncbatchnorm::kData].shape_.Size());
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
    Tensor<xpu, 1> slope = in_data[syncbatchnorm::kGamma].get<xpu, 1, real_t>(s);
    Tensor<xpu, 1> bias = in_data[syncbatchnorm::kBeta].get<xpu, 1, real_t>(s);
    Tensor<xpu, 1> moving_mean = aux_states[syncbatchnorm::kMovingMean].get<xpu, 1, real_t>(s);
    Tensor<xpu, 1> moving_var = aux_states[syncbatchnorm::kMovingVar].get<xpu, 1, real_t>(s);

    if (param_.fix_gamma) slope = 1.f;

    // whether use global statistics
    if (ctx.is_train && !param_.use_global_stats) {
      // get my rank
      pthread_barrier_t *globalBarrier = globalSharedBarrier.Register(param_.key, param_.ndev);
      int *globalRank = globalSharedRank.Register(param_.key);
      pthread_mutex_lock(&mm);
      int myRank = *globalRank;
      *globalRank += 1;
      pthread_mutex_unlock(&mm);
      // get the mean and var
      Tensor<xpu, 1> mean = out_data[syncbatchnorm::kMean].get<xpu, 1, real_t>(s);
      Tensor<xpu, 1> var = out_data[syncbatchnorm::kVar].get<xpu, 1, real_t>(s);
      CHECK(req[syncbatchnorm::kMean] == kNullOp || req[syncbatchnorm::kMean] == kWriteTo);
      CHECK(req[syncbatchnorm::kVar] == kNullOp || req[syncbatchnorm::kVar] == kWriteTo);
      // E(x) and E(x^2)
      mean = scale * sumall_except_dim<1>(data);
      var = scale * sumall_except_dim<1>(F<mshadow_op::square>(data));
      SharedND<mshadow::Tensor<cpu, 1, real_t>> *sharedMean =
        globalSharedMean.Register(param_.key, param_.ndev);
      SharedND<mshadow::Tensor<cpu, 1, real_t>> *sharedVar =
        globalSharedVar.Register(param_.key, param_.ndev);
      // copy to cpu
      Tensor<cpu, 1, real_t> mean_cpu = NewTensor<cpu, real_t>(mean.shape_, 0.0f);
      mshadow::Copy(mean_cpu, mean, s);
      Tensor<cpu,1,real_t> var_cpu = NewTensor<cpu, real_t>(var.shape_, 0.0f);
      mshadow::Copy(var_cpu,var,s);
      // push and pull
      sharedMean->Push(mean_cpu, myRank);
      sharedVar->Push(var_cpu, myRank);
      pthread_barrier_wait(globalBarrier);
      *globalRank = 0;
      pthread_mutex_lock(&mm);
      mean_cpu = sharedMean->Pop(myRank);
      var_cpu = sharedVar->Pop(myRank);
      pthread_mutex_unlock(&mm);
      // copy back to gpu
      mshadow::Copy(mean, mean_cpu, s);
      mshadow::Copy(var, var_cpu, s);

      var = var-F<mshadow_op::square>(mean);
      Assign(out, req[syncbatchnorm::kOut], broadcast<1>(slope, out.shape_) *
             (data - broadcast<1>(mean, data.shape_)) /
             F<mshadow_op::square_root>(broadcast<1>(var + param_.eps, data.shape_)) +
             broadcast<1>(bias, out.shape_));
    } else {
      Assign(out, req[syncbatchnorm::kOut], broadcast<1>(slope /
                                          F<mshadow_op::square_root>(moving_var + param_.eps),
                                          data.shape_) * data +
             broadcast<1>(bias - (slope * moving_mean) /
                          F<mshadow_op::square_root>(moving_var + param_.eps), data.shape_));
    }
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
    CHECK_EQ(out_grad.size(), param_.output_mean_var ? 3U : 1U);
    CHECK_EQ(in_data.size(), 3U);
    CHECK_EQ(out_data.size(), 3U);
    CHECK_EQ(in_grad.size(), 3U);

    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4> data, grad, grad_in;
    const real_t scale = static_cast<real_t>(out_grad[syncbatchnorm::kOut].shape_[1]) /
      static_cast<real_t>(out_grad[syncbatchnorm::kOut].shape_.Size());
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
    Tensor<xpu, 1> var = out_data[syncbatchnorm::kVar].get<xpu, 1, real_t>(s);
    Tensor<xpu, 1> slope = in_data[syncbatchnorm::kGamma].get<xpu, 1, real_t>(s);
    // Tensor<xpu, 1> bias = in_data[kBeta].get<xpu, 1, real_t>(s);
    Tensor<xpu, 1> gslope = in_grad[syncbatchnorm::kGamma].get<xpu, 1, real_t>(s);
    Tensor<xpu, 1> gbias = in_grad[syncbatchnorm::kBeta].get<xpu, 1, real_t>(s);
    // update moving avg
    Tensor<xpu, 1> moving_mean = aux_states[syncbatchnorm::kMovingMean].get<xpu, 1, real_t>(s);
    Tensor<xpu, 1> moving_var = aux_states[syncbatchnorm::kMovingVar].get<xpu, 1, real_t>(s);

    if (param_.fix_gamma) slope = 1.f;

    if (ctx.is_train && !param_.use_global_stats) {
      // get my rank
      pthread_barrier_t *globalBarrier = globalSharedBarrier.Register(param_.key, param_.ndev);
      int *globalRank = globalSharedRank.Register(param_.key);
      pthread_mutex_lock(&mm);
      int myRank = *globalRank;
      *globalRank += 1;
      pthread_mutex_unlock(&mm);
      // get requested temp space
      Tensor<xpu, 2> workspace = ctx.requested[syncbatchnorm::kTempSpace].get_space<xpu>(
          mshadow::Shape2(5, mean.shape_[0]), s);
      Tensor<xpu, 1> gmean = workspace[0];
      Tensor<xpu, 1> gvar = workspace[1];
      // Tensor<xpu, 1> tmp = workspace[2];

      moving_mean = moving_mean * param_.momentum + mean * (1 - param_.momentum);
      moving_var = moving_var * param_.momentum + var * (1 - param_.momentum);
      // cal
      Tensor<xpu, 1> sumGrad = workspace[3];
      Tensor<xpu, 1> sumProd = workspace[4];
      sumGrad = sumall_except_dim<1>(grad);
      sumProd = sumall_except_dim<1>(grad * data);

      SharedND<mshadow::Tensor<cpu, 1, real_t>> *sharedGrad=
        globalSharedGrad.Register(param_.key, param_.ndev);
      SharedND<mshadow::Tensor<cpu, 1, real_t>> *sharedProd =
        globalSharedProd.Register(param_.key, param_.ndev);

      Tensor<cpu, 1, real_t> grad_cpu = NewTensor<cpu, real_t>(sumGrad.shape_, 0.0f);
      mshadow::Copy(grad_cpu, sumGrad, s);
      Tensor<cpu,1,real_t> prod_cpu = NewTensor<cpu, real_t>(sumProd.shape_, 0.0f);
      mshadow::Copy(prod_cpu, sumProd, s);
      // push and pull
      sharedGrad->Push(grad_cpu, myRank);
      sharedProd->Push(prod_cpu, myRank);
      pthread_barrier_wait(globalBarrier);
      *globalRank = 0;
      pthread_mutex_lock(&mm);
      grad_cpu = sharedGrad->Pop(myRank);
      prod_cpu = sharedProd->Pop(myRank);
      pthread_mutex_unlock(&mm);
      // copy back to gpu
      mshadow::Copy(sumGrad, grad_cpu, s);
      mshadow::Copy(sumProd, prod_cpu, s);

      gvar = (sumProd - sumGrad * mean) * slope * (-0.5f) *
        F<mshadow_op::power>(var + param_.eps, -1.5f);
      gmean =  sumGrad * slope;
      gmean *= -1.0f / F<mshadow_op::square_root>(var + param_.eps);
      // assign
      if (!param_.fix_gamma) {
        Assign(gslope, req[syncbatchnorm::kGamma],
               sumall_except_dim<1>(
                   grad * (data - broadcast<1>(mean, data.shape_)) /
                   F<mshadow_op::square_root>(broadcast<1>(var + param_.eps, data.shape_))));
      } else {
        Assign(gslope, req[syncbatchnorm::kGamma], 0.0f);
      }
      Assign(grad_in, req[syncbatchnorm::kData],
             (grad * broadcast<1>(slope, data.shape_)) *
             broadcast<1>(1.0f / F<mshadow_op::square_root>(var + param_.eps), data.shape_) +
             broadcast<1>(gvar, data.shape_) * //(1 - scale / param_.ndev) *
              scale * 2.0f * (data - broadcast<1>(mean, data.shape_)) +
             broadcast<1>(gmean, data.shape_) * scale);
      Assign(gbias, req[syncbatchnorm::kBeta], sumall_except_dim<1>(grad));
    } else {
      // use global statistics with freeze moving mean and var.
      if (!param_.fix_gamma) {
        Assign(gslope, req[syncbatchnorm::kGamma],
               sumall_except_dim<1>(
                 grad * (data - broadcast<1>(moving_mean, data.shape_)) /
                 F<mshadow_op::square_root>(broadcast<1>(moving_var + param_.eps, data.shape_))));
      } else {
        Assign(gslope, req[syncbatchnorm::kGamma], 0.0f);
      }
      Assign(gbias, req[syncbatchnorm::kBeta], sumall_except_dim<1>(grad));
      Assign(grad_in, req[syncbatchnorm::kData], (grad * broadcast<1>(slope, data.shape_)) *
             broadcast<1>(
               1.0f / F<mshadow_op::square_root>(moving_var + param_.eps), data.shape_));
    }
  }

 private:
  SyncBatchNormParam param_;
};  // class SyncBatchNorm

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
    CHECK_EQ(in_shape->size(), 3U) << "Input:[data, gamma, beta]";
    const TShape &dshape = in_shape->at(0);
    if (dshape.ndim() == 0) return false;
    in_shape->at(1) = TShape(Shape1(dshape[1]));
    in_shape->at(2) = TShape(Shape1(dshape[1]));
    out_shape->clear();
    out_shape->push_back(dshape);
    out_shape->push_back(Shape1(dshape[1]));
    out_shape->push_back(Shape1(dshape[1]));

    aux_shape->clear();
    aux_shape->push_back(Shape1(dshape[1]));
    aux_shape->push_back(Shape1(dshape[1]));
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
    for (index_t i = 0; i < aux_type->size(); ++i) {
      if ((*aux_type)[i] != -1) {
        UNIFORM_TYPE_CHECK((*aux_type)[i], dtype_param, ListArguments()[i]);
      }
    }
    int n_aux = this->ListAuxiliaryStates().size();
    aux_type->clear();
    for (int i = 0; i < n_aux; ++i ) aux_type->push_back(dtype_param);
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
            out_data[syncbatchnorm::kMean],
            out_data[syncbatchnorm::kVar],
            in_data[syncbatchnorm::kData],
            in_data[syncbatchnorm::kGamma]
           };
  }

  std::vector<ResourceRequest> BackwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  int NumVisibleOutputs() const override {
    if (param_.output_mean_var) {
      return 3;
    }
    return 1;
  }

  int NumOutputs() const override {
    return 3;
  }

  std::vector<std::string> ListArguments() const override {
    return {"data", "gamma", "beta"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output", "mean", "var"};
  }

  std::vector<std::string> ListAuxiliaryStates() const override {
    return {"moving_mean", "moving_var"};
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
#endif  // MXNET_OPERATOR_CONTRIB_SYNC_BATCH_NORM_INL_H_
