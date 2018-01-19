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
 * Copyright (c) 2015 by Contributors
 * \file sync_batch_norm.cc
 * \brief cpu sync BN operator
 * \author Hang Zhang
 * Adapted from BatchNormV1
*/
#include "sync_batch_norm-inl.h"
#include <nnvm/op_attr_types.h>

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(SyncBatchNormParam param, int dtype) {
  return new SyncBatchNormOp<cpu>(param);
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *SyncBatchNormProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
    std::vector<int> *in_type) const {
    std::vector<TShape> out_shape, aux_shape;
    std::vector<int> out_type, aux_type;
    CHECK(InferType(in_type, &out_type, &aux_type));
    CHECK(InferShape(in_shape, &out_shape, &aux_shape));
    DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0]);
}

DMLC_REGISTER_PARAMETER(SyncBatchNormParam);

MXNET_REGISTER_OP_PROPERTY(SyncBatchNorm, SyncBatchNormProp)
.describe(R"code(Synchronized Cross-GPU Batch normalization.
TODO FIXME
)code" ADD_FILELINE)
.add_argument("data", "NDArray-or-Symbol", "Input data to batch normalization")
.add_argument("gamma", "NDArray-or-Symbol", "gamma array")
.add_argument("beta", "NDArray-or-Symbol", "beta array")
.add_argument("mean", "NDArray-or-Symbol", "mean array")
.add_argument("std", "NDArray-or-Symbol", "std array")
.add_arguments(SyncBatchNormParam::__FIELDS__());

NNVM_REGISTER_OP(SyncBatchNorm);

}  // namespace op
}  // namespace mxnet
