# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# pylint: disable=broad-except, redefined-builtin
"""Synchronized DataParallel"""
import threading
from ... import autograd
from ...ndarray import AllReduce, NDArray
from ..utils import split_and_load
from ..nn import BatchNorm
from ... import test_utils

__all__ = ['DataParallelModel', 'DataParallelLoss', 'SyncBatchNorm', 'Barrier']


class DataParallelModel(object):
    """Data Parallelism

    Hide the difference of single/multiple GPUs to the user.
    This container parallelizes the application of the given module by
    splitting the input across the specified devices by chunking in the
    batch dimension.
    In the forward pass, the module is replicated on each device,
    and each replica handles a portion of the input. During the backwards pass,
    gradients from each replica are summed into the original module.
    Note that the outputs are not gathered, we recommand using compatible
    :class:`mxnet.gluon.contrib.DataParallelLoss`.

    The batch size should be larger than the number of GPUs used. It should
    also be an integer multiple of the number of GPUs so that each chunk is
    the same size (so that each GPU processes the same number of samples).

    Parameters
    ----------
    module : object
        Network to be parallelized.
    ctx_list : list
        A list of contexts
    sync : bool
        enable synchronization (default: False).


    Inputs:

        - **inputs**: list of input (NDArrays)

    Outputs:

        - **outputs**: list of output (NDArrays)

    Example::

        >>> ctx = [mx.gpu(0), mx.gpu(1)]
        >>> net = DataParallelModel(model, ctx=ctx)
        >>> criterion = DataParallelLoss(criterion)
        >>> with autograd.record()
        >>>     y = net(x)
        >>>     loss = criterion(y, t)
        >>>     autograd.backward(loss)
    """
    def __init__(self, module, ctx_list=None, sync=False):
        module.collect_params().reset_ctx(ctx=ctx_list)
        self.ctx_list = ctx_list
        self.module = module
        self.sync = sync

    def __call__(self, *inputs, **kwargs):
        if not self.ctx_list:
            return self.module(*inputs, **kwargs)
        inputs, kwargs = _split_load_kwargs(inputs, kwargs, self.ctx_list)
        assert(len(inputs) == len(self.ctx_list))
        if len(self.ctx_list) == 1:
            return (tuple_map(self.module(*inputs[0], **kwargs[0])),)
        return parallel_apply(self.module, inputs, kwargs, self.sync)

    def __repr__(self):
        return 'DataParallel:\n module = {' + self.module.__repr__() + '}'


class DataParallelLoss(object):
    """Data Parallelism

    Hide the difference of single/multiple GPUs to the user.
    The targets are splitted across the specified devices by chunking in
    the batch dimension.
    In the forward pass, the module is replicated on each device,
    and each replica handles a portion of the input. During the backwards pass,
    gradients from each replica are summed into the original module.
    Note that the outputs are not gathered, please use compatible
    :class:`mxnet.gluon.contrib.DataParallelModel`

    Parameters
    ----------
    module : object
        Network to be parallelized.
    ctx_list : list
        A list of contexts to use.
    sync : bool
        enable synchronization (default: False).


    Inputs:

        - **inputs**: list of inputs (NDArrays)
        - **targets**: list of labels (NDArrays)

    Outputs:

        - **outputs**: list of output (NDArrays)

    Example::

        >>> ctx = [mx.gpu(0), mx.gpu(1)]
        >>> net = DataParallelModel(model, ctx=ctx)
        >>> criterion = DataParallelLoss(criterion)
        >>> with autograd.record()
        >>>     y = net(x)
        >>>     loss = criterion(y, t)
        >>>     autograd.backward(loss)
    """
    def __init__(self, module, ctx_list=None, sync=False):
        self.module = module
        self.ctx_list = ctx_list
        self.sync = sync

    def __call__(self, inputs, *targets, **kwargs):
        # the inputs should be the outputs of DataParallelModel
        if not self.ctx_list:
            return self.module(inputs, *targets, **kwargs)
        targets, kwargs = _split_load_kwargs(targets, kwargs, self.ctx_list)
        assert(len(targets) == len(self.ctx_list))
        if len(self.ctx_list) == 1:
            return tuple_map(self.module(*(inputs[0] + targets[0]), **kwargs[0]))
        assert(len(inputs) == len(self.ctx_list))
        return loss_parallel_apply(self.module, inputs, targets, kwargs, self.sync)


class Barrier(object):
    """Barrier for cross-device operation.

    A cross device operation that allows synchronized push and pull. It can be used in
    Cross-gpu Sycnhronized Batch Normalization.

    Parameters
    ----------
    counter : int
        Number of deivces.
    operation : callable
        The cross device operation is applying (e.g. AllReduce).
    """
    def __init__(self, counter, operation):
        self._mutex = threading.Lock()
        self.all_tasks_done = threading.Condition(self._mutex)
        self.counter = counter
        self.op = operation
        self._clear()

    def __call__(self, x):
        """Push a NDArray from one of the device.
        Input:
            x (NDArray)

        Output:
            idx (int), the output index
        """
        with self._mutex:
            if self.push_tasks == 0:
                self._clear()
            self.list.append(x)
            idx = len(self.list) - 1
            self.push_tasks -= 1

        with self.all_tasks_done:
            if self.push_tasks == 0:
                self.all_tasks_done.notify_all()
            while self.push_tasks:
                self.all_tasks_done.wait()

        self._sync_op()
        return self.out[idx]

    def _sync_op(self):
        with self._mutex:
            if self.reduce_tasks == 1:
                assert(len(self.list) == self.counter)
                self.out = self.op(*self.list)
                if isinstance(self.out, (list, tuple)):
                    for xi in self.out:
                        xi.wait_to_read()
                else:
                    self.out.wait_to_read()
                self.reduce_tasks -= 1
            else:
                self.reduce_tasks -= 1

        with self.all_tasks_done:
            if self.reduce_tasks == 0:
                self.all_tasks_done.notify_all()
            while self.reduce_tasks:
                self.all_tasks_done.wait()

    def _clear(self):
        self.list = []
        self.push_tasks = self.counter
        self.reduce_tasks = self.counter

    def __len__(self):
        return len(self.list)

    def __repr__(self):
        return 'Barrier'


class SyncBatchNorm(BatchNorm):
    """Cross-GPU Synchronized Batch normalization (SyncBN)
    Standard BN [1]_ implementation only normalize the data within each device.
    SyncBN normalizes the input within the whole mini-batch.
    We follow the sync-onece implmentation described in the paper [2]_ .

    Parameters
    ----------
    axis : int, default 1
        The axis that should be normalized. This is typically the channels
        (C) axis. For instance, after a `Conv2D` layer with `layout='NCHW'`,
        set `axis=1` in `BatchNorm`. If `layout='NHWC'`, then set `axis=3`.
    momentum: float, default 0.9
        Momentum for the moving average.
    epsilon: float, default 1e-5
        Small float added to variance to avoid dividing by zero.
    center: bool, default True
        If True, add offset of `beta` to normalized tensor.
        If False, `beta` is ignored.
    scale: bool, default True
        If True, multiply by `gamma`. If False, `gamma` is not used.
        When the next layer is linear (also e.g. `nn.relu`),
        this can be disabled since the scaling
        will be done by the next layer.
    use_global_stats: bool, default False
        If True, use global moving statistics instead of local batch-norm. This will force
        change batch-norm into a scale shift operator.
        If False, use local batch-norm.
    beta_initializer: str or `Initializer`, default 'zeros'
        Initializer for the beta weight.
    gamma_initializer: str or `Initializer`, default 'ones'
        Initializer for the gamma weight.
    moving_mean_initializer: str or `Initializer`, default 'zeros'
        Initializer for the moving mean.
    moving_variance_initializer: str or `Initializer`, default 'ones'
        Initializer for the moving variance.
    in_channels : int, default 0
        Number of channels (feature maps) in input data. If not specified,
        initialization will be deferred to the first time `forward` is called
        and `in_channels` will be inferred from the shape of input data.
    nGPUs : int, default number of visible GPUs
    Inputs:
        - **data**: input tensor with arbitrary shape.
    Outputs:
        - **out**: output tensor with the same shape as `data`.


    Reference:

        .. [1] Ioffe, Sergey, and Christian Szegedy. "Batch normalization: Accelerating
        deep network training by reducing internal covariate shift." *ICML 2015*

        .. [2] Hang Zhang, Kristin Dana, Jianping Shi, Zhongyue Zhang, Xiaogang Wang,
        Ambrish Tyagi, and Amit Agrawal. "Context Encoding for Semantic Segmentation." *CVPR 2018*
    """
    # pylint: disable=arguments-differ
    def __init__(self, axis=1, momentum=0.9, epsilon=1e-5, nGPUs=None, **kwargs):
        super(SyncBatchNorm, self).__init__(axis, momentum, epsilon, **kwargs)

        self.eps = epsilon
        self.momentum = momentum

        if nGPUs is None:
            nGPUs = self._get_nGPUs()
        self.xsum = Barrier(nGPUs, AllReduce)
        self.xsqu = Barrier(nGPUs, AllReduce)
        self.updater = _SharedUpdater(nGPUs)

    def _get_nGPUs(self):
        # caution: if not using all the GPUs, please mannually set nGPUs
        nGPUs = len(test_utils.list_gpus())
        # for CPU
        nGPUs = nGPUs if nGPUs > 0 else 1
        return nGPUs

    def hybrid_forward(self, F, x, gamma, beta, running_mean, running_var):
        """Hybrid forward"""
        if not autograd.is_training():
            return F.BatchNorm(x, gamma, beta, running_mean, running_var, name='fwd',
                               **self._kwargs)
        isum, isqu = F.SumSquare(x)
        # reduce sum for E(x) and E(x^2)
        osum = self.xsum(isum)
        osqu = self.xsqu(isqu)
        assert len(self.xsum) == len(self.xsqu)
        N = len(self.xsum)*x.shape[0]*x.shape[2]*x.shape[3]
        # calc mean and std
        mean = osum / N
        sumvar = osqu - osum * osum / N
        bias_var = sumvar / N
        std = F.sqrt(F.clip(bias_var, a_min=self.eps, a_max=bias_var.max().asscalar()))
        # update running mean and var
        with autograd.pause():
            unbias_var = sumvar / (N - 1)
            ctx = x.context
            self.updater(self.running_mean, self.running_var, mean, unbias_var,
                         self.momentum, ctx)
        return F.DecoupleBatchNorm(x, gamma, beta, mean, std)


class _SharedUpdater(object):
    # update only once
    def __init__(self, nGPUs):
        self.mutex = threading.Lock()
        self.nGPUs = nGPUs
        self._clear()

    def _clear(self):
        self.tasks = self.nGPUs

    def __call__(self, running_mean, running_var, mean, unbias_var, momentum, ctx):
        with self.mutex:
            if self.tasks == self.nGPUs:
                running_mean.set_data(momentum * running_mean.data(ctx) + \
                    (1.0 - momentum) * mean)
                running_var.set_data(momentum * running_var.data(ctx) + \
                    (1.0 - momentum) * unbias_var)
            self.tasks -= 1
        if self.tasks == 0:
            self._clear()


def _split_load_kwargs(inputs, kwargs, ctx_list, batch_axis=0):
    r"""Split with support for kwargs dictionary"""
    def split_map(obj):
        if isinstance(obj, NDArray):
            return split_and_load(obj, ctx_list, batch_axis, even_split=False)
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(split_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            return list(map(list, zip(*map(split_map, obj))))
        if isinstance(obj, dict) and len(obj) > 0:
            return list(map(type(obj), zip(*map(split_map, obj.items()))))
        return [obj for targets in ctx_list]
    inputs = split_map(inputs) if inputs else []
    kwargs = split_map(kwargs) if kwargs else []
    if len(inputs) < len(kwargs):
        inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
    elif len(kwargs) < len(inputs):
        kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
    inputs = tuple(inputs)
    kwargs = tuple(kwargs)
    return inputs, kwargs


def tuple_map(obj):
    if isinstance(obj, NDArray):
        return (obj,)
    if isinstance(obj, list) and len(obj) > 0:
        return tuple(obj)
    return obj


def parallel_apply(module, inputs, kwargs_tup=None, sync=False):
    """Parallel applying model forward"""
    if kwargs_tup is not None:
        assert len(inputs) == len(kwargs_tup)
    else:
        kwargs_tup = ({},) * len(inputs)

    lock = threading.Lock()
    results = {}

    def _worker(i, module, input, kwargs, results, is_recording, is_training, lock):
        try:
            if is_recording:
                with autograd.record(is_training):
                    output = tuple_map(module(*input, **kwargs))
                    for out in output:
                        out.wait_to_read()
            else:
                output = tuple_map(module(*input, **kwargs))
                for out in output:
                    out.wait_to_read()
            with lock:
                results[i] = output
        except Exception as e:
            with lock:
                results[i] = e

    is_training = autograd.is_training()
    is_recording = autograd.is_recording()
    threads = [threading.Thread(target=_worker,
                                args=(i, module, input, kwargs, results,
                                      is_recording, is_training, lock),
                               )
               for i, (input, kwargs) in
               enumerate(zip(inputs, kwargs_tup))]

    if sync:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        outputs = []
        for i in range(len(inputs)):
            output = results[i]
            if isinstance(output, Exception):
                raise output
            outputs.append(output)
        return tuple(outputs)
    else:
        outputs = [tuple_map(module(*input, **kwargs))
                   for (input, kwargs) in zip(inputs, kwargs_tup)]
        return tuple(outputs)


def loss_parallel_apply(module, inputs, targets, kwargs_tup=None, sync=False):
    """Data Parallel Criterion"""
    if kwargs_tup:
        assert len(inputs) == len(kwargs_tup)
    else:
        kwargs_tup = ({},) * len(inputs)

    lock = threading.Lock()
    results = {}

    def _worker(i, module, input, target, kwargs, results, is_recording, is_training, lock):
        try:
            if is_recording:
                with autograd.record(is_training):
                    output = module(*(input + target), **kwargs)
                    output.wait_to_read()
            else:
                output = module(*(input + target), **kwargs)
                output.wait_to_read()
            with lock:
                results[i] = output
        except Exception as e:
            with lock:
                results[i] = e

    is_training = bool(autograd.is_training())
    is_recording = autograd.is_recording()

    threads = [threading.Thread(target=_worker,
                                args=(i, module, input, target,
                                      kwargs, results, is_recording, is_training, lock),
                               )
               for i, (input, target, kwargs) in
               enumerate(zip(inputs, targets, kwargs_tup))]

    if sync:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        outputs = []
        for i in range(len(inputs)):
            output = results[i]
            if isinstance(output, Exception):
                raise output
            outputs.append(output)
        return tuple(outputs)
    else:
        outputs = [module(*(input + target), **kwargs) \
            for (input, target, kwargs) in zip(inputs, targets, kwargs_tup)]
        return tuple(outputs)
