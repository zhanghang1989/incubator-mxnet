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
from ... import ndarray as F
from ... import kv
from ... import autograd
from ...ndarray import NDArray
from ..utils import split_and_load
from ..nn import BatchNorm
from ... import test_utils
from ...operator import CustomOp, CustomOpProp, register

__all__ = ['DataParallelModel', 'DataParallelLoss', 'SyncBatchNorm', 'parallel_backward']


class SharedTensor(object):
    def __init__(self, key, nchannels, nGPUs):
        self.mutex = threading.Lock()
        self.all_tasks_done = threading.Condition(self.mutex)
        self._key = key
        self.nGPUs = int(nGPUs)
        self.out = F.zeros(nchannels)
        self._clear()

    def _clear(self):
        self.list = []
        self.push_tasks = self.nGPUs
        self.reduce_tasks = self.nGPUs

    def push(self, t):
        """push to _SharedTensor"""
        with self.mutex:
            if self.push_tasks == 0:
                self._clear()
            #t.wait_to_read()
            self.list.append(t)
            self.push_tasks -= 1
        print('self.push_tasks', self.push_tasks)
        with self.all_tasks_done:
            if self.push_tasks == 0:
                self.all_tasks_done.notify_all()
            while self.push_tasks:
                self.all_tasks_done.wait()

    def _reduce(self, kv):
        with self.mutex:
            if self.reduce_tasks == 1:
                assert(len(self.list) == self.nGPUs)
                kv.push(self._key, self.list)
                self.reduce_tasks -= 1
            else:
                self.reduce_tasks -= 1
        with self.all_tasks_done:
            if self.reduce_tasks == 0:
                self.all_tasks_done.notify_all()
            while self.reduce_tasks:
                self.all_tasks_done.wait()

    def pull(self, kv):
        """Get form _SharedTensor"""
        self._reduce(kv)
        kv.pull(self._key, out=self.out)
        return self.out

    def __len__(self):
        return len(self.list)


class SharedTDict(object):
    def __init__(self):
        self.stdict = {}
        self.keys = []
        self.mutex = threading.Lock()
        self.kv = kv.create('local')

    def register(self, key, nchannels, nGPUs):
        with self.mutex:
            if key in self.keys:
                return
            print('registerring {}'.format(key))
            self.stdict[key] = SharedTensor(key, nchannels, nGPUs)
            self.kv.init(key, F.zeros(nchannels))
            self.keys.append(key)

    def push(self, key, value):
        print('pushing {}'.format(key))
        self.stdict[key].push(value)

    def pull(self, key):
        print('pulling {}'.format(key))
        out = self.stdict[key].pull(self.kv)
        return out

sharedTensorDict = SharedTDict()

class AllReduce(autograd.Function):
    def __init__(self, key):
        super(AllReduce, self).__init__()
        self.xsumkey = key + 'sum'
        self.xsqukey = key + 'squ'

    def forward(self, isum, isqu):
        print('SyncFunc forwarding')
        # TODO allreduce here
        sharedTensorDict.push(self.xsumkey, isum)
        sharedTensorDict.push(self.xsqukey, isqu)
        osum = sharedTensorDict.pull(self.xsumkey).as_in_context(isum.context)
        osqu = sharedTensorDict.pull(self.xsqukey).as_in_context(isqu.context)
        return osum, osqu

    def backward(self, dsum, dsqu):
        # TODO FIXME
        print('SyncFunc backwarding')
        sharedTensorDict.push(self.xsumkey, dsum)
        sharedTensorDict.push(self.xsqukey, dsqu)
        disum = sharedTensorDict.pull(self.xsumkey).as_in_context(dsum.context)
        disqu = sharedTensorDict.pull(self.xsqukey).as_in_context(dsqu.context)
        return disum, disqu


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
    def __init__(self, in_channels, axis=1, momentum=0.9, epsilon=1e-5, ndevices=None, **kwargs):
        super(SyncBatchNorm, self).__init__(axis, momentum, epsilon, in_channels=in_channels, **kwargs)

        self.eps = epsilon
        self.momentum = momentum
        self.in_channels = in_channels
        self.ndevices = self._get_nGPUs() if ndevices is None else ndevices
        self.updater = _SharedUpdater(ndevices)
        sharedTensorDict.register(self._prefix + 'sum', in_channels, ndevices)
        sharedTensorDict.register(self._prefix + 'squ', in_channels, ndevices)

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
        #isum = x.sum(axis=1, exclude=True)
        #isqu = (x**2).sum(axis=1, exclude=True)
        N = self.ndevices * x.shape[0] * x.shape[2] * x.shape[3]
        allreduce = AllReduce(self._prefix)
        osum, osqu = allreduce(isum, isqu)
        # calc mean and std
        mean = osum / N
        sumvar = osqu - osum * osum / N
        bias_var = sumvar / N
        std = F.sqrt(F.maximum(bias_var, self.eps))
        # update running mean and var
        with autograd.pause():
            unbias_var = sumvar / (N - 1)
            self.updater(self.running_mean, self.running_var, mean, unbias_var,
                         self.momentum, x.context)
        # update running mean and var
        output = F.DecoupleBatchNorm(x, gamma, beta, mean, std)
        return output
        """
        isum, isqu = F.SumSquare(x)
        #isum = x.sum(axis=1, exclude=True)
        #isqu = (x**2).sum(axis=1, exclude=True)
        # reduce sum for E(x) and E(x^2)
        id1 = self.xsum.push(isum)
        id2 = self.xsqu.push(isqu)
        osum = self.xsum.get(id1)
        osqu = self.xsqu.get(id2)
        assert len(self.xsum) == len(self.xsqu)
        N = len(self.xsum)*x.shape[0]*x.shape[2]*x.shape[3]
        # calc mean and std
        mean = osum / N
        sumvar = osqu - osum * osum / N
        bias_var = sumvar / N
        std = F.sqrt(F.clip(bias_var, self.eps))
        # update running mean and var
        with autograd.pause():
            unbias_var = sumvar / (N - 1)
            ctx = x.context
            self.updater(self.running_mean, self.running_var, mean, unbias_var,
                         self.momentum, ctx)
        return F.DecoupleBatchNorm(x, gamma, beta, mean, std)
        #output = (x - mean.reshape(1, -1, 1, 1)) / std.reshape(1, -1, 1, 1) * \
        #    gamma.reshape(1, -1, 1, 1) + beta.reshape(1, -1, 1, 1)
        #return output
        """


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

#losses= [(loss1,),(loss2,)]
def parallel_backward(losses, sync=True):
    lock = threading.Lock()
    def _worker(loss):
        print('parallel backwarding')
        loss[0].backward()

    threads = [threading.Thread(target=_worker, args=(loss,)) for loss in losses]
    if sync:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    else:
        for loss in losses:
            loss.backward()


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
