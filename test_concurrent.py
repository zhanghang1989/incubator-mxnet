import mxnet as mx
import time

class Concurrent(mx.operator.CustomOp):
    def forward(self, is_train, req, in_data, out_data, aux):
        for i in range(5):
            time.sleep(1)
            print(i)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        for i in range(5):
            time.sleep(1)
            print(i)

@mx.operator.register("concurrent")
class ConcurrentProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(ConcurrentProp, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['data']

    def list_outputs(self):
        return ['output']

    def create_operator(self, ctx, shapes, dtypes):
        return Concurrent()

data = mx.nd.ones((2,2))
a = mx.nd.Custom(data, op_type='concurrent')
b = mx.nd.Custom(data, op_type='concurrent')
c = mx.nd.Custom(data, op_type='concurrent')
d = mx.nd.Custom(data, op_type='concurrent')
mx.nd.waitall()
