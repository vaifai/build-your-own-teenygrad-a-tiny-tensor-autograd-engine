"""
Build Your Own teenygrad: A Tiny Tensor Autograd Engine

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - prod
def prod(shape):
    # TODO: Multiply together the elements of a shape tuple to get the total number of elements.
    output = 1
    for i in shape:
        if isinstance(i, int):
            output = output * i
        elif isinstance(i, (list,tuple)):
            output = output*prod(i)
    return output

# Step 2 - argsort
def argsort(values):
    # TODO: Return the indices that would sort values in ascending order.
    args = {}
    for ind, item in enumerate(values):
        args[ind] = item
    sorted_args = dict(sorted(args.items(), key=lambda item: item[1]))
    return list(sorted_args.keys())

# Step 3 - make_op_enums
def make_op_enums():
    # TODO: create four enum classes naming every supported operation kind
    UnaryOps = enum.Enum("UnaryOps", ["NEG", "RELU", "LOG", "EXP", "SQRT", "SIGMOID"])
    BinaryOps = enum.Enum("BinaryOps", ['ADD', 'SUB', 'MUL', 'DIV', 'CMPLT', 'MAX'])
    ReduceOps = enum.Enum("ReduceOps", ['SUM', 'MAX'])
    MovementOps = enum.Enum("MovementOps", ['RESHAPE', 'EXPAND', 'PERMUTE'])

    return (UnaryOps, BinaryOps, ReduceOps, MovementOps)

# Step 4 - LazyBuffer
class LazyBuffer:
    def __init__(self, np_array):
        # TODO: wrap np_array as an ndarray and expose shape and dtype
        self._np = np.asarray(np_array)
        self.shape = tuple(int(d) for d in self._np.shape)
        self.dtype = self._np.dtype
    
    def __array__(self, dtype=None):
        return np.asarray(self._np, dtype = self.dtype)
    
    def __float__(self):
        return float(self._np)

# Step 5 - lazybuffer_const
def const(value, shape):
    # TODO: Create a new LazyBuffer of the given shape filled with a constant value.
    return LazyBuffer(np.full(shape, value, dtype=np.float32))
LazyBuffer.const = staticmethod(const)

# Step 6 - rand
def rand(shape, seed=None):
    # TODO: return a LazyBuffer of uniform random floats in [0, 1) with given shape
    rng = np.random.RandomState(seed)
    data = rng.random(shape).astype(np.float32)
    return LazyBuffer(data)

# Step 7 - lazybuffer_unary_e
import enum
def e(self, op):
    # TODO: apply a unary elementwise op (NEG, RELU, LOG, EXP, SQRT, SIGMOID)
    if op.name == "RELU":
        return LazyBuffer(np.maximum(0, self._np))
    elif op.name == "NEG":
        return LazyBuffer(-1*self._np)
    elif op.name == "LOG":
        return LazyBuffer(np.log(self._np))
    elif op.name == "EXP":
        return LazyBuffer(np.exp(self._np))
    elif op.name == "SQRT":
        return LazyBuffer(np.sqrt(self._np))
    return LazyBuffer(1/(1 + np.exp(-self._np)))

LazyBuffer.e = e

# Step 8 - lazybuffer_binary_e
import enum
def lazybuffer_binary_e(self, op, other):
    # TODO: apply a binary elementwise op between two LazyBuffers, return a new LazyBuffer
    a,b = self._np, other._np
    if op.name == "ADD":
        return LazyBuffer(a + b)
    elif op.name == "SUB":
        return LazyBuffer(a - b)
    elif op.name == "MUL":
        return LazyBuffer(a * b)
    elif op.name == "DIV":
        return LazyBuffer(a / b)
    elif op.name == "MAX":
        return LazyBuffer(np.maximum(a,b))
    return LazyBuffer((a < b).astype(a.dtype))

# Step 9 - lazybuffer_r
import enum
def r(self, op, axis):
    # TODO: reduce the underlying array along axis (SUM or MAX), keeping reduced dims as size 1
    if op.name == "SUM":
        return LazyBuffer(np.sum(self._np, axis=axis, keepdims=True))
    else:
        return LazyBuffer(np.max(self._np, axis=axis, keepdims=True))

# Step 10 - lazybuffer_reshape
def reshape(self, new_shape):
    # TODO: return a new LazyBuffer with the array reshaped to new_shape
    return LazyBuffer(self._np.reshape(new_shape))

# Step 11 - lazybuffer_expand
def expand(self, new_shape):
    # TODO: broadcast this buffer's size-1 dims out to new_shape
    if isinstance(new_shape, int):
        new_shape = tuple(new_shape)
    else:
        new_shape = tuple(int(d) for d in new_shape)
    
    return LazyBuffer(np.array(np.broadcast_to(self._np, new_shape)))

# Step 12 - lazybuffer_permute (not yet solved)
# TODO: implement

# Step 13 - Function (not yet solved)
# TODO: implement

# Step 14 - function_forward_backward_stubs (not yet solved)
# TODO: implement

# Step 15 - apply (not yet solved)
# TODO: implement

# Step 16 - Neg (not yet solved)
# TODO: implement

# Step 17 - Relu (not yet solved)
# TODO: implement

# Step 18 - Log (not yet solved)
# TODO: implement

# Step 19 - Exp (not yet solved)
# TODO: implement

# Step 20 - Sqrt (not yet solved)
# TODO: implement

# Step 21 - Sigmoid (not yet solved)
# TODO: implement

# Step 22 - Add (not yet solved)
# TODO: implement

# Step 23 - Sub (not yet solved)
# TODO: implement

# Step 24 - Mul (not yet solved)
# TODO: implement

# Step 25 - Div (not yet solved)
# TODO: implement

# Step 26 - sum_function_forward (not yet solved)
# TODO: implement

# Step 27 - sum_function_backward (not yet solved)
# TODO: implement

# Step 28 - max_function_forward (not yet solved)
# TODO: implement

# Step 29 - max_function_backward (not yet solved)
# TODO: implement

# Step 30 - Reshape (not yet solved)
# TODO: implement

# Step 31 - expand_function_forward (not yet solved)
# TODO: implement

# Step 32 - expand_function_backward (not yet solved)
# TODO: implement

# Step 33 - permute_function_forward_backward (not yet solved)
# TODO: implement

# Step 34 - Tensor (not yet solved)
# TODO: implement

# Step 35 - tensor_from_data (not yet solved)
# TODO: implement

# Step 36 - tensor_creation_helpers (not yet solved)
# TODO: implement

# Step 37 - tensor_randn (not yet solved)
# TODO: implement

# Step 38 - build_topological_order (not yet solved)
# TODO: implement

# Step 39 - tensor_backward (not yet solved)
# TODO: implement

# Step 40 - bind_unary_tensor_methods (not yet solved)
# TODO: implement

# Step 41 - broadcasted (not yet solved)
# TODO: implement

# Step 42 - bind_binary_tensor_methods (not yet solved)
# TODO: implement

# Step 43 - bind_movement_tensor_methods (not yet solved)
# TODO: implement

# Step 44 - bind_reduce_tensor_methods (not yet solved)
# TODO: implement

# Step 45 - tensor_mean (not yet solved)
# TODO: implement

# Step 46 - tensor_transpose (not yet solved)
# TODO: implement

# Step 47 - tensor_matmul_2d (not yet solved)
# TODO: implement

# Step 48 - tensor_softmax (not yet solved)
# TODO: implement

# Step 49 - tensor_log_softmax (not yet solved)
# TODO: implement

# Step 50 - sparse_categorical_cross_entropy (not yet solved)
# TODO: implement

# Step 51 - Linear (not yet solved)
# TODO: implement

# Step 52 - MLP (not yet solved)
# TODO: implement

# Step 53 - sgd_step (not yet solved)
# TODO: implement

# Step 54 - zero_grad (not yet solved)
# TODO: implement

# Step 55 - make_toy_digit_dataset (not yet solved)
# TODO: implement

# Step 56 - accuracy (not yet solved)
# TODO: implement

# Step 57 - train_mlp (not yet solved)
# TODO: implement

# Step 58 - evaluate_mlp (not yet solved)
# TODO: implement

