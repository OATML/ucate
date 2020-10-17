import tensorflow as tf
import tensorflow.keras as tfk
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine.input_spec import InputSpec


class MultiBranchDense(tf.keras.layers.Layer):
    def __init__(
            self,
            units,
            num_branches,
            num_examples,
            activation=None,
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            **kwargs
    ):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(MultiBranchDense, self).__init__(
            activity_regularizer=tfk.regularizers.get(activity_regularizer), **kwargs)

        self.units = int(units) if not isinstance(units, int) else units
        self.num_groups = num_branches
        self.num_examples = num_examples
        self.activation = tfk.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = tfk.initializers.get(kernel_initializer)
        self.bias_initializer = tfk.initializers.get(bias_initializer)
        self.kernel_constraint = tfk.constraints.get(kernel_constraint)
        self.bias_constraint = tfk.constraints.get(bias_constraint)

        self.supports_masking = True
        self.input_spec = [InputSpec(min_ndim=2), InputSpec(min_ndim=2)]

    def build(self, input_shape):
        dtype = dtypes.as_dtype(self.dtype or K.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError('Unable to build `Dense` layer with non-floating point '
                            'dtype %s' % (dtype,))
        input_shape_x, input_shape_g = tensor_shape.TensorShape(input_shape[0]), tensor_shape.TensorShape(
            input_shape[1])
        if tensor_shape.dimension_value(input_shape_x[-1]) is None:
            raise ValueError('The last dimension of the inputs to `Dense` '
                             'should be defined. Found `None`.')
        last_dim_x = tensor_shape.dimension_value(input_shape_x[-1])
        last_dim_g = tensor_shape.dimension_value(input_shape_g[-1])
        self.input_spec = [
            InputSpec(
                min_ndim=2,
                axes={-1: last_dim_x}
            ),
            InputSpec(
                min_ndim=2,
                axes={-1: last_dim_g}
            )
        ]
        num_examples = tf.cast(self.num_examples, self.dtype)
        self.kernel_shape = [last_dim_x, self.units]
        self.kernel = self.add_weight(
            'kernel',
            shape=[last_dim_g, last_dim_x, self.units],
            initializer=self.kernel_initializer,
            regularizer=branch_l2(num_examples),
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True
        )

        if self.use_bias:
            self.bias = self.add_weight(
                'bias',
                shape=[last_dim_g, self.units],
                initializer=self.bias_initializer,
                regularizer=branch_bias_l2(num_examples),
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):
        x, g = inputs
        x = math_ops.cast(x, self._compute_dtype)
        g = math_ops.cast(g, self._compute_dtype)
        kernel = tf.reshape(self.kernel, [-1, tf.reduce_prod(self.kernel_shape)])
        kernel = gen_math_ops.mat_mul(g, kernel)
        kernel = tf.reshape(kernel, [-1, ] + self.kernel_shape)
        outputs = tf.keras.backend.batch_dot(x, kernel)
        if self.use_bias:
            bias = gen_math_ops.mat_mul(g, self.bias)
            outputs = outputs + bias
        if self.activation is not None:
            return self.activation(outputs)  # pylint: disable=not-callable
        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        if tensor_shape.dimension_value(input_shape[-1]) is None:
            raise ValueError(
                'The innermost dimension of input_shape must be defined, but saw: %s'
                % input_shape)
        return input_shape[:-1].concatenate(self.units)

    def get_config(self):
        config = {
            'units': self.units,
            'activation': tfk.activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': tfk.initializers.serialize(self.kernel_initializer),
            'bias_initializer': tfk.initializers.serialize(self.bias_initializer),
            'activity_regularizer':
                tfk.regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': tfk.constraints.serialize(self.kernel_constraint),
            'bias_constraint': tfk.constraints.serialize(self.bias_constraint)
        }
        base_config = super(MultiBranchDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class BatchDropout(tf.keras.layers.Dropout):
    def _get_noise_shape(self, inputs):
        return ops.convert_to_tensor_v2([1, array_ops.shape(inputs)[-1]])


def branch_l2(rate):
    def func(x):
        l2 = tf.reduce_sum(tf.square(x), axis=[1, 2])
        return 0.5 * tf.reduce_sum(l2 / rate)
    return func


def branch_bias_l2(rate):
    def func(x):
        l2 = tf.reduce_sum(tf.square(x), axis=-1)
        return 0.5 * tf.reduce_sum(l2 / rate)
    return func
