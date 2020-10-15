import tensorflow as tf
from ucate.library.models import core
from ucate.library.modules import samplers


class BayesianConvolutionalNeuralNetwork(core.BaseModel):
    def __init__(
            self,
            num_examples,
            dim_hidden,
            regression,
            depth,
            dropout_rate=0.1,
            *args,
            **kwargs
    ):
        super(BayesianConvolutionalNeuralNetwork, self).__init__(
            *args,
            **kwargs
        )
        self.conv_1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=5,
            strides=(1, 1),
            padding='same',
            activation='elu',
            kernel_regularizer=tf.keras.regularizers.l2(0.5 * (1 / num_examples) * (1 / 25)),
            bias_regularizer=tf.keras.regularizers.l2(0.5 * (1 / num_examples))
        )
        self.conv_1_pool = tf.keras.layers.MaxPool2D()
        self.conv_1_drop = tf.keras.layers.SpatialDropout2D(rate=dropout_rate)
        self.conv_2 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=5,
            strides=(1, 1),
            padding='same',
            activation='elu',
            kernel_regularizer=tf.keras.regularizers.l2(0.5 * (1. - dropout_rate) * (1 / num_examples) * (1 / 25)),
            bias_regularizer=tf.keras.regularizers.l2(0.5 * (1 / num_examples))
        )
        self.conv_2_pool = tf.keras.layers.MaxPool2D()
        self.flatten = tf.keras.layers.Flatten()
        self.conv_2_drop = tf.keras.layers.Dropout(rate=dropout_rate)
        self.hidden_1 = tf.keras.layers.Dense(
            units=dim_hidden,
            activation='elu',
            kernel_regularizer=tf.keras.regularizers.l2(0.5 * (1. - dropout_rate) * (1 / num_examples)),
            bias_regularizer=tf.keras.regularizers.l2(0.5 * (1 / num_examples))
        )
        self.dropout_1 = tf.keras.layers.Dropout(rate=dropout_rate)
        self.hidden_2 = tf.keras.layers.Dense(
            units=dim_hidden,
            activation='elu',
            kernel_regularizer=tf.keras.regularizers.l2(0.5 * (1. - dropout_rate) * (1 / num_examples)),
            bias_regularizer=tf.keras.regularizers.l2(0.5 * (1 / num_examples))
        )
        self.dropout_2 = tf.keras.layers.Dropout(rate=0.5)
        self.sampler = samplers.NormalSampler(
            dim_output=1,
            num_branches=1,
            num_examples=num_examples,
            beta=0.0
        ) if regression else samplers.BernoulliSampler(
            dim_output=1,
            num_branches=1,
            num_examples=num_examples,
            beta=0.0
        )
        self.regression = regression
        self.mc_sample_function = None

    def stem(
            self,
            inputs,
            training
    ):
        outputs = self.conv_1(inputs)
        outputs = self.conv_1_pool(outputs)
        outputs = self.conv_1_drop(outputs, training=training)
        outputs = self.conv_2(outputs)
        outputs = self.conv_2_pool(outputs)
        outputs = self.flatten(outputs)
        outputs = self.conv_2_drop(outputs, training=training)
        outputs = self.hidden_1(outputs)
        outputs = self.dropout_1(outputs, training=training)
        outputs = self.hidden_2(outputs)
        outputs = self.dropout_2(outputs, training=training)
        return outputs

    def call(
            self,
            inputs,
            training=None
    ):
        x, y = inputs
        h = self.stem(
            x,
            training=training
        )
        py = self.sampler(h)
        self.add_loss(tf.reduce_mean(-py.log_prob(y)))
        return py.sample(), py.entropy()

    def mc_sample_step(
            self,
            inputs
    ):
        h = self.stem(
            inputs,
            training=True
        )
        py = self.sampler(h)
        mu = py.loc if self.regression else tf.sigmoid(py.logits)
        return mu, py.sample()
