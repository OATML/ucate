import tensorflow as tf
from ucate.library import layers
from ucate.library.modules.core import identity


class Dense(tf.keras.Model):
    def __init__(
        self,
        units,
        use_bias=True,
        num_examples=1000,
        dropout_rate=0.0,
        batch_norm=False,
        num_branches=1,
        activation='linear',
        kernel_initializer='he_normal',
        name=None,
        *args,
        **kwargs
    ):
        super(Dense, self).__init__(
            *args,
            **kwargs
        )
        if num_branches > 1:
            self.dense = layers.MultiBranchDense(
                units=max(units // num_branches, 1),
                num_branches=num_branches,
                num_examples=num_examples,
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                name='{}_dense'.format(name)
            )
        else:
            weight_decay = 1 / num_examples
            self.dense = tf.keras.layers.Dense(
                units=units,
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=tf.keras.regularizers.l2(
                    0.5 * (1. - dropout_rate) * weight_decay
                ),
                bias_regularizer=tf.keras.regularizers.l2(0.5 * weight_decay),
                name='{}_dense'.format(name)
            )
        self.norm = tf.keras.layers.BatchNormalization(name='{}_norm'.format(name)) if batch_norm else identity()
        self.activation = tf.keras.layers.Activation(
            activation,
            name='{}_act'.format(name)
        )
        self.dropout = tf.keras.layers.Dropout(
            rate=dropout_rate,
            name='{}_drop'.format(name)
        )

    def call(self, inputs, training=None):
        outputs = self.dense(inputs)
        outputs = self.norm(outputs, training=training)
        outputs = self.activation(outputs)
        return self.dropout(outputs, training=training)
