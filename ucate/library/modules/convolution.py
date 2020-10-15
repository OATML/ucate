import tensorflow as tf


class ConvHead(tf.keras.Model):
    def __init__(
        self,
        base_filters,
        num_examples=1000,
        dropout_rate=0.0,
        kernel_initializer='he_normal',
        *args,
        **kwargs
    ):
        super(ConvHead, self).__init__(
            *args,
            **kwargs
        )
        self.conv_1 = tf.keras.layers.Conv2D(
            filters=base_filters,
            kernel_size=5,
            strides=(1, 1),
            padding='same',
            activation='elu',
            kernel_initializer=kernel_initializer,
            kernel_regularizer=tf.keras.regularizers.l2(0.5 * (1 / num_examples) * (1 / 25)),
            bias_regularizer=tf.keras.regularizers.l2(0.5 * (1 / num_examples))
        )
        self.conv_1_pool = tf.keras.layers.MaxPool2D()
        self.conv_1_drop = tf.keras.layers.SpatialDropout2D(rate=dropout_rate)
        self.conv_2 = tf.keras.layers.Conv2D(
            filters=base_filters * 2,
            kernel_size=5,
            strides=(1, 1),
            padding='same',
            activation='elu',
            kernel_initializer=kernel_initializer,
            kernel_regularizer=tf.keras.regularizers.l2(0.5 * (1. - dropout_rate) * (1 / num_examples) * (1 / 25)),
            bias_regularizer=tf.keras.regularizers.l2(0.5 * (1 / num_examples))
        )
        self.conv_2_pool = tf.keras.layers.MaxPool2D()
        self.flatten = tf.keras.layers.Flatten()
        self.conv_2_drop = tf.keras.layers.Dropout(rate=dropout_rate)

    def call(
            self,
            inputs,
            training=None
    ):
        outputs = self.conv_1(inputs)
        outputs = self.conv_1_pool(outputs)
        outputs = self.conv_1_drop(outputs, training=True)
        outputs = self.conv_2(outputs)
        outputs = self.conv_2_pool(outputs)
        outputs = self.flatten(outputs)
        outputs = self.conv_2_drop(outputs, training=True)
        return outputs


class ConvTail(tf.keras.Model):
    def __init__(
        self,
        base_filters,
        num_examples=1000,
        dropout_rate=0.0,
        kernel_initializer='he_normal',
        *args,
        **kwargs
    ):
        super(ConvTail, self).__init__(
            *args,
            **kwargs
        )
        self.conv_1 = tf.keras.layers.Conv2DTranspose(
            filters=base_filters * 2,
            kernel_size=7,
            activation='elu',
            kernel_initializer=kernel_initializer,
            kernel_regularizer=tf.keras.regularizers.l2(0.5 * (1. - dropout_rate) * (1 / num_examples)),
            bias_regularizer=tf.keras.regularizers.l2(0.5 * (1 / num_examples))
        )
        self.conv_1_pool = tf.keras.layers.UpSampling2D()
        self.conv_1_drop = tf.keras.layers.SpatialDropout2D(rate=dropout_rate)
        self.conv_2 = tf.keras.layers.Conv2DTranspose(
            filters=base_filters * 1,
            kernel_size=5,
            strides=(1, 1),
            padding='same',
            activation='elu',
            kernel_initializer=kernel_initializer,
            kernel_regularizer=tf.keras.regularizers.l2(0.5 * (1. - dropout_rate) * (1 / num_examples) * (1 / 25)),
            bias_regularizer=tf.keras.regularizers.l2(0.5 * (1 / num_examples))
        )
        self.conv_2_pool = tf.keras.layers.UpSampling2D()
        self.conv_2_drop = tf.keras.layers.Dropout(rate=dropout_rate)

    def call(
            self,
            inputs,
            training=None
    ):
        outputs = tf.keras.backend.expand_dims(inputs, 1)
        outputs = tf.keras.backend.expand_dims(outputs, 1)
        outputs = self.conv_1(outputs)
        outputs = self.conv_1_pool(outputs)
        outputs = self.conv_1_drop(outputs, training=True)
        outputs = self.conv_2(outputs)
        outputs = self.conv_2_pool(outputs)
        outputs = self.conv_2_drop(outputs, training=True)
        return outputs
