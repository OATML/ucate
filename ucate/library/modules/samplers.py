import tensorflow as tf
from tensorflow_probability import distributions
from ucate.library.modules import dense


class NormalSampler(tf.keras.Model):
    def __init__(
            self,
            dim_output,
            num_branches,
            num_examples,
            beta,
            *args,
            **kwargs
    ):
        super(NormalSampler, self).__init__(
            *args,
            **kwargs
        )
        self.mu = dense.Dense(
            units=dim_output,
            num_examples=num_examples,
            dropout_rate=0.0,
            num_branches=num_branches,
            activation='linear',
            name='normal_mu'
        )
        self.sigma = dense.Dense(
            units=dim_output,
            num_examples=num_examples,
            dropout_rate=0.0,
            num_branches=num_branches,
            activation='softplus',
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
            name='normal_sigma'
        )
        self.beta = beta

    def call(
            self,
            inputs,
            training=None
    ):
        mu = self.mu(inputs)
        sigma = self.sigma(inputs) + tf.keras.backend.epsilon()
        q = distributions.MultivariateNormalDiag(
            loc=mu,
            scale_diag=sigma
        )
        if self.beta > 0.0:
            kld = q.kl_divergence(
                distributions.MultivariateNormalDiag(
                    loc=tf.zeros_like(mu)
                )
            )
            self.add_loss(self.beta * tf.reduce_mean(kld))
        return q


class BernoulliSampler(tf.keras.Model):
    def __init__(
            self,
            dim_output,
            num_branches,
            num_examples,
            beta,
            *args,
            **kwargs
    ):
        super(BernoulliSampler, self).__init__(
            *args,
            **kwargs
        )
        self.logits = dense.Dense(
            units=dim_output,
            num_examples=num_examples,
            dropout_rate=0.0,
            num_branches=num_branches,
            activation='linear',
            name='encoder_mu'
        )
        self.beta = beta

    def call(
            self,
            inputs,
            training=None
    ):
        logits = self.logits(inputs)
        q = distributions.Bernoulli(
            logits=logits,
            dtype=tf.float32,
        )
        if self.beta > 0.0:
            kld = q.kl_divergence(
                distributions.Bernoulli(
                    logits=tf.zeros_like(logits),
                    dtype=tf.float32,
                )
            )
            self.add_loss(tf.reduce_mean(kld))
        return q


class CategoricalSampler(tf.keras.Model):
    def __init__(
            self,
            dim_output,
            num_branches,
            num_examples,
            beta,
            *args,
            **kwargs
    ):
        super(CategoricalSampler, self).__init__(
            *args,
            **kwargs
        )
        self.logits = dense.Dense(
            units=dim_output,
            num_examples=num_examples,
            dropout_rate=0.0,
            num_branches=num_branches,
            activation='linear',
            name='categorical_logits'
        )
        self.beta = beta

    def call(
            self,
            inputs,
            training=None
    ):
        logits = self.logits(inputs)
        q = distributions.OneHotCategorical(
            logits=logits,
            dtype=tf.float32,
        )
        if self.beta > 0.0:
            kld = q.kl_divergence(
                distributions.OneHotCategorical(
                    logits=tf.zeros_like(logits),
                    dtype=tf.float32,
                )
            )
            self.add_loss(tf.reduce_mean(kld))
        return q


class ConvNormalSampler(tf.keras.Model):
    def __init__(
            self,
            dim_output,
            num_examples,
            beta,
            *args,
            **kwargs
    ):
        super(ConvNormalSampler, self).__init__(
            *args,
            **kwargs
        )
        self.mu = tf.keras.layers.Conv2D(
            filters=dim_output,
            kernel_size=3,
            padding='same',
            activation='linear',
            kernel_initializer='he_normal',
            kernel_regularizer=tf.keras.regularizers.l2(0.5 * (1 / num_examples)),
            bias_regularizer=tf.keras.regularizers.l2(0.5 * (1 / num_examples)),
            name='normal_mu'
        )
        # self.sigma = tf.keras.layers.Conv2D(
        #     filters=dim_output,
        #     kernel_size=3,
        #     padding='same',
        #     activation='softplus',
        #     kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
        #     kernel_regularizer=tf.keras.regularizers.l2(0.5 * (1 / num_examples)),
        #     bias_regularizer=tf.keras.regularizers.l2(0.5 * (1 / num_examples)),
        #     name='normal_sigma'
        # )
        self.beta = beta

    def call(
            self,
            inputs,
            training=None
    ):
        mu = self.mu(inputs)
        # sigma = self.sigma(inputs) + tf.keras.backend.epsilon()
        q = distributions.Independent(
            distributions.MultivariateNormalDiag(
                loc=mu,
                scale_diag=tf.ones_like(mu)
            )
        )
        if self.beta > 0.0:
            kld = q.kl_divergence(
                distributions.MultivariateNormalDiag(
                    loc=tf.zeros_like(mu)
                )
            )
            self.add_loss(0.5 * tf.reduce_mean(kld))
        return q
