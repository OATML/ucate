import tensorflow as tf
from ucate.library.models import core
from ucate.library.modules import dense
from ucate.library.modules import samplers
from ucate.library.modules import convolution


class TARNet(core.BaseModel):
    def __init__(
        self,
        do_convolution,
        num_examples,
        dim_hidden,
        regression,
        dropout_rate=0.1,
        beta=1.0,
        mode="tarnet",
        *args,
        **kwargs
    ):
        super(TARNet, self).__init__(*args, **kwargs)
        self.conv = (
            convolution.ConvHead(
                base_filters=32,
                num_examples=sum(num_examples),
                dropout_rate=0.1,
            )
            if do_convolution
            else dense.identity()
        )
        self.hidden_1 = dense.Dense(
            units=dim_hidden,
            num_examples=sum(num_examples),
            dropout_rate=dropout_rate,
            activation="elu",
            name="tarnet_hidden_1",
        )
        self.hidden_2 = dense.Dense(
            units=dim_hidden,
            num_examples=sum(num_examples),
            dropout_rate=dropout_rate,
            activation="elu",
            name="tarnet_hidden_2",
        )
        self.hidden_3 = dense.Dense(
            units=dim_hidden,
            num_examples=sum(num_examples),
            dropout_rate=dropout_rate,
            activation="elu",
            name="tarnet_hidden_3",
        )
        self.hidden_4 = dense.Dense(
            units=dim_hidden,
            num_examples=num_examples,
            dropout_rate=dropout_rate,
            num_branches=2,
            activation="elu",
            name="encoder_hidden_4",
        )
        self.hidden_5 = dense.Dense(
            units=dim_hidden,
            num_examples=num_examples,
            dropout_rate=0.5,
            num_branches=2,
            activation="elu",
            name="encoder_hidden_5",
        )
        y_sampler = samplers.NormalSampler if regression else samplers.BernoulliSampler
        self.y_sampler = y_sampler(
            dim_output=1, num_branches=2, num_examples=num_examples, beta=0.0
        )
        self.regression = regression
        self.beta = beta
        if mode == "dragon":
            self.t_sampler = samplers.CategoricalSampler(
                2, num_branches=1, num_examples=sum(num_examples), beta=0.0
            )
        self.mode = mode
        self.y_loss = (
            tf.keras.losses.MeanSquaredError()
            if regression
            else tf.keras.losses.BinaryCrossentropy()
        )

    def call(self, inputs, training=None):
        x, t, y = inputs
        py, pt = self.forward([x, t], training=training)
        self.add_loss(tf.reduce_mean(-py.log_prob(y)))
        mu = py.loc if self.regression else tf.sigmoid(py.logits)
        if self.mode == "dragon":
            self.add_loss(tf.reduce_mean(-pt.log_prob(t)))
            self.add_loss(
                self.beta
                * propensity_loss(
                    t_true=t,
                    t_pred=tf.keras.backend.softmax(pt.logits),
                    y_true=y,
                    y_pred=mu,
                    y_loss=self.y_loss,
                )
            )
        elif self.mode == "mmd":
            self.add_loss(self.beta * mmd(t, pt))
        return mu, py.sample()

    def forward(self, inputs, training=None):
        x, t = inputs
        phi = self.conv(x, training=training)
        phi = self.hidden_1(phi, training=training)
        phi = self.hidden_2(phi, training=training)
        phi = self.hidden_3(phi, training=training)
        pt = self.t_sampler(phi) if self.mode == "dragon" else phi
        outputs = self.hidden_4([phi, t], training=training)
        outputs = self.hidden_5([outputs, t], training=training)
        return self.y_sampler([outputs, t], training=training), pt

    def mc_sample_step(self, inputs):
        x, t = inputs
        py, pt = self.forward([x, t], training=True)
        mu = py.loc if self.regression else tf.sigmoid(py.logits)
        return mu, py.sample()


def propensity_loss(t_true, t_pred, y_true, y_pred, y_loss):
    eps = tf.keras.backend.epsilon()
    loss_value = tf.reduce_mean(
        tf.keras.losses.categorical_crossentropy(t_true, t_pred)
    )
    t_pred = (t_pred + 0.001) / 1.002
    h = tf.reduce_sum(t_true / t_pred, -1, keepdims=True)
    q_tilde = y_pred + eps * h
    loss_value += tf.reduce_mean(y_loss(y_true, q_tilde))
    return loss_value


def mmd(t, phi):
    t_0 = tf.expand_dims(t[:, 0], -1)
    t_1 = tf.expand_dims(t[:, 1], -1)
    phi_0 = phi * t_0
    phi_1 = phi * t_1
    mu_0 = tf.reduce_sum(phi_0, 0) / (tf.reduce_sum(t_0) + tf.keras.backend.epsilon())
    mu_1 = tf.reduce_sum(phi_1, 0) / (tf.reduce_sum(t_1) + tf.keras.backend.epsilon())
    return tf.reduce_sum(tf.square(mu_0 - mu_1))
