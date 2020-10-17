import tensorflow as tf
from ucate.library.models import core
from ucate.library.modules import dense
from ucate.library.modules import samplers
from ucate.library.modules import convolution


class BayesianCEVAE(core.BaseModel):
    def __init__(
        self,
        dim_x,
        dim_t,
        dim_y,
        regression,
        dim_latent,
        num_examples,
        dim_hidden,
        dropout_rate=0.1,
        beta=1.0,
        negative_sampling=True,
        do_convolution=False,
        *args,
        **kwargs
    ):
        super(BayesianCEVAE, self).__init__(*args, **kwargs)
        if isinstance(dim_x, list) and not do_convolution:
            self.dim_x = dim_x[0]
            self.dim_x_bin = dim_x[1]
            dim_x = sum(dim_x)
        else:
            self.dim_x = dim_x
            self.dim_x_bin = None
        self.regression = regression
        self.encoder = Encoder(
            do_convolution=do_convolution,
            dim_latent=dim_latent,
            num_examples=num_examples,
            dim_hidden=dim_hidden,
            dropout_rate=dropout_rate,
            beta=beta,
            negative_sampling=negative_sampling,
        )
        self.decoder = Decoder(
            dim_x=dim_x,
            dim_t=dim_t,
            dim_y=dim_y,
            regression=regression,
            num_examples=num_examples,
            dim_hidden=dim_hidden,
            dropout_rate=dropout_rate,
        )
        if do_convolution:
            self.x_sampler = samplers.ConvNormalSampler(
                dim_output=self.dim_x[-1], num_examples=sum(num_examples), beta=0.0
            )
        else:
            self.x_sampler = samplers.NormalSampler(
                dim_output=self.dim_x,
                num_branches=1,
                num_examples=sum(num_examples),
                beta=0.0,
            )
            if self.dim_x_bin is not None:
                self.x_sampler_bin = samplers.BernoulliSampler(
                    dim_output=self.dim_x_bin,
                    num_branches=1,
                    num_examples=sum(num_examples),
                    beta=0.0,
                )

    def call(self, inputs, training=None):
        x, t, y = inputs
        qz = self.encoder([x, t], training=training)
        hx, pt, py = self.decoder([qz.sample(), t], training=training)
        px = self.x_sampler(hx, training=training)
        if self.dim_x_bin is not None:
            px_bin = self.x_sampler_bin(hx, training=training)
            x_cont, x_bin = tf.split(x, [self.dim_x, self.dim_x_bin], axis=-1)
            self.add_loss(tf.reduce_mean(-px.log_prob(x_cont)))
            self.add_loss(tf.reduce_mean(-px_bin.log_prob(x_bin)))
        else:
            self.add_loss(tf.reduce_mean(-px.log_prob(x)))
        self.add_loss(tf.reduce_mean(-pt.log_prob(t)))
        self.add_loss(tf.reduce_mean(-py.log_prob(y)))
        mu = py.loc if self.regression else tf.sigmoid(py.logits)
        return mu, py.sample()

    def mc_sample_model(self, inputs):
        x, t = inputs
        qz = self.encoder([x, t], training=False)
        hx, pt, py = self.decoder([qz.loc, t], training=True)
        mu = py.loc if self.regression else tf.sigmoid(py.logits)
        return mu, py.sample()

    def mc_sample_latent_step(self, inputs):
        x, t = inputs
        qz = self.encoder([x, t], training=False)
        hx, pt, py = self.decoder([qz.sample(), t], training=False)
        mu = py.loc if self.regression else tf.sigmoid(py.logits)
        return mu, py.sample()

    def mc_sample_step(self, inputs):
        x, t = inputs
        qz = self.encoder([x, t], training=False)
        hx, pt, py = self.decoder([qz.sample(), t], training=True)
        mu = py.loc if self.regression else tf.sigmoid(py.logits)
        return mu, py.sample()


class Encoder(tf.keras.Model):
    def __init__(
        self,
        do_convolution,
        dim_latent,
        num_examples,
        dim_hidden,
        dropout_rate=0.1,
        beta=1.0,
        negative_sampling=True,
        *args,
        **kwargs
    ):
        super(Encoder, self).__init__(*args, **kwargs)
        self.conv = (
            convolution.ConvHead(
                base_filters=32,
                num_examples=sum(num_examples),
                dropout_rate=dropout_rate,
            )
            if do_convolution
            else dense.identity()
        )
        self.hidden_1 = dense.Dense(
            units=dim_hidden,
            num_examples=sum(num_examples),
            dropout_rate=dropout_rate,
            activation="elu",
            name="encoder_hidden_1",
        )
        self.hidden_2 = dense.Dense(
            units=dim_hidden,
            num_examples=sum(num_examples),
            dropout_rate=dropout_rate,
            activation="elu",
            name="encoder_hidden_2",
        )
        self.hidden_3 = dense.Dense(
            units=dim_hidden,
            num_examples=num_examples,
            dropout_rate=dropout_rate,
            num_branches=2,
            activation="elu",
            name="encoder_hidden_3",
        )
        self.hidden_4 = dense.Dense(
            units=dim_hidden,
            num_examples=num_examples,
            dropout_rate=dropout_rate,
            num_branches=2,
            activation="elu",
            name="encoder_hidden_4",
        )
        self.sampler = samplers.NormalSampler(
            dim_output=dim_latent,
            num_branches=2,
            num_examples=num_examples,
            beta=beta / 2 if negative_sampling else beta,
        )
        self.negative_sampling = negative_sampling

    def call(self, inputs, training=None):
        x, t = inputs
        q = self.forward([x, t], training=training)
        if self.negative_sampling:
            t_cf = 1.0 - t
            _ = self.forward([x, t_cf], training=training)
        return q

    def forward(self, inputs, training=None):
        x, t = inputs
        outputs = self.conv(x, training=training)
        outputs = self.hidden_1(outputs, training=training)
        outputs = self.hidden_2(outputs, training=training)
        outputs = self.hidden_3([outputs, t], training=training)
        outputs = self.hidden_4([outputs, t], training=training)
        return self.sampler([outputs, t], training=training)


class Decoder(tf.keras.Model):
    def __init__(
        self,
        dim_x,
        dim_t,
        dim_y,
        regression,
        num_examples,
        dim_hidden,
        dropout_rate=0.1,
        *args,
        **kwargs
    ):
        super(Decoder, self).__init__(*args, **kwargs)
        do_convolution = isinstance(dim_x, (tuple, list))
        self.x_hidden_1 = dense.Dense(
            units=dim_hidden,
            num_examples=sum(num_examples),
            dropout_rate=dropout_rate,
            num_branches=1,
            activation="elu",
            name="decoder_x_hidden_1",
        )
        self.x_hidden_2 = dense.Dense(
            units=dim_hidden,
            num_examples=sum(num_examples),
            dropout_rate=dropout_rate if do_convolution else 0.1,
            num_branches=1,
            activation="elu",
            name="decoder_x_hidden_2",
        )
        self.x_conv = (
            convolution.ConvTail(
                base_filters=32,
                num_examples=sum(num_examples),
                dropout_rate=dropout_rate,
            )
            if do_convolution
            else dense.identity()
        )
        self.t_hidden_1 = dense.Dense(
            units=dim_hidden,
            num_examples=sum(num_examples),
            dropout_rate=dropout_rate,
            num_branches=1,
            activation="elu",
            name="decoder_t_hidden_1",
        )
        self.t_hidden_2 = dense.Dense(
            units=dim_hidden,
            num_examples=sum(num_examples),
            dropout_rate=0.5,
            num_branches=1,
            activation="elu",
            name="decoder_t_hidden_2",
        )
        self.t_sampler = samplers.CategoricalSampler(
            dim_output=dim_t, num_branches=1, num_examples=sum(num_examples), beta=0.0
        )
        self.y_hidden_1 = dense.Dense(
            units=dim_hidden,
            num_examples=sum(num_examples),
            dropout_rate=dropout_rate,
            activation="elu",
            name="decoder_y_hidden_1",
        )
        self.y_hidden_2 = dense.Dense(
            units=dim_hidden,
            num_examples=sum(num_examples),
            dropout_rate=dropout_rate,
            activation="elu",
            name="decoder_y_hidden_2",
        )
        self.y_hidden_3 = dense.Dense(
            units=dim_hidden,
            num_examples=num_examples,
            dropout_rate=dropout_rate,
            num_branches=2,
            activation="elu",
            name="decoder_y_hidden_3",
        )
        self.y_hidden_4 = dense.Dense(
            units=dim_hidden,
            num_examples=num_examples,
            dropout_rate=0.5,
            num_branches=2,
            activation="elu",
            name="decoder_y_hidden_4",
        )
        y_sampler = samplers.NormalSampler if regression else samplers.BernoulliSampler
        self.y_sampler = y_sampler(
            dim_output=dim_y, num_branches=2, num_examples=num_examples, beta=0.0
        )

    def call(self, inputs, training=None):
        z, t = inputs
        hx = self.x_hidden_1(z, training=training)
        hx = self.x_hidden_2(hx, training=training)
        hx = self.x_conv(hx, training=training)
        ht = self.t_hidden_1(z, training=training)
        ht = self.t_hidden_2(ht, training=training)
        pt = self.t_sampler(ht, training=training)
        hy = self.y_hidden_1(z, training=training)
        hy = self.y_hidden_2(hy, training=training)
        hy = self.y_hidden_3([hy, t], training=training)
        hy = self.y_hidden_4([hy, t], training=training)
        py = self.y_sampler([hy, t], training=training)
        return hx, pt, py
