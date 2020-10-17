from ucate.library.models.mlp import BayesianNeuralNetwork

from ucate.library.models.cnn import BayesianConvolutionalNeuralNetwork

from ucate.library.models.cevae import BayesianCEVAE

from ucate.library.models.tarnet import TARNet

from ucate.library.models.core import BaseModel


MODELS = {
    "mlp": BayesianNeuralNetwork,
    "cnn": BayesianConvolutionalNeuralNetwork,
    "cevae": BayesianCEVAE,
    "tarnet": TARNet
}
