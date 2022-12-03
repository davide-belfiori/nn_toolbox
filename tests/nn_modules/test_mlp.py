import numpy as np
from nn_modules.mlp import MLP
import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

def test_mlp_shape():
    batch_size = 4
    in_size = 8
    input_tensor = np.random.random(size=(batch_size, in_size))
    mlp = MLP(unit_list=[32, 16])
    output_tensor = mlp(input_tensor)
    assert output_tensor.shape == (batch_size, 16)
