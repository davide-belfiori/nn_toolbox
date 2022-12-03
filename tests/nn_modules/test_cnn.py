import numpy as np
from nn_modules.cnn import *

def test_conv2d_block() -> None:
    shape = (4, 16, 16, 3)
    filters = 8
    input_tensor = np.zeros(shape = shape)
    conv_block = Conv2DBlock(filters = filters, kernel_size=3)
    output_tensor = conv_block(input_tensor)
    assert output_tensor.shape == (shape[0], shape[1], shape[2], filters)

def test_conv2d_res_block() -> None:
    shape = (4, 16, 16, 32)
    filters = 16
    input_tensor = np.zeros(shape = shape)
    for res_type in [RES_TYPE_ORIGIN, RES_TYPE_NORM_AFTER_ADD, RES_TYPE_ACTIV_AFTER_ADD, RES_TYPE_PRE_ACTIV, RES_TYPE_FULL_PRE_ACTIV]:
        conv_block = Conv2DResBlock(filters = filters, block_type=res_type)
        output_tensor = conv_block(input_tensor)
        assert output_tensor.shape == (shape[0], shape[1], shape[2], filters)

def test_conv2d_res_down_block() -> None:
    shape = (4, 16, 16, 32)
    filters = 16
    pool_size = 2
    input_tensor = np.zeros(shape = shape)
    for res_type in [RES_TYPE_ORIGIN, RES_TYPE_NORM_AFTER_ADD, RES_TYPE_ACTIV_AFTER_ADD, RES_TYPE_PRE_ACTIV, RES_TYPE_FULL_PRE_ACTIV]:
        conv_block = Conv2DResDownBlock(filters = filters, pool_size = pool_size, block_type=res_type)
        output_tensor = conv_block(input_tensor)
        assert output_tensor.shape == (shape[0], shape[1] // pool_size, shape[2] // pool_size, filters)

def test_conv2d_res_up_block() -> None:
    shape = (4, 16, 16, 32)
    filters = 16
    sample_size = 2
    input_tensor = np.zeros(shape = shape)
    for res_type in [RES_TYPE_ORIGIN, RES_TYPE_NORM_AFTER_ADD, RES_TYPE_ACTIV_AFTER_ADD, RES_TYPE_PRE_ACTIV, RES_TYPE_FULL_PRE_ACTIV]:
        conv_block = Conv2DResUpBlock(filters = filters, sample_size = sample_size, block_type=res_type)
        output_tensor = conv_block(input_tensor)
        assert output_tensor.shape == (shape[0], shape[1] * sample_size, shape[2] * sample_size, filters)