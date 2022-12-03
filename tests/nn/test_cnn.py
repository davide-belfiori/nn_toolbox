import numpy as np
from nn.cnn import Conv2DResnet
from nn_modules.cnn import RES_TYPE_FULL_PRE_ACTIV, Conv2DBlock
from nn_modules.mlp import MLP


def test_conv2d_resnet() -> None:
    image_shape = (16, 16, 3)
    tensor_shape = (8,) + image_shape
    filters = [8, 16, 32]
    input_tensor = np.random.random(size = tensor_shape)
    resnet = Conv2DResnet(input_shape=image_shape, 
                          filter_list=filters, 
                          pool_size=None, 
                          mlp = None, 
                          preprocessing=None).as_model()
    output_tensor = resnet.predict(input_tensor)
    assert output_tensor.shape == tensor_shape[:3] + (filters[-1], )

def test_conv2d_pool_resnet() -> None:
    image_shape = (32, 32, 3)
    tensor_shape = (8,) + image_shape
    filters = [8, 16, 32]
    pool_size = 2
    input_tensor = np.random.random(size = tensor_shape)
    resnet = Conv2DResnet(input_shape=image_shape, 
                          filter_list=filters, 
                          pool_size=pool_size, 
                          mlp = None, 
                          preprocessing=None).as_model()
    output_tensor = resnet.predict(input_tensor)
    target_out_height = image_shape[0] // (pool_size ** len(filters))
    target_out_width = image_shape[1] // (pool_size ** len(filters))
    assert output_tensor.shape == (tensor_shape[0], target_out_height, target_out_width, filters[-1])

def test_conv2d_resnet_mlp() -> None:
    image_shape = (16, 16, 3)
    tensor_shape = (8,) + image_shape
    filters = [8, 16, 32]
    unit_list = [32, 16]
    input_tensor = np.random.random(size = tensor_shape)
    mlp = MLP(unit_list=unit_list, flatten_input=True)
    resnet = Conv2DResnet(input_shape=image_shape, 
                         filter_list=filters, 
                         pool_size=None, 
                         mlp = mlp, 
                         preprocessing=None).as_model()
    output_tensor = resnet.predict(input_tensor)
    assert output_tensor.shape == (tensor_shape[0], ) + (unit_list[-1], )

def test_conv2d_resnet_preprocessing() -> None:
    image_shape = (16, 16, 3)
    tensor_shape = (8,) + image_shape
    filters = [8, 16, 32]
    unit_list = [32, 16]
    input_tensor = np.random.random(size = tensor_shape)
    res_block_type = RES_TYPE_FULL_PRE_ACTIV
    preprocessing = Conv2DBlock(filters=8)
    mlp = MLP(unit_list=unit_list, flatten_input=True)
    resnet = Conv2DResnet(input_shape=image_shape, 
                          filter_list=filters, 
                          pool_size=None, 
                          res_block_type=res_block_type, 
                          mlp = mlp, 
                          preprocessing = preprocessing).as_model()
    output_tensor = resnet.predict(input_tensor)
    assert output_tensor.shape == (tensor_shape[0], ) + (unit_list[-1], )
