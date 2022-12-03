from nn.masked_nn import MaskedModel, MaskedConv2DResnet
from nn.cnn import Conv2DResnet
from tests.mock import MockInput

def test_masked_model():
    mock = MockInput()
    patch_size = 2
    num_patches = (mock.image_height * mock.image_width) // (patch_size ** 2)
    masked_patches = 1
    mm = MaskedModel(input_shape=mock.shape[1:], 
                     patch_size=patch_size, 
                     num_masked_patches=masked_patches, 
                     reshape_patches=True).as_model()
    pred = mm.predict(mock.ones())
    assert pred[0].shape == (mock.batch_size, patch_size, patch_size, num_patches * mock.channels) and \
           pred[1].shape == (mock.batch_size, patch_size, patch_size, masked_patches * mock.channels)

def test_maskedConv2DResnet():
    mock = MockInput()
    patch_size = 2
    num_patches = (mock.image_height * mock.image_width) // (patch_size ** 2)
    masked_patches = 1
    resnet = Conv2DResnet(input_shape=(patch_size, patch_size, num_patches * mock.channels),
                         filter_list=[mock.channels * masked_patches])
    mres = MaskedConv2DResnet(input_shape=mock.shape[1:],
                              patch_size=patch_size,
                              num_masked_patches=masked_patches,
                              resnet=resnet).as_model()
    pred = mres.predict(mock.ones())
    assert pred.shape == (mock.batch_size, patch_size, patch_size, masked_patches * mock.channels)
