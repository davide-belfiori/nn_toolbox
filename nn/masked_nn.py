from keras.layers import Input, Lambda
from keras.models import Model
from keras.losses import MeanSquaredError
from nn_modules.preprocessing import Patches, MaskPatches, ReshapePatches
from nn.cnn import Conv2DResnet

class MaskedModel():

    def __init__(self, input_shape: tuple,
                       patch_size: int,
                       num_masked_patches: int,
                       reshape_patches: bool = True) -> None:
        self.input_shape = input_shape
        self.patch_size = patch_size
        self.num_masked_patches = num_masked_patches
        self.reshape_patches = reshape_patches
        # > Build the model
        # >> Input
        self.input = Input(shape=self.input_shape)
        # >> Patch conversion + masking
        patches = Patches(patch_size=self.patch_size)(self.input)
        self.masked_images, self.masked_patches = MaskPatches(num_masked_patches=self.num_masked_patches)(patches)
        # >> Reshape
        if self.reshape_patches:
            self.masked_images = ReshapePatches(patch_size=self.patch_size)(self.masked_images)
            self.masked_patches = ReshapePatches(patch_size=self.patch_size)(self.masked_patches)

    def as_model(self):
        return Model(self.input, [self.masked_images, self.masked_patches])

class MaskedConv2DResnet(MaskedModel):

    def __init__(self, input_shape: tuple, 
                       patch_size: int, 
                       num_masked_patches: int,
                       resnet: Conv2DResnet) -> None:
        super().__init__(input_shape, patch_size, num_masked_patches, True)
        self.resnet = resnet
        self.output = resnet.as_model()(self.masked_images)
        self.mse = MeanSquaredError()

        self.loss = Lambda(lambda x : self.mse(x[0], x[1]))([self.masked_patches, self.output])
        self.model = Model(inputs = self.input, outputs = self.output)

        self.model.add_loss(self.loss)

    def as_model(self):
        return self.model