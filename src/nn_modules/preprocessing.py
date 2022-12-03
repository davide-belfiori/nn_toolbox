from keras.layers import Layer, Reshape
from tensorflow import Tensor, Variable, image, argsort, random, shape, gather, repeat, newaxis, concat, transpose

class Patches(Layer):
    '''
        Extract pacthes from images.    

        Input: 
        
            4D tensor of shape `(batch_size, width, height, channels)`

        Output:
            
            3D tensor of shape `(batch_size, num_patches_per_image, patch_dimension)` where `patch_dimension = patch_size * patch_size * channels`
    '''
    def __init__(self, patch_size : int, stride: int = None, **kwargs):
        super().__init__(trainable = False, **kwargs)
        self.patch_size = patch_size
        self.stride = stride if stride != None else self.patch_size
        self.num_patches_per_image = None
        self.image_height = None
        self.image_width = None
        self.channels = None
        self.reshape = None

    def build(self, input_shape):
        super().build(input_shape)
        _, self.image_height, self.image_width, self.channels = input_shape
        self.num_patches_per_image = (self.image_height * self.image_width) // (self.patch_size ** 2)
        self.patch_dim = self.patch_size * self.patch_size * self.channels
        self.reshape = Reshape(target_shape=(self.num_patches_per_image, self.patch_dim))

    def call(self, inputs):
        # extract patches
        patches = image.extract_patches(
            images = inputs,
            sizes = [1, self.patch_size, self.patch_size, 1],
            strides = [1, self.stride, self.stride, 1],
            rates = [1, 1, 1, 1],
            padding = "VALID")
        # reshape
        patches = self.reshape(patches)
        return patches

    def get_config(self):
        config = super().get_config()
        config.update({
            "patch_size": self.patch_size,
            "stride" : self.stride
        })
        return config

class ReshapePatches(Layer):
    '''
        Reshape a 3D patch tensor of shape `(batch_size, num_patch_per_image, patch_dimension)` 
        into a 4D tensor of shape `(batch_size, patch_size, patch_size, num_patches_per_image * channels)`
    '''
    def __init__(self, patch_size: int, **kwargs):
        super().__init__(trainable = False, **kwargs)
        self.patch_size = patch_size
        self.num_patches_per_image = None
        self.channels = None

    def build(self, input_shape):
        _, num_patches_per_image, patch_dim = input_shape
        self.num_patches_per_image = num_patches_per_image
        self.channels = patch_dim // (self.patch_size ** 2)
        self.reshape1 = Reshape(target_shape=(self.num_patches_per_image, self.patch_size, self.patch_size, self.channels))
        self.reshape2 = Reshape(target_shape=(self.patch_size, self.patch_size, self.num_patches_per_image * self.channels))

    def call(self, patches):
        # 1: (batch_size, num_patches_per_image, patch_size, patch_size, channels)
        patches = self.reshape1(patches)
        # 2: (batch_size, patch_size, patch_size, channels, num_patches_per_image)
        patches = transpose(patches, [0, 2, 3, 4, 1])
        # 3: (batch_size, patch_size, patch_size, num_patches_per_image * channels)
        patches = self.reshape2(patches)
        return patches

    def get_config(self):
        config = super().get_config()
        config.update({
            "patch_size": self.patch_size
        })
        return config


class MaskPatches(Layer):

    def __init__(self, num_masked_patches: int = 1, mask: Tensor = None, indices = None, return_indices: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.num_masked_patches = num_masked_patches
        self.mask = mask
        self.indices = indices
        self.return_indices = return_indices

    def build(self, input_shape):
        patch_dim = input_shape[-1]
        if self.mask == None:
            self.mask = Variable(initial_value = random.normal([1, patch_dim]), 
                                trainable = True)

    def call(self, patches):
        # get batch size
        batch_size = shape(patches)[0]
        # get num pathces per image
        num_patches_per_image = shape(patches)[1]
        # calculate indices
        if self.indices != None:
            indices = self.indices
        else:
            indices = argsort(random.uniform(shape=(batch_size, num_patches_per_image)), axis=-1)
        # split indices
        mask_indices = indices[:, :self.num_masked_patches]
        unmask_indices = indices[:, self.num_masked_patches:]
        # masked patches
        masked_patches = gather(patches, mask_indices, axis=1, batch_dims=1)
        # unmasked patches
        unmasked_patches = gather(patches, unmask_indices, axis=1, batch_dims=1)
        # repeat mask token for each masked patch ...
        mask_tokens = repeat(self.mask, repeats = self.num_masked_patches, axis = 0)
        # ... and for each image
        mask_tokens = repeat(mask_tokens[newaxis, ...], repeats=batch_size, axis=0)
        # masked images
        masked_images = concat([mask_tokens, unmasked_patches], axis = 1)
        masked_images = gather(masked_images, argsort(indices), axis=1, batch_dims=1)

        if self.return_indices:
            return masked_images, masked_patches, mask_indices
        return masked_images, masked_patches

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_masked_patches" : self.num_masked_patches,
            "mask" : self.mask,
            "return_indices" : self.return_indices 
        })
        return config