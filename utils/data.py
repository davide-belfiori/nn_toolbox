from keras.utils import Sequence
from keras.preprocessing.image import ImageDataGenerator
from utils.image import mask_patches

class MaskedImageGenerator(Sequence):

    def __init__(self, patch_size: int, 
                       data_path, 
                       image_generator: ImageDataGenerator, 
                       target_size = (128, 128),
                       color_mode = "rgb",
                       shuffle = True, 
                       batch_size: int = 32,
                       seed = None,
                       subset = None,
                       num_masked_patches: int = 1):
        super().__init__()
        self.patch_size = patch_size
        self.num_masked_patches = num_masked_patches
        self.data_path = data_path
        self.target_size = target_size
        self.color_mode = color_mode
        self.shuffle = shuffle
        self.seed = seed
        self.subset = subset
        self.batch_size = batch_size
        self.generator = image_generator.flow_from_directory(directory=self.data_path,
                                                             target_size=self.target_size,
                                                             color_mode=self.color_mode,
                                                             class_mode=None,
                                                             batch_size = self.batch_size,
                                                             shuffle = self.shuffle,
                                                             seed = self.seed,
                                                             subset = self.subset)
                                                             
    def __getitem__(self, index):
        images = self.generator.__getitem__(index)
        x, y,_ = mask_patches(images=images, patch_size=self.patch_size, num_masked_patches=self.num_masked_patches)
        return x, y

    def __len__(self):
        return self.generator.__len__()
