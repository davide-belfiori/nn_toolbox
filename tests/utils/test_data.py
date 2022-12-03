from utils.data import MaskedImageGenerator, ImageDataGenerator

def test_MaskedImageGenerator() -> None:

    patch_size = 16
    data_path = "D:\\Workspace\\datasets\\mvtec_anomaly_detection\\bottle\\train"
    target_size = (128, 128)
    color_mode = "rgb"
    batch_size = 8
    num_masked_patches = 1

    image_generator = ImageDataGenerator(rescale=1./255, data_format="channels_last")

    mig = MaskedImageGenerator(patch_size = patch_size, 
                               data_path = data_path,
                               image_generator = image_generator,
                               target_size = target_size,
                               color_mode = color_mode,
                               shuffle = True,
                               batch_size = batch_size,
                               seed = 0,
                               subset = None,
                               num_masked_patches = num_masked_patches)

    x, y = mig.__getitem__(0)

    num_patches_per_image = (target_size[0] * target_size[1]) // (patch_size ** 2)
    assert x.shape == (batch_size, patch_size, patch_size, num_patches_per_image * 3)
    assert y.shape == (batch_size, patch_size, patch_size, num_masked_patches * 3)