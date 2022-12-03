from utils.image import mask_patches
import tensorflow as tf

def test_mask_patch() -> None:

    image_shape = (128, 128, 3)
    batch_size = 8
    patch_size = 16
    num_masked_patches = 1

    images = tf.ones(shape = (batch_size, ) + image_shape)

    masked_images, masked_patches, masked_indices = mask_patches(images, patch_size, num_masked_patches)

    num_patches_per_image = (image_shape[0] * image_shape[1]) // (patch_size ** 2)

    assert masked_images.shape == (batch_size, patch_size, patch_size, num_patches_per_image * image_shape[-1]) and \
           masked_patches.shape == (batch_size, patch_size, patch_size, image_shape[-1])

    mask = tf.zeros(shape = (patch_size, patch_size))
    for i in range(batch_size):
        image = masked_images[i]
        assert (image[:,:,masked_indices[i, 0]] == mask).numpy().all() and \
               (image[:,:,masked_indices[i, 0] + num_patches_per_image] == mask).numpy().all() and \
               (image[:,:,masked_indices[i, 0] + 2 * num_patches_per_image] == mask).numpy().all()
