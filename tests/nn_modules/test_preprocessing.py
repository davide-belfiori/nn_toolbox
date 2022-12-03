from nn_modules.preprocessing import Patches, ReshapePatches, MaskPatches
from tests.mock import MockInput
from tensorflow import zeros

def test_patches() -> None:
    images = MockInput()
    patch_size = 2
    num_patches_per_image = (images.image_width * images.image_height) // (patch_size**2)
    patch_dim = patch_size * patch_size * images.channels

    patches = Patches(patch_size=patch_size, stride=patch_size)(images.ones())

    assert patches.shape == (images.batch_size, num_patches_per_image, patch_dim)

def test_reshape_patches() -> None:
    images = MockInput()
    patch_size = 2
    num_patches_per_image = (images.image_width * images.image_height) // (patch_size**2)

    patches = Patches(patch_size=patch_size, stride=patch_size)(images.ones())
    reshape = ReshapePatches(patch_size=patch_size)(patches)

    assert reshape.shape == (images.batch_size, patch_size, patch_size, num_patches_per_image * images.channels)

def test_mask_patches() -> None:
    mock = MockInput()
    images = mock.ones()
    patch_size = 2
    patch_dim = patch_size * patch_size * mock.channels
    num_masked_patches = 3
    num_patches_per_image = (mock.image_width * mock.image_height) // (patch_size**2)
    mask = zeros(shape=[1, patch_dim])

    patches = Patches(patch_size=patch_size, stride=patch_size)(images)
    masked_images, masked_patches, masked_indices = MaskPatches(num_masked_patches=num_masked_patches, 
                                                                mask=mask, 
                                                                indices=None, 
                                                                return_indices=True)(patches)

    assert masked_images.shape == (mock.batch_size, num_patches_per_image, patch_dim) and \
           masked_patches.shape == (mock.batch_size, num_masked_patches, patch_dim) and \
           masked_indices.shape == (mock.batch_size, num_masked_patches)
        
    for i in range(mock.batch_size):
        for idx in masked_indices[i]:
            assert (masked_images[i][idx] == mask).numpy().all()
