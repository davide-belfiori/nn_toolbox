import numpy as np
from tensorflow import image, reshape, gather, argsort, random, zeros_like, concat, transpose

def mask_patches(images: np.ndarray, patch_size: int, num_masked_patches = 1):
    # dimensioni input
    batch_size, image_height, image_width, channels = images.shape
    # numero di patch per immagine
    num_patches_per_image = image_height * image_width // patch_size ** 2
    # estrai le patch
    patches = image.extract_patches(images=images,
                                sizes=[1, patch_size, patch_size, 1],
                                strides=[1, patch_size, patch_size, 1],
                                rates=[1,1,1,1],
                                padding="VALID")
    # reshape
    patches = reshape(patches, shape=(batch_size, num_patches_per_image, patch_size * patch_size * channels))
    # calcolo degli indici
    indices = argsort(random.uniform(shape=(batch_size, num_patches_per_image)), axis=-1)
    # split degli indici 
    mask_indices = indices[:, :num_masked_patches]
    unmask_indices = indices[:, num_masked_patches:]
    # patch mascherate
    masked_patches = gather(patches, mask_indices, axis=1, batch_dims=1)
    # patch in chiaro
    unmasked_patches = gather(patches, unmask_indices, axis=1, batch_dims=1)
    # maschera
    mask = zeros_like(masked_patches)
    # immagini mascherate
    masked_images = concat([mask, unmasked_patches], axis = 1)
    masked_images = gather(masked_images, argsort(indices), axis=1, batch_dims=1)
    # reshape immagini
    masked_images = reshape(masked_images, shape = (batch_size, num_patches_per_image, patch_size, patch_size, channels))
    masked_images = transpose(masked_images, [0,2,3,4,1])
    masked_images = reshape(masked_images, shape = (batch_size, patch_size, patch_size, num_patches_per_image * channels))
    # reshape patch mascherate
    masked_patches = reshape(masked_patches, shape=(batch_size, num_masked_patches, patch_size, patch_size, channels))
    masked_patches = transpose(masked_patches, [0,2,3,4,1])
    masked_patches = reshape(masked_patches, shape = (batch_size, patch_size, patch_size, num_masked_patches * channels))

    return masked_images, masked_patches, mask_indices

def patches_to_image(patches: np.ndarray, image_shape: tuple):

    image_h, image_w, image_d = image_shape
    n_images, patch_h, patch_w, patch_d = patches.shape
    patches_per_channel = patch_d // image_d
    ...
