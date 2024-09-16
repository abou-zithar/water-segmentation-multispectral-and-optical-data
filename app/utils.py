import os
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import tensorflow as tf
import tifffile as tiff


def normalize(bands):
    # Min-Max normalization
    bands_min = bands.min(axis=(1,2), keepdims=True)
    bands_max = bands.max(axis=(1,2), keepdims=True)
    bands_normalized = (bands - bands_min) / (bands_max - bands_min)
    return bands_normalized


# Example: Loading a Sentinel-2 image
def load_multispectral_image(file_path):
    try:
        with tiff.TiffFile(file_path) as tif:
            bands = tif.asarray()  # Reads the image data into a NumPy array
            profile = {
                "shape": bands.shape,
                "dtype": bands.dtype,
                "num_pages": len(tif.pages),  # Number of bands/pages
                "description": tif.pages[0].description if tif.pages else None
            }

         # Resize the image from (128, 128) to (32, 32) for each of the 13 channels
        
        return bands, profile
    except Exception as e:
        print(f"Error loading image: {e}")
        return None, None



def calculate_ndwi(green_band, nir_band):
    ndwi = (green_band - nir_band) / (green_band + nir_band)
    return ndwi


def append_new_channel(image):
    new_channel = calculate_ndwi(image[:,:,2], image[:,:,7])
    
    new_channel = np.expand_dims(new_channel, axis=-1)  # Shape becomes (height, width, 1)

    # Step 4: Append the new channel to the existing image
    img_with_new_channel = np.concatenate((image, new_channel), axis=-1)
    return img_with_new_channel


def open_image(image_path):
    multispectral_bands, _ = load_multispectral_image(image_path)

    multispectral_normalized = normalize(multispectral_bands)
    image  = append_new_channel(multispectral_normalized)
    img_resized = np.zeros((32, 32, 13))  # Create an empty array to store resized channels
    for i in range(13):
        # Resize each channel individually using PIL
        channel = Image.fromarray(image[..., i])
        channel_resized = channel.resize((32, 32), Image.BILINEAR)  # You can use other interpolation methods
        img_resized[..., i] = np.array(channel_resized)
    img_resized = np.expand_dims(img_resized, axis=0)
    return img_resized



# Function to postprocess the model's prediction
def postprocess_result(segmentation):
    segmentation = (segmentation < 0.5).astype(np.uint8)  # Apply threshold
    segmentation = np.squeeze(segmentation)  # Remove batch dimension
    segmentation_image = Image.fromarray(segmentation * 255)  # Convert to image
    return segmentation_image

def create_rgb_composite(bands, red, green, blue):
    rgb = np.stack([bands[:,:,:,red], bands[:,:,:,green], bands[:,:,:,blue]], axis=-1)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())  # Normalize to [0,1]
    return rgb


def create_patches(data, patch_size=32, overlap=0):
    patches = []
    
    step = patch_size - overlap
    
    for i in range(0, data.shape[0] - patch_size + 1, step):
        
        for j in range(0, data.shape[1] - patch_size + 1, step):

            # Extracting the patch for data
            patch = data[i:i + patch_size, j:j + patch_size, :]

            # Append patches to the respective lists
            patches.append(patch)

    # Convert lists to numpy arrays for easier handling
    return np.array(patches)