# This script aims to create augmented images from one image to create a larger dataset for our cnn model
# The augmentation this script will perform on each object is 
# orig_img,grayscaled_image,random_rotation_transformation_45_image,random_rotation_transformation_65_image,random_rotation_transformation_85_image,gausian_blurred_image_13_image,gausian_blurred_image_56_image,gausian_image_3,gausian_image_6,gausian_image_9,colour_jitter_image_1,colour_jitter_image_2,colour_jitter_image_3

#call the function creating file with augmented image give path of dataset and path of folder where you want the augmented images to be stored

import torch
from PIL import Image
from pathlib import Path
import torchvision.transforms as T
import os

# Define transformations

# Grayscale transformation
grayscale_transform = T.Grayscale(3)

# Random rotation
random_rotation_transformation_45 = T.RandomRotation(45)
random_rotation_transformation_85 = T.RandomRotation(85)
random_rotation_transformation_65 = T.RandomRotation(65)

# Gaussian Blur
gausian_blur_transformation_13 = T.GaussianBlur(kernel_size=(7, 13), sigma=(6, 9))
gausian_blur_transformation_56 = T.GaussianBlur(kernel_size=(7, 13), sigma=(5, 8))

# Gaussian Noise
def add_noise(input_image, noise_factor=0.3):
    inputs = T.ToTensor()(input_image)
    noisy = inputs + torch.rand_like(inputs) * noise_factor
    noisy = torch.clip(noisy, 0, 1)
    output_image = T.ToPILImage()(noisy)
    return output_image

# Color Jitter
colour_jitter_transformation_1 = T.ColorJitter(brightness=(0.5, 1.5), contrast=3, saturation=(0.3, 1.5), hue=(-0.1, 0.1))
colour_jitter_transformation_2 = T.ColorJitter(brightness=0.7, contrast=6, saturation=0.9, hue=(-0.1, 0.1))
colour_jitter_transformation_3 = T.ColorJitter(brightness=(0.5, 1.5), contrast=2, saturation=1.4, hue=(-0.1, 0.5))

# Main function to create augmented images
def augment_image(img_path):
    orig_img = Image.open(img_path).convert("RGB")

    # Apply transformations
    grayscaled_image = grayscale_transform(orig_img)
    random_rotation_transformation_45_image = random_rotation_transformation_45(orig_img)
    random_rotation_transformation_85_image = random_rotation_transformation_85(orig_img)
    random_rotation_transformation_65_image = random_rotation_transformation_65(orig_img)
    gausian_blurred_image_13_image = gausian_blur_transformation_13(orig_img)
    gausian_blurred_image_56_image = gausian_blur_transformation_56(orig_img)
    gausian_image_3 = add_noise(orig_img)
    gausian_image_6 = add_noise(orig_img, 0.6)
    gausian_image_9 = add_noise(orig_img, 0.9)
    colour_jitter_image_1 = colour_jitter_transformation_1(orig_img)
    colour_jitter_image_2 = colour_jitter_transformation_2(orig_img)
    colour_jitter_image_3 = colour_jitter_transformation_3(orig_img)

    return [
        orig_img,
        grayscaled_image,
        random_rotation_transformation_45_image,
        random_rotation_transformation_65_image,
        random_rotation_transformation_85_image,
        gausian_blurred_image_13_image,
        gausian_blurred_image_56_image,
        gausian_image_3,
        gausian_image_6,
        gausian_image_9,
        colour_jitter_image_1,
        colour_jitter_image_2,
        colour_jitter_image_3
    ]

def create_augmented_images_dataset(master_dataset_path, augmented_images_path):
    master_dataset_folder = Path(master_dataset_path)
    augmented_images_folder = Path(augmented_images_path)
    
    if not augmented_images_folder.exists():
        augmented_images_folder.mkdir(parents=True, exist_ok=True)

    for element in master_dataset_folder.iterdir():
        if element.is_file() and element.suffix.lower() == ".jpeg":
            print(f"Processing image: {element}")
            subdir = augmented_images_folder / element.stem
            subdir.mkdir(parents=True, exist_ok=True)
            
            augmented_images = augment_image(element)
                    
            for idx, augmented_image in enumerate(augmented_images):
                save_path = augmented_images_folder / f"{element.stem}_aug_{idx}.jpeg"
                print(f"Saving augmented image: {save_path}")
                augmented_image.save(save_path)

# Paths to your datasets
master_dataset = "D:\hasti\Object_detection\MD_images"  # Path to master dataset
augmented_dataset = "D:\hasti\Object_detection\AUG_images"  # Path to save augmented images

# Create augmented dataset
create_augmented_images_dataset(master_dataset, augmented_dataset)

