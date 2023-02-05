import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
                
#########################################################################################
# Clean mask, the mask must be either background or given a type of a classes
# mask for header = 1
# mask for question = 2
# mask for other = 3
#########################################################################################

def clean_mask(sub_folder):
    mask_dir = os.path.join(sub_folder, "masks")
    cleaned_dir = os.path.join(sub_folder, "cleaned_masks")
    os.makedirs(cleaned_dir, exist_ok=True)

    for subfolder_mask in os.listdir(mask_dir):
        for file in os.listdir(os.path.join(mask_dir, subfolder_mask)):
            if file[-4:] == ".png": 
                if "header" in file:
                    header_mask = cv2.imread(os.path.join(mask_dir, subfolder_mask, file))
                    binary_h_mask = np.where(header_mask > 0, 1, header_mask)
                    cleaned_path = os.path.join(cleaned_dir, subfolder_mask)
                    os.makedirs(cleaned_path, exist_ok=True)
                    #binary_h_mask = cv2.cvtColor(binary_h_mask, cv2.COLOR_BGR2GRAY)
                    cv2.imwrite(os.path.join(cleaned_path, file), binary_h_mask)
                elif "question" in file:
                    question_mask = cv2.imread(os.path.join(mask_dir, subfolder_mask, file))
                    binary_q_mask =  np.where(question_mask > 0, 2, question_mask)
                    cleaned_path = os.path.join(cleaned_dir, subfolder_mask)
                    os.makedirs(cleaned_path, exist_ok=True)
                    #binary_q_mask = cv2.cvtColor(binary_q_mask, cv2.COLOR_BGR2GRAY)
                    cv2.imwrite(os.path.join(cleaned_path, file), binary_q_mask)
                elif "other" in file:
                    other_mask = cv2.imread(os.path.join(mask_dir, subfolder_mask, file))
                    binary_o_mask =  np.where(other_mask > 0, 3, other_mask)
                    cleaned_path = os.path.join(cleaned_dir, subfolder_mask) 
                    os.makedirs(cleaned_path, exist_ok=True)
                    #binary_o_mask = cv2.cvtColor(binary_o_mask, cv2.COLOR_BGR2GRAY)
                    cv2.imwrite(os.path.join(cleaned_path, file), binary_o_mask)


data_dir = "./UNET/UNet_data/funsd" 

size_512_dir = os.path.join(data_dir, "size_512")

for sub_folder in os.listdir(size_512_dir):
    if sub_folder in ["train", "val", "test"]:
        sub_dir = os.path.join(size_512_dir, sub_folder)
        clean_mask(sub_dir)

# test mask dimenstion
#mask_file = "task-29-annotation-16-by-1-tag-header-0.png"
#mask = cv2.imread(os.path.join(data_dir, "size_512", "train", "cleaned_masks", "mask_83996357", mask_file))
#gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
#print(gray.shape)

print("The pixel be either BG or one of the labels: done!")
print("Images saved in folder cleaned_masks under size_512 directory.")
print("\n")

#########################################################################################
# Combine all masks that belongs to an image into one mask
#########################################################################################       
from skimage import io
from skimage.transform import rescale, resize

img_height = 512 
img_width = 512

def mutiple_masks_to_single(sub_folder, img_height, img_width):
    cleaned_dir = os.path.join(sub_folder, "cleaned_masks")
    single_mask_dir = os.path.join(sub_folder, "single_mask")
    os.makedirs(single_mask_dir, exist_ok=True)


    #Mask_train = np.zeros((img_height, img_width), dtype=np.uint8)
    
    for subfolder_mask in os.listdir(cleaned_dir):
        mask_name =  subfolder_mask.split("mask_")[1]+".png"
        mask = np.zeros((img_height, img_width, 3), dtype=np.uint8)
        for mask_file in os.listdir(os.path.join(cleaned_dir, subfolder_mask)):
            mask_file_path = os.path.join(cleaned_dir, subfolder_mask, mask_file)
            _mask = io.imread(mask_file_path)
            _mask =  resize(_mask, (img_height, img_width), mode="constant", preserve_range=True)
            mask = np.maximum(mask, _mask)
        
        cv2.imwrite(os.path.join(single_mask_dir, mask_name), mask)
            


data_dir = "./UNET/UNet_data/funsd" 

size_512_dir = os.path.join(data_dir, "size_512")

for sub_folder in os.listdir(size_512_dir):
    if sub_folder in ["train", "val", "test"]:
        sub_dir = os.path.join(size_512_dir, sub_folder)
        mutiple_masks_to_single(sub_dir, img_height, img_width)


# test mask dimenstion

#single_mask = cv2.imread(os.path.join(data_dir, "size_512", "train", "single_mask", "83996357.png"))
#gray = cv2.cvtColor(single_mask, cv2.COLOR_BGR2GRAY)
#print(gray.shape)
#plt.imshow(gray)

print("Images of lebels combined into one Label img:Done!")
print("Images saved in folder single_mask under size_512 directory.")
