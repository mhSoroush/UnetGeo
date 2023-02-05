import os
import cv2

data_dir = "./UNET/UNet_data/funsd" 

# Note: resize with cv2 because it keep the channel = 3

#########################################################################################
# Loop over each examples and resize them to (512, 512)
#########################################################################################

original_dir = os.path.join(data_dir, "original")
size_512_dir = os.path.join(data_dir, "size_512")
for sub_folder in os.listdir(original_dir):
    #  sub_dir are train, val and test
    if sub_folder in ["train", "val", "test"]:
        sub_orig_dir = os.path.join(original_dir, sub_folder, "images")
        sub_512_dir = os.path.join(size_512_dir, sub_folder, "images")
        os.makedirs(sub_512_dir, exist_ok=True) # Create path
        for file in os.listdir(sub_orig_dir):
            if file[-4:] == ".png":
                
                img = cv2.imread(os.path.join(sub_orig_dir, file))
                resized_img = cv2.resize(img, (512, 512), interpolation = cv2.INTER_AREA)
                cv2.imwrite(os.path.join(sub_512_dir, file), resized_img)  
                
print("Images are resized: done!")
print("Images saved in folder images under size_512 directory.")
                
