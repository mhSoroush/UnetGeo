

from keras.utils import normalize
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from model_unet import UNetModel




#from skimage.segmentation import mark_boundaries
from skimage.measure import label, regionprops

n_classes = 4 


data_dir = "./UNET/UNet_data/funsd"
trained_model_dir = "./UNET/trained_model"
############################################################################  
#  Test Data 
############################################################################
test_images = []
test_masks = []

test_path = os.path.join(data_dir, "size_512", "test")

for img_file in os.listdir(os.path.join(test_path, "images")):
    img_path = os.path.join(test_path, "images", img_file)
    mask_path = os.path.join(test_path, "single_mask", img_file)
    
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) 
    test_images.append(img)
    test_masks.append(mask)

test_images = np.array(test_images)
test_masks = np.array(test_masks)

test_images = normalize(test_images, axis=1)
# Add one channel dim for mask images
test_masks = np.expand_dims(test_masks, axis=3)

##################################################
IMG_HEIGHT = test_images.shape[1] # 512
IMG_WIDTH = test_images.shape[1] # 512
IMG_CHANNELS = test_images.shape[3] # 3


def get_model():
    return UNetModel(n_classes= n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS)

model1 = get_model()
model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#model1.summary()

traind_path = os.path.join(trained_model_dir, "Best_5and3kernals_1000_epochs_.hdf5")
model1.load_weights(traind_path)

test_image = test_images[0]
test_mask = test_masks[0]
test_image = np.expand_dims(test_image, axis=0) # (1, 512, 512, 3)
test_mask = np.expand_dims(test_mask, axis =0) # (1, 512, 512, 1)

#ground_truth = train_masks[0] 
#input_img = np.expand_dims(image1, 0) # (1, 512, 512, 3)

prediction = model1.predict(test_image)
predicted_mask = np.argmax(prediction, axis=3)[0,:,:]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (15, 5))

ax1.set_title("Test image")
ax2.set_title("Test label")
ax3.set_title("Pred test label")
ax1.imshow(test_image[0,:,:,0], cmap='gray')
ax2.imshow(test_mask[0,:,:,0], cmap='jet')
ax3.imshow(predicted_mask, cmap='jet')
plt.show()

# IOU
##################################################

#Using built in keras function
from keras.metrics import MeanIoU
n_classes = 4
IOU_keras = MeanIoU(num_classes=n_classes)  
IOU_keras.update_state(test_mask[:,:,:,0], predicted_mask)
print("Mean IoU =", IOU_keras.result().numpy())

# #To calculate I0U for each class...
values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
print(values)
class1_IoU = values[0,0]/(values[0,0] + values[0,1] + values[0,2] + values[0,3] + values[1,0]+ values[2,0]+ values[3,0])
class2_IoU = values[1,1]/(values[1,1] + values[1,0] + values[1,2] + values[1,3] + values[0,1]+ values[2,1]+ values[3,1])
class3_IoU = values[2,2]/(values[2,2] + values[2,0] + values[2,1] + values[2,3] + values[0,2]+ values[1,2]+ values[3,2])
class4_IoU = values[3,3]/(values[3,3] + values[3,0] + values[3,1] + values[3,2] + values[0,3]+ values[1,3]+ values[2,3])

print("IoU for class1 is: ", class1_IoU)
print("IoU for class2 is: ", class2_IoU)
print("IoU for class3 is: ", class3_IoU)
print("IoU for class4 is: ", class4_IoU)
