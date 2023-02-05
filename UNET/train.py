from model_unet import UNetModel
from keras.utils import normalize
import os
import cv2
import numpy as np
from keras.utils import to_categorical
import tensorflow as tf
from sklearn.utils import class_weight
from sklearn.preprocessing import LabelEncoder

n_classes = 4 

log_dir = "./UNET/logs"

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

parent_dir ="./UNET"
data_dir = "./UNET/UNet_data/funsd" 

#  load X_train, mask_train
############################################################################  
#  Train Data 
############################################################################
train_images = []
train_masks = []

train_path = os.path.join(data_dir, "size_512", "train")

for img_file in os.listdir(os.path.join(train_path, "images")):
    img_path = os.path.join(train_path, "images", img_file)
    mask_path = os.path.join(train_path, "single_mask", img_file)
    
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) 
    train_images.append(img)
    train_masks.append(mask)

train_images = np.array(train_images)
train_masks = np.array(train_masks)

train_images = normalize(train_images, axis=1)
# Add one channel dim for mask images
train_masks = np.expand_dims(train_masks, axis=3)

############################################################################  
#  Val Data 
############################################################################

val_images = []
val_masks = []

val_path = os.path.join(data_dir, "size_512", "val")

for img_file in os.listdir(os.path.join(val_path, "images")):
    img_path = os.path.join(val_path, "images", img_file)
    mask_path = os.path.join(val_path, "single_mask", img_file)
    
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) 
    val_images.append(img)
    val_masks.append(mask)

val_images = np.array(val_images)
val_masks = np.array(val_masks)


val_images = normalize(val_images, axis=1)
# Add one dim for the channel
val_masks = np.expand_dims(val_masks, axis=3)

############################################################################  
#  Other manipulation of data 
############################################################################

# Do category for each class
train_masks_cat = to_categorical(train_masks, num_classes=n_classes)
y_train_cat = train_masks_cat.reshape((train_masks.shape[0], train_masks.shape[1], train_masks.shape[2], n_classes))  

val_masks_cat = to_categorical(val_masks, num_classes=n_classes)
y_val_cat = val_masks_cat.reshape((val_masks.shape[0], val_masks.shape[1], val_masks.shape[2], n_classes))


labelencoder = LabelEncoder()
train_masks_reshaped = train_masks.reshape(-1,1)
train_masks_reshaped_encoded = labelencoder.fit_transform(train_masks_reshaped)

IMG_HEIGHT = train_images.shape[1]    # 512
IMG_WIDTH = train_images.shape[2]     # 512
IMG_CHANNELS = train_images.shape[3]  # 3


def get_model():
    return UNetModel(n_classes= n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS)

model1 = get_model()
model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model1.summary()

# save the model with highest accuracy
best_path = os.path.join(parent_dir, "high_acc_model.hdf5")
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath = best_path,
    monitor = 'val_accuracy',
    mode='max',
    save_best_only=True)


callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir= log_dir, histogram_freq=0, write_graph=True, write_images=False),
    model_checkpoint_callback,
]

#If starting with pre-trained weights. 
#model1.load_weights("./some_model_.hdf5")

history = model1.fit(train_images, y_train_cat, 
                    batch_size = 1, 
                    verbose=1, 
                    epochs=2, 
                    callbacks= callbacks,
                    validation_data=(val_images, y_val_cat), 
                    #class_weight=class_weight,
                    shuffle=False)

path_to_save = os.path.join(parent_dir, "unet_model_2_epochs.hdf5")
model1.save(path_to_save)

_, acc =  model1.evaluate(val_images, y_val_cat)
print("Accuracy is = ", (acc * 100.0), "%")

#plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
