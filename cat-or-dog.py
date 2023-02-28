'''
Cat or Dog?
CPSC 323-01
Sam Berkson

In this project, I will endeavor to create a convolutional neural network to classify images of cats and dogs.

I removed the code out of a ipynb and have gained permission to use it 

'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import cv2 as ocv
import random
#import PIL
import pathlib
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


# Data load
#/home/cbarker4/Pictures/Demo-cats/train/
# data_dir = pathlib.Path('/home/cbarker4/Pictures/Demo-cats/train')
start = '/home/cbarker4/Pictures/Demo-cats/train/'
pics = os.listdir('/home/cbarker4/Pictures/Demo-cats/train')
cats =[]
dogs = []
for path in pics:
    if "cat" in path:
        cats.append(start + path)
    else:
        dogs.append(start + path)
# dogs = list(data_dir.glob('dog/*'))
# cats = list(data_dir.glob('cat/*'))
# image_count = len(list(data_dir.glob('*/*.jpg')))


#PIL.Image.open(str(cats[0]))


# PIL.Image.open(str(dogs[0]))


pet_images_dict = {'cat': list(cats), 'dog': list(dogs)}
pet_labels_dict = {'cat': 0, 'dog': 1}


IMAGE_WIDTH=128
IMAGE_HEIGHT=128
X, Y = [], []

for pet_name, images in pet_images_dict.items():
    for image in images:
        img = ocv.imread(str(image))
        if isinstance(img,type(None)): 
            #print('image not found')
            continue
            
        elif ((img.shape[0] >= IMAGE_HEIGHT) and  (img.shape[1] >=IMAGE_WIDTH)):
            resized_img = ocv.resize(img,(IMAGE_WIDTH,IMAGE_HEIGHT))
            X.append(resized_img)
            Y.append(pet_labels_dict[pet_name])
        else:
            #print("Invalid Image")
            continue

X = np.array(X)
Y = np.array(Y)


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.25 ,random_state=0)


print('X_train shape: ', X_train.shape)
print('y_train shape: ', Y_train.shape)
print('X_test shape: ', X_test.shape)
print('y_test shape: ', Y_test.shape)


from keras.utils.vis_utils import plot_model

IMAGE_CHANNELS = 3

model = tf.keras.models.Sequential([
    # Convolutional network
    tf.keras.layers.Conv2D(32, (5, 5), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    # Fully connected feed-forward network
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid') # Sigmoid for multi-class classification
])

# plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)


callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor = 'val_accuracy',
        patience = 5,
        verbose = 1,
        min_delta = 0,
        mode = 'max',
        baseline = None,
        restore_best_weights = True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        os.path.join("/home/cbarker4/Documents/tfjs-tflite-model-runner-demo/quickNN/", 'cat', "{epoch:02d}-{val_loss:.2f}.hdf5"),
        monitor = 'val_accuracy',
        verbose = 1,
        save_best_only = True,
        save_weights_only = False,
        mode = 'max',
        save_freq = 'epoch',
        options = None,
        initial_value_threshold = None
    )   
]


model.compile(optimizer = 'adam',
              loss = 'binary_crossentropy', 
              metrics = ['accuracy'])

# Train the model
history = model.fit(X_train, Y_train, epochs = 1, 
                                      validation_split = 0.2, 
                                      callbacks = callbacks)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open('modelcatDog.tflite', 'wb') as f:
  f.write(tflite_model)



model.save("CatOrDog")
y_pred = model.predict(X_test)



from sklearn.metrics import confusion_matrix
import seaborn as sns

# Create confusion matrix
cm = confusion_matrix(Y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize = (10, 7))
sns.heatmap(cm, annot = True, fmt = 'd')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()