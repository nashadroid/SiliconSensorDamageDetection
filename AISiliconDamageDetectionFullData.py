from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import os
import random
import tensorflow as tf
from tensorflow import keras
from skimage.draw import line_aa

filepath="/fs/scratch/PAS0035/nashad/SensorImages/"
files = os.listdir(filepath)

# TODO: I will come back to this and add different types of damage
def addDamage(img):
    rr, cc, val = line_aa(np.random.randint(0,img.shape[0]), np.random.randint(0,img.shape[1]), np.random.randint(0,img.shape[0]), np.random.randint(0,img.shape[1]))
    img[rr, cc] = val * np.random.randint(0,255)
    return img


print("Loading in files")
for file in files:
    print(filepath+file)
    im = Image.open(filepath+file)
    p = np.array(im)[:,:,0]

    # Split the image into 64 even pieces
    split = np.array([np.vsplit(x, 8) for x in np.hsplit(p, 8)])

    # Reshape to create different images exist across first dimension of array
    split = np.reshape(split, (1, -1, split.shape[2], split.shape[3]))
    split = np.squeeze(split,0)

    if splitImages:
        splitImages = np.concatenate((splitImages, split))
    else:
        splitImages = np.copy(split)


print("Adding Damage")

# Create a undamaged and a damaged version
undamagedImages = np.copy(splitImages)
damagedImages = np.array([addDamage(img) for img in np.copy(splitImages)])

fullData = np.concatenate((undamagedImages, damagedImages))

# https://numpy.org/doc/stable/reference/generated/numpy.full.html
fullDataLabels = np.concatenate((np.full((len(undamagedImages),2),[0,1]),
                                np.full((len(damagedImages),2),[1,0])))

# Shuffle the two lists around:
# https://stackoverflow.com/questions/23289547/shuffle-two-list-at-once-with-same-order
# TODO: See if we can do that thing in the first assignment
c = list(zip(fullData, fullDataLabels))
random.shuffle(c)

fullData, fullDataLabels = zip(*c)
fullData=np.array(fullData)
fullDataLabels=np.array(fullDataLabels)

# Split into training and test splits
splitIndex = int(len(fullDataLabels)/10)

testData = fullData[:splitIndex]
trainData = fullData[splitIndex:]
testDataLabels = fullDataLabels[:splitIndex]
trainDataLabels = fullDataLabels[splitIndex:]


# Actually build the AI (ugh)
trainData = trainData.reshape((trainData.shape[0],128,153,1))
testData = testData.reshape((testData.shape[0],128,153,1))

# From: cnn_intro
# https://stackoverflow.com/questions/61742556/valueerror-shapes-none-1-and-none-2-are-incompatible

cnn_network = keras.models.Sequential()
#
# First convolutional layer
cnn_network.add(keras.layers.Conv2D(30,(5,5),activation='relu',input_shape=(128,153,1)))
# Pool
cnn_network.add(keras.layers.MaxPooling2D((2,2)))
#
# Second convolutional layer
cnn_network.add(keras.layers.Conv2D(25,(5,5),activation='relu'))
# Pool
cnn_network.add(keras.layers.MaxPooling2D((2,2)))
#
# Connect to a dense output layer - just like an FCN
cnn_network.add(keras.layers.Flatten())
cnn_network.add(keras.layers.Dense(64,activation='relu'))
cnn_network.add(keras.layers.Dense(2,activation='sigmoid'))
#
# Compile
cnn_network.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
#
# Fit/save/print summary
print("Fitting..")

callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=10),
             keras.callbacks.ModelCheckpoint(filepath='best_model.h5',
                                            monitor='val_loss',
                                            save_best_only=True)]

history = cnn_network.fit(trainData,trainDataLabels,epochs=10,batch_size=256,validation_data=(testData,testDataLabels), callbacks=callbacks)
cnn_network.save('fully_trained_model_cnn.h5')
print(cnn_network.summary())
