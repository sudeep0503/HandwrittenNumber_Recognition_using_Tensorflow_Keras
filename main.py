import os

import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import cv2
from keras.datasets import mnist

### Get the data and preprocessing

(x_train , y_train) , (x_test , y_test) = mnist.load_data()

x_train.shape , y_train.shape , x_test.shape , y_test.shape

plt.imshow(x_train[0])

plt.imshow(x_train[0] , cmap='binary')

def plot_input_image(i):
  plt.imshow(x_train[i] , cmap='binary')
  plt.title(y_train[i])
  plt.show()


for i in range(10):
  plot_input_image(i)


#  **Preprocess the data**
# Normalizing the image to [0,1] range


x_train = x_train.astype(np.float32) / 255
x_test = x_test.astype(np.float32) / 255


# Reshape / expand the dimension of images to (28,28,1)

x_train = np.expand_dims(x_train , -1)
x_test = np.expand_dims(x_test , -1)


# As y_train and y_test various from 0 to 9. We need to change its type to categorical data.
# We will be using One Hot Encoder here.

y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)


x_train.shape        
# become in 3D now



# **Build the Model**
# And for that we need to import couple of libraries and functions.

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout

# Droupout is to prevent OVERFITTING while trainning the model.

model = Sequential()

model.add(Conv2D(32 , (3,3) , input_shape = (28,28,1) , activation= 'relu') )       # 32 is the number of filter of 3x3
model.add(MaxPool2D ( (2,2) ) )                                                     # MaxPooling filter will be of 2x2 size (take the highest intensity numbers)


model.add(Conv2D(64 , (3,3) , activation= 'relu') )
model.add(MaxPool2D ( (2,2) ) )

model.add(Flatten())

model.add(Dropout(0.25))                                                            # As we take subset of neurons to train, 0.25 means 25% of total neurons we'll take in
every layer
model.add(Dense( 10 , activation= 'softmax'))

model.summary()

model.compile(optimizer='adam' , loss=keras.losses.CategoricalCrossentropy() , metrics=['accuracy'])

# Callbacks

from keras.callbacks import EarlyStopping , ModelCheckpoint

es = EarlyStopping(monitor='val_acc' , min_delta = 0.01 , patience= 4, verbose=1)

mc = ModelCheckpoint("./bestmodel.h5" , monitor="val_acc" , verbose = 1 , save_best_only = True)

cb = [es , mc]

his = model.fit(x_train, y_train, epochs=50, validation_split=0.3)

# No need to take care of over epochs because of EarlyStopping function, it will take care of it

his = model.fit(x_train , y_train , epochs=5 , validation_split=0.3 , callbacks=cb)

test_loss , test_acc = model.evaluate(x_test , y_test)

print('Test accuracy:' , test_acc)
print('Test loss:' , test_loss)

model.save('handwritten.model')

model = tf.keras.models.load_model('handwritten.model')

image_number = 0
while os.path.isfile(f"Digits/num{image_number}.png"):
    try:
        img = cv2.imread(f"Digits/num{image_number}.png")[:, :, 0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f"This digit is probably a {np.argmax(prediction)}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except:
        print("Error!!!")

    finally:
        image_number += 1
