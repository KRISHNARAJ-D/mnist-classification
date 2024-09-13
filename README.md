# Convolutional Deep Neural Network for Digit Classification

## AIM

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset
Digit classification and to verify the response for scanned handwritten images.

The MNIST dataset is a collection of handwritten digits. The task is to classify a given image of a handwritten digit into one of 10 classes representing integer values from 0 to 9, inclusively. The dataset has a collection of 60,000 handwrittend digits of size 28 X 28. Here we build a convolutional neural network model that is able to classify to it's appropriate numerical value.

## Neural Network Model

![image](https://github.com/aldrinlijo04/mnist-classification/assets/118544279/eef099d4-ccf0-4148-8d61-3cbe8c06ac37)

## DESIGN STEPS

### STEP 1:
Start by importing all the necessary libraries. And load the Data into Test sets and Training sets.

### STEP 2:
Then we move to normalization and encoding of the data.

### STEP 3:
The Model is then built using a Conv2D layer, MaxPool2D layer, Flatten layer, and 2 Dense layers of 16 and 10 neurons respectively.

### STEP 4:
Finally, we pass handwritten digits to the model for prediction.

## PROGRAM

### Name: KRISHNARAJ D
### Register Number: 212222230070
```python
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train.shape

X_test.shape

single_image= X_train[0]

single_image.shape

plt.imshow(single_image,cmap='gray')

y_train.shape

X_train.min()

X_train.max()

X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0

X_train_scaled.min()

X_train_scaled.max()

y_train[0]

y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)

type(y_train_onehot)

y_train_onehot.shape

single_image = X_train[500]
plt.imshow(single_image,cmap='gray')

y_train_onehot[500]

X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)

model = keras.Sequential()

model.add(layers.Input(shape=(28,28,1)))

model.add(layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu'))

model.add(layers.MaxPool2D(pool_size=(2,2)))

model.add(layers.Flatten())

model.add(layers.Dense(32,activation='relu'))

model.add(layers.Dense(64,activation='relu'))

model.add(layers.Dense(10,activation='softmax'))

model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train_scaled ,y_train_onehot, epochs=5,
          batch_size=64,
          validation_data=(X_test_scaled,y_test_onehot))
metrics = pd.DataFrame(model.history.history)

print('KRISHNARAJ D-212222230070')
metrics.head()

print('KRISHNARAJ D-212222230070')
metrics[['accuracy','val_accuracy']].plot()

print('KRISHNARAJ D-212222230070')
metrics[['loss','val_loss']].plot()
x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)

print(confusion_matrix(y_test,x_test_predictions))

print(classification_report(y_test,x_test_predictions))
img = image.load_img('NEWIMG.jpg')
type(img)

img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0
x_single_prediction = np.argmax(model.predict(img_28_gray_scaled.reshape(1,28,28,1)),axis=1)

print(x_single_prediction)
print('KRISHNARAJ D-212222230070')
plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')
print(x_single_prediction)

img1 = image.load_img('3.jpg')
img_tensor_1= tf.convert_to_tensor(np.asarray(img1))
img_28 = tf.image.resize(img_tensor_1,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0

x_single_prediction1 = np.argmax(model.predict(img_28_gray_scaled.reshape(1,28,28,1)),axis=1)
print(x_single_prediction1)
print("KRISHNARAJ D -212222230070")

plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')
img_28_gray_inverted = 255.0-img_28_gray
img_28_gray_inverted_scaled = img_28_gray_inverted.numpy()/255.0
print(x_single_prediction1)
````
## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot
![1DL](https://github.com/user-attachments/assets/4bc38f18-1f4b-4cbd-90f2-668e7bd36f35)

![2DL](https://github.com/user-attachments/assets/d87d7ff3-db22-4c45-9133-1e89f9721949)
![3DL](https://github.com/user-attachments/assets/96098930-8f61-4fdc-9dec-4f9b9d86cef3)

### Classification Report
![4DL](https://github.com/user-attachments/assets/af319983-84e1-402a-a4a3-65227127c519)


### Confusion Matrix
![5DL](https://github.com/user-attachments/assets/740036b8-9594-4df0-9339-89940a4299b6)



### New Sample Data Prediction
#### INPUT
<img src="https://github.com/user-attachments/assets/c543356e-5a6d-413e-beba-6d290b9b8dfa" width="300" height="200" alt="Reduced Image">


#### OUTPUT


![6DL](https://github.com/user-attachments/assets/4236f222-cfca-4425-95f7-4b8e7be8c3e8)

## RESULT
Thus, a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images is written and executed successfully.
