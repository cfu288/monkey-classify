# 3. Import libraries and modules
import numpy as np
import pandas as pd
import cv2
np.random.seed(123)  # for reproducibility

from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
K.set_image_dim_ordering('th')  

def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
		#plt.imshow(ims[i], interpolation=None if interp else 'none')
    f.savefig('tmp.png')

# Making a Panda's DataFrame from txt
data_labels = pd.read_csv("monkey_labels.txt")
data_labels.columns = data_labels.columns.str.strip()
print(data_labels)

# List of Common Names of Monkeys, ordered from N0 - N9
# labels = list(data_labels["Common Name"])
labels = list(data_labels["Label"])
labels = [x.strip() for x in labels]
print(labels)

height = 100
width = 100
channels = 3
batch_size = 10
seed=123
train_dir = 'training/'
test_dir = 'validation/'

# Generate minibatches of image data
train_datagen = ImageDataGenerator(rescale=1./255) # limit between 0-1
# Normalize Data
train_batches_generator = train_datagen.flow_from_directory(train_dir,
        target_size=(height,width),
        classes=labels,
        batch_size=batch_size)

test_datagen = ImageDataGenerator(rescale=1./255) # limit between 0-1
test_batches_generator = test_datagen.flow_from_directory(test_dir,
        target_size=(height,width),
        classes=labels,
        batch_size=batch_size)

imgs, lab = next(train_batches_generator)
plots(imgs, titles=lab) 

# 4. Load pre-shuffled MNIST data into train and test sets
#(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 5. Preprocess input data
# X_train = X_train.reshape(X_train.shape[0], 3, 28, 28) 
#X_test = X_test.reshape(X_test.shape[0], 3, 28, 28)
#X_train = X_train.astype('float32')
#X_test = X_test.astype('float32')
#X_train /= 255 
#X_test /= 255
 
# 6. Preprocess class labels
#Y_train = np_utils.to_categorical(y_train, 10)
#Y_test = np_utils.to_categorical(y_test, 10)
 
# 7. Define model architecture
model = Sequential()
 
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(1,28,28)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
 
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
 
# 8. Compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
 
# 9. Fit model on training data
# model.fit(X_train, Y_train, 
        #  batch_size=32, epochs=10, verbose=1)
 
# 10. Evaluate model on test data
# score = model.evaluate(X_test, Y_test, verbose=0)

print("DONE")
