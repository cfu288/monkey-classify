# 3. Import libraries and modules
import os
import numpy as np
import pandas as pd
import cv2


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import load_model
from keras import backend as K
K.set_image_dim_ordering('th') 
np.random.seed(123)  # for reproducibility


def main():
    TRAIN_IMG, TRAIN_CLS, TEST_IMG, TEST_CLS = ([] for i in range(4))
    COLS = ['Label', 'Latin Name', 'Common Name', 'Train Images', 'Validation Images']
    LABELS = pd.read_csv('./10-monkey-species/monkey_labels.txt', names=COLS, skiprows=1)
    CLASSES = [x for x in range(0, len(LABELS))]
    TRAIN_DIR = './10-monkey-species/training/'
    TEST_DIR = './10-monkey-species/validation/'

    # read in all images
    for x in range(0, len(LABELS)):
        train_dir = TRAIN_DIR + LABELS.loc[x,'Label'].strip() + '/'
        test_dir = TEST_DIR + LABELS.loc[x,'Label'].strip() + '/'
        for file in os.listdir(train_dir):
            img = cv2.imread(train_dir + file)
            if img is not None:
                img = cv2.resize(img, (100, 100))
                TRAIN_IMG.append(img)
                TRAIN_CLS.append(x)
        for file in os.listdir(test_dir):
            img = cv2.imread(test_dir + file)
            if img is not None:
                img = cv2.resize(img, (100, 100))
                TEST_IMG.append(img)
                TEST_CLS.append(x)


    # convert to numpy arrays
    TRAIN_IMG = np.array(TRAIN_IMG)
    TEST_IMG = np.array(TEST_IMG)
    TRAIN_CLS = np.array(TRAIN_CLS)
    TEST_CLS = np.array(TEST_CLS)

    # preprocess images
    TRAIN_IMG = TRAIN_IMG.reshape(TRAIN_IMG.shape[0], 3, 100, 100)
    TEST_IMG = TEST_IMG.reshape(TEST_IMG.shape[0], 3, 100, 100)
    TRAIN_IMG = TRAIN_IMG.astype('float32')
    TEST_IMG = TEST_IMG.astype('float32')
    TRAIN_IMG /= 255
    TEST_IMG /= 255

    # reshape class labels
    TRAIN_CLS = np_utils.to_categorical(TRAIN_CLS, 10)
    TEST_CLS = np_utils.to_categorical(TEST_CLS, 10)

    # construct the model
    model = Sequential()

    model.add(Conv2D(110, (3, 3), activation='relu', input_shape=(3, 100, 100)))
    model.add(Conv2D(110, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model on the training data
    history = model.fit(TRAIN_IMG, TRAIN_CLS, batch_size=32, epochs=1, verbose=1)
    model.save('test_model.h5')
    print history

    # Evaluate the model on the validation data
    score = model.evaluate(TEST_IMG, TEST_CLS, verbose=1)
    print score

    # Predict an image
    test_img = cv2.imread(TEST_DIR + 'n0/n000.jpg')
    test_img = cv2.resize(test_img, (100, 100))
    test_img = np.array(test_img)
    test_img = test_img.resize(test_img[0].shape, 3, 100, 100)
    prediction = model.predict(test_img, verbose=1)
    print prediction

        

main()
