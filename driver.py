# 3. Import libraries and modules
import os, cv2, argparse
import numpy as np
import pandas as pd

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.utils import np_utils
np.random.seed(123)  # for reproducibility

TRAIN_DIR = './training/'
TEST_DIR = './validation/'

def getArgs():
    p = argparse.ArgumentParser(description='Experiement with CNN for fine tuned Monkey Classificaton')
    p.add_argument('epochs',nargs='?',default=1,help='(OPTIONAL) Number of epochs to train for, defaults to 1')
    return p.parse_args()

def main():
    # Get CLA
    args = getArgs()
    EPOCHS = int(args.epochs)
    TRAIN_IMG, TRAIN_CLS, TEST_IMG, TEST_CLS = ([] for i in range(4))
    COLS = ['Label', 'Latin Name', 'Common Name', 'Train Images', 'Validation Images']
    LABELS = pd.read_csv('./monkey_labels.txt', names=COLS, skiprows=1)
    CLASSES = [x for x in range(0, len(LABELS))]

    # read in all images
    # resizing the images to 100x100 to make training faster
    print("READING IN AND FORMATTING IMAGES")
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

    # Preprocess images
    # Reshape them to theanos format (channels, hight, width)
    # Convert to 0-255 to value in [0-1]
    # TRAIN_IMG = TRAIN_IMG.reshape(TRAIN_IMG.shape[0], 3, 100, 100)
    # TEST_IMG = TEST_IMG.reshape(TEST_IMG.shape[0], 3, 100, 100)
    TRAIN_IMG = TRAIN_IMG.astype('float32')
    TEST_IMG = TEST_IMG.astype('float32')
    TRAIN_IMG /= 255
    TEST_IMG /= 255

    # Reshape class labels
    TRAIN_CLS = np_utils.to_categorical(TRAIN_CLS, 10)
    TEST_CLS = np_utils.to_categorical(TEST_CLS, 10)

    # Construct the model
    model = Sequential()

    model.add(Conv2D(110, (3, 3), activation='relu', input_shape=(100, 100, 3)))
    model.add(Conv2D(110, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    # Compile model
    print("COMPILING MODEL")
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model on the training data
    print("TRAINING FOR {} EPOCHS".format(EPOCHS)) 
    history = model.fit(TRAIN_IMG, TRAIN_CLS, batch_size=32, epochs=EPOCHS, verbose=1)

    # Save the model
    print("SAVE MODEL")
    model.save('test_model.h5')
    print(history.history.keys())

    # TODO - Print a plot of loss and accuracy over epochs and learning rates


    #model = load_model('test_model.h5')

    # Evaluate the model on the validation data
    loss, acc = model.evaluate(TEST_IMG, TEST_CLS, verbose=1)
    print("Loss: ", loss, " Accuracy: ", acc)

    # Predict images
    # TODO - Print mispredicted images, the label it predicted, and the correct label
    '''
    for i in range(len(TEST_IMG)):
        img = TEST_IMG[i]
        cls = TEST_CLS[i]
        img = np.array([img])
        prediction = model.predict(img, verbose=1, steps=1)
        print
        print "Class: ", cls
        print "Prediction: ", prediction[0]
        max_index = np.argmax(prediction[0])
        print "Predicted Class index: ", max_index
        print "Prediction Correct: ", True if cls[max_index] == 1. else False
    '''
        

main()
