
import os
import csv
import numpy as np
import pickle
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Activation, GlobalAveragePooling2D
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback
from keras.layers.normalization import BatchNormalization
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import DenseNet121
from keras.optimizers import Adam
from sklearn.metrics import cohen_kappa_score
import cv2 as cv
import pandas as pd

os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/extras/CUPTI/lib64"

def read_csv(filename, delimiter=','):

    csv_data = dict()
    keys = []

    with open(filename) as f:

        csv_reader = csv.reader(f, delimiter=delimiter)

        for i, row in enumerate(csv_reader):

            if i == 0:
                keys = row
                for key in keys:
                    csv_data[key] = []
            else:
                for key, data in zip(keys, row):
                    csv_data[key].append(data)

        print('Processed {} lines.'.format(i))

    return csv_data, i

def write_csv(filename, data, delimiter=',', index=True):

    keys = [key for key in data]
    rows = len(data[keys[0]])

    with open(filename, mode='w') as f:
        csv_writer = csv.writer(f, delimiter=delimiter)

        if index:
            csv_writer.writerow(keys)

        for i in range(rows):
            row = []
            for key in keys:
                row.append(data[key][i])
            csv_writer.writerow(row)

        print('Processed {} lines.'.format(i))


def preprocess_image(image_path, grayscale=False, width=224, height=224, output_channels=None):
    img = cv.imread(image_path)
    img = cv.resize(img, (width, height), interpolation=cv.INTER_CUBIC)

    if grayscale:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img = cv.addWeighted(img, 4, cv.GaussianBlur(img, (0,0) , (width+height)/20), -4, 128)

        if output_channels is None:
            img = np.expand_dims(img, axis=2)
        else:
            img = np.stack([img] * output_channels, axis=2)

    return img

class CustomMetric(Callback):

    def on_train_begin(self, logs={}):
        self.val_kappas = []

    def on_epoch_end(self, epoch, logs={}):

        X_val, y_val = self.validation_data[:2]
        y_val = y_val.sum(axis=1) - 1
        
        y_pred = self.model.predict(X_val) > 0.5
        y_pred = y_pred.astype(int).sum(axis=1) - 1

        _val_kappa = cohen_kappa_score(y_val, y_pred, weights='quadratic')

        self.val_kappas.append(_val_kappa)

        print("val_kappa: {}".format(_val_kappa))

        return

class Classifier():

    def __init__(self, labels, image_height=224, image_width=224, channels=3):
        self.model = None
        self.image_height = image_height
        self.image_width = image_width
        self.channels = channels
        self.input_dims = (image_height, image_width, channels)
        self.labels = labels
        self.label_size = len(labels)

    def build(self, densenet=True, verbose=True, 
              densenet_weights='/media/hdd/data/densenet/DenseNet-BC-121-32-no-top.h5'):

        if not densenet:

            self.model = Sequential()
            self.model.add(Conv2D(filters=16,
                                kernel_size=(5, 5),
                                padding="same",
                                input_shape=self.input_dims))
            self.model.add(BatchNormalization())
            self.model.add(Activation("relu"))
            self.model.add(MaxPooling2D(pool_size=(2,2), padding="same"))
            self.model.add(Conv2D(filters=32,
                                kernel_size=(5, 5),
                                padding="same"))
            self.model.add(BatchNormalization())
            self.model.add(Activation("relu"))
            self.model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
            self.model.add(Conv2D(filters=64,
                                kernel_size=(5, 5),
                                padding="same"))
            self.model.add(BatchNormalization())
            self.model.add(Activation("relu"))

            self.model.add(Flatten())
            self.model.add(Dense(units=256))
            self.model.add(BatchNormalization())
            self.model.add(Activation("relu"))
            self.model.add(Dropout(0.25))
            self.model.add(Dense(units= 128))
            self.model.add(BatchNormalization())
            self.model.add(Activation("relu"))
            self.model.add(Dropout(0.25))
            self.model.add(Dense(units=64))
            self.model.add(BatchNormalization())
            self.model.add(Activation("relu"))
            self.model.add(Dropout(0.25))
            self.model.add(Dense(units=self.label_size, activation="softmax"))

        else:
            densenet = DenseNet121(
                weights=densenet_weights,
                include_top=False,
                input_shape=self.input_dims
            )

            self.model = Sequential()
            self.model.add(densenet)
            self.model.add(GlobalAveragePooling2D())
            self.model.add(Dropout(0.5))
            self.model.add(Dense(self.label_size, activation='sigmoid'))


        if verbose:
            self.model.summary()

        self._compile()

    def _compile(self, loss='binary_crossentropy'):
        optimizer = Adam(lr=0.00005)
        self.model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    def train(self, data, data_split=0.15, epochs=100, batch_size=8, use_callbacks=True):

        x, y = data["x"], data["y"]
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=data_split)

        train_datagen = ImageDataGenerator(zoom_range=0.15,
                                           fill_mode='constant',
                                           cval=0.,
                                           horizontal_flip=True,
                                           vertical_flip=True)
        train_datagen.fit(x_train)

        # val_datagen = ImageDataGenerator(rescale=1./255)
        # val_datagen.fit(x_val)

        if use_callbacks:
            early_stopping = EarlyStopping(monitor='val_loss', patience=5)
            mcp_save = ModelCheckpoint('best_model.hdf5', 
                                       save_best_only=True, 
                                       monitor='val_loss')
            # reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', 
            #                                    factor=0.5, 
            #                                    patience=5, 
            #                                    verbose=1, 
            #                                    epsilon=1e-4)

                                            
            qwk = CustomMetric()

            callbacks = [early_stopping, mcp_save, qwk]
        else:
            callbacks = None

        # fits the model on batches with real-time data augmentation:
        self.model.fit_generator(train_datagen.flow(x_train, y_train, batch_size=batch_size),
                                 steps_per_epoch=len(x_train) / batch_size, 
                                 epochs=epochs,
                                #  validation_data=val_datagen.flow(x_val, y_val, batch_size=batch_size),
                                 validation_data=(x_val, y_val),
                                 validation_steps=x_val.shape[0]//batch_size, 
                                 callbacks=callbacks)

    def train_flow_from_directory(self, train_data_dir, data_split=0.5, epochs=100, batch_size=4, use_callbacks=True):
        
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            zoom_range=0.2,
            validation_split=data_split) # set validation split

        train_generator = train_datagen.flow_from_directory(
            train_data_dir,
            target_size=(self.image_height, self.image_width),
            color_mode=self.get_colormode(),
            batch_size=batch_size,
            class_mode='categorical',
            subset='training')

        validation_generator = train_datagen.flow_from_directory(
            train_data_dir,
            target_size=(self.image_height, self.image_width),
            color_mode=self.get_colormode(),
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation')

        validation_steps = validation_generator.samples // batch_size

        if use_callbacks:
            early_stopping = EarlyStopping(monitor='val_loss', patience=8)
            mcp_save = ModelCheckpoint('best_model.hdf5', 
                                       save_best_only=True, 
                                       monitor='val_loss')
            reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', 
                                               factor=0.5, 
                                               patience=5, 
                                               verbose=1, 
                                               epsilon=1e-4)

            qwk = CustomMetric()
            callbacks = [early_stopping, mcp_save, reduce_lr_loss, qwk]
        else:
            callbacks = None

        # fits the model on batches with real-time data augmentation:
        self.model.fit_generator(train_generator,
                                 steps_per_epoch=train_generator.samples // batch_size,
                                 epochs=epochs,
                                 validation_data = validation_generator, 
                                 validation_steps = validation_steps,
                                 callbacks=callbacks)

    def predict(self, data):
        return self.model.predict(data)
            
    def load(self, filename):
        self.model = load_model(filename)

    def save(self, filename):
        self.model.save(filename)

    def get_colormode(self):

        if self.channels == 1:
            return "grayscale"
        elif self.channels == 3:
            return "rgb"
        elif self.channels == 4:
            return "rgba"
        else:
            ValueError("Invalid input channels!")

def get_multilabel(x, n_labels, dtype=np.uint8):
    y = np.zeros((n_labels), dtype=dtype)
    for i in range(int(x)+1):
        y[i] = 1

    return y

def train_model(csv_file, train_dir, n_classes=5, train_cache_file="x_train.npy", epochs=20,
                densenet_weights='/media/hdd/data/densenet/DenseNet-BC-121-32-no-top.h5'):

    data, N = read_csv(csv_file)

    if not os.path.exists(train_cache_file):

        x_train = np.empty((N, 224, 224, 3), dtype=np.uint8)

        for i, (image_id, diagnosis) in enumerate(zip(data['id_code'], data['diagnosis'])):
            f_path = os.path.join(train_dir, diagnosis, image_id + ".png")
            x_train[i, :, :, :] = preprocess_image(f_path, grayscale=True, output_channels=3)
            print("processed {} of {}".format(i+1, N))

        np.save(train_cache_file, x_train)

    else:
        x_train = np.load(train_cache_file)

    y_train = np.zeros((N, n_classes))
    for i, diagnosis in enumerate(data["diagnosis"]):
        y_train[i,:] = get_multilabel(diagnosis, n_classes)

    train_data = {"x" : x_train, "y" : y_train}

    classifier = Classifier(labels=["0", "1", "2", "3", "4"])
    classifier.build(densenet_weights=densenet_weights)
    classifier.train(train_data, epochs=epochs)

def test_model(csv_file, 
               test_dir, 
               n_classes=5, 
               model="best_model.hdf5",
               test_cache_file="x_test.npy", 
               cached_predictions="predictions.npy", 
               submission_file="submission.csv"):

    data, N = read_csv(csv_file)
    
    if not os.path.exists(test_cache_file):

        x_test = np.empty((N, 224, 224, 3), dtype=np.uint8)

        for i, image_id in enumerate(data['id_code']):
            f_path = os.path.join(test_dir, image_id + ".png")
            x_test[i, :, :, :] = preprocess_image(f_path, grayscale=True, output_channels=3)
            print("processed {} of {}".format(i+1, N))

        np.save(test_cache_file, x_test)

    else:
        x_test = np.load(test_cache_file)
    
    if not os.path.exists(cached_predictions):
        classifier = Classifier(labels=["0", "1", "2", "3", "4"])
        classifier.load(model)
        predictions = classifier.predict(x_test)
        np.save(cached_predictions, predictions)
    else:
        predictions = np.load(cached_predictions)

    y_test = predictions > 0.5
    y_test = y_test.astype(int).sum(axis=1) - 1

    submission_df = pd.read_csv(csv_file)
    submission_df['diagnosis'] = y_test
    submission_df.to_csv('submission.csv', index=False)


if __name__ == "__main__":

    kaggle_run = False

    if not kaggle_run:
        train_dir = "/media/hdd/data/train_images"
        test_dir = "/media/hdd/data/test_images"
        train_csv = "/media/hdd/data/train.csv"
        test_csv = "/media/hdd/data/test.csv"
        densenet_weights = '/media/hdd/data/densenet/DenseNet-BC-121-32-no-top.h5'
    else:
        train_dir = "../input/aptos2019-blindness-detection/train_images"
        train_csv = "../input/aptos2019-blindness-detection/train.csv"
        test_dir = "../input/aptos2019-blindness-detection/test_images"
        test_csv = "../input/aptos2019-blindness-detection/test.csv"
        densenet_weights = '../input/densenet-keras/DenseNet-BC-121-32-no-top.h5'

    train_model(train_csv, train_dir, epochs=20, densenet_weights=densenet_weights)
    test_model(test_csv, test_dir)