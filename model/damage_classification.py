#####################################################################################################################################################################
# xView2                                                                                                                                                            #
# Copyright 2019 Carnegie Mellon University.                                                                                                                        #
# NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS" BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO    #
# WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY,          # 
# EXCLUSIVITY, OR RESULTS OBTAINED FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM FROM PATENT, # 
# TRADEMARK, OR COPYRIGHT INFRINGEMENT.                                                                                                                             #
# Released under a MIT (SEI)-style license, please see LICENSE.md or contact permission@sei.cmu.edu for full terms.                                                 #
# [DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.  Please see Copyright notice for non-US Government use  #
# and distribution.                                                                                                                                                 #
# This Software includes and/or makes use of the following Third-Party Software subject to its own license:                                                         #
# 1. SpaceNet (https://github.com/motokimura/spacenet_building_detection/blob/master/LICENSE) Copyright 2017 Motoki Kimura.                                         #
# DM19-0988                                                                                                                                                         #
#####################################################################################################################################################################

from PIL import Image
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import math
import random
import argparse
import logging
import json
import cv2
import datetime

from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
import shapely.wkt
import shapely
from shapely.geometry import Polygon
from collections import defaultdict

import tensorflow as tf
import keras
import ast
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Add, Input, Concatenate
from keras.models import Model
from keras.applications.resnet50 import ResNet50
from keras import backend as K

from model import *

logging.basicConfig(level=logging.INFO)

# Configurations
NUM_WORKERS = 4 
NUM_CLASSES = 4
BATCH_SIZE = 64
NUM_EPOCHS = 100 
LEARNING_RATE = 0.0001
RANDOM_SEED = 123
LOG_STEP = 150
LOG_DIR = '/path/to/logs' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

damage_intensity_encoding = dict()
damage_intensity_encoding[3] = '3'
damage_intensity_encoding[2] = '2' 
damage_intensity_encoding[1] = '1' 
damage_intensity_encoding[0] = '0' 


###
# Function to compute unweighted f1 scores, just for reference
###
def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision


    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


###
# Creates data generator for validation set
###
def validation_generator(test_csv, test_dir):
    df = pd.read_csv(test_csv)
    df = df.replace({"labels" : damage_intensity_encoding })

    gen = keras.preprocessing.image.ImageDataGenerator(
                             rescale=1/255.)


    return gen.flow_from_dataframe(dataframe=df,
                                   directory=test_dir,
                                   x_col='uuid',
                                   y_col='labels',
                                   batch_size=BATCH_SIZE,
                                   shuffle=False,
                                   seed=RANDOM_SEED,
                                   class_mode="categorical",
                                   target_size=(128, 128))


###
# Applies random transformations to training data
###
def augment_data(df, in_dir):

    df = df.replace({"labels" : damage_intensity_encoding })
    gen = keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True,
                             vertical_flip=True,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             rescale=1/255.)
    return gen.flow_from_dataframe(dataframe=df,
                                   directory=in_dir,
                                   x_col='uuid',
                                   y_col='labels',
                                   batch_size=BATCH_SIZE,
                                   seed=RANDOM_SEED,
                                   class_mode="categorical",
                                   target_size=(128, 128))


# Run training and evaluation based on existing or new model
def train_model(train_data, train_csv, test_data, test_csv, model_in, model_out):

    model = generate_xBD_baseline_model()

    # Add model weights if provided by user
    if model_in is not None:
        model.load_weights(model_in)

    df = pd.read_csv(train_csv)
    class_weights = compute_class_weight('balanced', np.unique(df['labels'].to_list()), df['labels'].to_list());
    d_class_weights = dict(enumerate(class_weights))

    samples = df['uuid'].count()
    steps = np.ceil(samples/BATCH_SIZE)

    # Augments the training data
    train_gen_flow = augment_data(df, train_data)

    #Set up tensorboard logging
    tensorboard_callbacks = keras.callbacks.TensorBoard(log_dir=LOG_DIR,
                                                        batch_size=BATCH_SIZE)

    
    #Filepath to save model weights
    filepath = model_out + "-saved-model-{epoch:02d}-{accuracy:.2f}.hdf5"
    checkpoints = keras.callbacks.ModelCheckpoint(filepath,
                                                    monitor=['loss', 'accuracy'],
                                                    verbose=1,
                                                    save_best_only=False,
                                                    mode='max')

    #Adds adam optimizer
    adam = keras.optimizers.Adam(lr=LEARNING_RATE,
                                    beta_1=0.9,
                                    beta_2=0.999,
                                    decay=0.0,
                                    amsgrad=False)


    model.compile(loss=ordinal_loss, optimizer=adam, metrics=['accuracy', f1])

    #Training begins
    model.fit_generator(generator=train_gen_flow,
                        steps_per_epoch=steps,
                        epochs=NUM_EPOCHS,
                        workers=NUM_WORKERS,
                        use_multiprocessing=True,
                        class_weight=d_class_weights,
                        callbacks=[tensorboard_callbacks, checkpoints],
                        verbose=1)


    #Evalulate f1 weighted scores on validation set
    validation_gen = validation_generator(test_csv, test_data)
    predictions = model.predict(validation_gen)

    val_trues = validation_gen.classes
    val_pred = np.argmax(predictions, axis=-1)

    f1_weighted = f1_score(val_trues, val_pred, average='weighted')
    print(f1_weighted)


def main():

    parser = argparse.ArgumentParser(description='Run Building Damage Classification Training & Evaluation')
    parser.add_argument('--train_data',
                        required=True,
                        metavar="/path/to/xBD_train",
                        help="Full path to the train data directory")
    parser.add_argument('--train_csv',
                        required=True,
                        metavar="/path/to/xBD_split",
                        help="Full path to the train csv")
    parser.add_argument('--test_data',
                        required=True,
                        metavar="/path/to/xBD_test",
                        help="Full path to the test data directory")
    parser.add_argument('--test_csv',
                        required=True,
                        metavar="/path/to/xBD_split",
                        help="Full path to the test csv")
    parser.add_argument('--model_in',
                        default=None,
                        metavar='/path/to/input_model',
                        help="Path to save model")
    parser.add_argument('--model_out',
                        required=True,
                        metavar='/path/to/save_model',
                        help="Path to save model")

    args = parser.parse_args()

    train_model(args.train_data, args.train_csv, args.test_data, args.test_csv, args.model_in, args.model_out)


if __name__ == '__main__':
    main()
