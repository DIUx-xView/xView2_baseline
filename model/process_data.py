"""
xView2
Copyright 2019 Carnegie Mellon University. All Rights Reserved.

NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS"
BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER
INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED
FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM
FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.

Released under a MIT (SEI)-style license, please see license.txt or contact permission@sei.cmu.edu for full terms.
This material has been approved for public release and unlimited distribution.
Please see Copyright notice for non-US Government use and distribution.

This Software includes and/or makes use of the following Third-Party Software subject to its own license:
1. Matterport/Mask_RCNN - MIT Copyright 2017 Matterport, Inc.
   https://github.com/matterport/Mask_RCNN/blob/master/LICENSE
"""
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

import shapely.wkt
import shapely
from shapely.geometry import Polygon
from collections import defaultdict
from sklearn.model_selection import train_test_split
logging.basicConfig(level=logging.INFO)

# Configurations
NUM_WORKERS = 4
NUM_CLASSES = 4
BATCH_SIZE = 64
NUM_EPOCHS = 120
LEARNING_RATE = 0.0001
RANDOM_SEED = 123
LOG_STEP = 150

damage_intensity_encoding = defaultdict(lambda: 0)
damage_intensity_encoding['destroyed'] = 3
damage_intensity_encoding['major-damage'] = 2
damage_intensity_encoding['minor-damage'] = 1
damage_intensity_encoding['no-damage'] = 0


def process_img(img_array, polygon_pts, scale_pct):
    """Process Raw Data into

            Args:
                img_array (numpy array): numpy representation of image.
                polygon_pts (array): corners of the building polygon.

            Returns:
                numpy array: .

    """

    height, width, _ = img_array.shape

    xcoords = polygon_pts[:, 0]
    ycoords = polygon_pts[:, 1]
    xmin, xmax = np.min(xcoords), np.max(xcoords)
    ymin, ymax = np.min(ycoords), np.max(ycoords)

    xdiff = xmax - xmin
    ydiff = ymax - ymin

    #Extend image by scale percentage
    xmin = max(int(xmin - (xdiff * scale_pct)), 0)
    xmax = min(int(xmax + (xdiff * scale_pct)), width)
    ymin = max(int(ymin - (ydiff * scale_pct)), 0)
    ymax = min(int(ymax + (ydiff * scale_pct)), height)

    return img_array[ymin:ymax, xmin:xmax, :]


def process_data(input_path, output_path, output_csv_path, val_split_pct):
    """Process Raw Data into

        Args:
            dir_path (path): Path to the xBD dataset.
            data_type (string): String to indicate whether to process
                                train, test, or holdout data.

        Returns:
            x_data: A list of numpy arrays representing the images for training
            y_data: A list of labels for damage represented in matrix form

    """
    x_data = []
    y_data = []

    disasters = [folder for folder in os.listdir(input_path) if not folder.startswith('.')]
    disaster_paths = ([input_path + "/" +  d + "/images" for d in disasters])
    image_paths = []
    image_paths.extend([(disaster_path + "/" + pic) for pic in os.listdir(disaster_path)] for disaster_path in disaster_paths)
    img_paths = np.concatenate(image_paths)

    for img_path in tqdm(img_paths):

        img_obj = Image.open(img_path)
        img_array = np.array(img_obj)

        #Get corresponding label for the current image
        label_path = img_path.replace('png', 'json').replace('images', 'labels')
        label_file = open(label_path)
        label_data = json.load(label_file)

        for feat in label_data['features']['xy']:

            # only images post-disaster will have damage type
            try:
                damage_type = feat['properties']['subtype']
            except: # pre-disaster damage is default no-damage
                damage_type = "no-damage"
                continue

            poly_uuid = feat['properties']['uid'] + ".png"

            y_data.append(damage_intensity_encoding[damage_type])

            polygon_geom = shapely.wkt.loads(feat['wkt'])
            polygon_pts = np.array(list(polygon_geom.exterior.coords))
            poly_img = process_img(img_array, polygon_pts, 0.8)
            cv2.imwrite(output_path + "/" + poly_uuid, poly_img)
            x_data.append(poly_uuid)
    
    output_train_csv_path = os.path.join(output_csv_path, "train.csv")

    if(val_split_pct > 0):
       x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=val_split_pct)
       data_array_train = {'uuid': x_train, 'labels': y_train}
       data_array_test = {'uuid': x_test, 'labels': y_test}
       output_test_csv_path = os.path.join(output_csv_path, "test.csv")
       df_train = pd.DataFrame(data_array_train)
       df_test = pd.DataFrame(data_array_test)
       df_train.to_csv(output_train_csv_path)
       df_test.to_csv(output_test_csv_path)
    else: 
       data_array = {'uuid': x_data, 'labels': y_data}
       df = pd.DataFrame(data = data_array)
       df.to_csv(output_train_csv_path)
    

def main():

    parser = argparse.ArgumentParser(description='Run Building Damage Classification Training & Evaluation')
    parser.add_argument('--input_dir',
                        required=True,
                        metavar="/path/to/xBD_input",
                        help="Full path to the parent dataset directory")
    parser.add_argument('--output_dir',
                        required=True,
                        metavar='/path/to/xBD_output',
                        help="Path to new directory to save images")
    parser.add_argument('--output_dir_csv',
                        required=True,
                        metavar='/path/to/xBD_output_csv',
                        help="Path to new directory to save csv")
    parser.add_argument('--val_split_pct', 
                        required=False,
                        default=0.0,
                        metavar='Percentage to split validation',
                        help="Percentage to split ")
    args = parser.parse_args()

    logging.info("Started Processing for Data")
    process_data(args.input_dir, args.output_dir, args.output_dir_csv, float(args.val_split_pct))
    logging.info("Finished Processing Data")


if __name__ == '__main__':
    main()
