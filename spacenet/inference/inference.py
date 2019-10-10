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

import resource
import sys
sys.path.append('../src/models')

from segmentation_cpu import SegmentationModel as Model
from os import path
from PIL import Image
import numpy as np
from uuid import uuid4
import json
from imantics import Polygons, Mask
from simplification.cutil import simplify_coords_vwp

def create_wkt(polygon):
    """
    :param polygon: a single polygon in the format [(x1,y1), (x2,y2), ...]
    :returns: a wkt formatted string ready to be put into the json 
    """
    wkt = 'POLYGON (('

    for coords in polygon:
        wkt += "{} {},".format(coords[0], coords[1])

    wkt = wkt[:-1] + '))'

    return wkt


def create_json(adjusted_polygons):
    """
    :param polygons: list of polygons in the format [(x1,y1), (x2,y2), ...]
    :returns: json with found and adjusted polygon pixel x,y values in WKT format
    """
    # Create a blank json that matched the labeler provided jsons with null or default values
    output_json = {
        "features": {
            "lng_lat": [],
            "xy": []
        }, 
        "metadata": {
            "sensor": "",
            "provider_asset_type": "",
            "gsd": 0,
            "capture_date": "", 
            "off_nadir_angle": 0, 
            "pan_resolution": 0, 
            "sun_azimuth": 0, 
            "sun_elevation": 0, 
            "target_azimuth": 0, 
            "disaster": "", 
            "disaster_type": "", 
            "catalog_id": "", 
            "original_width": 0, 
            "original_height": 0, 
            "width": 0, 
            "height": 0, 
            "id": "", 
            "img_name": ""
        }
    }

    # Using a lambda function to place the WKT string in the list of polygons 
    polygon_template = lambda poly, uuid: {
        'properties': {
            'feature_type': 'building',
            'uid': uuid
        },
        'wkt': poly
    }

    # For each adjusted polygon add the wkt for the polygon points
    for polygon in adjusted_polygons:
        wkt = create_wkt(polygon)
        uuid = gen_uuid()
        poly = polygon_template(wkt, uuid)
        output_json['features']['xy'].append(poly)

    return output_json

def gen_uuid():
    return str(uuid4())

def inference(image, score, output_file):
    building_score = score[1]
    
    building_mask_pred = (np.argmax(score, axis=0) == 1)
    polygons = Mask(building_mask_pred).polygons()
    
    new_predictions = []
    
    for poly in polygons:
        if len(poly) >= 3:
            f = poly.reshape(-1, 2)
            simplified_vw = simplify_coords_vwp(f, .3)
            if len(simplified_vw) > 2:
                    mpoly = []
                    # Rebuilding the polygon in the way that PIL expects the values [(x1,y1),(x2,y2)]
                    for i in simplified_vw:
                        mpoly.append((i[0], i[1]))
                    # Adding the first point to the last to close the polygon
                    mpoly.append((simplified_vw[0][0], simplified_vw[0][1]))
                    new_predictions.append(mpoly)
            
    # Creating the json with the predicted and then adjusted polygons
    output_json = create_json(new_predictions)
    
    with open(output_file, 'w') as out_file:
        json.dump(output_json, out_file)

if __name__ == "__main__": 
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description=
        """inference.py: takes an image and creates inferred polygons json off the VW algorithm and the unet model predictions"""
    )
    parser.add_argument('--input',
                        required=True,
                        metavar='/path/to/input/image.png')
    parser.add_argument(
        '--weights',
        required=True,
        metavar='/full/path/to/mode_iter_XXXX',
        help="Must be the output to a unet model weights trained for xView2"
    )
    parser.add_argument(
        '--mean',
        required=True,
        metavar='/full/path/to/mean.npy',
        help="a numpy data structure file that is the mean of the training images (found by running ./src/features/compute_mean.py)"
    )
    parser.add_argument('--output',
                        required=True,
                        metavar="/path/to/output/file.json")
    args = parser.parse_args()

    # Load trained model
    # Modify the paths based on your trained model location if needed.
    mean = np.load(args.mean)
    model = Model(args.weights, mean)
 
    image = np.array(Image.open(args.input))
    score = model.apply_segmentation(image)
    inference(image, score, args.output)

