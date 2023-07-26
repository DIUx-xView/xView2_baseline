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


import json
from os import path, walk, makedirs
from sys import exit, stderr

from cv2 import fillPoly, imwrite
import numpy as np
from shapely import wkt
from shapely.geometry import mapping, Polygon
from skimage.io import imread
from tqdm import tqdm
import imantics 

# This removes the massive amount of scikit warnings of "low contrast images"
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def get_dimensions(file_path): # give the path to images folder
    """
    :param file_path: The path of the file 
    :return: returns (width,height,channels)
    """
    # Open the image we are going to mask

    # The function reads the image and stores it as a PIL (Python Imaging Library) image object.
    pil_img = imread(file_path)

    #converts image object to a numpy array
    img = np.array(pil_img)

    #retrieves the dimensions of the image: width, height, and channels.
    w, h, c = img.shape
    return (w, h, c)

# -------------------------------------------------------------------------------------------------
# modified
def mask_polygons_separately(size, shapes):
    """
    :param size: A tuple of the (width,height,channels)
    :param shapes: A list of points in the polygon from get_feature_info
    :returns: a dict of masked polygons with the shapes filled in from cv2.fillPoly
    """
    # Mapping between subtype and color (RGB tuples)
    subtype_color_map = {
        "no-damage": (0, 0, 0),     # Black color for "no-damage"
        "minor-damage": (1, 1, 1),   # White color for "minor-damage"
        "major-damage": (2, 2, 2),   # Gray color for "major-damage"
        "destroyed": (3, 3, 3),     # Custom color for "destroyed"
        # Add more subtype-color mappings as needed
    }

    # For each WKT polygon, read the WKT format and fill the polygon as an image
    masked_polys = {}

    for u in shapes:
        sh = shapes[u]
        mask_img = np.zeros(size, np.uint8)
        
        # Get the subtype of the current polygon
        subtype = shapes[u]['subtype']
        
        # Get the color from the subtype_color_map, default to (0, 0, 0) if subtype not found
        color = subtype_color_map.get(subtype, (0, 0, 0))
        
        # Fill the polygon with the corresponding color
        i = fillPoly(mask_img, [sh], color)
        masked_polys[u] = i

    return masked_polys

# ---------------------------------------------------------------------------------------------------

# modified
def mask_polygons_together(size, shapes, damage_classification):
    """
    :param size: A tuple of the (width,height,channels)
    :param shapes: A list of points in the polygon from get_feature_info
    :param damage_classification: A dictionary containing UID: damage_classification pairs from the JSON file
    :returns: A numpy array with the polygons filled with appropriate pixel intensities based on damage_classification
    """
    # For each WKT polygon, read the WKT format and fill the polygon as an image
    mask_img = np.zeros(size, np.uint8)

    for u in shapes:
        blank = np.zeros(size, np.uint8)
        poly = shapes[u] 

        # Get the damage classification for the current polygon UID
        if u in damage_classification:
            damage_class = damage_classification[u]
        else:
            damage_class = "no-damage"  # Default to "no-damage" if damage classification is not available

        # Map damage classification to pixel intensity value
        if damage_class == "no-damage":
            intensity = 0
        elif damage_class == "destroyed":
            intensity = 1
        elif damage_class == "major-damage":
            intensity = 2
        elif damage_class == "minor-damage":
            intensity = 3
        else:
            intensity = 0  # Default to 0 for unknown or unrecognized classifications
        
        fillPoly(blank, [poly], (intensity, intensity, intensity))
        mask_img += blank
    
    return mask_img

# --------------------------------------------------------------------------------------------------

#modified
def mask_polygons_together_with_border(size, shapes, border):
    """
    :param size: A tuple of the (width,height,channels)
    :param shapes: A list of points in the polygon from get_feature_info
    :param border: A positive integer to add a border to the polygons
    :param subtype_mapping: A dictionary that maps subtypes to pixel intensities
    :returns: a dict of masked polygons with the shapes filled in from cv2.fillPoly
    """
    # Define the subtype mapping (example)
    subtype_mapping = {
    "no-damage": 0,
    "minor-damage": 1,
    "major-damage": 2,
    "destroyed": 3
    }
    # For each WKT polygon, read the WKT format and fill the polygon as an image
    mask_img = np.zeros(size, np.uint8)

    for u in shapes:
        # This blank array will be used to create a separate mask for the current polygon.
        blank =  np.zeros(size, np.uint8)
        # Each polygon stored in shapes is a np.ndarray
        poly = shapes[u]
        
        # Creating a shapely polygon object out of the numpy array 
        polygon = Polygon(poly)

        # Getting the center points from the polygon and the polygon points
        (poly_center_x, poly_center_y) = polygon.centroid.coords[0] #.centroid will find the centre of the polygon
        polygon_points = polygon.exterior.coords # .exterior extracts the coordinates of the exterior boundary of the current polygon 

        # Setting a new polygon with each X,Y manipulated based off the center point
        shrunk_polygon = []
        for (x,y) in polygon_points:
            if x < poly_center_x:
                x += border
            elif x > poly_center_x:
                x -= border

            if y < poly_center_y:
                y += border
            elif y > poly_center_y:
                y -= border

            shrunk_polygon.append([x,y])
        
        # Transforming the polygon back to a np.ndarray
        ns_poly = np.array(shrunk_polygon, np.int32)
  
        # Filling the shrunken polygon to add a border between close polygons
        fillPoly(blank, [ns_poly], subtype_mapping[u])
        mask_img += blank # adds up all the polygons to a single image in this for loop
    
    mask_img[mask_img > 1] = 0 # overlap regions of polygons are set to 0

    return mask_img # array of the masked images overcoming the neighboring and overlapping polygons
                    # this can be used to make images later


# -----------------------------------------------------------------------------------------------------

def save_masks(masks, output_path, mask_file_name):
    """
    This function is written to save the end result of mask_polygons_separately() function
    :param masks: dictionary of UID:masked polygons from mask_polygons_separately()
    :param output_path: path to save the masks
    :param mask_file_name: the file name the masks should have

    (?)It can be that 'masks' correspond to 'shapes' in mask_polygons_separately()
    and 'm' is corresponding to 'u'.
    """
    # For each filled polygon, write out a separate file, increasing the name
    for m in masks: # m is each UID (key of dict called 'masks')

        #resulting path will be something like output_path/mask_file_name_UID.png.
        final_out = path.join(output_path,
                              mask_file_name + '_{}.png'.format(m)) # .format(m) is a placeholder for m
                                                                    # which will be inserted instead of curly brackets
        
        #save the mask image to the specified file path. 
        # masks[m] represents the masked polygon corresponding to the current UID
        # imwrite is from the cv2 module, used to save the mask image to the specified file path. 
        # imwrite(path, image to be saved)
        imwrite(final_out, masks[m]) 

# --------------------------------------------------------------------------------------------------------

def save_one_mask(masks, output_path, mask_file_name):
    """
    This function is written to save the results of 
    mask_polygons_together() and mask_polygons_together_with_border()
    :param masks: list of masked polygons from the mask_polygons_separately function 
    :param output_path: path to save the masks
    :param mask_file_name: the file name the masks should have 
    """
    # For each filled polygon, write the mask shape out to the file per image
    mask_file_name = path.join(output_path, mask_file_name + '.png')
    imwrite(mask_file_name, masks)


# --------------------------------------------------------------------------------------------------------

def read_json(json_path):
    """
    :param json_path: path to load json from
    :returns: a python dictionary of json features
    """

    # The json.load() function is then used to parse the contents of the file object
    # as JSON data and convert it into a Python dictionary!!!!!
    annotations = json.load(open(json_path))
    return annotations # dictionary

# --------------------------------------------------------------------------------------------------------

def get_feature_info(feature):
    """
    :param feature: a python dictionary of json labels
    :returns: a list mapping of polygons contained in the image 
    """
    # Getting each polygon points from the json file and adding it to a dictionary of uid:polygons
    props = {}

    # 'xy' key in that dictionary maps to a list of features.
    for feat in feature['features']['xy']:
        '''
        wkt stands for "Well-Known Text," which is a text representation of geometric objects.
        The shapely.wkt.loads function is used to convert the WKT representation into a Shapely
        geometry object.
        '''
        feat_shape = wkt.loads(feat['wkt'])

        # mapping(feat_shape) converts feat_shape to a dict, coordinates is its key.
        #  [0] accesses the first outer list, which represents the exterior boundary of the polygon.
        coords = list(mapping(feat_shape)['coordinates'][0])

        # outermost borders of the polygons are saved in a dict with UIDs
        props[feat['properties']['uid']] = (np.array(coords, np.int32))

    # contains a mapping of UIDs to the corresponding polygon shapes as NumPy arrays.
    return props

# --------------------------------------------------------------------------------------------------------

def mask_chips(json_path, images_directory, output_directory, single_file, border):
    """
    :param json_path: path to find multiple json files for the chips
    :param images_directory: path to the directory containing the images to be masked
    :param output_directory: path to the directory where masks are to be saved
    :param single_file: a boolean value to see if masks should be saved a single file or multiple
    """
    # For each feature in the json we will create a separate mask
    # Getting all files in the directory provided for jsons
    jsons = [j for j in next(walk(json_path))[2] if '_pre' in j]

    # After removing non-json items in dir (if any)
    ##  each JSON file (j) 
    for j in tqdm([j for j in jsons if j.endswith('json')],
                  unit='poly',
                  leave=False):
        # Our chips start off in life as PNGs
        chip_image_id = path.splitext(j)[0] + '.png' # creates file names for each png
        mask_file = path.splitext(j)[0]

        # Loading the per chip json
        j_full_path = path.join(json_path, j)

        # returns a Python dictionary representing the JSON features.
        chip_json = read_json(j_full_path)

        # Getting the full chip path, and loading the size dimensions
        chip_file = path.join(images_directory, chip_image_id)

        # store a tuple containing these dimensions. width, height and channels
        chip_size = get_dimensions(chip_file)

        # Reading in the polygons from the json file
        # The 'polys' variable will store a dictionary mapping UIDs
        ## to corresponding polygon shapes as NumPy arrays.
        polys = get_feature_info(chip_json)

        # Getting a list of the polygons and saving masks as separate or single image files
        if len(polys) > 0: # checks if there are any polygons
            if single_file:
                if border > 0:
                    # The resulting mask (masked_polys) will have the polygons filled 
                    ## with different colors based on the border provided.
                    masked_polys = mask_polygons_together_with_border(chip_size, polys, border)
                else:
                    masked_polys = mask_polygons_together(chip_size, polys)
                save_one_mask(masked_polys, output_directory, mask_file)
            else:
                masked_polys = mask_polygons_separately(chip_size, polys)
                save_masks(masked_polys, output_directory, mask_file)

# --------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description=
        """mask_polygons.py: Takes in xBD dataset and masks polygons in the image\n\n
        WARNING: This could lead to hundreds of output images per input\n""")

    parser.add_argument('--input',
                        required=True,
                        metavar="/path/to/xBD/",
                        help='Path to parent dataset directory "xBD"')
    parser.add_argument('--single-file', 
                        action='store_true',
                        help='use to save all masked polygon instances to a single file rather than one polygon per mask file')
    parser.add_argument('--border',
                        default=0,
                        type=int,
                        metavar="positive integer for pixel border (e.g. 1)",
                        help='Positive integer used to shrink the polygon by')

    args = parser.parse_args()

    # Getting the list of the disaster types under the xBD directory
    disasters = next(walk(args.input))[1]

    for disaster in tqdm(disasters, desc='Masking', unit='disaster'):
        # Create the full path to the images, labels, and mask output directories
        image_dir = path.join(args.input, disaster, 'images')
        json_dir = path.join(args.input, disaster, 'labels')
        output_dir = path.join(args.input, disaster, 'masks')

        if not path.isdir(image_dir):
            print(
                "Error, could not find image files in {}.\n\n"
                .format(image_dir),
                file=stderr)
            exit(2)

        if not path.isdir(json_dir):
            print(
                "Error, could not find labels in {}.\n\n"
                .format(json_dir),
                file=stderr)
            exit(3)

        if not path.isdir(output_dir):
            makedirs(output_dir)

        mask_chips(json_dir, image_dir, output_dir, args.single_file, args.border)