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

from PIL import Image, ImageDraw
import rasterio.features
import shapely.geometry
import numpy as np

def save_img(path_to_image, path_to_localization, path_to_damage, path_to_output): 
    no_damage_polygons = []
    minor_damage_polygons = []
    major_damage_polygons = []
    destroyed_polygons = []

    # Load the challenge output localization image
    localization = Image.open(path_to_localization)
    loc_arr = np.array(localization)

    # If the localization has damage values convert all non-zero to 1
    # This helps us find where buildings are, and then use the damage file
    # to get the value of the classified damage
    loc_arr = (loc_arr >= 1).astype(np.uint8)

    # Load the challenge output damage image
    damage = Image.open(path_to_damage)
    dmg_arr = np.array(damage)

    # Use the localization to get damage only were they have detected buildings
    mask_arr = dmg_arr*loc_arr
    
    # Get the value of each index put into a dictionary like structure
    shapes = rasterio.features.shapes(mask_arr)
    
    # Iterate through the unique values of the shape files 
    # This is a destructive iterator or else we'd use the pythonic for x in shapes if x blah 
    for shape in shapes:
        if shape[1] == 1:
            no_damage_polygons.append(shapely.geometry.Polygon(shape[0]["coordinates"][0]))
        elif shape[1] == 2:
            minor_damage_polygons.append(shapely.geometry.Polygon(shape[0]["coordinates"][0]))
        elif shape[1] == 3:
            major_damage_polygons.append(shapely.geometry.Polygon(shape[0]["coordinates"][0]))
        elif shape[1] == 4:
            destroyed_polygons.append(shapely.geometry.Polygon(shape[0]["coordinates"][0]))
        elif shape[1] == 0:
            continue
        else:
            print("Found non-conforming damage type: {}".format(shape[1]))
    
    # Loading post image
    img = Image.open(path_to_image) 
    
    draw = ImageDraw.Draw(img, 'RGBA')
    
    damage_dict = {
        "no-damage": (0, 255, 0, 100),
        "minor-damage": (0, 0, 255, 125),
        "major-damage": (255, 69, 0, 125),
        "destroyed": (255, 0, 0, 125),
        "un-classified": (255, 255, 255, 125)
    }
    
    # Go through each list and write it to the post image we just loaded
    for polygon in no_damage_polygons:
        x,y = polygon.exterior.coords.xy
        coords = list(zip(x,y))
        draw.polygon(coords, damage_dict["no-damage"])

    for polygon in minor_damage_polygons:
        x,y = polygon.exterior.coords.xy
        coords = list(zip(x,y))
        draw.polygon(coords, damage_dict["minor-damage"])

    for polygon in major_damage_polygons:
        x,y = polygon.exterior.coords.xy
        coords = list(zip(x,y))
        draw.polygon(coords, damage_dict["major-damage"])

    for polygon in destroyed_polygons:
        x,y = polygon.exterior.coords.xy
        coords = list(zip(x,y))
        draw.polygon(coords, damage_dict["destroyed"])

    img.save(path_to_output)

if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description=
        """submission_to_overlay_polys.py: takes the submission file and overlays the predicted values over the image and saves the image with the polygons overtop to a new file"""
    )
    parser.add_argument('--image',
                        required=True,
                        metavar='/path/to/post_disaster.png',
                        help="Full path to the image to use to overlay the predictions onto")
    parser.add_argument('--damage',
                        required=True,
                        metavar='/path/to/challenge/prediction_damage.png',
                        help="Full path to the prediction json output from model"
    )
    parser.add_argument('--localization',
                        required=True,
                        metavar='/path/to/challenge/prediction_damage.png',
                        help="Full path to the prediction json output from model"
    )
    parser.add_argument('--output',
                        required=True,
                        metavar='/path/to/save/img_with_overlays.png',
                        help="Full path to save the final single output file to (include filename.png)"
    )

    args = parser.parse_args()

    # run main function 
    save_img(args.image, args.localization, args.damage, args.output)

