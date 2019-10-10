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

def combine_output(pred_polygons, pred_classification, output_file):
    """
    :param pred_polygons: the file path to the localization inference output json  
    :param pre_classification: the file path to the classification inference output json
    :param output_file: the file path to store the combined json file
    """

    # Skeleton of the json with null values 
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

    # Open the classification json 
    with open(pred_classification) as labels:
        label_json = json.load(labels)
        
        # Open the localization json 
        with open(pred_polygons) as polys:
            poly_json = json.load(polys)

            # Match UUIDs from the two jsons and combine in output_json skeleton 
            for p in poly_json['features']['xy']:
                p['properties']['subtype'] = label_json[p['properties']['uid']]
                output_json['features']['xy'].append(p)
    
    # Finally save out the combined json file 
    with open(output_file, 'w') as out: 
        json.dump(output_json, out)

if __name__ == '__main__': 
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description=
        """combine_jsons.py: combines the outputs of localization and classification inference into a single output json"""
    )
    parser.add_argument('--polys',
                        required=True,
                        metavar='/path/to/input/polygons.json',
                        help="Full path to the json from polygonize.py")
    parser.add_argument('--classes',
                        required=True,
                        metavar='/path/to/classifications.json',
                        help="Full path to the json from tensor_inf.py"
    )
    parser.add_argument('--output',
                        required=True,
                        metavar='/path/to/pred.json',
                        help="Full path to save the final single output file to"
    )

    args = parser.parse_args()

    # Combining the json based off the uuid assigned at the polygonize stage
    combine_output(args.polys, args.classes, args.output)
