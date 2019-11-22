#!/bin/bash 

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

set -euo pipefail

# this function is called when Ctrl-C is sent
function trap_ctrlc ()
{
    # perform cleanup here
    echo "Ctrl-C or Error caught...performing clean up check /tmp/inference.log"

    if [ -d /tmp/inference ]; then
           rm -rf /tmp/inference
    fi

    exit 99
}

# initialise trap to call trap_ctrlc function
# when signal 2 (SIGINT) is received
trap "trap_ctrlc" 2 9 13 3

help_message () {
        printf "${0}: Runs the polygonization in inference mode\n\t-x: path to xview-2 repository\n\t-i: /full/path/to/input/pre-disaster/image.png\n\t-p: /full/path/to/input/post-disaster/image.png\n\t-o: /path/to/output.png\n\t-l: path/to/localization_weights\n\t-c: path/to/classification_weights\n\t-e /path/to/virtual/env/activate\n\t-y continue with local environment and without interactive prompt\n\n"
}

input=""
input_post=""
inference_base="/tmp/inference"
LOGFILE="/tmp/inference_log"
XBDIR=""
virtual_env=""
localization_weights=""
classification_weights=""
continue_answer="n"

if [ "$#" -lt 13 ]; then
        help_message
        exit 1 
fi 

while getopts "i:p:o:x:l:e:c:hy" OPTION
do
     case $OPTION in
         h)
             help_message
             exit 0
             ;;
         y)
             continue_answer="y"
             ;;
         o)
             output_file="$OPTARG"
             ;;
         x)
             XBDIR="$OPTARG"
             virtual_env="$XBDIR/bin/activate"
             ;;
         i)
             input="$OPTARG"
             ;;
         p)
             input_post="$OPTARG"
             ;;
         l)
             localization_weights="$OPTARG"
             ;;
         c)
             classification_weights="$OPTARG"
             ;;
         e)
             virtual_env="$OPTARG"
             ;;
         ?)
             help_message
             exit 0
             ;;
     esac
done

# Create the output directory if it doesn't exist 
mkdir -p "$inference_base"

if ! [ -f "$LOGFILE" ]; then
    touch "$LOGFILE"
fi

printf "==========\n" >> "$LOGFILE"
echo `date +%Y%m%dT%H%M%S` >> "$LOGFILE"
printf "\n" >> "$LOGFILE"

input_image=${input##*/}

label_temp="$inference_base"/"${input_image%.*}"/labels
mkdir -p "$label_temp"

printf "\n"

printf "\n"

# Run in inference mode
# Because of the models _have_ to be in the correct directory, they use relative paths to find the source (e.g. "../src") 
# sourcing the virtual environment packages if they exist
# this is *necessary* or all packages must be installed globally
if [ -f  "$virtual_env" ]; then
    source "$virtual_env"
else
    if [ "$continue_answer" = "n" ]; then 
        printf "Error: cannot source virtual environment  \n\tDo you have all the dependencies installed and want to continue? [Y/N]: "
        read continue_answer 
        if [ "$continue_answer" == "N" ]; then 
               exit 2
        fi 
    fi
fi

cd "$XBDIR"/spacenet/inference/

# Quietly running the localization inference to output a json with the predicted polygons from the supplied input image
printf "Running localization\n"
python3 ./inference.py --input "$input" --weights "$localization_weights" --mean "$XBDIR"/weights/mean.npy --output "$label_temp"/"${input_image%.*}".json >> "$LOGFILE" 2>&1

printf "\n" >> "$LOGFILE"

# Classification inferences start below
cd "$XBDIR"/model

# Replace the pre image here with the post
# We need to do this so the classification inference pulls the images from the post 
# Since post is where the damage occurs
printf "Grabbing post image file for classification\n"
disaster_post_file="$input_post"

mkdir -p "$inference_base"/output_polygons

printf "Running classification\n" 

# Extracting polygons from post image 
python3 ./process_data_inference.py --input_img "$disaster_post_file" --label_path "$label_temp"/"${input_image%.*}".json --output_dir "$inference_base"/output_polygons --output_csv "$inference_base"/output.csv >> "$LOGFILE" 2>&1

# Classifying extracted polygons 
python3 ./damage_inference.py --test_data "$inference_base"/output_polygons --test_csv "$inference_base"/output.csv --model_weights "$classification_weights" --output_json /tmp/inference/classification_inference.json >> "$LOGFILE" 2>&1

printf "\n" >> "$LOGFILE"

# Combining the predicted polygons with the predicted labels, based off a UUID generated during the localization inference stage  
printf "Formatting json and scoring image\n"
python3 "$XBDIR"/utils/combine_jsons.py --polys "$label_temp"/"${input_image%.*}".json --classes /tmp/inference/classification_inference.json --output "$inference_base/inference.json" >> "$LOGFILE" 2>&1
printf "\n" >> "$LOGFILE"

# Transforming the inference json file to the image required for scoring
printf "Finalizing output file" 
python3 "$XBDIR"/utils/inference_image_output.py --input "$inference_base"/inference.json --output "$output_file"  >> "$LOGFILE" 2>&1

#Cleaning up by removing the temporary working directory we created
printf "Cleaning up\n"
rm -rf "$inference_base"

printf "==========\n" >> "$LOGFILE"
printf "Done!\n"

