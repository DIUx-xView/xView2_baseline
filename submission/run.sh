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

# Running inference using CLI arguments passed in
# 1) path to xview2-baseline 
# 2) path to input pre image 
# 3) path to input post image 
# 4) path to output localization image 
# 5) path to output localization+classification image

if [ $# -lt 5 ]; then 
        echo "run.sh: /path/to/xview2-baseline/ /path/to/input/pre/image /path/to/input/post/image /path/to/localization/output/image /path/to/classification/output/image" 
else
    "$1"/utils/inference.sh -x "$1" -i "$2"  -p "$3" -o "$4"  -l "$1"/weights/localization.h5 -c "$1"/weights/classification.hdf5 -y

    # The two images we will use for scoring will be identical so just copying the output localization png to the classification path 
    cp "$4" "$5"
fi
