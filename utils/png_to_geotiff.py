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

from osgeo import gdal
from osgeo.gdalconst import *

import json
import numpy as np


geotransforms = json.load(open("xview_geotransforms.json", "r"))
geomatrix = geotransforms["hurricane-michael_00000202_post_disaster.png"][0]
projection = geotransforms["hurricane-michael_00000202_post_disaster.png"][1]

inDs = gdal.Open("hurricane-michael_00000202_post_disaster.png")

rows = inDs.RasterYSize
cols = inDs.RasterXSize

outDs = gdal.GetDriverByName('GTiff').Create("hurricane-michael_00000202_post_disaster.tif", rows, cols, 3, GDT_Int16)

for i in range(1,4):
    outBand = outDs.GetRasterBand(i)
    outData = np.array(inDs.GetRasterBand(i).ReadAsArray())
    outBand.WriteArray(outData, 0, 0)
    outBand.FlushCache()
    outBand.SetNoDataValue(-99)

outDs.SetGeoTransform(geomatrix)
outDs.SetProjection(projection)

