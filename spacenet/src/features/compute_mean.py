#!/usr/bin/env python

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

import argparse
import sys
import os

import six
import numpy as np

try:
	from PIL import Image
	available = True
except ImportError as e:
	available = False
	_import_error = e

from chainer.dataset import dataset_mixin


def _check_pillow_availability():
	if not available:
		raise ImportError('PIL cannot be loaded. Install Pillow!\n'
						  'The actual import error is as follows:\n' +
						  str(_import_error))


def _read_image_as_array(path, dtype):
	f = Image.open(path)
	try:
		image = np.asarray(f, dtype=dtype)
	finally:
		# Only pillow >= 3.0 has 'close' method
		if hasattr(f, 'close'):
			f.close()
	return image


class ImageDataset(dataset_mixin.DatasetMixin):
	
	def __init__(self, paths, root='.', dtype=np.float32):
		_check_pillow_availability()
		if isinstance(paths, six.string_types):
			with open(paths) as paths_file:
				paths = [path.rstrip() for path in paths_file]
		self._paths = paths
		self._root = root
		self._dtype = dtype

	def __len__(self):
		return len(self._paths)

	def get_example(self, i):
		path = os.path.join(self._root, self._paths[i])
		image = _read_image_as_array(path, self._dtype)

		if image.ndim == 2:
			# image is greyscale
			image = image[:, :, np.newaxis]
		return image.transpose(2, 0, 1)


def compute_mean(dataset):
	print('compute mean image')
	sum_color = 0
	N = len(dataset)
	for i, image in enumerate(dataset):
		sum_color += image.mean(axis=2, keepdims=False).mean(axis=1, keepdims=False)
		sys.stderr.write('{} / {}\r'.format(i, N))
		sys.stderr.flush()
	sys.stderr.write('\n')
	return sum_color / N


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Compute images mean array')
	
	parser.add_argument('dataset',
						help='Path to training image-label list file')
	parser.add_argument('--root', '-R', default='.',
						help='Root directory path of image files')
	parser.add_argument('--output', '-o', default='mean.npy',
						help='path to output mean array')
	args = parser.parse_args()

	dataset = ImageDataset(args.dataset, args.root)
	mean = compute_mean(dataset)

	np.save(args.output, mean)
	
