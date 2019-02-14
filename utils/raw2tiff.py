from __future__ import division
from __future__ import print_function
import os, scipy.io
import glob
import argparse
import numpy as np
import rawpy
import tifffile

parser = argparse.ArgumentParser()

parser.add_argument('in_dir', metavar = 'input path', help = 'path to input folder')
parser.add_argument('out_dir', metavar = 'output path', help = 'path to output folder')

args = parser.parse_args()

input_dir = args.in_dir
output_dir = args.out_dir

if not os.path.isdir(input_dir):
    print('input path is not a path')
    exit(1)

if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

raw_filelist = glob.glob(os.path.join(input_dir, '*.ARW'))

for raw_file in raw_filelist:  
    raw = rawpy.imread(raw_file)
    processed = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)

    filename = os.path.splitext(os.path.split(raw_file)[-1])[0]
    tiff_file = os.path.join(output_dir, filename+'.tiff')
    tifffile.imwrite(tiff_file, data=processed)
    print('saved:', tiff_file)
