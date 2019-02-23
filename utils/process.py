from __future__ import division
from __future__ import print_function
import os, scipy.io
import numpy as np
import rawpy
import tifffile

def raw2tiff(raw_dir, output_dir):
    raw = rawpy.imread(raw_dir)
    processed = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
    filename = os.path.splitext(os.path.split(raw_dir)[-1])[0]
    tiff_file = os.path.join(output_dir, filename+'.tiff')
    tifffile.imwrite(tiff_file, data=processed)
    print('saved:', tiff_file)

def pack_raw(raw):
    #pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32) 
    im = np.maximum(im - 512,0)/ (16383 - 512) #subtract the black level

    im = np.expand_dims(im,axis=2) 
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2,0:W:2,:], 
                        im[0:H:2,1:W:2,:],
                        im[1:H:2,1:W:2,:],
                        im[1:H:2,0:W:2,:]), axis=2)
    return out
    