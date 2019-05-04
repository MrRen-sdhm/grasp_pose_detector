import h5py
import numpy as np
import sys
import zarr

h5_file = h5py.File(sys.argv[1])
images = h5_file.get('images')
labels = h5_file.get('labels')

root = zarr.open(sys.argv[2], mode='w')
images_out = root.create_dataset('images', shape=images.shape, dtype='float32')
labels_out = root.create_dataset('labels', shape=labels.shape, dtype='int32')

images_out[:] = images[:]
labels_out[:] = labels[:]

#zarr_out = zarr.open(sys.argv[2], mode='w', shape=images.shape)
#for i in range(images.shape[0]):
#    print i
#    zarr_out[i] = images[i]
#    if i % 1000 == 0:
#        print(i, images.shape[0])
