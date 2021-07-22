import os
from glob import glob

# Delete 2000 Images in train pneumonia

path = '/Users/srinivas/Desktop/Pneumonia_detection/data/train/PNEUMONIA/'
no_of_images = glob(path + '*')

length_before = len(no_of_images)

# Deleting 2000 Images

if length_before < 2000:
    print('No deletion Required')

else:
    for i in range(2000):
        os.remove(no_of_images[i])

length_after = len(glob(path + '*'))

print(length_before, length_after)
