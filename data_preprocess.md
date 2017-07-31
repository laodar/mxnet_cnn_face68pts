# data preprocess

## 1.crop with different boxs and fix the label,get cropped images and .lst file(for im2rec)

run

python crop_helen.py

for helen dataset

python crop_LFPW.py

for LFPW dataset

## 2.do data agumentation:rotate,shift and stretch the images,then fix the corresponding label

python agument.py

remember to change the save_dir and the .lst filenames used

## 3.remove the failed samples,and split data into training set and validation set

python remove_shuffle_and_split_lst.py

remember to change the save_dir and the .lst filenames used

## 4.run /mxnet/bin/im2rec to get the .rec files by .lst files
