import mxnet as mx
import numpy as np
import cv2

data_shape = (3,128,128)
batch_sz = 24
label_width = 68*2

train = mx.io.ImageRecordIter(
		path_imgrec = "train.rec",
		path_imglist = "train.lst",
		data_shape = data_shape,
		batch_size  = batch_sz,
		shuffle = True,
		label_width = label_width,
        max_random_contrast=0.3,
        max_random_illumination=0.3,
        max_random_h=30,
        max_random_l=40,
        max_random_s=40)

val = mx.io.ImageRecordIter(
		path_imgrec = "val.rec",
		path_imglist = "val.lst",
		data_shape = data_shape,
		batch_size  = batch_sz,
		shuffle = True,
		label_width = label_width)

for batch in train:
    for img,label in zip(batch.data[0].asnumpy(),batch.label[0].asnumpy()):
        img = cv2.cvtColor(img.transpose([1,2,0])/255.0,cv2.COLOR_BGR2RGB)
        pts_int = label.astype(int)
        for i in range(68):
            cv2.circle(img,(pts_int[i*2],pts_int[i*2+1]),1,(0,0,255),2)
        cv2.imshow('s',img)
	cv2.waitKey(-1)
