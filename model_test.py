import mxnet as mx
import numpy as np
import cv2
import os,sys
sys.path.append('..')
from mtcnn_detector import MtcnnDetector

save_model_prefix = "cpt/ibn2"
epoch = 55

np_x = np.array([range(128)]*128)*2.0
np_pos = np.stack([np.stack([np_x,np_x.transpose(1,0)],axis=0)]*1,axis=0)

detector = MtcnnDetector(model_folder='../model', ctx=mx.gpu(0), num_worker = 1 , accurate_landmark = True)
model = mx.model.FeedForward.load(save_model_prefix,epoch,ctx=mx.gpu())

camera = cv2.VideoCapture(0)

while True:
    grab, frame = camera.read()
    rs = detector.detect_face(frame)
    if rs is not None:
        bb = rs[0][0]
        pts = rs[1][0]
        nose_x = pts[2]
        nose_y = pts[7]
        r_left = abs(nose_x - bb[0])
        r_right = abs(nose_x - bb[2])
        r_top = abs(nose_y - bb[1])
        r_bottom = abs(nose_y - bb[3])
        r = max(r_left,r_right,r_top,r_bottom)
        R = 1.3 * r
        left = int(nose_x - R)
        right = int(nose_x + R)
        top = int(nose_y - R)
        bottom = int(nose_y + R)
        img_crop = cv2.resize(frame[top:bottom,left:right,:],dsize=(128,128))
        img = cv2.cvtColor(img_crop.copy(),cv2.COLOR_BGR2RGB)[None,:,:,:]
        img = np.concatenate([img.transpose(0,3,1,2),np_pos],axis=1)
        pts_int = model.predict(img)[0].astype(int)
        print pts_int
        for i in range(68):
            cv2.circle(img_crop,(pts_int[i*2],pts_int[i*2+1]),1,(255,0,0),1)
        cv2.imshow('marks',img_crop)
        cv2.waitKey(10)
