import mxnet as mx
import numpy as np
import cv2
import os,sys
sys.path.append('..')
from mtcnn_detector import MtcnnDetector

save_model_prefix = "cpt/cnet3"
epoch = 545

detector = MtcnnDetector(model_folder='../model', ctx=mx.gpu(0), num_worker = 1 , accurate_landmark = False)
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
        R = 1.25 * r
        left = int(nose_x - R)
        right = int(nose_x + R)
        top = int(nose_y - R)
        bottom = int(nose_y + R)
        img_crop = cv2.resize(frame[top:bottom,left:right,:],dsize=(64,64))
        img = cv2.cvtColor(img_crop.copy(),cv2.COLOR_BGR2RGB)[None,:,:,:]
        #img = np.concatenate([img.transpose(0,3,1,2),np_pos],axis=1)
        pts_int = model.predict(img.transpose(0,3,1,2))[0].astype(int)
        print pts_int
        for i in range(68):
            cv2.circle(img_crop,(pts_int[i*2],pts_int[i*2+1]),1,(255,0,0),1)
        cv2.imshow('marks',cv2.resize(img_crop,dsize=(0,0),fx=3,fy=3))
        cv2.waitKey(10)
