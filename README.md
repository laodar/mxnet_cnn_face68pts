# mxnet_cnn_face68pts



## requirement

-opencv

    only tested on 2.4.9.1

-mxnet

    only tested on 0.9.5

-mtcnn

    I use https://github.com/pangyupo/mxnet_mtcnn_face_detection to do face cropping and alignment
    more about cropping,see model_test_128.py
    
## best pretrained model up to now
 
you can download smpnet-0730.params here:https://pan.baidu.com/s/1nvfu3Nz

you can get a smaller pretrained model here:https://pan.baidu.com/s/1pLa7Ehd

## camera test:

python model_test_128.py

or

python model_test_64.py


