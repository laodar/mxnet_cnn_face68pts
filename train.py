import mxnet as mx
import numpy as np
import cv2
import inception_bn
import logging

logging.basicConfig(level=logging.DEBUG)

batch_sz = 32
img_sz = 128
np_x = np.array([range(128)]*128)*2.0
np_pos = np.stack([np.stack([np_x,np_x.transpose(1,0)],axis=0)]*batch_sz,axis=0)
nd_pos = mx.nd.array(np_pos)

class NewIter(mx.io.DataIter):
    def __init__(self, data_iter):
        super(NewIter, self).__init__()
        self.data_iter = data_iter
        self.batch_size = self.data_iter.batch_size

    @property
    def provide_label(self):
        label = self.data_iter.getlabel()
        return self.data_iter.provide_label

    @property
    def provide_data(self):
        return [('data',(batch_sz,5,img_sz,img_sz))]

    def next(self):
        batch = self.data_iter.next()

        data = [mx.nd.concatenate([batch.data[0],nd_pos],axis=1)]
        return mx.io.DataBatch(data, batch.label, pad=batch.pad, index=batch.index)

    def hard_reset(self):
        self.data_iter.hard_reset()

    def reset(self):
        self.data_iter.reset()

class Multi_Accuracy(mx.metric.EvalMetric):
    def __init__(self, num=None):
        super(Multi_Accuracy, self).__init__('multi-error', num)

    def update(self, labels, preds):
        preds = preds[0].asnumpy().astype('float32')
        labels = labels[0].asnumpy().astype('float32')
        for i in range(labels.shape[1]):
            pred_label = preds[:,i]
            label = labels[:,i]
            self.sum_metric[i] += (pred_label - label).T.dot(pred_label - label)
            self.num_inst[i] += pred_label.shape[0]

data_shape = (3, img_sz, img_sz)
ctx = mx.gpu(0)
num_epochs = 1000
label_width = 2*68
dataset_sz = 175157//100

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

train = NewIter(train)
val = NewIter(val)
epoch = 91
save_model_prefix = "cpt/ibn4"
checkpoint = mx.callback.do_checkpoint(save_model_prefix,1)
pretrained = mx.model.FeedForward.load(save_model_prefix,epoch,ctx=mx.gpu())

model = mx.model.FeedForward(
    ctx=ctx,
    #symbol=inception_bn.get_symbol(),
    symbol=pretrained.symbol,
    aux_params=pretrained.aux_params,
    arg_params=pretrained.arg_params,
    begin_epoch=epoch,
    num_epoch=num_epochs,
    learning_rate=0.00001,
    optimizer='adam',
    #momentum=0.9,
    wd=0.0008,
    initializer=mx.init.Xavier(factor_type="in", magnitude=2.34))

model.fit(
    X=train,
    eval_data=val,
    eval_metric=Multi_Accuracy(num=label_width),
    batch_end_callback=mx.callback.Speedometer(batch_sz, dataset_sz//batch_sz),
    epoch_end_callback=checkpoint)
