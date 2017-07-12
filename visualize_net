import mxnet as mx


save_model_prefix = "cpt/ibn2"
epoch = 50
checkpoint = mx.callback.do_checkpoint(save_model_prefix,1)
pretrained = mx.model.FeedForward.load(save_model_prefix,epoch,ctx=mx.gpu())
mx.viz.plot_network(symbol=pretrained.symbol,shape={'data':(32,5,128,128)}).view('s')

