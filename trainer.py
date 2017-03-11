from paddle.trainer_config_helpers import *

is_predict = get_config_arg("is_predict", bool, False)

####################Data Configuration ##################
if not is_predict:

    args = {
        'attrNum' : 1,
        'imageWidth' : 120,
        'imageHeight' : 120,
        'channel' : 3
    }

    define_py_data_sources2(
        train_list="files",
        test_list="files",
        module='dataprovider',
        obj='process',
        args=args)

######################Algorithm Configuration #############
settings(
    batch_size=128,
    learning_rate=0.1 / 128.0,
    learning_rate_decay_a=0.1,
    learning_rate_decay_b=50000 * 100,
    learning_rate_schedule='discexp',
    learning_method=MomentumOptimizer(0.9),
    regularization=L2Regularization(0.0005 * 128), )
          
#######################Network Configuration #############
def vgg_bn_drop(input):
    def conv_block(ipt, num_filter, groups, dropouts, num_channels=None):
        return img_conv_group(
            input=ipt,
            num_channels=num_channels,
            pool_size=2,
            pool_stride=2,
            conv_num_filter=[num_filter] * groups,
            conv_filter_size=3,
            conv_act=ReluActivation(),
            conv_with_batchnorm=True,
            conv_batchnorm_drop_rate=dropouts,
            pool_type=MaxPooling())

    conv1 = conv_block(input, 64, 1, [0.3, 0], 3)
    conv2 = conv_block(conv1, 128, 2, [0.4, 0])
    conv3 = conv_block(conv2, 256, 2, [0.4, 0.4, 0])
    #conv4 = conv_block(conv3, 512, 3, [0.4, 0.4, 0])
    #conv5 = conv_block(conv4, 512, 3, [0.4, 0.4, 0])

    #drop = dropout_layer(input=conv5, dropout_rate=0.5)
    drop = dropout_layer(input=conv3, dropout_rate=0.5)
    fc1 = fc_layer(input=drop, size=512, act=LinearActivation())
    bn = batch_norm_layer(
        input=fc1, act=ReluActivation(), layer_attr=ExtraAttr(drop_rate=0.5))
    fc2 = fc_layer(input=bn, size=512, act=LinearActivation())
    return fc2


datadim = 3 * 120 * 120
data = data_layer(name='image', size=datadim)
net = vgg_bn_drop(data)
#out = fc_layer(input=net, size=1, act=LinearActivation())
out = fc_layer(input=net, size=1, act=SigmoidActivation())
if not is_predict:
    lbl = data_layer(name="curve", size=1)
    cost = regression_cost(input=out, label=lbl)
    outputs(cost)
else:
    outputs(out)
