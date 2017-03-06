import h5py
from paddle.trainer.PyDataProvider2 import *

def hook(settings, attrNum, imageWidth, 
         imageHeight, channel, is_train, **kwargs):
    settings.input_types = [
        dense_vector(imageWidth*imageHeight*channel),
        dense_vector(attrNum)
    ]
    settings.is_train = is_train

@provider(init_hook=hook, min_pool_size=0, pool_size=1000 )
def process(settings, file_list):
    with open(file_list, 'rb') as file_data:
        lines = [line.strip() for line in file_data]
        for file_name in lines:
            data = h5py.File(file_name.strip(), 'r')
            image = data['image'].value
            curve = []
            curve.append(float(data['curve'].value))
            data.close()
            yield image, curve
