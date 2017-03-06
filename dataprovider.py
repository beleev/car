import h5py
from paddle.trainer.PyDataProvider2 import *

def hook(settings, attrNum, imageWidth, 
         imageHeight, channel, is_train, **kwargs):
    settings.input_types = [
        dense_vector(imageWidth*imageHeight*channel),
        dense_vector(attrNum)
    ]
    settings.is_train = is_train

def load_data(file_name):
    data = h5py.File(file_name, 'r')
    image = data['image'].value
    curve = []
    curve.append(float(data['curve'].value))
    data.close()
    return image, curve
    
@provider(init_hook=hook, min_pool_size=0, pool_size=200, cache=CacheType.NO_CACHE)
def process(settings, file_list):
    with open(file_list, 'rb') as file_data:
        lines = [line.strip() for line in file_data]
        for file_name in lines:
            yield load_data(file_name.strip())
