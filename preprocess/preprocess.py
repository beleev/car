import h5py
import numpy
import random
from optparse import OptionParser

size = 320

def resample(image, step):
    index = []
    for i in range(size):
        if i%step:
           index.append(i)
    image = numpy.delete(image, index, 0)
    image = numpy.delete(image, index, 1)
    return image


def split_h5_to_pic(h5_image, h5_attr, outpath, step):
    images = h5py.File(h5_image, 'r')
    attrs = h5py.File(h5_attr, 'r')
    imagelen = len(attrs["attrs"])
    mean = numpy.zeros((size*size*3/step/step),dtype="float32")
    for i in range(1000):
        num = random.randint(0, imagelen)
        image_time = "{:.3f}".format(attrs["attrs"][num][0])
        mean += resample(images[image_time].value, step).flatten().astype('float32')
    mean /= 1000
    for i in attrs["attrs"]:
        image_time = "{:.3f}".format(i[0])
        image = images[image_time].value
        resampled_image = resample(image, step)
        outfile = h5py.File((outpath + str(image_time)), 'w')
        outfile['image'] = resampled_image.flatten().astype('float32') - mean
        outfile['curve'] = i[3]
        outfile.close()
    images.close()
    attrs.close()

def option_parser():
    parser = OptionParser(usage="usage: python preprcoess.py "\
                          "-p data_dir -i data_id")
    parser.add_option(
        "-p",
        "--path",
        action="store",
        dest="path",
        help="data directory.")
    parser.add_option(
        "-i",
        "--id",
        action="store",
        dest="data_id",
        help="data id.")
    return parser.parse_args()

if __name__ == '__main__':
    options, args = option_parser()
    data_path = options.path
    data_id = options.data_id
    image = data_path + "image/" + data_id + ".h5"
    attr = data_path + "attr/" + data_id + ".h5"
    split_h5_to_pic(image, attr, data_path, 4)
