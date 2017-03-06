import h5py
import numpy
from optparse import OptionParser

def resample(image, step):
    index = [i*step for i in range(320/step)]
    image = numpy.delete(image, index, 0)
    image = numpy.delete(image, index, 1)
    return image

def split_h5_to_pic(h5_image, h5_attr, outpath):
    images = h5py.File(h5_image, 'r')
    attrs = h5py.File(h5_attr, 'r')
    for i in attrs["attrs"]:
        image_time = "{:.3f}".format(i[0])
        image = images[image_time].value
        resampled_image = resample(image, 2)
        outfile = h5py.File((outpath + str(image_time)), 'w')
        outfile['image'] = resampled_image.flatten().astype('float32')
        outfile['curve'] = i[3]
        outfile.close()
    images.close()
    attrs.close()

def option_parser():
    parser = OptionParser(usage="usage: python preprcoess.py "\
                          "-p data_dir -i data_id -c is_cluster")
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
    parser.add_option(
        "-c",
        "--is_cluster",
        action="store",
        dest="is_cluster",
        help="data for cluster.")
    return parser.parse_args()

if __name__ == '__main__':
    options, args = option_parser()
    data_path = options.path
    is_cluster = options.is_cluster
    data_id = options.data_id
    image = data_path + "image/" + data_id + ".h5"
    attr = data_path + "attr/" + data_id + ".h5"
    split_h5_to_pic(image, attr, data_path)
    if is_cluster:
        dispatch()
