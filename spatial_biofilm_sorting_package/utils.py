import numpy as np
from tifffile import imread
from pathlib import Path


def normalize_axis(x, axis, dtype=float):
    """
    Normalizes values of array x in the range 0...1 for each entry along axis.

    x: numpy array to be normalized
    axis: axis along which to normalize

    raises numpy.AxisError
    """
    x = x.astype(dtype)
    ind_list = [slice(None) for i in range(x.ndim)]
    try:
        for i in range(x.shape[axis]):
            ind_list[axis] = i
            ind = tuple(ind_list)
            minn = x[ind].min()
            maxx = x[ind].max()
            x[ind] = (x[ind]-minn) / (maxx-minn)
    except IndexError:
        raise np.AxisError(
            "axis {} is out of bounds for array of dimension {}".format(
                axis, x.ndim
            )
        )
    return x


def grayscale(x, axis):
    return normalize_axis(x, axis).sum(axis)


def read_as_grayscale(file, axis):
    return grayscale(imread_wrapper(file), axis)


def imread_wrapper(file):
    """
    Imread returns the first series of a tiff-file. It detects that ome.tiffs
    belong together. So if you try to open the second ome.tiff-file it still
    returns the first series. This wrapper tries to detect the series the user
    implies to open from the filename supplied and returns the image belonging
    to that series.

    file: the file to open. The file name must only consist of the series (int)
    with possible suffixes ome.tiff, ome.tif, tiff, tif.

    returns a series from selected ome.tiff-file.
    """
    p = Path(file)
    if len(p.suffixes) == 2:
        series = int(Path(p.stem).stem)
    else:
        series = int(p.stem)
    return imread(file, series=series)
