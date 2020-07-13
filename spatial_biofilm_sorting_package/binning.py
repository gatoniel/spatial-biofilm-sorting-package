import numpy as np


def bin_array_along_axis(x, num_bins, axis):
    """
    Creates equally spaced bins with respective labels for the bins from a
    given array x for each column along given axis. num_bins gives the number
    of created bins.

    x: Array to bin.
    num_bins: the number of bins to create.
    axis: Axis along which the bins are created.

    Returns:
    bins: Array of shape ((x.shape[axis], num_bins+1)) indicating start and
        stop for the bins.
    labels: Array of shape ((x.shape[axis], num_bins)) indicating the center
        of each bin.
    """
    try:
        bins = np.empty((x.shape[axis], num_bins + 1))
    except IndexError:
        raise np.AxisError(
            "axis {} is out of bounds for array of dimension {}".format(
                axis, x.ndim
            )
        )
    labels = np.empty((x.shape[axis], num_bins))

    # binning for each entry starts at minimum and ends at maximum value
    axis_list = list(range(x.ndim))
    del axis_list[axis]
    axis_list = tuple(axis_list)
    pos_min = x.min(axis=axis_list)
    pos_max = x.max(axis=axis_list)

    for i in range(x.shape[axis]):
        bins[i, :] = np.linspace(
            pos_min[i], pos_max[i], num_bins + 1, endpoint=True
        )
        labels[i, :] = bins[i, :-1] + (bins[i, 1]-bins[i, 0]) / 2

    return bins, labels
