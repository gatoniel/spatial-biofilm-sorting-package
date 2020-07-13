import numpy as np
import pandas as pd


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


def bin_a_from_b(a, b, num_bins):
    """
    Creates histograms for a based on b. Values in a are pooled together, when
    they fall into the same bin in b. num_bins gives the number of bins, that
    are created.

    a: array of shape ((num_elems, num_prop1))
    b: array of shape ((num_elems, num_prop2))
    num_bins: number of bins to create

    Returns:
    mu_sigma: array of shape((num_prop1, num_prop2, num_bins, 4)). For each bin
    in a for each property of a gives the mean, sigma and normalized values of
    those for b.
    """
    if not a.shape[0] == b.shape[0]:
        raise ValueError("a and b must have same length in first dimension")
    num_prop1 = a.shape[1]
    num_prop2 = b.shape[1]

    # Create bins for b
    bins, labels = bin_array_along_axis(b, num_bins, 1)
    # to be sure, that no elements at the left most and right most edges are
    # lost, we subtract -1 and add 1 add the lower and upper ends
    bins[:, 0] -= 1
    bins[:, -1] += 1

    mu_sigma = np.empty((num_prop1, num_prop2, num_bins, 2))
    dfs = [[] for i in range(num_prop1)]
    pdf = np.empty_like(labels)

    for i in range(num_prop1):
        for j in range(num_prop2):
            ab = np.stack((a[:, i], b[:, j]), axis=-1)

            df = pd.DataFrame(ab, columns=["prop1", "prop2"])
            df["binned"] = pd.cut(
                df["prop2"],
                bins=bins[j, :], labels=labels[j, :]
            )
            tmp_groups = df.groupby("binned")

            if i == 0:
                counts = tmp_groups["prop2"].count()
                pdf[j, :] = counts / counts.sum()

            mu_sigma[i, j, :, 0] = tmp_groups["prop1"].mean()
            mu_sigma[i, j, :, 1] = tmp_groups["prop1"].std()

            dfs[i].append(df)

    return mu_sigma, dfs, pdf, labels, bins
