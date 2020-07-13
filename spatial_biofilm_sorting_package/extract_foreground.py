import numpy as np
from skimage.filters import threshold_multiotsu
from skimage.measure import label, regionprops
from skimage.morphology import binary_dilation, binary_erosion, disk


def get_edge_mask(true_fg, dil_radius, ero_radius):
    selem_dilation = disk(dil_radius)
    selem_erosion = disk(ero_radius)
    edge_dilation = binary_dilation(true_fg, selem_dilation)
    edge_erosion = binary_erosion(true_fg, selem_erosion)

    return np.logical_not(np.logical_or(
        np.logical_not(edge_dilation),
        edge_erosion,
    ))


def extract_mask(img, mask):
    masked_img = np.zeros_like(img)
    masked_img[mask] = img[mask]
    return masked_img


def new_edge_threshold(img, mask, true_fg):
    thresholds = threshold_multiotsu(img[mask], classes=2)
    return np.logical_or(np.digitize(img, thresholds), true_fg)


def extract_foreground(img):
    """
    Extracts the single largest object from grayscale image img.
    Returns a boolean mask and a skimage RegionProperty for that object.
    """
    thresholds = threshold_multiotsu(img, classes=3)
    true_fg = np.digitize(img, [thresholds[0]])

    mask_edge = get_edge_mask(true_fg, 100, 20)
    masked_img = extract_mask(img, mask_edge)
    new_true_fg = new_edge_threshold(masked_img, mask_edge, true_fg)

    new_mask_edge = get_edge_mask(new_true_fg, 50, 10)
    new_masked_img = extract_mask(img, new_mask_edge)
    last_true_fg = new_edge_threshold(
        new_masked_img, new_mask_edge, new_true_fg,
    )

    labels = label(last_true_fg)
    props = regionprops(labels, img)

    areas = np.asarray([prop.area for prop in props])
    ind = areas.argmax()

    prop = props[ind]
    mask = labels == prop.label

    return mask, prop
