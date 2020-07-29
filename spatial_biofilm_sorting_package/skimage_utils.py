import numpy as np

from skimage.measure import regionprops


def regionprops_channels(label, intensity, props):
    """
    Extract certain region properties from intensity img with more than one
    channel and return them as array.
    Arguments:
        - label: the label map that defines separate objects in the image
        - intensity: the intensity img, from which the properties should be
        calculated.
        - props: list of properties that should be extracted. Properties are
        named with strings.
    Returns:
        - properties: Array of shape (num_objects, len(props), num_channels),
        where num_objects is the number of objects in label and num_channels is
        the length of the last axis of intensity.
    """
    num_channels = intensity.shape[-1]
    regions = regionprops(label)
    num_objects = len(regions)

    properties = np.empty((num_objects, len(props), num_channels))

    for i in range(num_channels):
        regions = regionprops(label, intensity[..., i])
        for j in range(len(props)):
            properties[:, j, i] = np.asarray([
                getattr(reg, props[j]) for reg in regions
            ])
    return properties


def crop_img_from_prop(img, prop, channels_last=True):
    bbox = prop.bbox
    sl1 = slice(bbox[0], bbox[2])
    sl2 = slice(bbox[1], bbox[3])
    if img.ndim == 2:
        return img[sl1, sl2]
    elif channels_last:
        return img[sl1, sl2, :]
    else:
        return img[:, sl1, sl2]
