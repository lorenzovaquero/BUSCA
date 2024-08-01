import cv2
import math
import numpy as np
from scipy.spatial.distance import cdist


def missing_candidate_bbox(seq_len=None, flavour='ltrb'):
    if flavour == 'ltrb':
        bbox = np.array([np.finfo('float32').min, np.finfo('float32').min, np.finfo('float32').min/100.0, np.finfo('float32').min/100.0])  # [ltrb]

    elif flavour == 'ltwh':
        bbox = np.array([np.finfo('float32').min, np.finfo('float32').min, -np.finfo('float32').min/100.0, -np.finfo('float32').min/100.0])  # [ltwh]
    
    else:
        raise ValueError('Unknown flavour: {}'.format(flavour))

    if seq_len is not None:
        bbox = np.tile(bbox, (seq_len, 1))

    return bbox


def center_distance(atracks, btracks, weight_size=False):
    """
    Compute center-to-center dictances
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype dist_matrix np.ndarray
    """

    if len(atracks)>0 and isinstance(atracks[0], np.ndarray):
        atlbrs = atracks
    else:
        atlbrs = np.array([track.tlbr for track in atracks])
    
    if len(btracks) > 0 and isinstance(btracks[0], np.ndarray):
        btlbrs = btracks
    else:
        btlbrs = np.array([track.tlbr for track in btracks])

    if len(atlbrs) == 0 or len(btlbrs) == 0:
        return np.zeros((len(atracks), len(btracks)), dtype=np.float)
    
    a_centers = (atlbrs[:, :2] + atlbrs[:, 2:]) / 2.0
    b_centers = (btlbrs[:, :2] + btlbrs[:, 2:]) / 2.0

    dist_matrix = cdist(a_centers, b_centers, metric='euclidean')  # Euclidean distance

    if weight_size:
        a_sizes = np.sqrt((atlbrs[:, 2] - atlbrs[:, 0]) * (atlbrs[:, 3] - atlbrs[:, 1]))
        b_sizes = np.sqrt((btlbrs[:, 2] - btlbrs[:, 0]) * (btlbrs[:, 3] - btlbrs[:, 1]))

        a_sizes = np.tile(a_sizes, (len(b_sizes), 1)).T
        b_sizes = np.tile(b_sizes, (len(a_sizes), 1))

        weights = np.maximum(a_sizes / b_sizes, b_sizes / a_sizes)
        dist_matrix = dist_matrix * weights
    
    return dist_matrix

def get_bbox_crop(im, bbox_real_scale, output_size=(128, 384), normalize=True, ghost_normalize=True):  # Like the one in network.py       
    if ghost_normalize:
        input_pixel_mean = np.array([0.406, 0.456, 0.485])  # BGR
        input_pixel_std = np.array([0.225, 0.224, 0.299])  # BGR
    else:
        input_pixel_mean = np.array([0.406, 0.456, 0.485])  # BGR
        input_pixel_std = np.array([0.225, 0.224, 0.229])  # BGR

    cutout = _cutout_with_pad(im=im, bbox=bbox_real_scale)
    crop = cv2.resize(cutout, output_size, interpolation=cv2.INTER_LINEAR)

    if normalize:
        crop = crop.astype(np.float32) / 255.0
        crop -= input_pixel_mean
        crop /= input_pixel_std

    return crop
    
def _cutout_with_pad(im, bbox):
    x1, y1, x2, y2 = bbox

    assert im is not None, "Image is None"

    # Instead of casting directly to int, we floor x1, y1 and ceil x2, y2 to ensure that the crop always contains the bbox (and to avoid empty cutouts)
    x1 = int(math.floor(x1))
    y1 = int(math.floor(y1))
    x2 = int(math.ceil(x2))
    y2 = int(math.ceil(y2))

    # ROI cutout of image
    bbox = np.array([y1, y2, x1, x2])
    
    # We clip so it's within the image
    im_limits = np.array([im.shape[0], im.shape[0], im.shape[1], im.shape[1]])
    clipped_bbox = np.clip(bbox, a_min=0, a_max=im_limits)

    # We get the crop
    crop = im[clipped_bbox[0]:clipped_bbox[1], clipped_bbox[2]:clipped_bbox[3]]

    # We pad the crop with the average so it has the same size as the original bbox
    pad = clipped_bbox - bbox
    pad = np.abs(pad)
    pad = pad.astype(np.int32)
    pad = np.array([[pad[0], pad[1]], [pad[2], pad[3]], [0, 0]])
    crop = np.pad(crop, pad, mode='constant', constant_values=np.mean(crop))

    # If crop is empty, we return an array [1, 1, 3] made of zeros
    if crop.shape[0] == 0 or crop.shape[1] == 0:
        print("WARNING! empty crop ({}) originated by bbox {}".format(crop.shape, bbox))
        crop = np.zeros((1, 1, 3), dtype=crop.dtype)

    return crop