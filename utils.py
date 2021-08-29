import numpy as np
import os
import sys
import shutil
import urllib.request

from tqdm import tqdm
from sklearn.metrics import confusion_matrix


def download_pix4d_dataset(dataset_path):
    """Download pix4d point cloud dataset from internet
    Parameters
    ----------
    dataset_path : str
        Path were unpacked dataset will be placed.
    """
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)
    print("Downloading dataset. It will take a while")
    with urllib.request.urlopen(
        "https://s3.amazonaws.com/mics.pix4d.com/research/pix4d_isprs2017_datasets.zip"
    ) as response:
        with open("datasets.zip", "wb") as tmp_file:
            shutil.copyfileobj(response, tmp_file)
        print("Unpacking dataset")
        shutil.unpack_archive("datasets.zip", dataset_path)
        os.remove("datasets.zip")


def estimate_ious(true, predicted, n_classes):
    """Intersection over Union score
    Parameters
    ----------
    true : 1d array-like
        Ground truth (correct) class labels.
    predicted : 1d array-like
        Predicted class labels.
    n_classes : int
        Number of classes

    Returns
    -------
    (miou, ious_list) : tuple
        Mean intersection over union and
        list of per-class intersection over unions
    """
    conf_matrix = confusion_matrix(true, predicted)
    gt_classes = np.sum(conf_matrix, axis=1)
    positive_classes = np.sum(conf_matrix, axis=0)
    true_positive_classes = np.diagonal(conf_matrix)

    iou_list = []
    for n in range(0, n_classes, 1):
        iou = true_positive_classes[n] / float(
            gt_classes[n] + positive_classes[n] - true_positive_classes[n]
        )
        iou_list.append(iou)
    mean_iou = sum(iou_list) / float(n_classes)

    return mean_iou, iou_list


def load_point_cloud(filename, point_cloud, use_colors=True):
    """Load point cloud and labels for given dataset
    Parameters
    ----------
    filename : str
        Name of the dataset for loading.
    point_cloud : pypcs.PointCloud
        Class representing thee point cloud
    use_colors : bool
        If False colors will not be loaded

    Returns
    -------
    (point_cloud, labels) : tuple
        Loaded point cloud and class labels
    """
    point_cloud.clear()
    labels = []
    with open(filename, "r") as f:
        lines = f.readlines()
        loop = tqdm(lines)
        loop.set_description("Loading {}".format(os.path.basename(filename)))
        for idx, line in enumerate(loop):
            x, y, z, r, g, b, l = (float(t) for t in line.split())
            labels.append(int(l))
            point = np.asarray([x, y, z], dtype=np.float64)
            if use_colors:
                color = np.asarray([r, g, b], dtype=np.float64)
                point_cloud.add_point_and_color(point, color)
            else:
                point_cloud.add_point(point)

    labels = np.array(labels, dtype=np.int)
    return point_cloud, labels


def generate_features(feature_estimator):
    """Generate features for all points in point cloud
    Parameters
    ----------
    feature_estimator : pypcs.FeatureEstimator
        Valid FeatureEstimator class for wich features will
        be calculated
    Returns
    -------
    features : 2-d numpy array of size (num_points, feature_size)
        Array of calculated features, where earch row represents a features
        for individual point
    """
    num_points = feature_estimator.num_points()
    batch_size = feature_estimator.batch_size()
    features = np.zeros(
        (num_points, feature_estimator.feature_size()), dtype=np.float64
    )
    loop = tqdm(feature_estimator)
    loop.set_description("Calculate point features")
    for batch_id, feat in enumerate(loop):
        start = batch_id * batch_size
        features[start : start + feat.shape[0], :] = feat

    return features


def write_ply(filename, field_list, field_names):
    """Write point cloud to ply file
    Parameters
    ----------
    filename : str
        Name of output file
    field_list : list
        List of np.ndarrays with values for writing
    field_names : list
        List of strings with field property names
    """

    def header_properties(field_list, field_names):
        lines = []
        lines.append("element vertex {}".format(field_list[0].shape[0]))
        i = 0
        for fields in field_list:
            for field in fields.T:
                lines.append("property {} {}".format(field.dtype.name, field_names[i]))
                i += 1
        return lines

    field_list = (
        list(field_list)
        if (type(field_list) == list or type(field_list) == tuple)
        else list((field_list,))
    )
    for i, field in enumerate(field_list):
        if field is None:
            raise ValueError("WRITE_PLY ERROR: a field is None")

        elif field.ndim > 2:
            raise ValueError("WRITE_PLY ERROR: a field have more than 2 dimensions")

        elif field.ndim < 2:
            field_list[i] = field.reshape(-1, 1)

    n_points = [field.shape[0] for field in field_list]
    if not np.all(np.equal(n_points, n_points[0])):
        raise ValueError("wrong field dimensions")

    n_fields = np.sum([field.shape[1] for field in field_list])
    if n_fields != len(field_names):
        raise ValueError("wrong number of field names")

    with open(filename, "w") as plyfile:
        header = ["ply"]
        header.append("format binary_" + sys.byteorder + "_endian 1.0")
        header.extend(header_properties(field_list, field_names))
        header.append("end_header")

        for line in header:
            plyfile.write("{}\n".format(line))

    with open(filename, "ab") as plyfile:
        i = 0
        type_list = []
        for fields in field_list:
            for field in fields.T:
                type_list += [(field_names[i], field.dtype.str)]
                i += 1
        data = np.empty(field_list[0].shape[0], dtype=type_list)
        i = 0
        for fields in field_list:
            for field in fields.T:
                data[field_names[i]] = field
                i += 1

        data.tofile(plyfile)
