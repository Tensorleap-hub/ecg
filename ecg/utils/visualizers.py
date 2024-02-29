import numpy.typing as npt
import numpy as np
from code_loader.contract.visualizer_classes import LeapHorizontalBar, LeapGraph, LeapImage
from ecg.config import CONFIG
import tensorflow as tf
import cv2


def display_waveform(data: npt.NDArray[np.float32], i) -> LeapGraph:
    data = data[..., np.newaxis]
    waveform = data[:, i]
    return LeapGraph(waveform)


def display_waveform_heatmap(data: npt.NDArray[np.float32], i) -> LeapGraph:
    data = data[..., np.newaxis]
    waveform = data[:, i]
    return waveform


def display_waveform_I(data: npt.NDArray[np.float32]) -> LeapGraph:
    return display_waveform(data, 0)


def display_waveform_I_heatmap(data: npt.NDArray[np.float32]) -> np.ndarray:
    return display_waveform_heatmap(data, 0)


def display_waveform_II(data: npt.NDArray[np.float32]) -> LeapGraph:
    return display_waveform(data, 1)


def display_waveform_II_heatmap(data: npt.NDArray[np.float32]) -> np.ndarray:
    return display_waveform_heatmap(data, 1)


def display_waveform_III(data: npt.NDArray[np.float32]) -> LeapGraph:
    return display_waveform(data, 2)


def display_waveform_III_heatmap(data: npt.NDArray[np.float32]) -> np.ndarray:
    return display_waveform_heatmap(data, 2)


def display_waveform_AVR(data: npt.NDArray[np.float32]) -> LeapGraph:
    return display_waveform(data, 3)


def display_waveform_AVR_heatmap(data: npt.NDArray[np.float32]) -> np.ndarray:
    return display_waveform_heatmap(data, 3)


def display_waveform_AVL(data: npt.NDArray[np.float32]) -> LeapGraph:
    return display_waveform(data, 4)


def display_waveform_AVL_heatmap(data: npt.NDArray[np.float32]) -> np.ndarray:
    return display_waveform_heatmap(data, 4)


def display_waveform_AVF(data: npt.NDArray[np.float32]) -> LeapGraph:
    return display_waveform(data, 5)


def display_waveform_AVF_heatmap(data: npt.NDArray[np.float32]) -> np.ndarray:
    return display_waveform_heatmap(data, 5)


def display_waveform_V1(data: npt.NDArray[np.float32]) -> LeapGraph:
    return display_waveform(data, 6)


def display_waveform_V1_heatmap(data: npt.NDArray[np.float32]) -> np.ndarray:
    return display_waveform_heatmap(data, 6)


def display_waveform_V2(data: npt.NDArray[np.float32]) -> LeapGraph:
    return display_waveform(data, 7)


def display_waveform_V2_heatmap(data: npt.NDArray[np.float32]) -> np.ndarray:
    return display_waveform_heatmap(data, 7)


def display_waveform_V3(data: npt.NDArray[np.float32]) -> LeapGraph:
    return display_waveform(data, 8)


def display_waveform_V3_heatmap(data: npt.NDArray[np.float32]) -> np.ndarray:
    return display_waveform_heatmap(data, 8)


def display_waveform_V4(data: npt.NDArray[np.float32]) -> LeapGraph:
    return display_waveform(data, 9)


def display_waveform_V4_heatmap(data: npt.NDArray[np.float32]) -> np.ndarray:
    return display_waveform_heatmap(data, 9)


def display_waveform_V5(data: npt.NDArray[np.float32]) -> LeapGraph:
    return display_waveform(data, 10)


def display_waveform_V5_heatmap(data: npt.NDArray[np.float32]) -> np.ndarray:
    return display_waveform_heatmap(data, 10)


def display_waveform_V6(data: npt.NDArray[np.float32]) -> LeapGraph:
    return display_waveform(data, 11)


def display_waveform_V6_heatmap(data: npt.NDArray[np.float32]) -> np.ndarray:
    return display_waveform_heatmap(data, 11)


def horizontal_bar_visualizer_with_labels_name(data: npt.NDArray[np.float32]) -> LeapHorizontalBar:
    labels_names = [CONFIG['LABELS'][index] for index in range(data.shape[-1])]
    return LeapHorizontalBar(data, labels_names)


def graph_visuaizer(image):
    image = image[..., np.newaxis]
    image = image[:, :, -1]
    return LeapGraph(image)


def graph_heatmap_visualizer(image):
    image = image[..., np.newaxis]
    image = np.expand_dims(np.mean(image[..., -1], axis=-1), axis=0)
    return image


def img_vis_T(image):
    image = image[..., np.newaxis]
    return LeapImage(np.transpose(image, (1, 0, 2)))


def heatmap_vis_T(image):
    image = image[..., np.newaxis]
    return np.transpose(image, (1, 0, 2))
