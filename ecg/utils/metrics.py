from tensorflow.python.ops import math_ops
from tensorflow.python.keras import backend
import tensorflow as tf
from code_loader.contract.datasetclasses import ConfusionMatrixElement  # type: ignore
from code_loader.contract.enums import ConfusionMatrixValue  # type: ignore
import numpy as np
from ecg.config import CONFIG


def TP(y_true, y_pred):
    y_pred = math_ops.cast(tf.math.greater_equal(y_pred, 0.5), backend.floatx())
    return tf.math.reduce_sum(
        math_ops.cast(math_ops.equal(y_true, y_pred) & math_ops.equal(y_true, 1), backend.floatx()), axis=1)


def TN(y_true, y_pred):
    y_pred = math_ops.cast(tf.math.greater_equal(y_pred, 0.5), backend.floatx())
    return tf.math.reduce_sum(
        math_ops.cast(math_ops.equal(y_true, y_pred) & math_ops.equal(y_true, 0), backend.floatx()), axis=1)


def FP(y_true, y_pred):
    y_pred = math_ops.cast(tf.math.greater_equal(y_pred, 0.5), backend.floatx())
    return tf.math.reduce_sum(
        math_ops.cast(~math_ops.equal(y_true, y_pred) & math_ops.equal(y_true, 0), backend.floatx()), axis=1)


def FN(y_true, y_pred):
    y_pred = math_ops.cast(tf.math.greater_equal(y_pred, 0.5), backend.floatx())
    return tf.math.reduce_sum(
        math_ops.cast(~math_ops.equal(y_true, y_pred) & math_ops.equal(y_true, 1), backend.floatx()), axis=1)


def pred_label(y_true, y_pred) -> float:

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    max_index = [float(np.argmax(y_pred))]

    return tf.convert_to_tensor(max_index)
