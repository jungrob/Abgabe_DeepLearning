import tensorflow as tf
import numpy as np


def confidence_acc(predictions):

    predictions = tf.nn.softmax(predictions)        # normalize predictions with values between 0 and 1
    # predictions = np.argmax(predictions, axis=1)
    confidence_level = np.max(predictions, 1)


    print(predictions_max)
