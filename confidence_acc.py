import tensorflow as tf
import numpy as np







def confidence_acc(predictions, ground_truth):

    class_predictions = np.argmax(predictions, axis=1)

    predictions = tf.nn.softmax(predictions)        # normalize predictions with values between 0 and 1
    predictions_max = np.max(predictions, 1)

    lo_conf_i = np.argmin(predictions_max)
    hi_conf_i = np.argmax(predictions_max)

    lo_conf = np.min(predictions_max)
    hi_conf = np.max(predictions_max)


    # print("Index", lo_conf_i)

    # /////////////////////////////////// print
    print("")
    print("lowest confidence level:", round(lo_conf * 100, 1), "%")
    print("associated class:", ground_truth[lo_conf_i], "(image-index: ", lo_conf_i, ")")
    print("")
    print("highest confidence level:", round(hi_conf * 100, 1), "%")
    print("associated class:", ground_truth[hi_conf_i], "(image-index: ", hi_conf_i, ")")

    print("")

   # print("max predictions", predictions_max)
