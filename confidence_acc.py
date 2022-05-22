import tensorflow as tf
import numpy as np


def print_lowest(predictions, ground_truth):
    min_val = np.min(predictions)                   # Find the smallest confidence-level
    min_val = round(min_val * 100, 1)               # Output with one decimal place and conversion as % value

    # Calculate the associated array index in order to subsequently determine the associated class.
    # The array index is the same as the image index.
    image_index = np.argmin(predictions)
    associated_class = ground_truth[image_index]

    # Output the results on the command line
    print("")
    print("Lowest confidence level:", min_val, "%")
    print("Associated class (ground_truth):", associated_class, "(image-index: ", image_index, ")")
    print("")
def print_highest(predictions, ground_truth):
    max_val = np.max(predictions)                   # Find the smallest confidence-level
    max_val = round(max_val * 100, 1)               # Output with one decimal place and conversion as % value

    # Calculate the associated array index in order to subsequently determine the associated class.
    # The array index is the same as the image index.
    image_index = np.argmax(predictions)
    associated_class = ground_truth[image_index]

    # Output the results on the command line
    print("")
    print("Highest confidence level:", max_val, "%")
    print("Associated class (ground_truth):", associated_class, "(image-index: ", image_index, ")")
    print("")



def accuracy(TP, predictions):
    return TP / np.len(predictions)

def calc_TP(predictions, tsh)
    for element in predictions
        if predictions[element] >= tsh
            predictions[element] = 1;
        else 


def confidence_acc(predictions, ground_truth, tsh):
    #   +++++ InputData +++++
    #
    #
    # predictions:      A 2-dimensional array containing the predictions for each sample and each class.
    #
    # ground_truth:     An array containing the class assignment of the validation data for each image
    #
    # threshold:        This parameter influences the calculation of accuracy.
    #                   Only from a confidence level above this threshold value is the sample considered to be
    #                   "correctly recognised" when calculating the accuracy, otherwise it is declared to be "false".
    #                   Example: tsh = 0.9 -> images must be recognised with at least 90% confidence level






    # Normalisation of the predictions to a value between 0 and 1
    predictions = tf.nn.softmax(predictions)


    # Conversion from a 2-dim to a 1-dim array. The highest value was filtered.
    # -> It now contains the highest confidence level for each test image
    predictions_max = np.max(predictions, 1)




    accuracy = accuracy()


    # /////////////////////////////////// print
    print_lowest(predictions_max, ground_truth)
    print_highest(predictions_max, ground_truth)


