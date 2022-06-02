import tensorflow as tf
import numpy as np


class Predictions:
    def __init__(self, predictions):
        self.all = predictions
        self.classes = np.argmax(predictions, axis=1)
        #self.confidence_all = tf.nn.softmax(predictions)
        self.confidence_max = np.max(tf.nn.softmax(predictions), 1)
        self.matches = np.zeros(len(self.classes))







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





# predicted.classes, ground_truth, predicted.matches
def find_matches(prediction, ground_truth):
    for i in range(0, len(prediction.classes)):
        if prediction.classes[i] == ground_truth[i]:
            prediction.matches[i] = True
        else:
            prediction.matches[i] = False


# Anpassung der predictions (nur confidence level >= Threshold zählt als Treffer (1) )
def filter_matches(confidence_max, matches, threshold):
    for i in range(0, len(confidence_max)):
        if confidence_max[i] < threshold:
            matches[i] = False


#count matches
def count_matches(matches):
    num_matches = 0
    for i in range(0, len(matches)):
        if matches[i] == True:
            num_matches += 1
    return num_matches



# calculate accuracy
def accuracy(num_matches, matches ):
    return num_matches / len(matches) * 100



def confidence_acc(input_predictions, ground_truth, threshold):
    #   +++++ InputData +++++
    #
    #
    # input_predictions:    A 2-dimensional array containing the predictions for each sample and each class.
    #
    # ground_truth:         An array containing the class assignment of the validation data for each image
    #
    # threshold:            This parameter influences the calculation of accuracy.
    #                       Only from a confidence level above this threshold value is the sample considered to be
    #                       "correctly recognised" when calculating the accuracy, otherwise it is declared to be "false".
    #                       Example: tsh = 0.9 -> images must be recognised with at least 90% confidence level



    # initialize predictions object
    predicted = Predictions(input_predictions)


    # Normalisation of the predictions to a value between 0 and 1
    #predictions = tf.nn.softmax(predictions)



    # Conversion from a 2-dim to a 1-dim array. The highest value was filtered.
    # -> It now contains the highest confidence level for each test image
    #predictions_max = np.max(predictions, 1)

    find_matches(predicted, ground_truth)
    filter_matches(predicted.confidence_max, predicted.matches, threshold)
    num_matches = count_matches(predicted.matches)
    my_accuracy = accuracy(num_matches, predicted.matches)

    print("Acc@",threshold*100,"%:", round(my_accuracy,1),"%")

    #predicted.matches[0] = True
    print("matches", predicted.matches)


   # accuracy = accuracy()


    # /////////////////////////////////// print
    #print_lowest(predictions_max, ground_truth)
    #print_highest(predictions_max, ground_truth)


