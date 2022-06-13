import tensorflow as tf
import numpy as np


############################################################################
#       DATA
############################################################################

class Predictions:
    def __init__(self, predictions):        #ctor
        # Filtering the highest confidence level and replacing the value with the array index
        # (Corresponds to the designation of the classes)
        self.classes = np.argmax(predictions, axis=1)

        # Filtering the highest confidence level
        self.confidence_max = np.max(tf.nn.softmax(predictions), 1)

        # Creating an empty array to store the boolean matches
        self.matches = np.zeros(len(self.classes))

        # Creating an empty array to store the boolean matches (filtered)
        self.filtered_matches = np.zeros(len(self.classes))

    # compare index of max confidence-level with ground_truth
    # --> create boolean-array (match = true, no match = false)
    def find_matches(self, ground_truth):
        for i in range(0, len(self.classes)):
            if self.classes[i] == ground_truth[i]:
                self.matches[i] = True
            else:
                self.matches[i] = False

    # uses threshold to filter the array. For predictions<threshold the boolean value in matches becomes false
    def filter_matches(self, threshold):
        self.filtered_matches = np.copy(self.matches)
        for i in range(0, len(self.confidence_max)):
            if self.confidence_max[i] < threshold:
                self.filtered_matches[i] = False

    # Counting the number of correctly predicted samples
    def count_matches(self, matches):
        num_matches = 0;
        for i in range(0, len(matches)):
            if matches[i]:
                num_matches += 1
        return num_matches

    # calculate accuracy
    def accuracy(self, num_matches, num_samples):
        acc = num_matches / num_samples * 100
        return acc

    def num_filtered(self, threshold):
        num_filtered = 0
        for i in self.confidence_max:
            if i < threshold:
                num_filtered += 1
        return num_filtered




############################################################################
#       FUNCTION
############################################################################

def confidence_acc(input_predictions, ground_truth, threshold=0.7):
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
    P = Predictions(input_predictions)


    # compare index of max confidence-level with ground_truth
    # --> create boolean-array (match = true, no match = false)
    P.find_matches(ground_truth)

    #  uses threshold to filter the array. For predictions<threshold the boolean value in matches becomes false
    P.filter_matches(threshold)

    # Counts the boolean true values in matches to get num_matches
    num_matches_filtered = P.count_matches(P.filtered_matches)

    # calculate accuracy (filtered matches/number of samples)
    acc_at_threshold = P.accuracy(num_matches_filtered, len(P.filtered_matches))

    # normal calculated accuracy
    num_matches = P.count_matches(P.matches)
    acc = P.accuracy(num_matches, len(P.matches))

    print("Acc@",threshold*100,"%:", round(acc_at_threshold,1),"%")
    print("Normal Acc: ", round(acc, 1), "%")
    print("Number of real matches:" , num_matches)
    print("Number of filtered elements:", P.num_filtered(threshold))



############################################################################
#       TESTING
############################################################################

class TestSet:
    def __init__(self,
    predictions = np.array([[0.018, 0.98, 0.002],
                            [0.019, 0.001, 0.98],
                            [0.25, 0.25, 0.50],
                            [0.80, 0.01, 0.19]]),
    test_ground_truth = np.array([1, 1, 2, 0]),
    matches_soll = np.array([1, 0, 1, 1]),
    filtered_matches_soll =  np.array([1, 0, 0, 1]),
    num_matches_soll = 2,
    threshold = 0.7,
    lowest_conf_level = 0.80,
    num_filtered = 1):

        self.predictions = predictions
        self.test_ground_truth = test_ground_truth
        self.matches_soll = matches_soll
        self.filtered_matches_soll = filtered_matches_soll
        self.num_matches_soll = num_matches_soll
        self.threshold = threshold
        self.lowest_conf_level = lowest_conf_level
        self.num_filtered = num_filtered

def test_find_matches(Test_Predictions, matches_soll, test_ground_truth):
    Test_Predictions.find_matches(test_ground_truth)
    if np.array_equal(Test_Predictions.matches, matches_soll):
        print("PASSED >>find_matches()<<")
        return True
    else: return False
def test_filtering(Test_Predictions, threshold, filtered_matches_soll):
    Test_Predictions.filter_matches(threshold)
    if np.array_equal(Test_Predictions.filtered_matches, filtered_matches_soll):
        print("PASSED >>filter_matches()<<")
        return True
    else:
        print("FAILED >>filter_matches()<<")
        return False
def test_count_matches(Test_Predictions, num_matches_soll):
    if np.array_equal(Test_Predictions.count_matches(Test_Predictions.filtered_matches), num_matches_soll):
        print("PASSED >>count_matches()<<")
        return True
    else:
        print("FAILED >>count_matches()<<")
        return False
def test_accuracy(Test_Predictions, test_accuracy):
    num_matches =  Test_Predictions.count_matches(Test_Predictions.filtered_matches)
    num_samples = len(Test_Predictions.filtered_matches)
    if Test_Predictions.accuracy(num_matches, num_samples) == test_accuracy*100:
        print("PASSED >>accuracy()<<")
        return True
    else:
        print("FAILED >>accuracy()<<")
        return False
def test_num_filtered(Test_Predictions, threshold, num_filtered):
    if Test_Predictions.num_filtered(threshold) == num_filtered:
        print("PASSED >>num_filtered()<<")
        return True
    else:
        print("FAILED >>num_filtered()<<")
        return False


def testing():

    #Open default testset:
    T1 = TestSet()

    # create test object
    Test_Predictions = Predictions(T1.predictions)
    Test_Predictions.confidence_max = np.max(T1.predictions, 1)

    if not test_find_matches(Test_Predictions, T1.matches_soll, T1.test_ground_truth): return False
    if not test_filtering(Test_Predictions, T1.threshold, T1.filtered_matches_soll): return False
    if not test_count_matches(Test_Predictions, T1.num_matches_soll): return False
    if not test_accuracy(Test_Predictions, T1.num_matches_soll/len(T1.filtered_matches_soll)): return False
    if not test_num_filtered(Test_Predictions, T1.threshold, T1.num_filtered): return False

    return True