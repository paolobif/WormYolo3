"""binary_perceptron.py
An edited version of a file provided for use in CSE 415, Spring 2021


Creates a binary perceptron that learns to classify data
"""



import numpy as np
import csv
import image_analysis as ia
import random as r
USE_DIF = False

class BinaryPerceptron:
    """
    Class representing the Binary Perceptron
    ---
    It is an algorithm that can learn a binary classifier
    """

    def __init__(self, size, weights=None, alpha=0.01, save_path = None):
        """
        Initialize the Binary Perceptron
        ---
        size: How many inputs will each example have, exclusing class
        weights: Weight vector of the form [w_0, w_1, ..., w_{n-1}, bias_weight]
        alpha: Learning rate
        """

        self.alpha = alpha
        self.mult_factors = [1/1000,1/1000,2*np.pi,1/100,1/10,1/10,1/100]
        self.weights = np.zeros(size + 1)
        if weights:
          for i in range(len(weights)):
            self.weights[i] = weights[i]

        self.save_path = save_path

    def classify(self, x_vector):
        """
        Method that classifies a given data point into one of 2 classes
        ---
        Inputs:
        x_vector = [x_0, x_1, ..., x_{n-1}]
        Note: y (correct class) is not part of the x_vector.

        Returns:
        y_hat: Predicted class
              +1 if the current weights classify x_vector as positive i.e. Σx_i*w_i>=0,
        else  -1 if it is classified as negative.
        """

        if (type(x_vector) == type([])):
          x_vector = np.asarray(x_vector)

        # Account for bias
        x_vector = np.append(x_vector, 1)

        if np.dot(x_vector, self.weights)>=0:
            return 1
        else:
            return -1

    def train_with_one_example(self, x_vector, y):
        """
        Method that updates the model weights using a particular training example (x_vector,y)
        and returns whether the model weights were actually changed or not
        ---
        Inputs:
        x_vector: Feature vector, same as method classify
        y: Actual class of x_vector
            +1 if x_vector represents a positive example,
        and -1 if it represents a negative example.
        Returns:
        weight_changed: True if there was a change in the weights
                        else False
        """


        if (type(x_vector) == type([])):
          x_vector = np.asarray(x_vector)



        #If the class is correct, do nothing
        if self.classify(x_vector) == y:

            return False
        #Otherwise, update
        else:
            x_vector = np.append(x_vector, 1)

            self.weights = np.add(self.weights, y*self.alpha*x_vector)
            return True

    def train_for_an_epoch(self, training_data):
        """
        Method that goes through the given training examples once, in the order supplied,
        passing each one to train_with_one_example.
        ---
        Input:
        training_data: Input training data {(x_vector_1, y_1), (x_vector_2, y_2), ...}
        where each x_vector is concatenated with the corresponding y value.

        Returns:
        changed_count: Return the number of weight updates.
        (If zero, then training has converged.)
        """
        changed_count = 0

        for x_vector in training_data:
            y = x_vector[-1]
            work_vect = x_vector[0:-1]
            if self.train_with_one_example(work_vect,y):
                changed_count += 1
        return changed_count

    def test_folder(self,file1, class_v = 1):
        """
        Determines how many files in the folder are classified incorrectly

        file1: The folder path
        class_v: (1 or -1) Whether the folder is 'positive' or 'negative'
        """
        csv_data = csv.reader(open(file1), delimiter=',')
        all_x = []
        row_list = []
        count = 0
        for row in csv_data:
            row_list.append(row)
        for i in range(len(row_list)):
            row = row_list[i]
            row2 = row_list[i-1]
            # Ignore header
            if not '#' in row[0] and i!=1:
                x_vector = np.array([])
                x_vector2 = np.array([])
                used_values = row[6:13]
                for item in used_values:
                    x_vector = np.append(x_vector,float(item))
                used_values = row2[6:13]
                for item in used_values:
                    x_vector2 = np.append(x_vector2,float(item))
                if USE_DIF:
                    x_vector = np.abs(np.array(x_vector) - np.array(x_vector2))
                else:
                    x_vector = np.array(x_vector)*np.array(self.mult_factors)
                if not self.classify(x_vector) == class_v:
                    count+=1

        return count
    def test_image(self,image_path):
        """
        Tells whether a given image is 'positive'
        """
        row = ia.single_data(image_path)
        x_vector = np.array([])
        used_values = row[6:13]
        for item in used_values:
            x_vector = np.append(x_vector,float(item))
        return self.classify(np.array(x_vector))

    def train_two_folders(self, fold1,fold2,class_v1,class_v2,epochs=1):
        """
        Trains the perceptron from two csv files

        fold1: The path to the first folder
        fold2: The path to the second folder
        class_v1: (1 or -1) Whether the first class is 'positive' or 'negative'
        class_v2: (1 or -1) Whether the second class is 'positive' or 'negative'
        epochs: How many iterations to train on both folders
        """
        csv_data = csv.reader(open(fold1), delimiter=',')
        all_x = []
        row_list = []
        for row in csv_data:
            row_list.append(row)
        for i in range(len(row_list)):
            row = row_list[i]
            row2 = row_list[i-1]
            # Ignore header
            if not '#' in row[0] and i!=1:
                x_vector = np.array([])
                x_vector2 = np.array([])
                used_values = row[6:13]
                for item in used_values:
                    x_vector = np.append(x_vector,float(item))
                used_values = row2[6:13]
                for item in used_values:
                    x_vector2 = np.append(x_vector2,float(item))
                if USE_DIF:
                    x_vector = np.abs(np.array(x_vector) - np.array(x_vector2))
                else:
                    x_vector = np.array(x_vector)*np.array(self.mult_factors)
                x_vector = np.append(x_vector, class_v1)
                all_x.append(x_vector)
        csv_data = csv.reader(open(fold2), delimiter=',')
        row_list = []
        for row in csv_data:
            row_list.append(row)
        for i in range(len(row_list)):
            row = row_list[i]
            row2 = row_list[i-1]
            # Ignore header
            if not '#' in row[0] and i!=1:
                x_vector = np.array([])
                x_vector2 = np.array([])
                used_values = row[6:13]
                for item in used_values:
                    x_vector = np.append(x_vector,float(item))
                used_values = row2[6:13]
                for item in used_values:
                    x_vector2 = np.append(x_vector2,float(item))

                if USE_DIF:
                    x_vector = np.abs(np.array(x_vector) - np.array(x_vector2))
                else:
                    x_vector = np.array(x_vector)*np.array(self.mult_factors)
                x_vector = np.append(x_vector, class_v2)

                all_x.append(x_vector)

        changed = 0
        for i in range(epochs):
            r.shuffle(all_x)
            t_vect = np.stack(tuple(all_x))
            changed += self.train_for_an_epoch(t_vect)
        return changed
    def save(self):
        """
        Saves the perceptron's trained values
        """
        np.savetxt(self.save_path, self.weights, delimiter=',')
    def load(self):
        """
        Loads the perceptron's values from it's path
        """
        self.weights = np.loadtxt(self.save_path, delimiter=',')




def sample_test():
    """
    Trains the binary perceptron using a synthetic training set
    Prints the weights obtained after training
    """
    DATA = [
        [-2, 7, +1],
        [1, 10, +1],
        [3, 2, -1],
        [5, -2, -1]]
    bp = BinaryPerceptron(2)
    print("Training Binary Perceptron for 3 epochs.")
    for i in range(3):
        bp.train_for_an_epoch(DATA)
    print("Binary Perceptron weights:")
    print("Done.")


if __name__ == '__main__':
    bp = BinaryPerceptron(7)
    for i in range(100):
        #bp.train_from_class_csv("C:/Users/cdkte/Downloads/yolo3/Worm-Yolo3/Day4/Day4.csv",class_v = -1)
        #bp.train_from_class_csv("C:/Users/cdkte/Downloads/yolo3/Worm-Yolo3/Day10/Day10.csv")

        bp.train_two_folders("C:/Users/cdkte/Downloads/yolo3/Worm-Yolo3/Day4/Day4.csv","C:/Users/cdkte/Downloads/yolo3/Worm-Yolo3/Day10/Day10.csv",-1,1,100)
        if i%10==0:
            print(bp.weights)
    #print(bp.test_from_two_csvs("C:/Users/cdkte/Downloads/yolo3/Worm-Yolo3/Day10/Day10.csv","C:/Users/cdkte/Downloads/yolo3/Worm-Yolo3/Day4/Day4.csv"))
    err1=bp.test_folder("C:/Users/cdkte/Downloads/yolo3/Worm-Yolo3/Day4/Day4.csv",-1)
    print(err1)
    err2=bp.test_folder("C:/Users/cdkte/Downloads/yolo3/Worm-Yolo3/Day10/Day10.csv")
    print(err2)
    count = 0
    csv_data = csv.reader(open("C:/Users/cdkte/Downloads/yolo3/Worm-Yolo3/Day4/Day4.csv"), delimiter=',')
    for row in csv_data:
        count+=1
    csv_data = csv.reader(open("C:/Users/cdkte/Downloads/yolo3/Worm-Yolo3/Day10/Day10.csv"), delimiter=',')
    for row in csv_data:
        count+=1
    print(count)
    print((err1+err2)/count)