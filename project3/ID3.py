import math

from DecisonTree import Leaf, Question, DecisionNode, class_counts
from utils import *

"""
Make the imports of python packages needed
"""


class ID3:
    def __init__(self, label_names: list,  target_attribute='diagnosis'):
        self.label_names = label_names
        self.target_attribute = target_attribute
        self.tree_root = None
        self.used_features = set()

    @staticmethod
    def entropy(rows: np.ndarray, labels: np.ndarray):
        """
        Calculate the entropy of a distribution for the classes probability values.
        :param rows: array of samples
        :param labels: rows data labels.
        :return: entropy value.
        """
        # TODO:
        #  Calculate the entropy of the data as shown in the class.
        #  - You can use counts as a helper dictionary of label -> count, or implement something else.

        counts = class_counts(rows, labels)
        #impurity = 0.0

        # ====== YOUR CODE: ======
        num_samples_in_node = len(rows)
        cls_to_prob_in_node = {cls: num_of_samples_with_cls_in_node / num_samples_in_node
                               for cls, num_of_samples_with_cls_in_node in counts.items()}

        impurity = - sum(cls_prob * np.log2(cls_prob) for cls_prob in cls_to_prob_in_node.values())
        # ========================

        return impurity

    def info_gain(self, left, left_labels, right, right_labels, current_uncertainty):
        """
        Calculate the information gain, as the uncertainty of the starting node, minus the weighted impurity of
        two child nodes.
        :param left: the left child rows.
        :param left_labels: the left child labels.
        :param right: the right child rows.
        :param right_labels: the right child labels.
        :param current_uncertainty: the current uncertainty of the current node
        :return: the info gain for splitting the current node into the two children left and right.
        """
        # TODO:
        #  - Calculate the entropy of the data of the left and the right child.
        #  - Calculate the info gain as shown in class.
        assert (len(left) == len(left_labels)) and (len(right) == len(right_labels)), \
            'The split of current node is not right, rows size should be equal to labels size.'

        info_gain_value = 0.0
        # ====== YOUR CODE: ======
        info_gain_value += current_uncertainty

        node_samples_num = len(left) + len(right)

        for child_samples, child_samples_labels in zip([left, right], [left_labels, right_labels]):
            child_samples_num = len(child_samples)
            child_weight = child_samples_num / node_samples_num
            child_entropy = self.entropy(child_samples, child_samples_labels)
            info_gain_value -= child_weight * child_entropy
        # ========================

        return info_gain_value

    def partition(self, rows, labels, question: Question, current_uncertainty):
        """
        Partitions the rows by the question.
        :param rows: array of samples
        :param labels: rows data labels.
        :param question: an instance of the Question which we will use to partition the data.
        :param current_uncertainty: the current uncertainty of the current node
        :return: Tuple of (gain, true_rows, true_labels, false_rows, false_labels)
        """
        # TODO:
        #   - For each row in the dataset, check if it matches the question.
        #   - If so, add it to 'true rows', otherwise, add it to 'false rows'.
        #   - Calculate the info gain using the `info_gain` method.

        gain, true_rows, true_labels, false_rows, false_labels = None, [], [], [], []
        assert len(rows) == len(labels), 'Rows size should be equal to labels size.'

        # ====== YOUR CODE: ======
        for sample, sample_label in zip(rows, labels):
            if question.match(sample):
                true_rows.append(sample)
                true_labels.append(sample_label)
            else:
                false_rows.append(sample)
                false_labels.append(sample_label)

        true_rows = np.array(true_rows)
        true_labels = np.array(true_labels)
        false_rows = np.array(false_rows)
        false_labels = np.array(false_labels)

        gain = self.info_gain(true_rows, true_labels, false_rows, false_labels, current_uncertainty)
        # ========================

        return gain, true_rows, true_labels, false_rows, false_labels

    def get_question_name(self, question_index):
        if self.target_attribute not in self.label_names:
            if 0 <= question_index < len(self.label_names):
                return self.label_names[question_index]
        else:
            target_index = self.label_names.index(self.target_attribute)
            if question_index < target_index:
                return self.label_names[question_index]
            elif question_index + 1 < len(self.label_names):
                return self.label_names[question_index + 1]
    def find_best_split(self, rows, labels):
        """
        Find the best question to ask by iterating over every feature / value and calculating the information gain.
        :param rows: array of samples
        :param labels: rows data labels.
        :return: Tuple of (best_gain, best_question, best_true_rows, best_true_labels, best_false_rows, best_false_labels)
        """
        # TODO:
        #   - For each feature of the dataset, build a proper question to partition the dataset using this feature.
        #   - find the best feature to split the data. (using the `partition` method)
        best_gain = - math.inf  # keep track of the best information gain
        best_question = None  # keep train of the feature / value that produced it
        best_false_rows, best_false_labels = None, None
        best_true_rows, best_true_labels = None, None
        current_uncertainty = self.entropy(rows, labels)

        # ====== YOUR CODE: ======
        for col_idx, feature_samples_values in enumerate(rows.T):
            sorted_feature_samples_values = np.sort(feature_samples_values)
            thresholds = [0.5 * (sorted_feature_samples_values[i] + sorted_feature_samples_values[i+1])
                          for i in range(len(sorted_feature_samples_values)-1)]
            for threshold in thresholds:
                question_col_name = self.get_question_name(col_idx)
                question = Question(question_col_name, col_idx, threshold)
                gain, samples_above, samples_above_labels, samples_below, samples_below_labels = \
                    self.partition(rows, labels, question, current_uncertainty)
                if gain >= best_gain:  # in case several features have max gain, only the last one (max idx) is chosen
                    best_gain = gain
                    best_question = question
                    best_true_rows, best_true_labels = samples_above, samples_above_labels
                    best_false_rows, best_false_labels = samples_below, samples_below_labels
        # ========================

        return best_gain, best_question, best_true_rows, best_true_labels, best_false_rows, best_false_labels

    def build_tree(self, rows, labels):
        """
        Build the decision tree recursively.
        :param rows: array of samples
        :param labels: rows data labels.
        :return: a DecisionNode, This records the best feature/value to ask at this point, depending on the answer.
                or a leaf node if we have to prune this branch (in which case all samples have the same label).
        """

        # Base case 1: If all labels are the same, return a Leaf node
        if len(set(labels)) == 1:
            return Leaf(rows, labels)

        # Base case 2: If no more features to split, return a Leaf node
        # (Assumption: Decision trees usually require stopping if there are no features left)
        if len(rows[0]) == 0:
            return Leaf(rows, labels)

        # Find the best question to ask
        best_gain, best_question, best_true_rows, best_true_labels, best_false_rows, best_false_labels = self.find_best_split(
            rows, labels)

        # Base case 3: If no information gain is achieved, return a Leaf node
        if best_gain == 0:
            return Leaf(rows, labels)

        # Recursively build the true branch
        true_branch = self.build_tree(best_true_rows, best_true_labels)

        # Recursively build the false branch
        false_branch = self.build_tree(best_false_rows, best_false_labels)

        # Return a DecisionNode, with the best question and the corresponding branches
        return DecisionNode(best_question, true_branch, false_branch)

    def fit(self, x_train, y_train):
        """
        Trains the ID3 model. By building the tree.
        :param x_train: A labeled training data.
        :param y_train: training data labels.
        """
        # TODO: Build the tree that fits the input data and save the root to self.tree_root

        # ====== YOUR CODE: ======
        self.tree_root = self.build_tree(x_train, y_train)
        # ========================

    def predict_sample(self, row, node: DecisionNode or Leaf = None):
        """
        Predict the most likely class for single sample in subtree of the given node.
        :param row: vector of shape (1,D).
        :return: The row prediction.
        """
        # TODO: Implement ID3 class prediction for set of data.
        #   - Decide whether to follow the true-branch or the false-branch.
        #   - Compare the feature / value stored in the node, to the example we're considering.

        if node is None:
            node = self.tree_root
        if node is None:
            node = self.tree_root

        if isinstance(node, Leaf):
            max_freq = max(node.predictions.values())
            predictions = [cls for cls, freq in node.predictions.items() if freq == max_freq]
            if len(predictions) == 1:
                prediction = predictions[0]  # leaf has a majority label
            else:
                prediction = max(predictions)  # leaf has 50-50 split labels, so classify as positive 'M'
        else:  # node is DecisionNode
            if node.question.match(row):
                prediction = self.predict_sample(row, node.true_branch)
            else:
                prediction = self.predict_sample(row, node.false_branch)

        return prediction

    def predict(self, rows):
        """
        Predict the most likely class for each sample in a given vector.
        :param rows: vector of shape (N,D) where N is the number of samples.
        :return: A vector of shape (N,) containing the predicted classes.
        """
        # TODO:
        #  Implement ID3 class prediction for set of data.

        #y_pred = None

        # ====== YOUR CODE: ======
        y_pred = np.array([self.predict_sample(sample) for sample in rows])
        # ========================

        return y_pred
