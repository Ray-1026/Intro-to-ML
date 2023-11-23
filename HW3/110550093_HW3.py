# You are not allowed to import any additional packages/libraries.
import numpy as np
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


# This function computes the gini impurity of a label array.
def gini(y):
    return 1 - np.sum((np.bincount(y) / y.shape[0]) ** 2)


# This function computes the entropy of a label array.
def entropy(y):
    prob_y = np.bincount(y) / y.shape[0]
    prob_y[prob_y == 0] = 1
    return -np.sum(prob_y * np.log2(prob_y))


# The decision tree classifier class.
# Tips: You may need another node class and build the decision tree recursively.
class TreeNode:
    def __init__(self, feature=None, threshold=None, value=None, left=None, right=None):
        self.feature = feature
        self.threshold = threshold
        self.value = value
        self.left = left
        self.right = right


class DecisionTree:
    def __init__(self, criterion="gini", max_depth=None, rand=False):
        self.criterion = criterion
        self.max_depth = max_depth
        self.rand = rand
        self.root = None

    # This function computes the impurity based on the criterion.
    def impurity(self, y):
        if self.criterion == "gini":
            return gini(y)
        elif self.criterion == "entropy":
            return entropy(y)

    def best_split(self, X, y):
        best_feature, best_threshold = None, None
        best_impurity = np.inf

        for feature in range(X.shape[1]):
            for threshold in np.unique(X[:, feature]):
                left = y[X[:, feature] <= threshold]
                right = y[X[:, feature] > threshold]

                information_gain = (
                    self.impurity(left) * left.shape[0]
                    + self.impurity(right) * right.shape[0]
                ) / X.shape[0]

                if information_gain < best_impurity:
                    best_impurity = information_gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold, best_impurity

    def random_split(self, X):
        best_feature = np.random.randint(X.shape[1])
        best_threshold = np.random.choice(X[:, best_feature])

        return best_feature, best_threshold, 1

    def generate_tree(self, X, y, depth=0):
        if depth == self.max_depth:
            return TreeNode(value=np.bincount(y).argmax())

        feature, threshold, impurity = (
            self.random_split(X) if self.rand else self.best_split(X, y)
        )

        if feature is None or impurity == 0:
            return TreeNode(value=np.bincount(y).argmax())

        left = X[:, feature] <= threshold
        right = X[:, feature] > threshold

        if len(y[left]) == 0 or len(y[right]) == 0:
            return TreeNode(value=np.bincount(y).argmax())

        left_subtree = self.generate_tree(X[left], y[left], depth + 1)
        node_subtree = self.generate_tree(X[right], y[right], depth + 1)

        return TreeNode(
            feature=feature, threshold=threshold, left=left_subtree, right=node_subtree
        )

    # This function fits the given data using the decision tree algorithm.
    def fit(self, X, y):
        self.root = self.generate_tree(X, y)

    # This function takes the input data X and predicts the class label y according to your trained model.
    def predict(self, X):
        pred = []
        for i in X:
            node = self.root
            while node.value is None:
                node = node.left if i[node.feature] <= node.threshold else node.right
            pred.append(node.value)

        return np.array(pred)

    # This function plots the feature importance of the decision tree.
    def plot_feature_importance_img(self, columns):
        feature_importance = np.zeros(columns)

        def traverse(node):
            if node.value is not None:
                return

            feature_importance[node.feature] += 1
            traverse(node.left)
            traverse(node.right)

        traverse(self.root)
        plt.barh(range(columns), feature_importance)
        plt.yticks(range(columns), ["age", "sex", "cp", "fbs", "thalach", "thal"])
        plt.title("Feature Importance")
        plt.show()


# The AdaBoost classifier class.
class AdaBoost:
    def __init__(self, criterion="gini", n_estimators=200):
        self.criterion = criterion
        self.n_estimators = n_estimators

        self.classifiers = []
        self.alphas = []

    # This function fits the given data using the AdaBoost algorithm.
    # You need to create a decision tree classifier with max_depth = 1 in each iteration.
    def fit(self, X, y):
        y_i = np.copy(y)
        y_i[y_i == 0] = -1
        D = np.ones(X.shape[0]) / X.shape[0]

        for _ in range(self.n_estimators):
            weak_clf = DecisionTree(criterion=self.criterion, max_depth=1, rand=True)
            weak_clf.fit(X, y)
            y_pred = weak_clf.predict(X)

            error = np.sum(D[y_pred != y])
            alpha = 0.5 * np.log((1 - error) / error)

            y_pred[y_pred == 0] = -1
            D = D * np.exp(-alpha * y_i * y_pred)
            D = D / np.sum(D)

            self.classifiers.append(weak_clf)
            self.alphas.append(alpha)

    # This function takes the input data X and predicts the class label y according to your trained model.
    def predict(self, X):
        pred = np.zeros(X.shape[0])

        for alpha, tree in zip(self.alphas, self.classifiers):
            temp = tree.predict(X)
            temp[temp == 0] = -1
            pred += alpha * temp

        pred[pred > 0] = 1
        pred[pred < 0] = 0
        return pred


# Do not modify the main function architecture.
# You can only modify the value of the arguments of your Logistic Regression class.
if __name__ == "__main__":
    # Data Loading
    train_df = DataFrame(read_csv("train.csv"))
    test_df = DataFrame(read_csv("test.csv"))
    X_train = train_df.drop(["target"], axis=1)
    y_train = train_df["target"]
    X_test = test_df.drop(["target"], axis=1)
    y_test = test_df["target"]

    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

    # Set random seed to make sure you get the same result every time.
    # You can change the random seed if you want to.
    np.random.seed(69)

    # Decision Tree
    print("Part 1: Decision Tree")
    data = np.array([0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1])
    print(f"gini of [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1]: {gini(data)}")
    print(f"entropy of [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1]: {entropy(data)}")
    tree = DecisionTree(criterion="gini", max_depth=7)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    print("Accuracy (gini with max_depth=7):", accuracy_score(y_test, y_pred))
    tree = DecisionTree(criterion="entropy", max_depth=7)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    print("Accuracy (entropy with max_depth=7):", accuracy_score(y_test, y_pred))

    # Plot the feature importance of your decision tree.
    tree = DecisionTree(criterion="gini", max_depth=15)
    tree.fit(X_train, y_train)
    tree.plot_feature_importance_img(X_train.shape[1])

    # AdaBoost
    print("Part 2: AdaBoost")
    # Tune the arguments of AdaBoost to achieve higher accuracy than your Decision Tree.
    ada = AdaBoost(criterion="gini", n_estimators=10)
    ada.fit(X_train, y_train)
    y_pred = ada.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
