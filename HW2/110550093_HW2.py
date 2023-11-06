# You are not allowed to import any additional packages/libraries.
import numpy as np
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


class LogisticRegression:
    def __init__(self, learning_rate=0.1, iteration=100):
        self.learning_rate = learning_rate
        self.iteration = iteration
        self.weights = None
        self.intercept = None

    # This function computes the gradient descent solution of logistic regression.
    def fit(self, X, y, beta_1=0.9, beta_2=0.999):
        """
        Logistic regression with gradient descent
        """
        total = X.shape[0]
        X_prime = np.hstack([np.ones((total, 1)), X])
        weights = np.zeros(X_prime.shape[1])
        grad = np.zeros(X_prime.shape[1])
        G = 0
        m_t = 0
        v_t = 0

        for i in range(self.iteration):
            pred = self.sigmoid((weights) @ X_prime.T)

            # # momentum
            # if i == 0:
            #     grad = self.learning_rate * (1 / total) * (X_prime.T @ (pred - y))
            # else:
            #     grad = 0.9 * grad + self.learning_rate * (1 / total) * (X_prime.T @ (pred - y))
            # weights = weights - grad

            # # adagrad
            # grad = (1 / total) * (X_prime.T @ (pred - y))
            # G += grad**2
            # weights = weights - (self.learning_rate / (np.sqrt(G) + 1e-8)) * grad

            # adam
            grad = (1 / total) * (X_prime.T @ (pred - y))
            m_t = beta_1 * m_t + (1 - beta_1) * grad
            v_t = beta_2 * v_t + (1 - beta_2) * grad**2
            m_t_hat = m_t / (1 - beta_1 ** (i + 1))
            v_t_hat = v_t / (1 - beta_2 ** (i + 1))
            weights -= (self.learning_rate / (np.sqrt(v_t_hat) + 1e-8)) * m_t_hat

            # # cross entropy loss
            # if (i + 1) % 10 == 0:
            #     loss = -np.mean(y * np.log(pred) + (1 - y) * np.log(1 - pred))
            #     print(f"iteration {i + 1}: loss = {loss:.5f}")

        self.intercept = weights[0]
        self.weights = weights[1:]

    # This function takes the input data X and predicts the class label y according to your solution.
    def predict(self, X):
        X_prime = np.hstack([np.ones((X.shape[0], 1)), X])
        weights = np.array([self.intercept])
        weights = np.hstack([weights, self.weights])

        return np.round(self.sigmoid(weights @ X_prime.T)).astype(int)

    # This function computes the value of the sigmoid function.
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


class FLD:
    def __init__(self):
        self.w = None
        self.m0 = None
        self.m1 = None
        self.sw = None
        self.sb = None
        self.slope = None

    # This function computes the solution of Fisher's Linear Discriminant.
    def fit(self, X, y):
        X0 = X[y == 0]
        X1 = X[y == 1]
        self.m0 = np.mean(X0, axis=0).reshape((1, 2))
        self.m1 = np.mean(X1, axis=0).reshape((1, 2))
        self.sw = (X0 - self.m0).T @ (X0 - self.m0) + (X1 - self.m1).T @ (X1 - self.m1)
        self.sb = (self.m1 - self.m0).T @ (self.m1 - self.m0)
        self.w = np.linalg.inv(self.sw) @ (self.m1 - self.m0).T
        self.slope = self.w[1, 0] / self.w[0, 0]
        self.m0 = self.m0.squeeze()
        self.m1 = self.m1.squeeze()

    # This function takes the input data X and predicts the class label y by comparing the distance between the projected result of the testing data with the projected means (of the two classes) of the training data.
    # If it is closer to the projected mean of class 0, predict it as class 0, otherwise, predict it as class 1.
    def predict(self, X):
        pred_0 = abs((X - self.m0) @ self.w).squeeze()
        pred_1 = abs((X - self.m1) @ self.w).squeeze()

        y_pred = np.zeros(X.shape[0])
        y_pred[pred_0 > pred_1] = 1

        return y_pred.astype(int)

    # This function plots the projection line of the testing data.
    # You don't need to call this function in your submission, but you have to provide the screenshot of the plot in the report.
    def plot_projection(self, X):
        pred = self.predict(X)
        X0 = X[pred == 0]
        X1 = X[pred == 1]
        pt = np.linspace(-50, -20, 100)
        projected = ((X @ self.w) * self.w.squeeze()) / (self.w.T @ self.w)

        plt.scatter(X0[:, 0], X0[:, 1], c="r", label="Class 0", s=8)
        plt.scatter(X1[:, 0], X1[:, 1], c="b", label="Class 1", s=8)
        plt.scatter(projected[(pred == 0), 0], projected[(pred == 0), 1], c="r", s=8)
        plt.scatter(projected[(pred == 1), 0], projected[(pred == 1), 1], c="b", s=8)
        plt.plot(pt, self.slope * pt, color="green")
        plt.plot(
            [X[:, 0], projected[:, 0]],
            [X[:, 1], projected[:, 1]],
            color="black",
            linewidth=0.1,
            zorder=1,
        )
        plt.title(f"Projection Line: m={self.slope:.6f}, b={0}")
        plt.legend()
        plt.show()


# Do not modify the main function architecture.
# You can only modify the value of the arguments of your Logistic Regression class.
if __name__ == "__main__":
    # Data Loading
    train_df = DataFrame(read_csv("train.csv"))
    test_df = DataFrame(read_csv("test.csv"))

    # Part 1: Logistic Regression
    # Data Preparation
    # Using all the features for Logistic Regression
    X_train = train_df.drop(["target"], axis=1)
    y_train = train_df["target"]
    X_test = test_df.drop(["target"], axis=1)
    y_test = test_df["target"]

    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

    # Model Training and Testing
    LR = LogisticRegression(learning_rate=0.05, iteration=100)
    LR.fit(X_train, y_train)
    y_pred = LR.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Part 1: Logistic Regression")
    print(f"Weights: {LR.weights}, Intercept: {LR.intercept}")
    print(f"Accuracy: {accuracy}")
    # You must pass this assertion in order to get full score for this part.
    assert accuracy > 0.75, "Accuracy of Logistic Regression should be greater than 0.75"

    # Part 2: Fisher's Linear Discriminant
    # Data Preparation
    # Only using two features for FLD
    X_train = train_df[["age", "thalach"]]
    y_train = train_df["target"]
    X_test = test_df[["age", "thalach"]]
    y_test = test_df["target"]

    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

    # Model Training and Testing
    FLD = FLD()
    FLD.fit(X_train, y_train)
    y_pred = FLD.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Part 2: Fisher's Linear Discriminant")
    print(f"Class Mean 0: {FLD.m0}, Class Mean 1: {FLD.m1}")
    print(f"With-in class scatter matrix:\n{FLD.sw}")
    print(f"Between class scatter matrix:\n{FLD.sb}")
    print(f"w:\n{FLD.w}")
    print(f"Accuracy of FLD: {accuracy}")
    # You must pass this assertion in order to get full score for this part.
    assert accuracy > 0.65, "Accuracy of FLD should be greater than 0.65"

    FLD.plot_projection(X_test)
