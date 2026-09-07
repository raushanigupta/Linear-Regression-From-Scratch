import numpy as np


class LinearRegressionScratch:

    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.theta = None
        self.cost_history = []

    def hypothesis(self, X):
        """Calculate predicted values using Xθ."""
        return np.dot(X, self.theta)

    def compute_cost(self, X, y):
        """Calculate Mean Squared Error."""
        predictions = self.hypothesis(X)
        errors = predictions - y

        cost = np.mean(errors ** 2)

        return cost

    def gradient_descent(self, X, y):
        """Perform one gradient descent update."""
        n = len(y)

        predictions = self.hypothesis(X)
        errors = predictions - y

        gradients = (2 / n) * np.dot(X.T, errors)

        self.theta = self.theta - self.learning_rate * gradients

    def fit(self, X, y):
        """Train the linear regression model."""

        # Add intercept column
        X = np.c_[np.ones((X.shape[0], 1)), X]

        # Initialize parameters with zeros
        self.theta = np.zeros(X.shape[1])

        # Reset cost history
        self.cost_history = []

        # Gradient descent
        for _ in range(self.iterations):

            self.gradient_descent(X, y)

            # Store cost after each iteration
            cost = self.compute_cost(X, y)
            self.cost_history.append(cost)

        return self

    def predict(self, X):
        """Generate predictions for input data."""

        # Add intercept column
        X = np.c_[np.ones((X.shape[0], 1)), X]

        return self.hypothesis(X)