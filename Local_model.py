import numpy as np

class LinearLocalModel:
    """
    A simple local interpretable model using softmax regression
    to approximate the predictions of a neural network in a small
    neighborhood around a given instance.

    Attributes:
        W (ndarray): Weight matrix of shape (n_features, n_classes).
        b (ndarray): Bias vector of shape (1, n_classes).
        lr (float): Learning rate for gradient descent.
        epochs (int): Number of training epochs.
    """

    def __init__(self, input_dim: int, num_classes: int = 3, lr: float = 0.1, epochs: int = 200):
        """
        Initialize the LinearLocalModel.

        Args:
            input_dim: Number of input features.
            num_classes: Number of output classes.
            lr: Learning rate for gradient updates.
            epochs: Number of epochs to run gradient descent.
        """
        self.W = np.zeros((input_dim, num_classes))
        self.b = np.zeros((1, num_classes))
        self.lr = lr
        self.epochs = epochs

    def softmax(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the softmax of each row of the input array.

        Args:
            x: Logit array of shape (n_samples, n_classes).

        Returns:
            probs: Array of shape (n_samples, n_classes) with
                   softmax-normalized probabilities.
        """
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train the model using gradient descent on cross-entropy loss.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: One-hot label matrix of shape (n_samples, n_classes).
        """
        for _ in range(self.epochs):
            # Compute logits and probabilities
            logits = X @ self.W + self.b
            probs = self.softmax(logits)

            # Compute gradient of loss w.r.t. weights and biases
            error = probs - y
            grad_W = (X.T @ error) / len(X)
            grad_b = np.mean(error, axis=0, keepdims=True)

            # Update parameters
            self.W -= self.lr * grad_W
            self.b -= self.lr * grad_b

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Compute class probabilities for input samples.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            probs: Softmax probabilities of shape (n_samples, n_classes).
        """
        logits = X @ self.W + self.b
        return self.softmax(logits)

    def get_weights(self) -> np.ndarray:
        """
        Retrieve the learned weight matrix.

        Returns:
            W: Array of shape (n_features, n_classes).
        """
        return self.W
