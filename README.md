# Linear Regression Implementation in Python

A simple implementation of linear regression from scratch using Python and NumPy.

```python
import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    """
    A simple linear regression model implemented from scratch.
    
    Attributes:
        weights (numpy.ndarray): The weights (coefficients) of the linear model.
        bias (float): The bias term (intercept) of the linear model.
    """
    
    def __init__(self):
        """Initialize the linear regression model."""
        self.weights = None
        self.bias = None
    
    def fit(self, X, y, learning_rate=0.01, n_iters=1000):
        """
        Fit the linear regression model to the training data using gradient descent.
        
        Args:
            X (numpy.ndarray): Training data of shape (n_samples, n_features).
            y (numpy.ndarray): Target values of shape (n_samples,).
            learning_rate (float): The learning rate for gradient descent.
            n_iters (int): Number of iterations for gradient descent.
        """
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for _ in range(n_iters):
            # Predictions
            y_pred = np.dot(X, self.weights) + self.bias
            
            # Compute gradients
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= learning_rate * dw
            self.bias -= learning_rate * db
    
    def predict(self, X):
        """
        Predict using the linear regression model.
        
        Args:
            X (numpy.ndarray): Input data of shape (n_samples, n_features).
            
        Returns:
            numpy.ndarray: Predicted values.
        """
        return np.dot(X, self.weights) + self.bias

# Example usage
if __name__ == "__main__":
    # Generate some random data
    np.random.seed(42)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)
    
    # Create and fit the model
    model = LinearRegression()
    model.fit(X, y.ravel())
    
    # Make predictions
    X_test = np.array([[0], [2]])
    y_pred = model.predict(X_test)
    
    # Plot the results
    plt.scatter(X, y, color='blue', label='Training data')
    plt.plot(X_test, y_pred, color='red', label='Linear regression')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Linear Regression Example')
    plt.legend()
    plt.show()
    
    print(f"Model weights: {model.weights}")
    print(f"Model bias: {model.bias}")
```

## How It Works

1. **Initialization**: The model starts with zero weights and bias.
2. **Gradient Descent**: The model learns by iteratively adjusting the weights to minimize the mean squared error.
3. **Prediction**: Once trained, the model can make predictions on new data.

## Dependencies

- Python 3.x
- NumPy
- Matplotlib (for visualization)

## Installation

```bash
pip install numpy matplotlib
