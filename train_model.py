import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import os

def train_model():
    """
    Train a simple model and save it to the models directory.
    This is just a placeholder - replace with your actual model training code.
    """
    # Create a simple dataset (replace with your actual data)
    np.random.seed(42)
    X = np.random.rand(100, 2)
    y = 3*X[:, 0] + 2*X[:, 1] + np.random.randn(100) * 0.1
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a simple model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Print model performance
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"Model R² on training data: {train_score:.4f}")
    print(f"Model R² on test data: {test_score:.4f}")
    
    # Save the model
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, 'model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model saved to {model_path}")
    
if __name__ == "__main__":
    train_model()
