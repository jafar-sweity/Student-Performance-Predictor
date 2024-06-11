import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import messagebox

# Perceptron class implementing the learning algorithm
class Perceptron:
    def __init__(self, learning_rate=0.01, max_epochs=1000):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # Initialize weights and bias
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        for epoch in range(self.max_epochs):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_function(linear_output)

                # Calculate the error
                error = y[idx] - y_predicted

                # Update weights and bias
                self.weights += self.learning_rate * error * x_i
                self.bias += self.learning_rate * error

            # Debugging output
            y_pred = self.predict(X)
            accuracy = np.mean(y_pred == y)
            print(f"Epoch {epoch + 1}/{self.max_epochs} - Accuracy: {accuracy:.4f}, Weights: {self.weights}, Bias: {self.bias}")

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_function(linear_output)
        return y_predicted

    def activation_function(self, x):
        return np.where(x >= 0, 1, 0)

# Function to prepare and train the model
def prepare_and_train_model(learning_rate, max_epochs, goal):
    # Sample dataset
    data = {
        'Math': [80, 90, 50, 60, 70],
        'Science': [85, 95, 55, 65, 75],
        'English': [78, 88, 58, 68, 72],
        'Pass': [1, 1, 0, 0, 1]
    }

    # Create a DataFrame and save to CSV
    df = pd.DataFrame(data)
    df.to_csv('student_scores.csv', index=False)

    # Load dataset
    df = pd.read_csv('student_scores.csv')
    X = df[['Math', 'Science', 'English']].values
    y = df['Pass'].values

    # Normalize the data
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    # Initialize the Perceptron model with specified hyperparameters
    perceptron = Perceptron(learning_rate=learning_rate, max_epochs=max_epochs)

    # Train the Perceptron model using all data (Leave-One-Out Cross-Validation approach)
    correct_predictions = 0
    for i in range(len(X)):
        X_train = np.concatenate((X[:i], X[i+1:]))
        y_train = np.concatenate((y[:i], y[i+1:]))
        X_test = X[i].reshape(1, -1)
        y_test = y[i]

        perceptron.fit(X_train, y_train)
        y_pred = perceptron.predict(X_test)
        correct_predictions += (y_pred[0] == y_test)

    accuracy = correct_predictions / len(X)
    print(f'Final Accuracy: {accuracy}')

    # Save the trained model to a file (simplified for demonstration purposes)
    np.savez('perceptron_model.npz', weights=perceptron.weights, bias=perceptron.bias)

    # Return the trained model and other relevant information
    return perceptron, accuracy, X, y

# GUI prediction function
def predict():
    try:
        math = float(entry_math.get())
        science = float(entry_science.get())
        english = float(entry_english.get())
        data = np.array([[math, science, english]])
        data = (data - data.mean(axis=0)) / data.std(axis=0)  # Normalize input
        prediction = perceptron.predict(data)
        result = 'Pass' if prediction[0] == 1 else 'Fail'
        messagebox.showinfo("Result", f'The prediction is: {result}')
    except ValueError:
        messagebox.showerror("Input error", "Please enter valid numerical values.")

# GUI train model function
def train_model():
    try:
        learning_rate = float(entry_learning_rate.get())
        max_epochs = int(entry_max_epochs.get())
        goal = float(entry_goal.get())
        global perceptron, X, y  # Add this line to access the global variables
        perceptron, accuracy, X, y = prepare_and_train_model(learning_rate, max_epochs, goal)
        if accuracy >= goal:
            messagebox.showinfo("Training Result", f'Model trained successfully with accuracy: {accuracy}')
        else:
            messagebox.showwarning("Training Result", f'Model accuracy ({accuracy}) did not meet the goal ({goal}).')
    except ValueError:
        messagebox.showerror("Input error", "Please enter valid numerical values for training parameters.")

# Load the trained perceptron model (initialize as None)
perceptron = None

# GUI setup
root = tk.Tk()
root.title("Pass/Fail Predictor")

# Training parameters
tk.Label(root, text="Learning Rate:").grid(row=0)
tk.Label(root, text="Max Epochs:").grid(row=1)
tk.Label(root, text="Goal (Accuracy):").grid(row=2)

entry_learning_rate = tk.Entry(root)
entry_max_epochs = tk.Entry(root)
entry_goal = tk.Entry(root)

entry_learning_rate.grid(row=0, column=1)
entry_max_epochs.grid(row=1, column=1)
entry_goal.grid(row=2, column=1)

tk.Button(root, text='Train Model', command=train_model).grid(row=3, column=1, pady=4)

# Prediction inputs
tk.Label(root, text="Math:").grid(row=4)
tk.Label(root, text="Science:").grid(row=5)
tk.Label(root, text="English:").grid(row=6)

entry_math = tk.Entry(root)
entry_science = tk.Entry(root)
entry_english = tk.Entry(root)

entry_math.grid(row=4, column=1)
entry_science.grid(row=5, column=1)
entry_english.grid(row=6, column=1)

tk.Button(root, text='Predict', command=predict).grid(row=7, column=1, pady=4)

root.mainloop()

