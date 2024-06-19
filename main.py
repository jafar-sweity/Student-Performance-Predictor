import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

# Perceptron functions (without class)
def activation_function(x):
    return np.where(x >= 0, 1, 0)

def fit(X, y, learning_rate=0.01, max_epochs=1000):
    weights = np.zeros(X.shape[1])
    bias = 0
    for epoch in range(max_epochs):
        for idx, x_i in enumerate(X):
            linear_output = np.dot(x_i, weights) + bias
            y_predicted = activation_function(linear_output)
            error = y[idx] - y_predicted
            weights += learning_rate * error * x_i
            bias += learning_rate * error
        y_pred = predict(X, weights, bias)
        accuracy = np.mean(y_pred == y)
        print(f"Epoch {epoch + 1}/{max_epochs} - Accuracy: {accuracy:.4f}, Weights: {weights}, Bias: {bias}")
    return weights, bias

def predict(X, weights, bias):
    linear_output = np.dot(X, weights) + bias
    y_predicted = activation_function(linear_output)
    return y_predicted

# Function to prepare and train the model
def prepare_and_train_model(learning_rate, max_epochs, goal):
    data = {
        'Math': [80, 90, 50, 60, 70],
        'Science': [85, 95, 55, 65, 75],
        'English': [78, 88, 58, 68, 72],
        'Pass': [1, 1, 0, 0, 1]
    }
    df = pd.DataFrame(data)
    df.to_csv('student_scores.csv', index=False)
    df = pd.read_csv('student_scores.csv')
    X = df[['Math', 'Science', 'English']].values
    y = df['Pass'].values
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_normalized = (X - X_mean) / X_std
    weights, bias = fit(X_normalized, y, learning_rate, max_epochs)
    correct_predictions = 0
    for i in range(len(X)):
        X_train = np.concatenate((X_normalized[:i], X_normalized[i+1:]))
        y_train = np.concatenate((y[:i], y[i+1:]))
        X_test = (X[i] - X_mean) / X_std  # Normalize test input using training mean and std
        X_test = X_test.reshape(1, -1)
        y_test = y[i]
        weights, bias = fit(X_train, y_train, learning_rate, max_epochs)
        y_pred = predict(X_test, weights, bias)
        correct_predictions += (y_pred[0] == y_test)
    accuracy = correct_predictions / len(X)
    print(f'Final Accuracy: {accuracy}')
    np.savez('perceptron_model.npz', weights=weights, bias=bias)
    return weights, bias, accuracy, X_mean, X_std

# GUI functions
def predict_gui():
    try:
        global X_mean, X_std  # Access global variables for mean and std
        math = float(entry_math.get())
        science = float(entry_science.get())
        english = float(entry_english.get())
        data = np.array([[math, science, english]])
        data_normalized = (data - X_mean) / X_std  # Normalize input using training mean and std
        prediction = predict(data_normalized, weights, bias)
        result = 'Pass' if prediction[0] == 1 else 'Fail'
        messagebox.showinfo("Result", f'The prediction is: {result}')
    except ValueError:
        messagebox.showerror("Input error", "Please enter valid numerical values.")

def train_model_gui():
    try:
        learning_rate = float(entry_learning_rate.get())
        max_epochs = int(entry_max_epochs.get())
        goal = float(entry_goal.get())
        global weights, bias, X_mean, X_std
        weights, bias, accuracy, X_mean, X_std = prepare_and_train_model(learning_rate, max_epochs, goal)
        if accuracy >= goal:
            messagebox.showinfo("Training Result", f'Model trained successfully with accuracy: {accuracy}')
        else:
            messagebox.showwarning("Training Result", f'Model accuracy ({accuracy}) did not meet the goal ({goal}).')
    except ValueError:
        messagebox.showerror("Input error", "Please enter valid numerical values for training parameters.")

# GUI setup
root = tk.Tk()
root.title("Pass/Fail Predictor")

# Maximize the window and remove decorations
root.attributes('-fullscreen', True)
root.attributes('-zoomed', True)
root.configure(bg='black')  # Set background color to black for transitions

# Load and set background image
bg_image = Image.open("background.png")
bg_image = bg_image.resize((root.winfo_screenwidth(), root.winfo_screenheight()), Image.LANCZOS)
bg_photo = ImageTk.PhotoImage(bg_image)

# Create a label to hold the background image and place it using grid layout to fill entire window
bg_label = tk.Label(root, image=bg_photo)
bg_label.place(x=0, y=0, relwidth=1, relheight=1)

# Training parameters
frame = tk.Frame(root, bg='lightblue', bd=5)
frame.place(relx=0.5, rely=0.1, relwidth=0.75, relheight=0.1, anchor='n')

tk.Label(frame, text="Learning Rate:").grid(row=0, column=0)
tk.Label(frame, text="Max Epochs:").grid(row=0, column=2)
tk.Label(frame, text="Goal (Accuracy):").grid(row=0, column=4)

entry_learning_rate = tk.Entry(frame)
entry_max_epochs = tk.Entry(frame)
entry_goal = tk.Entry(frame)

entry_learning_rate.grid(row=0, column=1)
entry_max_epochs.grid(row=0, column=3)
entry_goal.grid(row=0, column=5)

train_button = tk.Button(frame, text='Train Model', command=train_model_gui, bg='#80c1ff')
train_button.grid(row=0, column=6, padx=10)

# Prediction inputs
frame_pred = tk.Frame(root, bg='lightblue', bd=5)
frame_pred.place(relx=0.5, rely=0.3, relwidth=0.75, relheight=0.1, anchor='n')

tk.Label(frame_pred, text="Math:").grid(row=1, column=0)
tk.Label(frame_pred, text="Science:").grid(row=1, column=2)
tk.Label(frame_pred, text="English:").grid(row=1, column=4)

entry_math = tk.Entry(frame_pred)
entry_science = tk.Entry(frame_pred)
entry_english = tk.Entry(frame_pred)

entry_math.grid(row=1, column=1)
entry_science.grid(row=1, column=3)
entry_english.grid(row=1, column=5)

predict_button = tk.Button(frame_pred, text='Predict', command=predict_gui, bg='#80c1ff')
predict_button.grid(row=1, column=6, padx=10)

root.mainloop()

