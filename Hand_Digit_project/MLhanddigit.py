import tensorflow as tf
from keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
import os

# Load and preprocess data
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

# Build CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))

# Evaluate model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc:.4f}")

# Save model
model.save("digit_recognition_model.h5")

# GUI App to draw digits and predict
import tkinter as tk
from PIL import Image, ImageDraw, ImageOps

class DigitRecognizerApp:
    def __init__(self, master):
        self.master = master
        master.title("Draw a digit")

        self.canvas = tk.Canvas(master, width=280, height=280, bg='white')
        self.canvas.pack()

        self.button_predict = tk.Button(master, text="Predict", command=self.predict_digit)
        self.button_predict.pack()

        self.button_clear = tk.Button(master, text="Clear", command=self.clear_canvas)
        self.button_clear.pack()

        self.label_result = tk.Label(master, text="Draw a digit and click Predict")
        self.label_result.pack()

        self.image = Image.new("L", (280, 280), 255)
        self.draw = ImageDraw.Draw(self.image)
        self.canvas.bind("<B1-Motion>", self.paint)

        self.model = tf.keras.models.load_model("digit_recognition_model.h5")

    def paint(self, event):
        x1, y1 = (event.x - 8), (event.y - 8)
        x2, y2 = (event.x + 8), (event.y + 8)
        self.canvas.create_oval(x1, y1, x2, y2, fill='black')
        self.draw.ellipse([x1, y1, x2, y2], fill=0)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, 280, 280], fill=255)
        self.label_result.config(text="Draw a digit and click Predict")

    def predict_digit(self):
        img = self.image.resize((28, 28))
        img = ImageOps.invert(img)
        img = np.array(img).reshape(1, 28, 28, 1) / 255.0
        pred = self.model.predict(img)
        digit = np.argmax(pred)
        confidence = np.max(pred)
        self.label_result.config(text=f"Predicted: {digit} (Confidence: {confidence:.2f})")

# Run the app
if __name__ == '__main__':
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()
