#import the libraries
import numpy as np
import tensorflow as tf
import struct

#tensorflow
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Activation, experimental
from tensorflow.keras import Sequential, Input, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras import regularizers


#to make a GUI for the hand writing
from tkinter import Tk, Canvas, Button
import matplotlib.pyplot as plt

#image processing
from PIL import Image, ImageOps, ImageGrab


 #Load the dataset
def load_images(file_path):
    with open(file_path, 'rb') as file:
        # Read magic number, number of images, rows, and columns
        magic, num_images, rows, cols = struct.unpack('>IIII', file.read(16))
        # Read the image data and reshape it
        images = np.fromfile(file, dtype=np.uint8).reshape(num_images, rows, cols, 1)
        return images / 255.0  # Normalize pixel values

def load_labels(file_path):
    with open(file_path, 'rb') as file:
        # Read magic number and number of labels
        magic, num_labels = struct.unpack('>II', file.read(8))
        # Read the label data
        labels = np.fromfile(file, dtype=np.uint8)
        return to_categorical(labels, 10)  # One-hot encode the labels

base_path = r"your path for the datasets......"

# Load training and test data using the modified path
train_images = load_images(f"{base_path}\\train-images.idx3-ubyte")
train_labels = load_labels(f"{base_path}\\train-labels.idx1-ubyte")
test_images = load_images(f"{base_path}\\t10k-images.idx3-ubyte")
test_labels = load_labels(f"{base_path}\\t10k-labels.idx1-ubyte")


# Define the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), 
                  kernel_regularizer=regularizers.l2(0.001)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])


# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# Set up the learning rate scheduler
lr_scheduler = ReduceLROnPlateau(monitor='val_accuracy', patience=2, factor=0.5, min_lr=0.00001)

# Train the model
model.fit(train_images, train_labels, epochs=20, batch_size=32, validation_split=0.1)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc}")

model.save('digit_recognizer.h5')


model = models.load_model('digit_recognizer.h5')

# Function to preprocess the canvas image and predict the digit
def predict_digit(image):
    image = image.resize((28, 28))  # Resize to 28x28 pixels
    image = ImageOps.grayscale(image)  # Convert to grayscale
    image = np.array(image) / 255.0  # Normalize pixel values
    image = image.reshape(1, 28, 28, 1)  # Reshape to match model input shape
    prediction = model.predict(image)
    return np.argmax(prediction)  # Return the digit with the highest probability


# Train the model and save the history
history = model.fit(train_images, train_labels, 
                    epochs=20, 
                    batch_size=32,
                    validation_split=0.1,
                    callbacks=[lr_scheduler])

# Plot the training and validation accuracy over epochs
plt.figure(figsize=(12, 4))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.show()


model.summary()

#GUI 
class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Gridded Digit Recognizer")

        # Parameters for the grid
        self.grid_size = 28
        self.square_size = 20  # Size of each square (pixel)
        self.canvas_size = self.grid_size * self.square_size

        # Initialize a grid to store the "pixels" state (0 or 1)
        self.grid_data = np.zeros((self.grid_size, self.grid_size), dtype=int)

        # Create a canvas with a gridded layout
        self.canvas = Canvas(self.root, width=self.canvas_size, height=self.canvas_size, bg="white")
        self.canvas.grid(row=0, column=0, pady=10, padx=10, columnspan=2)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<Button-1>", self.draw)

        # Button to clear the canvas
        self.clear_button = Button(self.root, text="Clear", command=self.clear_canvas, 
                                   font=("Helvetica", 14), width=10, height=2, bg="#f5f5f5")
        self.clear_button.grid(row=1, column=0, padx=20, pady=20)

        # Button to recognize the digit
        self.predict_button = Button(self.root, text="Predict", command=self.recognize_digit, 
                                     font=("Helvetica", 14), width=10, height=2, bg="#4CAF50", fg="white")
        self.predict_button.grid(row=1, column=1, padx=20, pady=20)

        # Label to display the prediction result
        self.result_label = Label(self.root, text="Draw a digit and click 'Predict'", font=("Helvetica", 12))
        self.result_label.grid(row=2, column=0, columnspan=2, pady=10)

        # Load the trained model
        self.model = tf.keras.models.load_model('digit_recognizer.h5')

    def draw(self, event):
        # Get the row and column of the grid square being clicked
        col = event.x // self.square_size
        row = event.y // self.square_size

        if 0 <= row < self.grid_size and 0 <= col < self.grid_size:
            # Mark the square as "active" (1) and color it black
            self.grid_data[row, col] = 1
            self.canvas.create_rectangle(
                col * self.square_size, row * self.square_size,
                (col + 1) * self.square_size, (row + 1) * self.square_size,
                fill="#00a4bd"
            )

    def clear_canvas(self):
        # Reset the canvas and grid data
        self.canvas.delete("all")
        self.grid_data = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.result_label.config(text="Draw a digit and click 'Predict'")

    def recognize_digit(self):
        # Convert the grid data to the format required by the model
        img = self.grid_data.reshape(1, self.grid_size, self.grid_size, 1).astype("float32")
        
        # Predict the digit using the model
        prediction = self.model.predict(img)
        digit = np.argmax(prediction)
        
        # Display the prediction result
        self.result_label.config(text=f"Prediction: {digit}")
        
       
# Main function to run the GUI
def main():
    root = Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()

