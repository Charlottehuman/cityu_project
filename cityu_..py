import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


# Example model creation
def create_model():
   model = models.Sequential([
       layers.Conv2D(32, (3, 3), activation='relu', input_shape=(300, 300, 3)),
       layers.MaxPooling2D((2, 2)),
       layers.Flatten(),
       layers.Dense(64, activation='relu'),
       layers.Dense(3)  # Output layer for width, height, length
   ])
   model.compile(optimizer='adam', loss='mean_squared_error')
   return model


# Save the model
model = create_model()
model.save('your_model.keras')  # Save the model in .keras format


# Load the model
model = tf.keras.models.load_model('your_model.keras')


def capture_image():
   # Capture image from device camera
   cap = cv2.VideoCapture(0)
   ret, frame = cap.read()
   cap.release()
   cv2.imwrite('captured_image.jpg', frame)  # Save as .jpg
   return 'captured_image.jpg'


def detect_object(image_path):
   # Load image
   image = cv2.imread(image_path)
   # Preprocess image for model
   input_image = cv2.resize(image, (300, 300))  # Adjust size based on your model
   input_image = np.expand_dims(input_image, axis=0)


   # Object detection
   predictions = model.predict(input_image)
   # Process predictions to extract dimensions
   dimensions = extract_dimensions(predictions)
   return dimensions


def extract_dimensions(predictions):
   # Ensure predictions are in the expected format (e.g., [width, height, length])
   width = predictions[0][0]  # Adjust indexing as necessary
   height = predictions[0][1]
   length = predictions[0][2]
   return width, height, length


def calculate_volume(dimensions):
   width, height, length = dimensions
   volume = width * height * length
   return volume


def main():
   image_path = capture_image()
   dimensions = detect_object(image_path)
   volume = calculate_volume(dimensions)
   print(f"Dimensions: {dimensions}")
   print(f"Maximum Volume: {volume}")


if __name__ == "__main__":
   main()
