import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


def create_model():
   model = models.Sequential([
       layers.Conv2D(32, (3, 3), activation='relu', input_shape=(300, 300, 3)),
       layers.MaxPooling2D((2, 2)),
       layers.Flatten(),
       layers.Dense(64, activation='relu'),
       layers.Dense(3)  
   ])
   model.compile(optimizer='adam', loss='mean_squared_error')
   return model


model = create_model()
model.save('your_model.keras')  


model = tf.keras.models.load_model('your_model.keras')


def capture_image():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Camera not accessible.")
        return None
    
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Unable to capture image.")
        return None
    
    cap.release()
    cv2.imwrite('captured_image.jpg', frame)
    return 'captured_image.jpg'


def detect_object(image_path):
   image = cv2.imread(image_path)
   input_image = cv2.resize(image, (300, 300)) 
   input_image = np.expand_dims(input_image, axis=0)


   predictions = model.predict(input_image)
   dimensions = extract_dimensions(predictions)
   return dimensions


def extract_dimensions(predictions):
   width = predictions[0][0] 
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
   print(f"Maximum Volume: {volume}")


if __name__ == "__main__":
   main()
