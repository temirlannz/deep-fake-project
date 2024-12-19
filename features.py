import cv2
import numpy as np
import tensorflow as tf
# from tensorflow.keras.preprocessing.image import img_to_array, load_img
# from tensorflow.keras.utils import array_to_img

# Load a pre-trained TensorFlow/Keras model for deepfake detection
model = tf.keras.models.load_model('./models/deepfake2.keras')
"""
This line loads a deep learning model saved in the specified file path ('deepfake2.keras').
- `tf.keras.models.load_model`: Loads a complete Keras model, including architecture, weights, and optimizer settings.
- The model is assumed to be trained for detecting deepfake images/videos.
"""


def crop_face(img_arr):
    """
    Detects and crops the face from the given image, resizes it, and normalizes it for model input.

    Args:
        img_arr (numpy.ndarray): The input image array (expected in BGR format as returned by OpenCV).

    Returns:
        numpy.ndarray: A 224x224 normalized image array of the cropped face if a face is detected.
        int: Returns -1 if no face is detected.
    """
    # Convert the image from BGR (OpenCV default) to RGB (used for processing or visualization)
    img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
    """
    - OpenCV reads images in BGR format, but most image processing libraries like TensorFlow/Keras use RGB.
    - `cv2.cvtColor`: Converts the color space of the image.
    """

    # Load the pre-trained Haar Cascade for frontal face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    """
    - `cv2.CascadeClassifier`: Detects objects (in this case, faces) in images using the Haar feature-based cascade.
    - `cv2.data.haarcascades`: Provides the path to OpenCV's pre-trained cascade classifiers.
    - 'haarcascade_frontalface_default.xml': A cascade file trained specifically to detect frontal faces.
    """

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(
        img_arr, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    """
    - `detectMultiScale`: Detects multiple faces in the image.
    - `scaleFactor=1.1`: The image is scaled down by 10% at each step to detect faces of varying sizes.
    - `minNeighbors=5`: Ensures detected regions are grouped into a face if they are close enough.
    - `minSize=(30, 30)`: Sets the minimum size of the face to be detected (30x30 pixels).
    - Returns: A list of rectangles [(x, y, w, h)] where each rectangle represents a detected face.
    """

    # Check if any faces were detected
    if len(faces) > 0:
        # Extract the first detected face (assuming one face per image)
        x, y, w, h = faces[0]

        # Define margins around the face to include extra context in the crop
        margin = 200
        x_margin = max(0, x - margin)  # Ensure margins do not go out of bounds
        y_margin = max(0, y - margin)
        w_margin = min(img_arr.shape[1], x + w + margin)  # Ensure width stays within image bounds
        h_margin = min(img_arr.shape[0], y + h + margin)  # Ensure height stays within image bounds
        """
        - `margin=200`: Adds extra pixels around the detected face for better context.
        - `max(0, ...)` and `min(...)`: Prevent cropping beyond the image boundaries.
        """

        # Crop the region of interest (ROI) containing the face with margins
        cropped_face = img_arr[y_margin:h_margin, x_margin:w_margin]

        # Resize the cropped face to the model's expected input size (224x224)
        cropped_face = cv2.resize(cropped_face, (224, 224)) / 255.0
        """
        - `cv2.resize`: Resizes the image to 224x224 pixels.
        - `/ 255.0`: Normalizes pixel values to the range [0, 1] for better model performance.
        """

        # Return the processed face image
        return cropped_face

    # If no faces are detected, return -1
    return -1
    """
    Returning -1 indicates that no face was detected in the image.
    """


def image_classifier(img_path):
    """
    Classifies an image by detecting a face, processing it, and using a pre-trained model for prediction.

    Args:
        img_path (str): The file path to the image to be classified.

    Returns:
        int: The classification result as an integer.
        -1: Returned if no face is detected in the image.
    """
    # Read the image from the given file path using OpenCV
    img_arr = cv2.imread(img_path)
    """
    - `cv2.imread(img_path)`: Reads the image at `img_path` into a NumPy array.
    - The image is read in BGR format (default for OpenCV).
    - `img_arr`: Stores the image as a multi-dimensional array.
    """

    # Attempt to crop the face from the image
    face = crop_face(img_arr)
    """
    - `crop_face(img_arr)`: Calls the previously defined function to detect and crop the face in the image.
    - `face`: The cropped and preprocessed face image (224x224 normalized array) or -1 if no face is detected.
    """

    # Check if the cropped face is valid (i.e., if a face was detected)
    if not isinstance(face, np.ndarray):
        return -1
    """
    - `isinstance(face, np.ndarray)`: Checks if the returned `face` is a valid NumPy array.
    - If `face` is not a NumPy array (e.g., -1 was returned), the function exits and returns -1 to indicate failure.
    """

    # Add an extra dimension to the face array to match the model's input shape
    input = np.expand_dims(face, axis=0)
    """
    - `np.expand_dims(face, axis=0)`: Expands the dimensions of the face array to make it suitable for model input.
      Example: Converts shape (224, 224, 3) to (1, 224, 224, 3).
    - Neural networks typically expect a batch dimension (even for a single input), which is added here.
    - `input`: The processed face array ready for prediction.
    """

    # Perform prediction using the pre-trained model
    pred = model.predict(input)
    """
    - `model.predict(input)`: Passes the input image to the deepfake detection model for classification.
    - `pred`: The model's output, typically a probability distribution over the classes.
    """

    # Debug print: Log the raw prediction output
    print("PREDUCTIONNNNNN", pred)
    """
    - Debugging statement to inspect the raw prediction results.
    - `pred` is usually an array of probabilities (e.g., `[0.1, 0.9]` for a binary classification).
    """

    # Get the index of the class with the highest predicted probability
    res = np.argmax(pred)
    """
    - `np.argmax(pred)`: Returns the index of the highest value in the prediction array.
      Example: If `pred = [0.1, 0.9]`, `res = 1`.
    - `res`: Represents the predicted class label as an integer.
    """

    # Debug print: Log the predicted class label
    print("weoinfowenofnwoenfoweinf", res)
    """
    - Another debugging statement to inspect the predicted class label.
    """

    # Return the final classification result as an integer
    return int(res)
    """
    - `int(res)`: Ensures the returned value is an integer, representing the predicted class.
    - The function ends by returning the classification result.
    """


def video_classifier(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Unable to open video")

    count = 0
    noFrame = 0
    fakeFrame = 0
    realFrame = 0
    
    while cap.isOpened():
        ret,frame = cap.read()

        if not ret:
            break;

        count+=1
        if(not count%3==0):
            continue;
        
        face = crop_face(frame)

        if not isinstance(face,np.ndarray):
            continue;
    
        count+=1
        data = np.expand_dims(face,axis=0)
        pred = np.argmax(model.predict(data))
        print(pred)

        if pred==1:
            return 1

    cap.release()

    if count==0:
        return -1
    
    return 0


