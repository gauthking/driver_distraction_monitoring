import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img
from tensorflow.keras.preprocessing import image
import os

test_images_path = r"D:\MainFolders\programcodestuff\gigs\driver_distraction_sheenamaam\train_Scriptvgg16\train\concentrating"
model_path = r'D:\\MainFolders\\programcodestuff\\gigs\\driver_distraction_sheenamaam\\test_script\\model\\model_vgg16_13epochs.h5'
model = load_model(model_path)

# print("Model Summary")
# model.summary()

# Function to preprocess the image


def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to predict the class of the image


def predict_image_class(img_path):
    img = preprocess_image(img_path)
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)
    return predicted_class, confidence


def predict_images_in_directory(directory):
    print(len(os.listdir(directory)))
    count = 0
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            image_path = os.path.join(directory, filename)
            predicted_class, confidence = predict_image_class(image_path)
            if predicted_class == 2 or predicted_class == 3:
                count += 1

            print(
                f"Image: {filename}, Predicted Class: {predicted_class}, Confidence: {confidence}")
    print(count/len(os.listdir(directory)))


# Test on a single image
predict_images_in_directory(test_images_path)


# predicted_class, confidence = predict_image_class(test_images_path)
# print("Predicted Class:", predicted_class)
# print("Confidence:", confidence)

# img = cv2.imread(test_image_path)
# cv2.putText(img, f"Predicted: {predicted_class}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
# cv2.imshow("Test Image", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
