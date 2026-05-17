import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

#
# Load your saved model
model = load_model("eye_disease_model.keras")  # or your HDF5 file

# Load the new image
img_path = "NORMAL-1001666-1.jpeg"
img = image.load_img(
    img_path, target_size=(224, 224)
)  # match the size used in training
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # add batch dimension
img_array = img_array / 255.0  # normalize same as training

class_names = ["CNV", "DME", "DRUSEN", "NORMAL"]


# Predict
pred = model.predict(img_array)
predicted_index = np.argmax(pred)  # get the index of the highest probability
predicted_class = class_names[predicted_index]  # get class name

for cls, prob in zip(class_names, pred[0]):
    print(f"{cls}: {prob*100:.2f}%")


print("Predicted class:", predicted_class)
