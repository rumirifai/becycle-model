from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import cloudinary
import cloudinary.uploader

app = Flask(__name__)

MODEL_PATH = "models/best_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# Config Cloudinary
cloudinary.config(
  cloud_name = "dp7xpq6hb",
  api_key = "351219748492825",
  api_secret = "BD2FiRS5mzrxcQT-WrvO870fxzo",
  secure = True
)

# Mapping index ke label
index_to_label = {
    0: "aerosol_cans", 1: "aluminum_food_cans", 2: "aluminum_soda_cans",
    3: "cardboard_boxes", 4: "cardboard_packaging", 5: "clothing",
    6: "coffee_grounds", 7: "disposable_plastic_cutlery", 8: "eggshells",
    9: "food_waste", 10: "glass_beverage_bottles", 11: "glass_cosmetic_containers",
    12: "glass_food_jars", 13: "magazines", 14: "newspaper", 15: "office_paper",
    16: "paper_cups", 17: "plastic_cup_lids", 18: "plastic_detergent_bottles",
    19: "plastic_food_containers", 20: "plastic_shopping_bags", 21: "plastic_soda_bottles",
    22: "plastic_straws", 23: "plastic_trash_bags", 24: "plastic_water_bottles",
    25: "shoes", 26: "steel_food_cans", 27: "styrofoam_cups", 28: "styrofoam_food_containers",
    29: "tea_bags"
}

# Daftar recyclable (berdasarkan nama label)
recyclable_labels = {
    "aluminum_food_cans", "aluminum_soda_cans", "cardboard_boxes",
    "cardboard_packaging", "glass_beverage_bottles", "glass_cosmetic_containers",
    "glass_food_jars", "magazines", "newspaper", "office_paper",
    "plastic_detergent_bottles", "plastic_food_containers", "plastic_soda_bottles",
    "plastic_water_bottles", "steel_food_cans"
}

THRESHOLD = 0.7

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0]
    pred_index = np.argmax(prediction)
    confidence = float(np.max(prediction))

    if confidence < THRESHOLD:
        return "unknown", confidence, None

    label = index_to_label[pred_index]
    is_recyclable = label in recyclable_labels

    return label, confidence, is_recyclable

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    
    upload_result = cloudinary.uploader.upload(file)
    
    img_url = upload_result["secure_url"]
    
    file_path = "temp.jpg"
    file.save(file_path)

    label, conf, recyclable = predict_image(file_path)
    os.remove(file_path)

    response = {
        'label': label,
        'confidence': f"{conf*100:.2f}%",
        'image_url': img_url
    }

    # Tambahkan recyclable hanya jika labelnya bukan unknown
    if label != "unknown":
        response['recyclable'] = recyclable

    return jsonify(response)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
