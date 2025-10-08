from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)
CORS(app)

# Load your uploaded model
model = tf.keras.models.load_model("mnist_cnn_model.h5")

def preprocess_image(image_base64):
    image_data = image_base64.split(",")[1]
    image = Image.open(BytesIO(base64.b64decode(image_data)))
    image = image.convert("L").resize((28, 28))
    arr = np.array(image) / 255.0
    arr = arr.reshape(1, 28, 28)
    return arr

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        img_arr = preprocess_image(data["image"])
        preds = model.predict(img_arr)
        digit = int(np.argmax(preds))
        return jsonify({"prediction": digit})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(port=5000, debug=True)
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)
CORS(app)

# Load your uploaded model
model = tf.keras.models.load_model("mnist_cnn_model.h5")

def preprocess_image(image_base64):
    image_data = image_base64.split(",")[1]
    image = Image.open(BytesIO(base64.b64decode(image_data)))
    image = image.convert("L").resize((28, 28))
    arr = np.array(image) / 255.0
    arr = arr.reshape(1, 28, 28)
    return arr

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        img_arr = preprocess_image(data["image"])
        preds = model.predict(img_arr)
        digit = int(np.argmax(preds))
        return jsonify({"prediction": digit})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(port=5000, debug=True)

