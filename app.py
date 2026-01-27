import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="model_quantized.tflite")
interpreter.allocate_tensors()

# Get input & output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Class labels
class_names = ["Cardboard", "Food Organics", "Glass", "Metal", "Other"]

def predict_image(img):
    # Resize image (same as training)
    img = img.resize((225, 225))
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], img)

    # Run inference
    interpreter.invoke()

    # Get prediction
    prediction = interpreter.get_tensor(output_details[0]['index'])

    return class_names[np.argmax(prediction)]

# Gradio Interface
interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(),
    title="Garbage Classification System",
    description="Upload an image of garbage to identify its type"
)

interface.launch()
