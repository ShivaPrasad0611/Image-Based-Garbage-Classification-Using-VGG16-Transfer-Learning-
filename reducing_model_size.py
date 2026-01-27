import tensorflow as tf

model = tf.keras.models.load_model("vgg16_garbage_classification_model.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

with open("model_quantized.tflite", "wb") as f:
    f.write(tflite_model)

print("Quantized model saved!")
