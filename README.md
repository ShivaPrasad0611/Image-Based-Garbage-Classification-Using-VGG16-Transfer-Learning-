# Image-Based-Garbage-Classification-Using-VGG16-Transfer-Learning-
## 📚 Project Description

This project presents a deep learning-based **Garbage Classification System** using **VGG16** with transfer learning. The model classifies garbage images into categories like **plastic, paper, metal, glass, and organic waste**. By leveraging a pre-trained VGG16 model, we achieve high classification accuracy and make waste sorting efficient for recycling and environmental sustainability.

Once trained, the model is saved as an `.h5` file, ready for deployment. It can be integrated into **waste sorting factories**, **automated machines**, and **robotic arms** to automatically identify and separate waste materials. This reduces manual labor, improves sorting accuracy, and enhances recycling efficiency. The model is highly scalable and can be embedded into real-time smart bins or industrial waste management systems.

The system is trained using a labeled image dataset, resized to 224x224 pixels. It applies data augmentation techniques such as rotation, flipping, and zooming to improve generalization. The architecture uses VGG16 as a feature extractor and adds custom dense layers for classification.



## 🚀 Features

* ✅ Multi-class garbage classification
* ✅ Uses VGG16 pre-trained on ImageNet
* ✅ Transfer learning with custom layers
* ✅ Data augmentation for improved generalization
* ✅ Saves best model using checkpoints
* ✅ Ready for deployment in smart waste sorting systems

## 🖼️ Classes

* **Plastic**
* **Paper**
* **Metal**
* **Glass**
* **Organic Waste**

## 🛠️ Tech Stack

* **Python**
* **TensorFlow / Keras**
* **NumPy & Matplotlib**

## 📂 Dataset

Images are labeled into 5 classes. Preprocessing includes:

* Resizing to **224x224** pixels
* Normalization
* Data augmentation (rotation, flipping, zooming)

## 🔥 Model Architecture

* **Base Model**: VGG16 (without top layers, `include_top=False`)
* **Custom Layers**: Flatten + Dense layers + BatchNorm + Dropout
* **Activation**: ReLU (hidden layers), Softmax (output layer)

## ✅ Loss & Metrics

* **Loss**: `sparse_categorical_crossentropy`
* **Optimizer**: `Adam`
* **Metric**: `accuracy`

## 📊 Evaluation

Model achieves high test accuracy and performs well on unseen garbage images. Best model is saved as `.h5` file for deployment. The evaluation script not only tests the model's accuracy but also demonstrates its real-world applicability in automating waste sorting. This model can be installed in **waste sorting factories**, integrated into **industrial machines**, or embedded in **robots**. By using image-based classification, it helps automate the process of garbage separation, reducing manual labor, minimizing errors, and improving operational efficiency in recycling plants and waste management facilities.

## 📝 Usage

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/garbage-classification-vgg16.git
cd garbage-classification-vgg16
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model

```python
python train.py
```

### 4. Load and predict

```python
from tensorflow.keras.models import load_model
model = load_model("vgg16_garbage_classification_model.h5")
```

## 📦 Saved Model

* **vgg16\_garbage\_classification\_model.h5**

## 💡 Future Work

* Add more garbage classes
* Build a web or mobile app interface
* Deploy in real-time smart bins and factory sorting lines

---

⭐ If you find this useful, give it a star on GitHub!

