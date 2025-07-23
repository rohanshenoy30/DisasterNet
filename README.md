# Human Identification using CNN

This project implements a Convolutional Neural Network (CNN) using TensorFlow and Keras for binary human detection in images. It takes an image dataset and classifies whether a human is present (`class 1`) or not (`class 0`). The model is trained and evaluated with visualizations of accuracy and loss.

## 📂 Project Structure

```
human-identification-cnn/
├── human_identification_using_cnn.ipynb
├── dataset/
│   ├── train/
│   │   ├── class0/
│   │   └── class1/
│   └── test/
│       ├── class0/
│       └── class1/
├── model/
│   └── human_detection_model.h5
└── predictions/
    └── *.jpg
```

## 🚀 Features

* Binary image classification using a CNN.
* Data preprocessing and augmentation.
* Training and validation performance visualization.
* Model saving and loading.
* Inference on test images with predicted class output.

## 🛠️ Requirements

Install the required libraries using:

```bash
pip install tensorflow keras matplotlib numpy opencv-python
```

## 🧠 Model Architecture

The CNN model consists of:

* Convolutional layers with ReLU activation
* MaxPooling layers
* Dropout layers for regularization
* Dense layers for classification

Model summary is printed in the notebook.

## 📊 Training

The training includes:

* Image augmentation with rotation, flipping, and zoom
* 10 epochs of training
* `categorical_crossentropy` loss and `adam` optimizer
* Accuracy and loss plots

## 📈 Results

The notebook generates:

* Accuracy and loss curves
* Predictions on test images
* Saved model file: `human_detection_model.h5`

## 🖼️ Inference

Test images are passed to the model and predictions are printed on the images using OpenCV. Predicted class (`Human` or `Not Human`) is displayed with the image.

## 📦 Output

Example output:

```
File: image1.jpg → Predicted Class: Human
File: image2.jpg → Predicted Class: Not Human
```

## 📁 Dataset

Ensure that your dataset follows this structure:

```
dataset/
├── train/
│   ├── class0/ (e.g. non-human images)
│   └── class1/ (e.g. human images)
└── test/
    ├── class0/
    └── class1/
```

## 📌 Usage

Run the notebook `human_identification_using_cnn.ipynb` step-by-step in Jupyter Notebook or Google Colab.

## 👨‍💻 Author

Rohan Shenoy

## 📜 License

This project is licensed under the MIT License.

