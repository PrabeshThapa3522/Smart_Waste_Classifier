# ♻️ Smart Waste Classifier

A deep learning-based image classification system that identifies different types of waste using Convolutional Neural Networks (CNN) and Transfer Learning (MobileNet). The application helps users correctly classify waste for better recycling and waste management.

---

## Features

- Image classification into five waste categories
- Custom CNN implementation
- Transfer Learning using MobileNet
- Image preprocessing and data augmentation
- Model comparison
- Performance evaluation
- Streamlit web application for prediction

---

## Waste Categories

- Cardboard
- Glass
- Metal
- Paper
- Plastic


## Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn
- Streamlit

---

## Models

I implemented two different models and compared their performance.

- Custom CNN
- MobileNet (Transfer Learning)

After training both models, I compared their accuracy and selected the better-performing model for prediction.

---

## Dataset

The dataset contains images of different waste materials divided into training, validation, and testing folders.

```
dataset/
├── train/
├── validation/
└── test/
```

---

## Project Structure

```
Smart_Waste_Classifier/
│
├── app.py
├── train.py
├── predict.py
├── requirements.txt
├── models/
├── utils/
├── saved_model/
└── dataset/
```

---

## How to Run

Install the required packages.

```bash
pip install -r requirements.txt
```

Train the model.

```bash
python train.py
```

Run the prediction script.

```bash
python predict.py
```

Start the Streamlit application.

```bash
python -m streamlit run app.py
```

---

## Features

- Image preprocessing
- Data augmentation
- Waste image classification
- Model comparison
- Streamlit interface for prediction

---

## Future Work

Some improvements that can be added in the future:

- Add more waste categories
- Improve prediction accuracy
- Deploy the application online
- Support real-time webcam prediction

---

## Author

Prabesh Thapa