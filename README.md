# Deep Fake Video Detection Using Deep Learning

## 📌 Project Overview
This project focuses on detecting deep fake videos by analyzing facial features extracted from video frames using deep learning techniques. The system uses Convolutional Neural Networks (CNNs) with MobileNetV2 transfer learning to classify videos as Real or Fake.

Deep fake videos often contain subtle artifacts such as facial asymmetry, blending errors, unnatural eye movements, and temporal inconsistencies that are difficult for humans to detect. This model learns these hidden patterns to provide reliable video authenticity verification.

---

## 🎯 Objectives
- Detect manipulated (deep fake) videos accurately
- Extract facial features from video frames
- Classify videos as Real or Fake using deep learning
- Provide a scalable and efficient detection pipeline

---

## 🧠 Technologies Used
- Python
- TensorFlow / Keras
- OpenCV
- MobileNetV2 (Transfer Learning)
- Haar Cascade Classifier
- CNN

---

## ⚙️ Methodology
1. Extract frames from input videos at fixed intervals
2. Detect and crop faces using Haar Cascade classifier
3. Preprocess facial images
4. Train CNN model based on MobileNetV2
5. Perform frame-level prediction
6. Use majority voting for final video classification

---

## 📊 Results
- Training Accuracy: ~89.95%
- Validation Accuracy: ~70.77%
- Evaluated using accuracy/loss graphs and confusion matrix
---

 ## 📸 Sample Outputs

### Accuracy & Loss
![Accuracy and Loss](results/accuracy_loss.png)

### Confusion Matrix
![Confusion Matrix](results/confusion_matrix.png)

### Real Video Prediction
![Real Output](results/real_video_output.jpg)


### Fake Video Prediction
![Fake Output](results/fake_video_output.png)


---

## 🚀 Setup & Usage

### Step 1: Clone the Repository
```bash
git clone https://github.com/ramavathsrujana/deepfakevideodetection.git
cd deepfakevideodetection
```
---

### Step 2: Install Dependencies
Make sure Python (3.8 or higher) is installed, then run:
```bash
pip install -r requirements.txt
```

### 🔹 Step 3: Prepare Dataset
```md
Arrange the dataset in the following structure:
dataset/
├── real/
└── fake/
```

> ⚠️ The dataset is not included in this repository due to size limitations.
Public datasets such as DFDC (DeepFake Detection Challenge) from Kaggle can be used.
---

### Step 4: Train the Model
Run the training script:
```bash
python src/train_model.py
```

This will:
- Load images from the dataset
- Train a MobileNetV2-based CNN model
- Save the trained model
---

### Step 5: Predict on a Video
Place a video file (e.g., `sample.mp4`) in the project directory and run:
```bash
python src/predict_video.py
```

The script will:
- Extract frames from the video
- Detect faces
- Predict Real/Fake labels for frames
- Output the final prediction
- 📌 The model was trained using Google Colab.
Dataset and trained model files are not included due to size constraints.

---

## 📁 Project Structure
```
deepfakevideodetection/
├── src/
│ ├── frame_extraction.py
│ ├── face_detection.py
│ ├── train_model.py
│ └── predict_video.py
├── results/
├── requirements.txt
├── .gitignore
└── README.md
```
---
## 🏁 Conclusion
The project demonstrates that lightweight deep learning models such as MobileNetV2 can effectively detect deep fake videos with good accuracy while remaining computationally efficient. This system can be extended for real-time detection and deployment in digital forensics and media verification.

---

## 👩‍💻 Authors
- Ramavath Srujana  
- Gurram Vijendar Reddy  
- Gandla Swathi  
- Chenna Chaitanya
---

## 📜 License
This project is licensed under the MIT License.
