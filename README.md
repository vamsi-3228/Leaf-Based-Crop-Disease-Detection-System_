# 🌿 Leaf-Based Crop Disease Detection System

## 📌 Problem Statement
Farmers often fail to identify crop diseases early, which results in crop loss and incorrect pesticide usage.
Most disease detection requires experts or laboratory testing which is slow and expensive.

This project builds an AI-based system that detects crop diseases using only leaf images.
The system identifies:
- Crop name
- Disease name
- Confidence percentage
- Treatment suggestion

No soil data, weather data, or sensor information is used.
Only leaf images are analyzed.


## 🛠 Tech Stack

| Layer | Technology |
|------|-----------|
| Frontend | HTML, CSS |
| Backend | Python, Flask |
| AI Model | TensorFlow, Keras |
| Image Processing | OpenCV, Pillow |
| Dataset | PlantVillage Leaf Dataset |
| Deployment | Localhost (Flask Web Server) |



## ⚙️ Setup Instructions

1. Clone the repository  
   git clone https://github.com/yourname/Leaf-Based-Crop-Disease-Detection-System

2. Open the project folder  
   cd Leaf-Based-Crop-Disease-Detection-System/website

3. Install required libraries  
   pip install -r requirements.txt

4. Run the website  
   python app.py

5. Open browser and go to  
   http://127.0.0.1:5000/


   ## 🤖 AI Tools Used

- TensorFlow – Used for building the deep learning model
- Keras – Used for CNN and transfer learning
- MobileNetV2 – Pre-trained model for leaf classification
- OpenCV – Used for image preprocessing
- Pillow – Used to load images
- Flask – Used to create the web interface



## 🧠 Prompt Strategy

AI tools were used to design:
- The system architecture
- Deep learning model
- Image preprocessing pipeline
- Flask web application

Prompts were carefully written to ensure:
- Only leaf images are used
- No soil, weather, or sensor data
- The project is hackathon-ready



## 💻 Source Code

This repository includes:
- Model training code
- Trained CNN model
- Flask backend
- HTML frontend
- Disease to treatment mapping



## 📊 Final Output

The system allows users to upload a leaf image and receive:
- Crop name
- Disease name
- Confidence score
- Treatment advice

![Output](docs/output.png)



## 🔁 Build Reproducibility

To recreate this project:

1. Download the PlantVillage leaf dataset
2. Organize images into train and validation folders
3. Run train.py to train the model
4. Generate leaf_model.h5
5. Place the model inside website/model/
6. Run python app.py
This allows anyone to rebuild this project from scratch.
