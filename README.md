# 🐔 Poultry Disease Detector

A deep learning web application for detecting common poultry diseases using images of chickens. Powered by **MobileNetV2** and a custom-trained Keras model, this project aims to assist poultry farmers with early disease detection and better management.

![App Screenshot](static/images/sick_chicken.jpg)

---

## 🚀 Features

- 🖼 Upload a chicken image for analysis
- 🧠 Uses a trained `MobileNetV2` model (`quick_model.h5`)
- ⚙️ Built with Flask (Python backend)
- 🎨 Clean UI with HTML + CSS
- 🐤 Supports detection of multiple poultry diseases

---

## 🛠 Tech Stack

- **Python** (Flask, TensorFlow/Keras)
- **HTML/CSS** (Frontend templates)
- **MobileNetV2** (Transfer Learning model)
- **VS Code + Git** (Dev environment and version control)

---

## 📁 Folder Structure

project_folder/
│
├── app.py # Flask backend
├── train_model.py # Model training script
├── quick_model.h5 # Trained model file
├── dataset/ # Your training/test/val image folders
│ └── train/val/test/
├── static/ # Static files (CSS/images/uploads)
│ ├── uploads/ # Uploaded test images
│ └── images/ # App visuals
├── templates/ # HTML templates
│ └── index.html
│ └── predict.html
└── requirements.txt # Python dependencies


---

## 🧪 How to Run the App

### 1. Install Dependencies

```bash
pip install -r requirements.txt

2. Run the Web App
bash
Copy
Edit
python app.py
The app will start at:
http://127.0.0.1:5000

🧠 How to Train Your Own Model
If you'd like to retrain the model:

bash
Copy
Edit
python train_model.py
Make sure your dataset is placed under the dataset/ directory with subfolders like train, test, and val, each containing class folders (e.g., healthy, coccidiosis, etc.).

📈 Sample Dataset Structure
bash
Copy
Edit
dataset/
├── train/
│   ├── healthy/
│   ├── coccidiosis/
│   └── ...
├── val/
│   ├── healthy/
│   ├── coccidiosis/
│   └── ...
└── test/
    ├── healthy/
    ├── coccidiosis/
    └── ...
📸 Example Predictions
Image	Prediction
	Coccidiosis
	Healthy

👤 Author
G. Hema Harshith Reddy
GitHub: Harshith-45
Email: hemaharshithreddygulimcherla@gmail.com

📌 License
This project is for educational and research purposes. Attribution appreciated.

💡 Future Enhancements
Deploy to Hugging Face or Render

Real-time webcam disease detection

Integration with farmer-friendly mobile apps

yaml
Copy
Edit

---

### ✅ To Use:
1. Copy this into a file named `README.md`
2. Save it in your project folder
3. Run:

```bash
git add README.md
git commit -m "Add README.md"
git push
