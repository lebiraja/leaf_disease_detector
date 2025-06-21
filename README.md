# 🌿 Leaf Disease Detector

A machine learning-powered web application built with Flask to detect plant leaf diseases from images. Using a Convolutional Neural Network (CNN) trained on the PlantVillage dataset, this tool classifies leaf conditions to help farmers and agricultural enthusiasts identify diseases early and take timely action.

---

## 🚀 Features

* 🌱 **Disease Detection**: Identifies plant leaf diseases from uploaded images.
* 🧠 **CNN Model**: High-accuracy model trained on the PlantVillage dataset.
* 🌐 **Flask Web Interface**: User-friendly UI for image uploads and predictions.
* 📸 **Image Preview**: Displays uploaded leaf images with results.
* 📊 **Confidence Scores**: Shows predicted disease class with probability.
* 🪴 **Multi-Crop Support**: Covers crops like Apple, Corn, Grape, Potato, and more.

---

## 🛠️ Technologies Used

* Python 🐍
* TensorFlow/Keras 🦖 (or PyTorch, based on `plant_disease_model.pth`)
* Flask 🌐
* OpenCV 🖼️
* NumPy, Pandas 📊
* HTML/CSS & Bootstrap 🎨 (for responsive UI)

---

## 📂 Project Structure

```
leaf_disease_detector/
├── static/
│   └── uploads/             # Directory for uploaded images
├── templates/
│   ├── index.html           # Home page UI
│   └── result.html          # Prediction results page
├── model/
│   └── plant_disease_model.pth  # Trained CNN model
├── dataset/                 # PlantVillage dataset (optional)
├── app.py                   # Main Flask application
├── s.py                     # Dataset preprocessing script
├── requirements.txt         # Project dependencies
└── README.md                # This file
```

---

## 🧪 How to Run Locally

**Clone the Repository:**

```bash
git clone https://github.com/lebiraja/leaf_disease_detector.git
cd leaf_disease_detector
```

**Set Up a Virtual Environment:**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Install Dependencies:**

```bash
pip install -r requirements.txt
```

**Run the Flask App:**

```bash
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.
Upload a leaf image to get disease predictions.

**Preprocessing Dataset (if training):**

```bash
python s.py
```

Ensure the dataset is placed in the `dataset/` directory.

---

## 🧠 Model Information

* **Architecture**: Convolutional Neural Network (CNN), potentially using transfer learning (e.g., ResNet or custom CNN).
* **Dataset**: PlantVillage Dataset with \~54,000 images across multiple crops and diseases.
* **Classes**: Includes Apple Scab, Black Rot, Cedar Apple Rust, Healthy Leaves, and more.
* **Training**: Pre-trained model (`plant_disease_model.pth`) provided in the `model/` directory.

---

## 📊 Sample Results

| Input Image | Prediction | Confidence |
| ----------- | ---------- | ---------- |
| Apple Leaf  | Apple Scab | 92.3%      |

---

## 📌 To-Do

* [ ] Deploy on cloud platforms (Heroku, Render, etc.)
* [ ] Enhance UI/UX with improved styling and responsiveness
* [ ] Expand dataset with additional crop types and diseases
* [ ] Add mobile camera support for direct uploads

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a feature branch:

   ```bash
   git checkout -b feature/YourFeature
   ```
3. Commit changes:

   ```bash
   git commit -m "Add YourFeature"
   ```
4. Push to the branch:

   ```bash
   git push origin feature/YourFeature
   ```
5. Open a Pull Request.

Check the [issues page](https://github.com/lebiraja/leaf_disease_detector/issues) for open tasks.

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🙌 Acknowledgments

* PlantVillage Dataset for training data
* TensorFlow/Keras and PyTorch communities
* Flask and Bootstrap for web development

---

## 📨 Contact

**Lebi Raja C**
📧 [lebiraja2007@gmail.com](mailto:lebiraja2007@gmail.com)
🌐 [GitHub Profile](https://github.com/lebiraja)

---

> 🌿 *Saving plants, one pixel at a time!* 💻
