# 🔬 Skin Lesion Detection and Classification Pipeline

## 📌 Overview
This project implements a modular pipeline for skin lesion detection and classification using:
- YOLOv12 for lesion localization
- MobileNetV2 for multi-class classification (Herpes, Melanoma, MonkeyPox, Varicela)

## 🧠 Objectives
- Detect and crop lesion regions from raw images using YOLOv12
- Train a MobileNetV2 classifier on cropped lesions
- Evaluate model performance using F1 score, classification report, and confusion matrix
- Save predictions and cropped images for downstream analysis


## ⚙️ Pipeline Steps

### 1. YOLOv12 Detection
model = YOLO("yolo12n")
model.train(data="data.yaml", epochs=100, imgsz=500, batch=50, patience=20)
result = model.predict("Test/images", save=True)

### 2. Cropping Detected Lesions
- Extract bounding boxes
- Save cropped regions to cropped_dataset/
- Export detection metadata to data_skin.csv
  
### 3. Data Augmentation & Class Balancing
ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    validation_split=0.2
)

### 4. MobileNetV2 Classification
- Load pretrained weights
- Freeze base layers
- Add custom dense layers
- Train with class weights
model.fit(train_generator, validation_data=val_generator, epochs=20, class_weight=class_weights_dict)

## 📊 Evaluation Metrics
#### ✅ Weighted F1 Score: 0.3473


### 🧪 Prediction Example
img = image.load_img("Varicela (52).jpg_0.png", target_size=(128, 128))
prediction = model.predict(img_array)


### 📦 Output
- Cropped lesion images
- Trained .keras model
- CSV with bounding box metadata
- Evaluation metrics
### 🧭 Next Steps
- Improve class balance and sample diversity
- Fine-tune MobileNetV2 layers
- Integrate Grad-CAM for interpretability
- Prepare annotated notebook for migration portfolio


## 📫 Contact
Feel free to reach out via LinkedIn or GitHub for collaboration or freelance opportunities.
#### \https://www.linkedin.com/in/alireza-sobhani-385134245


```python
