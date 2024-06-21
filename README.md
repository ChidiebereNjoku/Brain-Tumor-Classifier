# Brain-Tumor-Classifier
## Description
Brain-Tumor-Classifier is a comprehensive image classification project designed to detect and classify four classes of brain tumor images: 'glioma_tumor', 'meningioma_tumor', 'no_tumor', and 'pituitary_tumor'. This project compares eight different deep learning algorithms: CNN (baseline model), DenseNet-201, VGG16, VGG19, InceptionV3, MobileNetV2, Xception, and ResNet50. The primary objectives are to evaluate the impact of data augmentation, transfer learning, and hyperparameter tuning, and to deploy a user-friendly web application for public use. This model is crucial for improving the accuracy and efficiency of brain tumor diagnosis using medical imaging data.

## Preprocessing
Scaling: All images are scaled to a range of 0 to 1.
Resizing: Images are resized to 128x128 pixels to ensure uniformity and compatibility with the models.
Data Augmentation: Techniques such as rotation, zoom, and flip are used to increase the diversity of the training dataset and improve model robustness.
SMOTE: Synthetic Minority Over-sampling Technique is employed to address class imbalance, ensuring that the model is well-trained across all classes.
Hyperparameter Tuining: Hyperparameter tuining was carried out
The deployment of this model as a web app allows for real-time predictions and can be accessed here. Brain tumor classifiers are critical in various sectors such as healthcare, research, and education, providing support for early diagnosis, treatment planning, and medical training.

## Features
Comprehensive Classification: Detects and classifies four types of brain tumors: 'glioma_tumor', 'meningioma_tumor', 'no_tumor', and 'pituitary_tumor'.
High Accuracy: Utilizes advanced deep learning models to achieve high accuracy in classification.
Real-time Prediction: Provides real-time predictions for uploaded images.
Class Imbalance Mitigation: Implements Synthetic Minority Over-sampling Technique (SMOTE) and data augmentation to address class imbalance.
Data Augmentation: Uses data augmentation techniques to enhance the training dataset and improve model robustness.
End-to-End Solution: Comprehensive workflow from image collection to full deployment as a web application using this url https://braintumorapp.streamlit.app/
## Project Objectives
Comparison of Deep Learning Algorithms: Compare eight different deep learning algorithms to detect and classify four classes of brain tumor images: pituitary tumor, no tumor, meningioma tumor, and glioma tumor.
Evaluation of Techniques: Evaluate the impact of data augmentation, transfer learning, and hyperparameter tuning to enhance the model and mitigate against class imbalance.
Performance Metrics: Evaluate accuracy, precision, recall, and F1-score in categorizing brain tumors using medical imaging data. The deep learning models include CNN (baseline model), DenseNet-201, VGG16, VGG19, InceptionV3, MobileNetV2, Xception, and ResNet50.
Web Application Deployment: Deploy a deep learning-enabled web app for brain tumor detection and classification for public use at Brain Tumor Classifier.
## Importance 
Brain tumor classifiers are crucial in various sectors to ensure timely and accurate diagnosis:

Healthcare: Assists radiologists and healthcare professionals in diagnosing brain tumors early, leading to better treatment outcomes.
Research: Supports medical research by providing tools for the analysis of brain tumor images.
Education: Aids in the training of medical students and professionals by offering practical experience with brain tumor classification.
Public Health: Enhances public health initiatives by enabling large-scale screenings and studies.

## Acknowledgments
TensorFlow
Keras
OpenCV
Special thanks to all contributors and the open-source community.
streamlit
## Deployment
This project is deployed as a web enabled streamlit application and can be accessed https://braintumorapp.streamlit.app/. The deployment process includes end-to-end implementation from image collection to full deployment, ensuring a seamless user experience.

