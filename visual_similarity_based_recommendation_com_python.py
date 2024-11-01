# Instalação das bibliotecas necessárias
!pip install tensorflow keras opencv-python matplotlib scikit-learn

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from google.colab import drive

drive.mount('/content/drive')

image_directory = '/content/drive/MyDrive/Colab Notebooks/dataset-Recommendation'

def load_images_from_folder(folder):
    images = []
    labels = []
    for class_folder in os.listdir(folder):
        class_path = os.path.join(folder, class_folder)
        if os.path.isdir(class_path):
            for filename in os.listdir(class_path):
                img = cv2.imread(os.path.join(class_path, filename))
                if img is not None:
                    images.append(img)
                    labels.append(class_folder)
                else:
                    print(f"Erro ao carregar a imagem: {filename}")
    return images, labels

images, labels = load_images_from_folder(image_directory)

model = VGG16(weights='imagenet', include_top=False, pooling='avg')

def extract_features(images):
    features = []
    for img in images:
        img = cv2.resize(img, (224, 224))
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        feature = model.predict(img)
        features.append(feature.flatten())
    return np.array(features)

features = extract_features(images)

def recommend_images(test_image_path, top_n=5):
    test_image = cv2.imread(test_image_path)
    if test_image is None:
        print(f"Erro ao carregar a imagem de teste: {test_image_path}")
        return
    test_image = cv2.resize(test_image, (224, 224))
    test_image = np.expand_dims(test_image, axis=0)
    test_image = preprocess_input(test_image)

    test_features = model.predict(test_image).flatten().reshape(1, -1)

    similarities = cosine_similarity(test_features, features).flatten()

    recommended_indices = similarities.argsort()[-top_n:][::-1]

    plt.figure(figsize=(15, 10))
    for i, idx in enumerate(recommended_indices):
        plt.subplot(1, top_n, i + 1)
        plt.imshow(cv2.cvtColor(images[idx], cv2.COLOR_BGR2RGB))
        plt.title(labels[idx])
        plt.axis('off')
    plt.show()

test_image_path = '/content/drive/MyDrive/Colab Notebooks/fiat500.jpg'
recommend_images(test_image_path, top_n=5)
