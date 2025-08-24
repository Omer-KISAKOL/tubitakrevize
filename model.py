import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import pickle

# Model ve label encoder yolları
MODEL_PATH = 'hasar_siniflandirma_modeli.h5'
LABEL_ENCODER_PATH = 'label_encoder.pkl'

# Görüntü boyutları (eğitimde kullanılanlarla aynı olmalı)
IMG_HEIGHT = 224
IMG_WIDTH = 224

# Global değişkenler
model = None
label_encoder = None

def load_model():
    """Modeli ve label encoder'ı yükle"""
    global model, label_encoder
    
    if model is None:
        if os.path.exists(MODEL_PATH):
            model = keras.models.load_model(MODEL_PATH)
            print("Model başarıyla yüklendi.")
        else:
            raise FileNotFoundError(f"Model dosyası bulunamadı: {MODEL_PATH}")
    
    if label_encoder is None:
        if os.path.exists(LABEL_ENCODER_PATH):
            with open(LABEL_ENCODER_PATH, 'rb') as f:
                label_encoder = pickle.load(f)
            print("Label encoder başarıyla yüklendi.")
        else:
            raise FileNotFoundError(f"Label encoder dosyası bulunamadı: {LABEL_ENCODER_PATH}")
    
    return model, label_encoder

def preprocess_image(image_path):
    """Görüntüyü model için önişle"""
    try:
        # Görüntüyü yükle
        img = Image.open(image_path).convert('RGB')
        
        # Boyutlandır
        img = img.resize((IMG_WIDTH, IMG_HEIGHT))
        
        # NumPy dizisine çevir ve normalize et
        img_array = np.array(img) / 255.0
        
        # Batch boyutu ekle (1, 224, 224, 3)
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        raise Exception(f"Görüntü işlenirken hata oluştu: {str(e)}")

def predict_damage_type(image_path):
    """
    Verilen görüntüdeki hasar türünü tahmin et
    
    Args:
        image_path: Görüntü dosyasının yolu
        
    Returns:
        dict: Tahmin sonuçları
            - hasar_turu: 'kesme' veya 'korozyon'
            - olasiliklar: Her sınıf için olasılık değerleri
            - en_yuksek_olasilik: En yüksek olasılık değeri
    """
    # Model ve label encoder'ı yükle
    model, label_encoder = load_model()
    
    # Görüntüyü önişle
    processed_image = preprocess_image(image_path)
    
    # Tahmin yap
    predictions = model.predict(processed_image)
    
    # En yüksek olasılığa sahip sınıfı bul
    predicted_class_index = np.argmax(predictions[0])
    predicted_class = label_encoder.inverse_transform([predicted_class_index])[0]
    confidence = float(predictions[0][predicted_class_index])
    
    # Tüm sınıfların olasılıklarını hazırla
    class_probabilities = {}
    for i, class_name in enumerate(label_encoder.classes_):
        class_probabilities[class_name] = float(predictions[0][i])
    
    return {
        'hasar_turu': predicted_class,
        'olasiliklar': class_probabilities,
        'en_yuksek_olasilik': confidence
    }

def get_model_info():
    """Model hakkında bilgi döndür"""
    model, label_encoder = load_model()
    
    return {
        'siniflar': list(label_encoder.classes_),
        'model_girdi_boyutu': f"{IMG_HEIGHT}x{IMG_WIDTH}x3",
        'model_parametreleri': model.count_params()
    } 