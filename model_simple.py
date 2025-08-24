import os
import numpy as np
from PIL import Image
import pickle
import joblib

# Model ve bileşen yolları
MODEL_PATH = 'hasar_siniflandirma_modeli.pkl'
SCALER_PATH = 'feature_scaler.pkl'
LABEL_MAPPING_PATH = 'label_mapping.pkl'

# TensorFlow model yolları (eski)
TF_MODEL_PATH = 'hasar_siniflandirma_modeli.h5'
TF_LABEL_ENCODER_PATH = 'label_encoder.pkl'

# Görüntü boyutları
IMG_HEIGHT = 64
IMG_WIDTH = 64

# Global değişkenler
model = None
scaler = None
label_mapping = None

def check_model_type():
    """Hangi model tipinin mevcut olduğunu kontrol et"""
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        return 'sklearn'
    elif os.path.exists(TF_MODEL_PATH):
        return 'tensorflow'
    else:
        return None

def load_sklearn_model():
    """Scikit-learn modelini yükle"""
    global model, scaler, label_mapping
    
    if model is None:
        model = joblib.load(MODEL_PATH)
        print("Scikit-learn modeli yüklendi.")
    
    if scaler is None:
        scaler = joblib.load(SCALER_PATH)
        print("Feature scaler yüklendi.")
    
    if label_mapping is None:
        with open(LABEL_MAPPING_PATH, 'rb') as f:
            label_mapping = pickle.load(f)
        print("Label mapping yüklendi.")
    
    return model, scaler, label_mapping

def extract_features(image_path):
    """Görüntüden özellik çıkar (train_model_simple.py ile aynı)"""
    try:
        # Görüntüyü yükle ve boyutlandır
        img = Image.open(image_path).convert('RGB')
        img = img.resize((IMG_WIDTH, IMG_HEIGHT))
        
        # NumPy dizisine çevir
        img_array = np.array(img)
        
        # Basit özellik çıkarımı
        # 1. Renk histogramı (RGB kanalları için)
        hist_r = np.histogram(img_array[:,:,0], bins=16, range=(0, 256))[0]
        hist_g = np.histogram(img_array[:,:,1], bins=16, range=(0, 256))[0]
        hist_b = np.histogram(img_array[:,:,2], bins=16, range=(0, 256))[0]
        
        # 2. Ortalama ve standart sapma
        mean_r = np.mean(img_array[:,:,0])
        mean_g = np.mean(img_array[:,:,1])
        mean_b = np.mean(img_array[:,:,2])
        std_r = np.std(img_array[:,:,0])
        std_g = np.std(img_array[:,:,1])
        std_b = np.std(img_array[:,:,2])
        
        # 3. Kenar yoğunluğu (basit Sobel benzeri)
        gray = np.mean(img_array, axis=2)
        dx = np.abs(np.diff(gray, axis=0))
        dy = np.abs(np.diff(gray, axis=1))
        edge_density = np.mean(dx) + np.mean(dy)
        
        # Tüm özellikleri birleştir
        features = np.concatenate([
            hist_r, hist_g, hist_b,
            [mean_r, mean_g, mean_b, std_r, std_g, std_b, edge_density]
        ])
        
        return features / 255.0  # Normalize
        
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
    model_type = check_model_type()
    
    if model_type == 'sklearn':
        # Scikit-learn modeli kullan
        model, scaler, label_mapping = load_sklearn_model()
        
        # Özellikleri çıkar
        features = extract_features(image_path)
        features = features.reshape(1, -1)  # (1, n_features) şeklinde
        
        # Özellikleri ölçekle
        features_scaled = scaler.transform(features)
        
        # Tahmin yap
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        
        # Sınıf isimlerini al
        classes = label_mapping['classes']
        
        # Sonuçları hazırla
        class_probabilities = {}
        for i, class_name in enumerate(classes):
            class_probabilities[class_name] = float(probabilities[i])
        
        # En yüksek olasılığı bul
        max_prob_index = np.argmax(probabilities)
        predicted_class = classes[max_prob_index]
        confidence = float(probabilities[max_prob_index])
        
    elif model_type == 'tensorflow':
        # TensorFlow modeli için eski kodu kullan
        try:
            # Dinamik import
            import tensorflow as tf
            from tensorflow import keras
            
            # Eski model yükleme kodu
            if model is None:
                model = keras.models.load_model(TF_MODEL_PATH)
            
            # Label encoder yükle
            with open(TF_LABEL_ENCODER_PATH, 'rb') as f:
                label_encoder = pickle.load(f)
            
            # Görüntüyü işle (TensorFlow için 224x224)
            img = Image.open(image_path).convert('RGB')
            img = img.resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Tahmin yap
            predictions = model.predict(img_array)
            predicted_class_index = np.argmax(predictions[0])
            predicted_class = label_encoder.inverse_transform([predicted_class_index])[0]
            confidence = float(predictions[0][predicted_class_index])
            
            # Olasılıkları hazırla
            class_probabilities = {}
            for i, class_name in enumerate(label_encoder.classes_):
                class_probabilities[class_name] = float(predictions[0][i])
                
        except ImportError:
            raise Exception("TensorFlow yüklü değil. Lütfen önce 'python train_model_simple.py' ile scikit-learn modelini eğitin.")
    else:
        raise FileNotFoundError("Hiçbir model dosyası bulunamadı. Lütfen önce 'python train_model_simple.py' ile modeli eğitin.")
    
    # Güven eşiği kontrolü
    GUVEN_ESIGI = 0.1  # %65
    alakasiz_gorsel = confidence < GUVEN_ESIGI
    
    # Eğer her iki sınıf için de olasılık birbirine çok yakınsa (örn. %50-%50)
    # bu da alakasız görsel göstergesi olabilir
    olasilik_farki = abs(class_probabilities.get('kesme', 0) - class_probabilities.get('korozyon', 0))
    belirsiz_tahmin = olasilik_farki < 0.01  # %20'den az fark varsa
    
    result = {
        'hasar_turu': predicted_class,
        'olasiliklar': class_probabilities,
        'en_yuksek_olasilik': confidence,
        'alakasiz_gorsel_uyarisi': alakasiz_gorsel or belirsiz_tahmin
    }
    
    # Uyarı mesajı ekle
    if alakasiz_gorsel:
        result['uyari'] = f"DİKKAT: Model bu görsel için düşük güvenilirlik gösteriyor ({confidence:.1%}). Görsel, kesme veya korozyon hasarı içermiyor olabilir."
    elif belirsiz_tahmin:
        result['uyari'] = f"DİKKAT: Model bu görsel için belirsiz bir tahmin yaptı. Kesme ve korozyon olasılıkları birbirine çok yakın."
    
    return result

def get_model_info():
    """Model hakkında bilgi döndür"""
    model_type = check_model_type()
    
    if model_type == 'sklearn':
        model, scaler, label_mapping = load_sklearn_model()
        
        # Model tipini belirle
        model_name = type(model).__name__
        
        # Özellik sayısını al
        n_features = scaler.n_features_in_
        
        return {
            'model_tipi': 'Scikit-learn',
            'algoritma': model_name,
            'siniflar': label_mapping['classes'],
            'ozellik_sayisi': n_features,
            'model_girdi_boyutu': f"{IMG_HEIGHT}x{IMG_WIDTH}x3"
        }
    
    elif model_type == 'tensorflow':
        try:
            import tensorflow as tf
            from tensorflow import keras
            
            model = keras.models.load_model(TF_MODEL_PATH)
            with open(TF_LABEL_ENCODER_PATH, 'rb') as f:
                label_encoder = pickle.load(f)
            
            return {
                'model_tipi': 'TensorFlow/Keras',
                'siniflar': list(label_encoder.classes_),
                'model_girdi_boyutu': "224x224x3",
                'model_parametreleri': model.count_params()
            }
        except ImportError:
            return {
                'durum': 'TensorFlow modeli mevcut ama TensorFlow yüklü değil',
                'cozum': "Scikit-learn modelini kullanmak için 'python train_model_simple.py' çalıştırın"
            }
    else:
        return {
            'durum': 'Model bulunamadı',
            'cozum': "'python train_model_simple.py' komutu ile modeli eğitin"
        } 