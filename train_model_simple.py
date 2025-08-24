import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import joblib

# Veri yolları
data_dir = "images"
kesme_dir = os.path.join(data_dir, "kesme")
korozyon_dir = os.path.join(data_dir, "korozyon")

# Görüntü boyutları
IMG_HEIGHT = 64
IMG_WIDTH = 64

def extract_features(image_path):
    """Görüntüden özellik çıkar"""
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
        print(f"Hata: {image_path} işlenemedi - {str(e)}")
        return None

def load_dataset():
    """Veri setini yükle"""
    features = []
    labels = []
    
    # Kesme hasarı görüntüleri
    print("Kesme hasarı görüntüleri işleniyor...")
    for filename in os.listdir(kesme_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(kesme_dir, filename)
            feat = extract_features(img_path)
            if feat is not None:
                features.append(feat)
                labels.append('kesme')
                print(f"[OK] {filename}")
    
    # Korozyon görüntüleri
    print("\nKorozyon görüntüleri işleniyor...")
    for filename in os.listdir(korozyon_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(korozyon_dir, filename)
            feat = extract_features(img_path)
            if feat is not None:
                features.append(feat)
                labels.append('korozyon')
                print(f"[OK] {filename}")
    
    return np.array(features), np.array(labels)

def main():
    print("=== Basit Makine Öğrenmesi Model Eğitimi ===\n")
    
    # Veri setini yükle
    X, y = load_dataset()
    print(f"\nToplam görüntü sayısı: {len(X)}")
    print(f"Özellik sayısı: {X.shape[1]}")
    print(f"Kesme hasarı: {np.sum(y == 'kesme')}")
    print(f"Korozyon: {np.sum(y == 'korozyon')}")
    
    # Veriyi eğitim ve test olarak böl
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nEğitim seti: {len(X_train)} görüntü")
    print(f"Test seti: {len(X_test)} görüntü")
    
    # Özellikleri ölçekle
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Model 1: Support Vector Machine
    print("\n1. Support Vector Machine (SVM) eğitiliyor...")
    svm_model = SVC(kernel='rbf', probability=True, random_state=42)
    svm_model.fit(X_train_scaled, y_train)
    
    # SVM değerlendirme
    svm_pred = svm_model.predict(X_test_scaled)
    svm_acc = accuracy_score(y_test, svm_pred)
    print(f"SVM Test Doğruluğu: {svm_acc:.2%}")
    
    # Model 2: Random Forest
    print("\n2. Random Forest eğitiliyor...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    
    # Random Forest değerlendirme
    rf_pred = rf_model.predict(X_test_scaled)
    rf_acc = accuracy_score(y_test, rf_pred)
    print(f"Random Forest Test Doğruluğu: {rf_acc:.2%}")
    
    # En iyi modeli seç
    if svm_acc > rf_acc:
        best_model = svm_model
        best_name = "SVM"
        best_acc = svm_acc
    else:
        best_model = rf_model
        best_name = "Random Forest"
        best_acc = rf_acc
    
    print(f"\n{best_name} modeli seçildi (Doğruluk: {best_acc:.2%})")
    
    # Detaylı rapor
    print("\nSınıflandırma Raporu:")
    print(classification_report(y_test, best_model.predict(X_test_scaled)))
    
    # Modeli ve diğer bileşenleri kaydet
    joblib.dump(best_model, 'hasar_siniflandirma_modeli.pkl')
    joblib.dump(scaler, 'feature_scaler.pkl')
    
    # Label mapping kaydet
    label_mapping = {'kesme': 0, 'korozyon': 1}
    inverse_label_mapping = {v: k for k, v in label_mapping.items()}
    
    with open('label_mapping.pkl', 'wb') as f:
        pickle.dump({
            'label_to_int': label_mapping,
            'int_to_label': inverse_label_mapping,
            'classes': ['kesme', 'korozyon']
        }, f)
    
    print("\nModel dosyaları kaydedildi:")
    print("- hasar_siniflandirma_modeli.pkl")
    print("- feature_scaler.pkl")
    print("- label_mapping.pkl")
    
    print("\nEğitim tamamlandı!")

if __name__ == "__main__":
    main() 