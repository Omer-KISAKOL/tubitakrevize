import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# Veri yolları
data_dir = "images"
kesme_dir = os.path.join(data_dir, "kesme")
korozyon_dir = os.path.join(data_dir, "korozyon")

# Görüntü boyutları
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32

def load_and_preprocess_image(image_path):
    """Görüntüyü yükle ve önişle"""
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((IMG_WIDTH, IMG_HEIGHT))
        img_array = np.array(img) / 255.0  # Normalize
        return img_array
    except Exception as e:
        print(f"Hata: {image_path} yüklenemedi - {str(e)}")
        return None

def load_dataset():
    """Veri setini yükle"""
    images = []
    labels = []
    
    # Kesme hasarı görüntüleri
    print("Kesme hasarı görüntüleri yükleniyor...")
    for filename in os.listdir(kesme_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(kesme_dir, filename)
            img = load_and_preprocess_image(img_path)
            if img is not None:
                images.append(img)
                labels.append('kesme')
                print(f"✓ {filename}")
    
    # Korozyon görüntüleri
    print("\nKorozyon görüntüleri yükleniyor...")
    for filename in os.listdir(korozyon_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(korozyon_dir, filename)
            img = load_and_preprocess_image(img_path)
            if img is not None:
                images.append(img)
                labels.append('korozyon')
                print(f"✓ {filename}")
    
    return np.array(images), np.array(labels)

def create_cnn_model(input_shape, num_classes):
    """CNN modeli oluştur"""
    model = keras.Sequential([
        # Konvolüsyon katmanları
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        
        # Tam bağlantılı katmanlar
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def main():
    print("=== Makine Öğrenmesi Model Eğitimi ===\n")
    
    # Veri setini yükle
    X, y = load_dataset()
    print(f"\nToplam görüntü sayısı: {len(X)}")
    print(f"Kesme hasarı: {np.sum(y == 'kesme')}")
    print(f"Korozyon: {np.sum(y == 'korozyon')}")
    
    # Label encoding
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_categorical = keras.utils.to_categorical(y_encoded)
    
    # Label encoder'ı kaydet
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # Veriyi eğitim ve test olarak böl
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_categorical, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nEğitim seti: {len(X_train)} görüntü")
    print(f"Test seti: {len(X_test)} görüntü")
    
    # Model oluştur
    model = create_cnn_model(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        num_classes=len(label_encoder.classes_)
    )
    
    # Model özeti
    model.summary()
    
    # Model derleme
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Data augmentation
    datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.2
    )
    
    # Model eğitimi
    print("\nModel eğitimi başlıyor...")
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        epochs=50,
        validation_data=(X_test, y_test),
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001
            )
        ]
    )
    
    # Model değerlendirme
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"\nTest Doğruluğu: {test_accuracy:.2%}")
    print(f"Test Kaybı: {test_loss:.4f}")
    
    # Modeli kaydet
    model.save('hasar_siniflandirma_modeli.h5')
    print("\nModel 'hasar_siniflandirma_modeli.h5' olarak kaydedildi.")
    
    # Eğitim geçmişini kaydet
    import json
    with open('training_history.json', 'w') as f:
        json.dump(history.history, f)
    
    print("\nEğitim tamamlandı!")

if __name__ == "__main__":
    main() 