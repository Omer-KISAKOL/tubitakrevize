import requests
import os
import json

# API base URL
API_URL = "http://localhost:8000"

def test_model_info():
    """Model bilgilerini test et"""
    print("=== Model Bilgisi Testi ===")
    response = requests.get(f"{API_URL}/model/bilgi/")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}\n")

def test_damage_classification(image_path):
    """Hasar türü sınıflandırmasını test et"""
    print(f"=== Hasar Türü Tespiti: {os.path.basename(image_path)} ===")
    
    with open(image_path, 'rb') as f:
        files = {'dosya': (os.path.basename(image_path), f, 'image/jpeg')}
        response = requests.post(f"{API_URL}/tespit/hasar-turu/", files=files)
    
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Dosya: {result['dosya']}")
        print(f"Hasar Türü: {result['hasar_turu']}")
        print(f"Güvenilirlik: {result['guvence_orani']}")
        
        # Uyarı kontrolü
        if 'uyari' in result:
            print(f"\n⚠️  {result['uyari']}")
            print(f"Alakasız Görsel: {'Evet' if result.get('alakasiz_gorsel', False) else 'Hayır'}")
        
        print("Detaylar:")
        print(f"  - Kesme Olasılığı: {result['detaylar']['kesme_olasiligi']}")
        print(f"  - Korozyon Olasılığı: {result['detaylar']['korozyon_olasiligi']}")
    else:
        print(f"Hata: {response.json()}")
    print()

def test_crack_detection(image_path):
    """Çatlak tespitini test et"""
    print(f"=== Çatlak Açı Tespiti: {os.path.basename(image_path)} ===")
    
    with open(image_path, 'rb') as f:
        files = {'dosya': (os.path.basename(image_path), f, 'image/jpeg')}
        response = requests.post(f"{API_URL}/tespit/catlak/", files=files)
    
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Dosya: {result['dosya']}")
        print(f"Açıklama: {result['aciklama']}")
        print(f"Açı: {result['aci_derece']}°")
        print(f"Sonuç Dosyası: {result['sonuc_dosyasi']}")
    else:
        print(f"Hata: {response.json()}")
    print()

def main():
    print("=== Açı Tespit API Demo ===\n")
    
    # Model bilgilerini kontrol et
    test_model_info()
    
    # Test edilecek görüntüler
    test_images = [
        "images/kesme/kesme hasar 1.jpg",
        "images/kesme/kesme hasar 2.JPG",
        "images/korozyon/korozyon 1.JPG",
        "images/korozyon/korozyon 2.JPG"
    ]
    
    # Her görüntü için testleri çalıştır
    for image_path in test_images:
        if os.path.exists(image_path):
            # Hasar türü tespiti
            test_damage_classification(image_path)
            
            # Eğer kesme hasarı ise çatlak tespiti yap
            if "kesme" in image_path:
                test_crack_detection(image_path)
        else:
            print(f"Uyarı: {image_path} bulunamadı.\n")
    
    print("Demo tamamlandı!")
    print("\nSwagger UI için tarayıcınızda açın: http://localhost:8000/docs")

if __name__ == "__main__":
    main() 