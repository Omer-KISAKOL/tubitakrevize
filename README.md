# Açı Tespit API

Bu API, görüntülerdeki çatlakların veya kırmızı noktaların açılarını tespit etmek ve makine öğrenmesi ile hasar türünü sınıflandırmak için kullanılır.

## Özellikler

1. **Çatlak Açı Tespiti**: Görüntüdeki çatlağın yatay ile yaptığı açıyı tespit eder
2. **Kırmızı Nokta Açı Tespiti**: Görüntüdeki kırmızı noktaların dikey ile yaptığı açıyı tespit eder
3. **Hasar Türü Sınıflandırma** (Makine Öğrenmesi): Görüntüdeki hasarın kesme mi yoksa korozyon mu olduğunu tespit eder

## Kurulum

1. Gerekli paketleri yükleyin:

```bash
pip install -r requirements.txt
```

2. Makine öğrenmesi modelini eğitin:

```bash
python train_model_simple.py
```

**Not**: Python 3.13 kullanıyorsanız, TensorFlow henüz desteklenmiyor. Bu yüzden scikit-learn tabanlı `train_model_simple.py` scriptini kullanmanız önerilir.

3. API'yi çalıştırın:

```bash
python api.py
```

API varsayılan olarak `http://localhost:8000` adresinde çalışacaktır.

## API Endpointleri

### Ana Sayfa

- **URL**: `/`
- **Metod**: `GET`
- **Açıklama**: API'nin çalıştığını doğrulamak için temel bir endpoint.

### Çatlak Tespiti

- **URL**: `/tespit/catlak/`
- **Metod**: `POST`
- **Form Parametreleri**: `dosya` (resim dosyası)
- **Açıklama**: Bir görüntüdeki çatlağı tespit eder ve yatay ile arasındaki açıyı döndürür.
- **Yanıt**:
  ```json
  {
    "dosya": "örnek.jpg",
    "aciklama": "kesme hasarı",
    "sonuc_dosyasi": "/sonuclar/sonuc_123456.jpg",
    "aci_derece": 45.7
  }
  ```

### Kırmızı Nokta Tespiti

- **URL**: `/tespit/kirmizi-nokta/`
- **Metod**: `POST`
- **Form Parametreleri**: `dosya` (resim dosyası)
- **Açıklama**: Bir görüntüdeki kırmızı noktaları tespit eder ve dikey ile arasındaki açıyı döndürür.
- **Yanıt**:
  ```json
  {
    "dosya": "örnek.jpg",
    "aciklama": "kesme hasarı",
    "sonuc_dosyasi": "/sonuclar/kirmizi_nokta_sonuc_123456.jpg",
    "aci_derece": 30.5
  }
  ```

### Hasar Türü Tespiti (Makine Öğrenmesi)

- **URL**: `/tespit/hasar-turu/`
- **Metod**: `POST`
- **Form Parametreleri**: `dosya` (resim dosyası)
- **Açıklama**: Makine öğrenmesi modeli kullanarak hasar türünü tespit eder.
- **Yanıt**:
  ```json
  {
    "dosya": "örnek.jpg",
    "hasar_turu": "kesme",
    "guvence_orani": "95.2%",
    "detaylar": {
      "kesme_olasiligi": "95.2%",
      "korozyon_olasiligi": "4.8%"
    }
  }
  ```

### Model Bilgisi

- **URL**: `/model/bilgi/`
- **Metod**: `GET`
- **Açıklama**: Makine öğrenmesi modeli hakkında bilgi döndürür.
- **Yanıt**:
  ```json
  {
    "durum": "Model hazır",
    "model_bilgileri": {
      "siniflar": ["kesme", "korozyon"],
      "model_girdi_boyutu": "224x224x3",
      "model_parametreleri": 3458946
    }
  }
  ```

## Swagger ile Test Etme

API'yi Swagger UI ile test etmek için tarayıcınızda aşağıdaki adresi açın:

```
http://localhost:8000/docs
```

Bu arayüz, API endpointlerini interaktif olarak test etmenizi sağlar ve tüm API işlevlerini belgeler.

## Kullanım Örnekleri

### curl ile Çatlak Tespiti

```bash
curl -X POST "http://localhost:8000/tespit/catlak/" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "dosya=@örnek.jpg"
```

### curl ile Kırmızı Nokta Tespiti

```bash
curl -X POST "http://localhost:8000/tespit/kirmizi-nokta/" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "dosya=@örnek.jpg"
```

### curl ile Hasar Türü Tespiti

```bash
curl -X POST "http://localhost:8000/tespit/hasar-turu/" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "dosya=@örnek.jpg"
```

## Notlar

- Tespit edilen resimler `api_sonuclar` klasörüne kaydedilir.
- API, açı değerini derece cinsinden float olarak döndürür.
- Çatlak veya kırmızı nokta tespit edilemediğinde 404 hatası döndürülür.
- Makine öğrenmesi modeli eğitilmeden hasar türü tespiti yapılamaz.
- Model eğitimi sırasında `hasar_siniflandirma_modeli.h5` ve `label_encoder.pkl` dosyaları oluşturulur. 