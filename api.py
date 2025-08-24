from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import tempfile
from typing import Optional
import uuid
import subprocess
import sys

# opencv.py'deki fonksiyonları içe aktar
from opencv import tespit_et_ve_ciz, kirmizi_nokta_tespit

# Model dosyalarını kontrol et
MODEL_FILES = ['hasar_siniflandirma_modeli.pkl', 'feature_scaler.pkl', 'label_mapping.pkl']
models_exist = all(os.path.exists(f) for f in MODEL_FILES)

if not models_exist:
    print("Model dosyaları bulunamadı. Model eğitimi başlatılıyor...")
    try:
        # train_model_simple.py'yi çalıştır
        result = subprocess.run([sys.executable, 'train_model_simple.py'], 
                              capture_output=True, text=True, encoding='utf-8')
        if result.returncode == 0:
            print("Model eğitimi başarıyla tamamlandı!")
            print(result.stdout)
        else:
            print("Model eğitimi sırasında hata oluştu:")
            print(result.stderr)
    except Exception as e:
        print(f"Model eğitimi başlatılamadı: {e}")

# Model fonksiyonlarını içe aktar
try:
    # Önce basit modeli dene
    from model_simple import predict_damage_type, get_model_info
    MODEL_AVAILABLE = True
    print("Model modülü (scikit-learn) başarıyla yüklendi.")
except ImportError:
    try:
        # Eğer model_simple yoksa, eski modeli dene
        from model import predict_damage_type, get_model_info
        MODEL_AVAILABLE = True
        print("Model modülü (TensorFlow) başarıyla yüklendi.")
    except Exception as e:
        print(f"Model yüklenemedi: {e}")
        MODEL_AVAILABLE = False

app = FastAPI(
    title="Açı Tespit API",
    description="Görüntülerdeki çatlakların veya kırmızı noktaların açılarını tespit eden API",
    version="1.0.0",
)

# CORS ayarları
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Sonuç dosyaları için statik dosya sunucusu
sonuc_klasoru = "api_sonuclar"
os.makedirs(sonuc_klasoru, exist_ok=True)
app.mount("/sonuclar", StaticFiles(directory=sonuc_klasoru), name="sonuclar")

@app.get("/")
async def root():
    return {"message": "Açı Tespit API'ye Hoş Geldiniz"}

@app.post("/tespit/catlak/")
async def catlak_tespit(dosya: UploadFile = File(...)):
    """
    Bir görüntüdeki çatlağı tespit eder ve yatay ile arasındaki açıyı döndürür.
    """
    try:
        # Geçici dosya oluştur
        suffix = os.path.splitext(dosya.filename)[1] if dosya.filename else ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
            # Yüklenen dosyayı geçici dosyaya yaz
            temp.write(await dosya.read())
            temp_path = temp.name
        
        # Dosyayı işle ve sonuç klasörünü oluştur
        sonuc_klasoru = "api_sonuclar"
        os.makedirs(sonuc_klasoru, exist_ok=True)
        
        # Benzersiz dosya adı oluştur
        benzersiz_id = str(uuid.uuid4())
        sonuc_dosya_adi = f"sonuc_{benzersiz_id}{suffix}"
        sonuc_yolu = os.path.join(sonuc_klasoru, sonuc_dosya_adi)
        
        # Çatlak tespiti yap - artık fonksiyon iki değer döndürüyor
        sonuc, aci_derece = tespit_et_ve_ciz(temp_path, sonuc_yolu, gui_goster=False)
        
        # Geçici dosyayı sil
        os.unlink(temp_path)
        
        # Eğer sonuç yoksa hata döndür
        if sonuc is None:
            return JSONResponse(
                status_code=404,
                content={"mesaj": "Çatlak tespit edilemedi veya görüntü işlenemedi."}
            )
        
        # Web erişilebilir URL oluştur
        sonuc_url = f"/sonuclar/{sonuc_dosya_adi}"
        
        # Açı değerine göre hasar türünü belirle
        hasar_turu = "kesme hasarı" if 20 <= float(aci_derece) <= 55 else "burkulma hasarı"
        
        # Sonuçları döndür
        return {
            "dosya": dosya.filename,
            "aciklama": hasar_turu,
            "sonuc_dosyasi": sonuc_url,
            "aci_derece": float(aci_derece)  # Float olarak döndür
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hata oluştu: {str(e)}")

@app.post("/tespit/kirmizi-nokta/")
async def kirmizi_nokta_tespit_api(dosya: UploadFile = File(...)):
    """
    Bir görüntüdeki kırmızı noktaları tespit eder ve dikey ile arasındaki açıyı döndürür.
    """
    try:
        # Geçici dosya oluştur
        suffix = os.path.splitext(dosya.filename)[1] if dosya.filename else ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
            # Yüklenen dosyayı geçici dosyaya yaz
            temp.write(await dosya.read())
            temp_path = temp.name
        
        # Dosyayı işle ve sonuç klasörünü oluştur
        sonuc_klasoru = "api_sonuclar"
        os.makedirs(sonuc_klasoru, exist_ok=True)
        
        # Benzersiz dosya adı oluştur
        benzersiz_id = str(uuid.uuid4())
        sonuc_dosya_adi = f"kirmizi_nokta_sonuc_{benzersiz_id}{suffix}"
        sonuc_yolu = os.path.join(sonuc_klasoru, sonuc_dosya_adi)
        
        # Kırmızı nokta tespiti yap - artık fonksiyon iki değer döndürüyor
        sonuc, aci_derece = kirmizi_nokta_tespit(temp_path, sonuc_yolu, gui_goster=False)
        
        # Geçici dosyayı sil
        os.unlink(temp_path)
        
        # Eğer sonuç yoksa hata döndür
        if sonuc is None:
            return JSONResponse(
                status_code=404,
                content={"mesaj": "Kırmızı noktalar tespit edilemedi veya görüntü işlenemedi."}
            )
        
        # Web erişilebilir URL oluştur
        sonuc_url = f"/sonuclar/{sonuc_dosya_adi}"
        
        # Açı değerine göre hasar türünü belirle
        hasar_turu = "kesme hasarı" if 20 <= float(aci_derece) <= 55 else "burkulma hasarı"
        
        # Sonuçları döndür
        return {
            "dosya": dosya.filename,
            "aciklama": hasar_turu,
            "sonuc_dosyasi": sonuc_url,
            "aci_derece": float(aci_derece)  # Float olarak döndür
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hata oluştu: {str(e)}")

@app.post("/tespit/hasar-turu/")
async def hasar_turu_tespit(dosya: UploadFile = File(...)):
    """
    Makine öğrenmesi modeli kullanarak hasar türünü tespit eder (kesme veya korozyon).
    """
    if not MODEL_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Makine öğrenmesi modeli henüz yüklenmedi. Lütfen önce modeli eğitin."
        )
    
    try:
        # Geçici dosya oluştur
        suffix = os.path.splitext(dosya.filename)[1] if dosya.filename else ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
            # Yüklenen dosyayı geçici dosyaya yaz
            temp.write(await dosya.read())
            temp_path = temp.name
        
        # Makine öğrenmesi ile tahmin yap
        tahmin_sonucu = predict_damage_type(temp_path)
        
        # Geçici dosyayı sil
        os.unlink(temp_path)
        
        # Sonuçları hazırla
        response_data = {
            "dosya": dosya.filename,
            "hasar_turu": tahmin_sonucu['hasar_turu'],
            "guvence_orani": f"{tahmin_sonucu['en_yuksek_olasilik'] * 100:.1f}%",
            "detaylar": {
                "kesme_olasiligi": f"{tahmin_sonucu['olasiliklar'].get('kesme', 0) * 100:.1f}%",
                "korozyon_olasiligi": f"{tahmin_sonucu['olasiliklar'].get('korozyon', 0) * 100:.1f}%"
            }
        }
        
        # Alakasız görsel uyarısı varsa ekle
        if tahmin_sonucu.get('alakasiz_gorsel_uyarisi', False):
            response_data['uyari'] = tahmin_sonucu.get('uyari', 'Bu görsel kesme veya korozyon hasarı içermiyor olabilir.')
            response_data['alakasiz_gorsel'] = True
        
        return response_data
        
    except Exception as e:
        # Geçici dosyayı temizle
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)
        raise HTTPException(status_code=500, detail=f"Hata oluştu: {str(e)}")

@app.get("/model/bilgi/")
async def model_bilgisi():
    """
    Makine öğrenmesi modeli hakkında bilgi döndürür.
    """
    if not MODEL_AVAILABLE:
        return {
            "durum": "Model yüklenmedi",
            "mesaj": "Makine öğrenmesi modeli henüz eğitilmemiş. 'python train_model.py' komutu ile modeli eğitin."
        }
    
    try:
        bilgi = get_model_info()
        return {
            "durum": "Model hazır",
            "model_bilgileri": bilgi
        }
    except Exception as e:
        return {
            "durum": "Hata",
            "mesaj": str(e)
        }

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True) 