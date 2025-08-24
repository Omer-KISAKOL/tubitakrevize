import cv2
import numpy as np
import argparse
import os
import pathlib 
import math

def tespit_et_ve_ciz(resim_yolu, sonuc_yolu=None, gui_goster=True):
    # Resmi oku
    resim = cv2.imread(resim_yolu)
    if resim is None:
        print(f"Hata: {resim_yolu} yolu üzerindeki resim okunamadı.")
        return None, None

    # Dosya adını al (uzantısız)
    dosya_adi = os.path.splitext(os.path.basename(resim_yolu))[0]
    
    # Varsayılan sonuç yolu oluştur
    if sonuc_yolu is None:
        # images klasörünü kontrol et ve yoksa oluştur
        images_klasoru = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
        pathlib.Path(images_klasoru).mkdir(parents=True, exist_ok=True)
        
        # Sonuç dosya yollarını belirle
        sonuc_yolu = os.path.join(images_klasoru, f"sonuc_{dosya_adi}.jpg")
        catlak_tespiti_yolu = os.path.join(images_klasoru, f"catlak_tespiti_{dosya_adi}.jpg")
    else:
        # Kullanıcı özel bir yol belirttiyse, çatlak tespiti için de benzer bir yol oluştur
        sonuc_dizini = os.path.dirname(sonuc_yolu)
        sonuc_dosya_adi = os.path.splitext(os.path.basename(sonuc_yolu))[0]
        catlak_tespiti_yolu = os.path.join(sonuc_dizini, f"catlak_tespiti_{sonuc_dosya_adi}.jpg")

    # Orijinal resmin kopyasını oluştur
    sonuc = resim.copy()
    
    # Resmi gri tonlamaya çevir
    gri = cv2.cvtColor(resim, cv2.COLOR_BGR2GRAY)
    
    # Gürültüyü azaltmak için Gaussian bulanıklaştırma uygula
    gri = cv2.GaussianBlur(gri, (5, 5), 0)
    
    # Adaptif eşikleme uygula - kolon üzerindeki çatlaklar genelde daha koyu
    th = cv2.adaptiveThreshold(gri, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                               cv2.THRESH_BINARY_INV, 11, 2)
    
    # Morfolojik işlemler
    kernel = np.ones((3, 3), np.uint8)
    morfolojik = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
    morfolojik = cv2.morphologyEx(morfolojik, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Canny kenar tespiti (ek filtre olarak)
    kenarlar = cv2.Canny(gri, 50, 150)
    
    # İki yöntemi birleştir
    kombinasyon = cv2.bitwise_or(morfolojik, kenarlar)
    
    # Gürültüyü temizle
    temizlenmis = cv2.morphologyEx(kombinasyon, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # HoughLinesP kullanarak doğru parçalarını tespit et
    # Sadece ikinci görüntü üzerinde çalış (beyaz olan - eşiklenmiş görüntü)
    lines = cv2.HoughLinesP(
        th,  # th görüntüsünü kullan (beyaz olan çatlak)
        rho=1,
        theta=np.pi/180,
        threshold=50,
        minLineLength=50,  # Minimum çizgi uzunluğu
        maxLineGap=10      # İki çizgi arasında izin verilen maksimum boşluk
    )
    
    # Eğer hiç çizgi tespit edilemezse
    if lines is None or len(lines) == 0:
        print("Çatlak tespit edilemedi.")
        return None, None
    
    # Çizgileri en uzundan en kısaya doğru sırala
    line_lengths = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = float(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))  # float'a dönüştür
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        # -180 ile 180 arasındaki açıyı 0-180 arasına normalize et
        if angle < 0:
            angle += 180
        line_lengths.append((length, line[0], angle))
    
    # En uzun 10 çizgiyi seç - key kullanarak sırala
    line_lengths.sort(key=lambda x: x[0], reverse=True)
    top_lines = line_lengths[:10]
    
    # Debug görüntüsü için
    line_image = np.zeros_like(resim)
    
    # Tespit edilen tüm çizgileri göster (debug amaçlı)
    for _, line, _ in top_lines:
        x1, y1, x2, y2 = line
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # En uzun çizgiyi al
    longest_length, longest_line, longest_angle = top_lines[0]
    x1, y1, x2, y2 = longest_line
    
    # Çizgi merkezini bul
    merkez_x = (x1 + x2) // 2
    merkez_y = (y1 + y2) // 2
    merkez = (merkez_x, merkez_y)
    
    # Çatlak doğrultusunu uzat
    uzunluk = 500
    aci_radyan = math.atan2(y2 - y1, x2 - x1)
    dx = math.cos(aci_radyan)
    dy = math.sin(aci_radyan)
    
    nokta1 = (int(merkez_x - uzunluk * dx), int(merkez_y - uzunluk * dy))
    nokta2 = (int(merkez_x + uzunluk * dx), int(merkez_y + uzunluk * dy))
    
    # Çatlak doğrultusuna göre çizgi çiz (mavi)
    cv2.line(sonuc, nokta1, nokta2, (255, 0, 0), 2)
    
    # Yatay çizgi çiz (kırmızı)
    yatay_nokta1 = (merkez_x - uzunluk, merkez_y)
    yatay_nokta2 = (merkez_x + uzunluk, merkez_y)
    cv2.line(sonuc, yatay_nokta1, yatay_nokta2, (0, 0, 255), 2)
    
    # Çatlak merkez noktasını görselleştir
    cv2.circle(sonuc, merkez, 5, (0, 255, 0), -1)
    
    # Çatlağın yönü ile yatay arasındaki açıyı hesapla
    aci_derece = abs(longest_angle)
    if aci_derece > 90:
        aci_derece = 180 - aci_derece
    
    # Açıyı görselleştir - yazı olarak
    cv2.putText(sonuc, f"Aci: {aci_derece:.1f}°", 
                (merkez_x + 60, merkez_y - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    
    # Açıyı yay şeklinde görselleştir
    # Çizgiler arasındaki açıyı hesapla (çizim için)
    yay_radiusu = 50  # Yayın yarıçapı
    
    # Çatlak açısının yönü
    catlak_aci = math.degrees(aci_radyan)
    # Normalizasyon (0-360 aralığına getir)
    while catlak_aci < 0:
        catlak_aci += 360
    while catlak_aci >= 360:
        catlak_aci -= 360
    
    # Yatay çizgi açısı (0 derece)
    yatay_aci = 0
    
    # Başlangıç ve bitiş açılarını belirle
    baslangic_aci = min(yatay_aci, catlak_aci)
    bitis_aci = max(yatay_aci, catlak_aci)
    
    # Eğer açı farkı 180'den büyükse, kısa yolu seç
    if bitis_aci - baslangic_aci > 180:
        baslangic_aci, bitis_aci = bitis_aci, baslangic_aci + 360
    
    # Açı yayını çiz (sarı renk)
    cv2.ellipse(sonuc, merkez, (yay_radiusu, yay_radiusu), 
                0, baslangic_aci, bitis_aci, (0, 255, 255), 2)
    
    # Açının köşelerine çizgiler çiz
    ucgen_uzunluk = 40
    # Yatay çizgi için açı işaretçisi
    yatay_isaretci_x = int(merkez_x + ucgen_uzunluk * math.cos(math.radians(yatay_aci)))
    yatay_isaretci_y = int(merkez_y - ucgen_uzunluk * math.sin(math.radians(yatay_aci)))
    cv2.line(sonuc, merkez, (yatay_isaretci_x, yatay_isaretci_y), (0, 255, 255), 2)
    
    # Çatlak çizgisi için açı işaretçisi
    catlak_isaretci_x = int(merkez_x + ucgen_uzunluk * math.cos(math.radians(catlak_aci)))
    catlak_isaretci_y = int(merkez_y - ucgen_uzunluk * math.sin(math.radians(catlak_aci)))
    cv2.line(sonuc, merkez, (catlak_isaretci_x, catlak_isaretci_y), (0, 255, 255), 2)
    
    # Açı değerini yayın ortasına yaz
    orta_aci = (baslangic_aci + bitis_aci) / 2
    aci_yazisi_x = int(merkez_x + (yay_radiusu + 15) * math.cos(math.radians(orta_aci)))
    aci_yazisi_y = int(merkez_y - (yay_radiusu + 15) * math.sin(math.radians(orta_aci)))
    cv2.putText(sonuc, f"{aci_derece:.1f}°", 
                (aci_yazisi_x - 20, aci_yazisi_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Görüntüyü çizdirme kısmı için gelişmiş görselleştirme - tüm tespit aşamaları
    debug_goruntusu = np.zeros_like(sonuc)
    
    # Tespit edilen konturları görselleştir
    cizgi_goruntusu = cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY)
    
    # Debug görüntüsünü oluştur
    debug_satiri_1 = np.hstack([gri, th, kombinasyon])
    debug_satiri_2 = np.hstack([temizlenmis, cizgi_goruntusu, cv2.cvtColor(sonuc, cv2.COLOR_BGR2GRAY)])
    
    if debug_satiri_1.shape[1] == debug_satiri_2.shape[1]:
        debug_goruntusu = np.vstack([debug_satiri_1, debug_satiri_2])
    
    # GUI penceresini yalnızca gui_goster=True ise göster
    if gui_goster:
        # Sonuçları göster
        cv2.imshow("Orijinal Görüntü", resim)
        cv2.imshow("Çatlak Tespiti Aşamaları", debug_goruntusu)
        cv2.imshow("Sonuç", sonuc)
        
        # En uzun çizgiyle oluşturulan özel görüntü
        uzun_cizgi_goruntusu = np.zeros_like(resim)
        cv2.line(uzun_cizgi_goruntusu, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.imshow("En Uzun Çatlak Çizgisi", uzun_cizgi_goruntusu)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        # API için yalnızca bir uzun çizgi görüntüsü oluştur (gösterilmeyecek)
        uzun_cizgi_goruntusu = np.zeros_like(resim)
        cv2.line(uzun_cizgi_goruntusu, (x1, y1), (x2, y2), (255, 255, 255), 2)
    
    # Sonuç ve çatlak tespiti resimlerini kaydet
    cv2.imwrite(sonuc_yolu, sonuc)
    cv2.imwrite(catlak_tespiti_yolu, debug_goruntusu)
    
    # Ayrıca en uzun çatlak çizgisini de kaydet
    uzun_cizgi_yolu = os.path.join(os.path.dirname(sonuc_yolu), f"uzun_cizgi_{dosya_adi}.jpg")
    cv2.imwrite(uzun_cizgi_yolu, uzun_cizgi_goruntusu)
    
    print(f"Sonuç resmi {sonuc_yolu} yoluna kaydedildi.")
    print(f"Çatlak tespiti resmi {catlak_tespiti_yolu} yoluna kaydedildi.")
    print(f"En uzun çatlak çizgisi {uzun_cizgi_yolu} yoluna kaydedildi.")
    print(f"Çatlak ile yatay arasındaki açı: {aci_derece:.1f} derece")
    
    return sonuc, aci_derece

def kirmizi_nokta_tespit(resim_yolu, sonuc_yolu=None, gui_goster=True):
    # Resmi oku
    resim = cv2.imread(resim_yolu)
    if resim is None:
        print(f"Hata: {resim_yolu} yolu üzerindeki resim okunamadı.")
        return None, None

    # Dosya adını al (uzantısız)
    dosya_adi = os.path.splitext(os.path.basename(resim_yolu))[0]
    
    # Varsayılan sonuç yolu oluştur
    if sonuc_yolu is None:
        # images klasörünü kontrol et ve yoksa oluştur
        images_klasoru = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
        pathlib.Path(images_klasoru).mkdir(parents=True, exist_ok=True)
        
        # Sonuç dosya yolunu belirle
        sonuc_yolu = os.path.join(images_klasoru, f"kirmizi_nokta_sonuc_{dosya_adi}.jpg")
    
    # Orijinal resmin kopyasını oluştur
    sonuc = resim.copy()
    
    # Kırmızı rengi tespit etmek için HSV uzayına dönüştür
    hsv = cv2.cvtColor(resim, cv2.COLOR_BGR2HSV)
    
    # Kırmızı renk için eşik değerleri (HSV'de kırmızı iki aralıkta bulunur)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    
    # Kırmızı rengi mask'le
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    
    # Gürültüyü azaltmak için morfolojik işlemler
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Kırmızı bölgelerin merkezlerini bul
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Eğer hiç kontur bulunamazsa
    if len(contours) < 2:
        print("En az iki kırmızı nokta tespit edilemedi.")
        return None, None
    
    # Konturların merkezlerini ve alanlarını hesapla
    kirmizi_noktalar = []
    for c in contours:
        # Çok küçük konturları filtrele
        if cv2.contourArea(c) < 50:  # Minimum alan değeri (ayarlanabilir)
            continue
        
        # Konturun merkezini hesapla
        M = cv2.moments(c)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            kirmizi_noktalar.append((cx, cy, cv2.contourArea(c)))
    
    # Eğer 2'den az kırmızı nokta bulunduysa
    if len(kirmizi_noktalar) < 2:
        print("En az iki kırmızı nokta tespit edilemedi.")
        return None, None
    
    # Konturları alan büyüklüğüne göre sırala
    kirmizi_noktalar.sort(key=lambda x: x[2], reverse=True)
    
    # En büyük iki kırmızı noktayı seç
    nokta1 = (kirmizi_noktalar[0][0], kirmizi_noktalar[0][1])
    nokta2 = (kirmizi_noktalar[1][0], kirmizi_noktalar[1][1])
    
    # Noktaları görselleştir
    cv2.circle(sonuc, nokta1, 5, (0, 255, 0), -1)  # Yeşil
    cv2.circle(sonuc, nokta2, 5, (0, 255, 0), -1)  # Yeşil
    
    # İki nokta arasında çizgi çiz
    cv2.line(sonuc, nokta1, nokta2, (255, 0, 0), 2)  # Mavi
    
    # Dikey çizgiyi çiz (merkez noktasından geçen)
    merkez_x = (nokta1[0] + nokta2[0]) // 2
    merkez_y = (nokta1[1] + nokta2[1]) // 2
    merkez = (merkez_x, merkez_y)
    
    uzunluk = 500
    dikey_nokta1 = (merkez_x, merkez_y - uzunluk)
    dikey_nokta2 = (merkez_x, merkez_y + uzunluk)
    cv2.line(sonuc, dikey_nokta1, dikey_nokta2, (0, 0, 255), 2)  # Kırmızı
    
    # Çizginin dikey ile yaptığı açıyı hesapla
    dx = nokta2[0] - nokta1[0]
    dy = nokta2[1] - nokta1[1]
    
    # Çizginin eğimini hesapla (dikey açı)
    aci_radyan = math.atan2(dx, dy)  # Dikey için x ve y'nin yerini değiştiriyoruz
    aci_derece = math.degrees(aci_radyan)
    
    # Açıyı 0-90 arasında normalize et
    aci_derece = abs(aci_derece)
    if aci_derece > 90:
        aci_derece = 180 - aci_derece
    
    # Açıyı görselleştir
    cv2.putText(sonuc, f"Aci: {aci_derece:.1f}°", 
                (merkez_x + 60, merkez_y - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    
    # Açıyı yay şeklinde görselleştir
    yay_radiusu = 50
    
    # Dikey ve çizgi açılarını hesapla
    dikey_aci = 90  # 90 derece yukarı
    cizgi_aci = math.degrees(math.atan2(nokta2[1] - nokta1[1], nokta2[0] - nokta1[0]))
    # Normalizasyon
    while cizgi_aci < 0:
        cizgi_aci += 360
    while cizgi_aci >= 360:
        cizgi_aci -= 360
    
    # Başlangıç ve bitiş açılarını belirle
    baslangic_aci = min(dikey_aci, cizgi_aci)
    bitis_aci = max(dikey_aci, cizgi_aci)
    
    # Eğer açı farkı 180'den büyükse, kısa yolu seç
    if bitis_aci - baslangic_aci > 180:
        baslangic_aci, bitis_aci = bitis_aci, baslangic_aci + 360
    
    # Açı yayını çiz (sarı renk)
    cv2.ellipse(sonuc, merkez, (yay_radiusu, yay_radiusu), 
                0, baslangic_aci, bitis_aci, (0, 255, 255), 2)
    
    # Açının köşelerine çizgiler çiz
    ucgen_uzunluk = 40
    # Dikey çizgi için açı işaretçisi
    dikey_isaretci_x = int(merkez_x + ucgen_uzunluk * math.cos(math.radians(dikey_aci)))
    dikey_isaretci_y = int(merkez_y - ucgen_uzunluk * math.sin(math.radians(dikey_aci)))
    cv2.line(sonuc, merkez, (dikey_isaretci_x, dikey_isaretci_y), (0, 255, 255), 2)
    
    # Kırmızı noktalar çizgisi için açı işaretçisi
    cizgi_isaretci_x = int(merkez_x + ucgen_uzunluk * math.cos(math.radians(cizgi_aci)))
    cizgi_isaretci_y = int(merkez_y - ucgen_uzunluk * math.sin(math.radians(cizgi_aci)))
    cv2.line(sonuc, merkez, (cizgi_isaretci_x, cizgi_isaretci_y), (0, 255, 255), 2)
    
    # Açı değerini yayın ortasına yaz
    orta_aci = (baslangic_aci + bitis_aci) / 2
    aci_yazisi_x = int(merkez_x + (yay_radiusu + 15) * math.cos(math.radians(orta_aci)))
    aci_yazisi_y = int(merkez_y - (yay_radiusu + 15) * math.sin(math.radians(orta_aci)))
    cv2.putText(sonuc, f"{aci_derece:.1f}°", 
                (aci_yazisi_x - 20, aci_yazisi_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # GUI penceresini yalnızca gui_goster=True ise göster
    if gui_goster:
        # Sonuçları göster
        cv2.imshow("Orijinal Görüntü", resim)
        cv2.imshow("Kırmızı Noktalar Tespiti", mask)
        cv2.imshow("Sonuç", sonuc)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # Sonuç resmini kaydet
    cv2.imwrite(sonuc_yolu, sonuc)
    
    print(f"Sonuç resmi {sonuc_yolu} yoluna kaydedildi.")
    print(f"İki kırmızı nokta arasındaki çizginin dikey ile açısı: {aci_derece:.1f} derece")
    
    return sonuc, aci_derece

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Kolon görselinde işlem yapma')
    parser.add_argument('resim_yolu', help='İşlenecek resim dosyasının yolu')
    parser.add_argument('--sonuc_yolu', help='Sonuç resminin kaydedileceği yol (opsiyonel)', default=None)
    parser.add_argument('--mod', help='İşlem modu: "catlak" veya "kirmizi_nokta"', default="catlak")
    parser.add_argument('--gui_goster', help='GUI penceresini göster (True/False)', type=bool, default=True)
    
    args = parser.parse_args()
    
    if args.mod == "catlak":
        tespit_et_ve_ciz(args.resim_yolu, args.sonuc_yolu, args.gui_goster)
    elif args.mod == "kirmizi_nokta":
        kirmizi_nokta_tespit(args.resim_yolu, args.sonuc_yolu, args.gui_goster)
    else:
        print(f"Geçersiz mod: {args.mod}. 'catlak' veya 'kirmizi_nokta' kullanın.")



