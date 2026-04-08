# 🏥 Akıllı Sigorta Poliçesi ve Risk Analiz Sistemi 

Bu proje, modern makine öğrenmesi algoritmalarını kullanarak sigorta sektöründeki risk analizi ve fiyatlandırma süreçlerini otomatize eden akademik tabanlı bir yazılım çözümüdür. Sistem, müşteri verilerini analiz ederek hem **Sınıflandırma** (Risk Grubu) hem de **Regresyon** (Poliçe Ücreti) tahmini yapar.

---

## 📂 Proje Yapısı
Proje iki temel bileşenden oluşmaktadır:
1. **main.py**: Veri işleme, model eğitimi, sürekli öğrenme döngüsü ve görselleştirme modüllerini içeren ana uygulama dosyası.
2. **sigorta_verisi.csv**: Sistemin tüm geçmiş ve yeni verilerini sakladığı, modelin beslendiği ana veri tabanı dosyası.

---

## 🚀 Öne Çıkan Özellikler

### 1. Sürekli ve Dinamik Öğrenme (Online Learning)
Sistem statik bir yapıda değildir. Kullanıcı tarafından girilen her yeni müşteri verisi anlık olarak `sigorta_verisi.csv` dosyasına kaydedilir. Algoritmalar, her yeni analiz öncesinde güncellenmiş bu veri kümesiyle tekrar eğitilerek zamanla daha "akıllı" hale gelir.

### 2. Hibrid Makine Öğrenmesi Yaklaşımı
Proje, ders notlarındaki temel kavramları üç farklı koldan uygular:
* **Sınıflandırma:** Müşterileri Düşük, Orta veya Yüksek risk gruplarına ayırır.
* **Regresyon:** Değişkenler arası doğrusal ilişkiyi kurarak net poliçe ücretini hesaplar.
* **Olasılıksal Tahmin:** Lojistik regresyon ile riskin gerçekleşme ihtimalini hesaplar.

---

## 📊 Veri Seti Metrikleri ve Anlamları

| Metrik | Açıklama | Teknik Rol |
| :--- | :--- | :--- |
| **KAYIT_TARIHI** | Verinin sisteme giriş zamanı. | Zaman Serisi Takibi |
| **YAS** | Müşterinin kronolojik yaşı. | Bağımsız Değişken (Feature) |
| **VKE** | Vücut Kitle Endeksi ($kg/m^2$). | Sağlık Göstergesi / Feature |
| **SIGARA** | Tütün kullanımı (0: Hayır, 1: Evet). | Kategorik (Binary) Değişken |
| **HASTALIK_SAYISI**| Mevcut kronik hastalık adedi (0-3). | Kesikli (Discrete) Değişken |
| **GELIR** | Yıllık toplam gelir. | Ekonomik Feature |
| **RISK_GRUBU** | 0: Düşük, 1: Orta, 2: Yüksek. | **Hedef (Classification Label)** |
| **POLICE_UCRETI** | Tahmin edilen yıllık prim bedeli. | **Hedef (Regression Value)** |

---

## 🧠 Uygulanan Algoritmalar ve Teknik Detaylar

### 🔹 Karar Ağaçları (Decision Trees)
* **Kriter:** Gini Impurity (Gini Kirliliği).
* **Fonksiyon:** Veriyi en saf alt kümelere ayırana kadar özellikler üzerinden hiyerarşik kurallar (If-Then) oluşturur.
* **Avantajı:** İnsan karar mekanizmasına en yakın ve en yorumlanabilir modeldir.

### 🔹 Destek Vektör Makineleri (SVM)
* **Kernel:** Linear (Doğrusal).
* **Fonksiyon:** Sınıflar arasındaki sınırı (Margin) en geniş tutacak şekilde optimal bir "Hiper-Düzlem" çizer.
* **Avantajı:** Yüksek boyutlu verilerde ve sınıfların net ayrıldığı durumlarda çok güçlüdür.

### 🔹 Lojistik Regresyon
* **Aktivasyon:** Sigmoid Fonksiyonu.
* **Fonksiyon:** Girdileri 0 ile 1 arasında bir olasılık değerine dönüştürür.
* **Avantajı:** Sadece sınıf tahmini değil, o sınıfın olma olasılığını da sunar.

### 🔹 Lineer Regresyon
* **Başarı Metrikleri:** $R^2$ (Belirlilik Katsayısı) ve MSE (Ortalama Kare Hata).
* **Fonksiyon:** $Y = wX + b$ denklemi üzerinden poliçe ücretini sürekli bir değer olarak tahmin eder.

---

## 📈 Görselleştirme ve Analiz Standartları
Sistem, sunum ve analiz için dört ana grafik türü üretir:
1.  **Poliçe Ücret Dağılımı:** Verideki ücretlerin yoğunluğunu ve ortalama (Mean) çizgisini gösterir.
2.  **Korelasyon Isı Haritası (Heatmap):** Değişkenlerin (Örn: Yaş ve Risk) birbirlerini ne kadar güçlü etkilediğini gösterir.
3.  **Hata (Residual) Analizi:** Regresyon modelinin tahmin hatalarının dağılımını ölçer. 0 etrafındaki rastgele dağılım, modelin başarılı olduğunu kanıtlar.
4.  **Model Kıyaslama:** Algoritmaların (DT, SVM, LR) başarı oranlarını (Accuracy) yan yana bar grafiği ile kıyaslar.

---

## 🛠 Kullanım Talimatı

1.  **Gereksinimler:** Python 3.x, NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn.
2.  **Çalıştırma:** Terminal üzerinden `python main.py` komutu ile başlatılır.
3.  **Modüller:** * `Seçenek 1`: Yeni müşteri verisi girişi yapılır. Veri anlık kaydedilir ve tahmin üretilir.
    * `Seçenek 2`: Modellerin başarıları teknik açıklamalarla kıyaslanır.
    * `Seçenek 3`: Veri setinin genel istatistikleri ve grafiksel dökümü alınır.

---
