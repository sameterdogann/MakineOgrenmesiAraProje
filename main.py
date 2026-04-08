import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, confusion_matrix

# Grafiklerin düzgün görünmesi için stil ayarı
sns.set_theme(style="whitegrid")

def veri_yukle_ve_hazirla():
    """
    Veriyi diskten okur. Eğer dosya yoksa hata verir. 
    Sürekli öğrenme için en güncel veriyi çeker.
    """
    dosya_yolu = 'sigorta_verisi.csv'
    if not os.path.exists(dosya_yolu):
        print(f"Hata: {dosya_yolu} bulunamadı!")
        return None
    
    df = pd.read_csv(dosya_yolu)
    # Eğitimde kullanılmayacak sütunları filtrele (Cleaning)
    if 'KAYIT_TARIHI' in df.columns:
        df_analiz = df.drop(columns=['KAYIT_TARIHI'])
    else:
        df_analiz = df
    return df_analiz

def modelleri_egit(df):
    """
    Ders notlarındaki algoritmaları (SVM, DT, LR, Regresyon) güncel veriyle eğitir.
    Sürekli öğrenme mekanizmasının kalbidir.
    """
    # Özellikler ve Hedefler
    X = df.drop(columns=['RISK_GRUBU', 'POLICE_UCRETI'])
    y_sinif = df['RISK_GRUBU']
    y_reg = df['POLICE_UCRETI']

    # Veriyi Eğitim ve Test seti olarak ayırıyoruz (Öğrenmeyi doğrulamak için)
    X_train, X_test, y_train_s, y_test_s = train_test_split(X, y_sinif, test_size=0.2, random_state=42)
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.2, random_state=42)

    # Ölçeklendirme (Özellikle SVM ve Mesafe bazlı ölçümler için standartlaştırma)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 1. Karar Ağacı (Decision Tree) - Gini kriterine göre dallanma
    dt = DecisionTreeClassifier(criterion='gini', max_depth=5).fit(X_train, y_train_s)
    
    # 2. SVM (Support Vector Machines) - Hiper düzlem ile sınıfları ayırma
    svm = SVC(kernel='linear', probability=True).fit(X_train_scaled, y_train_s)
    
    # 3. Lojistik Regresyon - Sigmoid fonksiyonu ile olasılık hesaplama
    lr = LogisticRegression(max_iter=1000).fit(X_train_scaled, y_train_s)
    
    # 4. Lineer Regresyon - Poliçe ücreti tahmini (Continuous value)
    reg_model = LinearRegression().fit(X_train_r, y_train_r)

    return {
        "dt": dt, "svm": svm, "lr": lr, "reg": reg_model, "scaler": scaler,
        "X_test": X_test, "X_test_scaled": X_test_scaled, "y_test_s": y_test_s,
        "X_test_r": X_test_r, "y_test_r": y_test_r, "full_df": df
    }

def yeni_veri_ekle_ve_tahmin(modeller):
    """Kullanıcıdan veri alır, CSV'ye ekler ve anlık sonuç üretir."""
    print("\n[YENİ VERİ GİRİŞİ VE ANALİZ]")
    try:
        yas = int(input("Yaş: "))
        vke = float(input("VKE (Vücut Kitle Endeksi): "))
        sigara = int(input("Sigara (0: Hayır, 1: Evet): "))
        hastalik = int(input("Kronik Hastalık Sayısı (0-10): "))
        gelir = int(input("Yıllık Gelir: "))

        # 1. Tahmin İşlemi
        input_data = np.array([[yas, vke, sigara, hastalik, gelir]])
        scaled_input = modeller["scaler"].transform(input_data)
        
        risk_tahmin = modeller["dt"].predict(input_data)[0]
        ucret_tahmin = modeller["reg"].predict(input_data)[0]

        # 2. Sürekli Öğrenme: Veriyi CSV'ye Kaydetme
        yeni_satir = pd.DataFrame([[
            pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'), 
            yas, vke, sigara, hastalik, gelir, risk_tahmin, int(ucret_tahmin)
        ]], columns=['KAYIT_TARIHI', 'YAS', 'VKE', 'SIGARA', 'HASTALIK_SAYISI', 'GELIR', 'RISK_GRUBU', 'POLICE_UCRETI'])
        
        yeni_satir.to_csv('sigorta_verisi.csv', mode='a', header=False, index=False)
        print("\n(+) Veri sisteme eklendi ve öğrenme süreklileştirildi.")

        risk_etiket = {0: "Düşük", 1: "Orta", 2: "Yüksek"}
        print(f"SONUÇ: Risk Grubu: {risk_etiket[risk_tahmin]} | Önerilen Poliçe: {ucret_tahmin:.2f} TL")

    except Exception as e:
        print(f"Hata oluştu: {e}. Lütfen sayısal değerler girin.")

def model_kiyaslama_detayli(modeller):
    """Modellerin teknik açıklamalarını basar ve başarılarını kıyaslar."""
    print("\n" + "="*60)
    print(" MODEL PERFORMANS VE TEKNİK ANALİZ ")
    print("="*60)
    
    # Başarı skorları
    s1 = accuracy_score(modeller["y_test_s"], modeller["dt"].predict(modeller["X_test"]))
    s2 = accuracy_score(modeller["y_test_s"], modeller["svm"].predict(modeller["X_test_scaled"]))
    s3 = accuracy_score(modeller["y_test_s"], modeller["lr"].predict(modeller["X_test_scaled"]))

    print(f"1. Karar Ağacı (Gini): %{s1*100:.2f}")
    print("   - Bilgi kazancını maksimize eder, kurallar hiyerarşisi oluşturur.")
    print(f"2. SVM (Linear): %{s2*100:.2f}")
    print("   - Sınıflar arasında maksimum marjlı bir hiper-düzlem çizer.")
    print(f"3. Lojistik Regresyon: %{s3*100:.2f}")
    print("   - Verinin belirli bir sınıfa ait olma olasılığını Sigmoid ile hesaplar.")

    # Grafiksel Kıyaslama
    plt.figure(figsize=(10, 5))
    modeller_ad = ['Karar Ağacı', 'SVM', 'Lojistik Reg.']
    skorlar = [s1, s2, s3]
    sns.barplot(x=modeller_ad, y=skorlar, palette="magma")
    plt.axhline(np.mean(skorlar), color='red', linestyle='--', label=f'Ortalama Başarı: {np.mean(skorlar):.2f}')
    plt.title('Modellerin Sınıflandırma Başarısı (Accuracy)')
    plt.legend()
    plt.show()

def genel_istatistik_ve_grafikler(modeller):
    """Kapsamlı görselleştirme sekmesi."""
    df = modeller["full_df"]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Ücret Dağılımı ve Ortalama
    sns.histplot(df['POLICE_UCRETI'], kde=True, ax=axes[0,0], color='blue')
    axes[0,0].axvline(df['POLICE_UCRETI'].mean(), color='red', label='Ortalama Ücret')
    axes[0,0].set_title('Poliçe Ücret Dağılımı')
    axes[0,0].legend()

    # 2. Risk Grubu vs Gelir (Box Plot)
    sns.boxplot(x='RISK_GRUBU', y='GELIR', data=df, ax=axes[0,1])
    axes[0,1].set_title('Risk Gruplarına Göre Gelir Standartları')

    # 3. Korelasyon Isı Haritası
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=axes[1,0])
    axes[1,0].set_title('Metriklerin Birbirine Etkisi (Korelasyon)')

    # 4. Hata Analizi (Regresyon)
    y_pred_r = modeller["reg"].predict(modeller["X_test_r"])
    residuals = modeller["y_test_r"] - y_pred_r
    sns.scatterplot(x=y_pred_r, y=residuals, ax=axes[1,1], color='purple')
    axes[1,1].axhline(0, color='black', linestyle='--')
    axes[1,1].set_title('Regresyon Hata (Residual) Analizi')

    plt.tight_layout()
    plt.show()

def main():
    while True:
        df = veri_yukle_ve_hazirla()
        if df is None: break
        
        # Her döngü başında modelleri en güncel veriyle tekrar eğitiyoruz (Sürekli Öğrenme)
        modeller = modelleri_egit(df)

        print("\n--- SİGORTA ANALİZ VE TAHMİN SİSTEMİ (v2.0) ---")
        print("1. Yeni Veri Girişi ve Anlık Tahmin (Kayıt + Öğrenme)")
        print("2. Model Başarı Kıyaslaması ve Teknik Açıklamalar")
        print("3. Genel İstatistikler ve Kapsamlı Grafikler")
        print("4. Çıkış")
        
        secim = input("\nİşlem Seçiniz: ")

        if secim == '1':
            yeni_veri_ekle_ve_tahmin(modeller)
        elif secim == '2':
            model_kiyaslama_detayli(modeller)
        elif secim == '3':
            genel_istatistik_ve_grafikler(modeller)
        elif secim == '4':
            print("Sistem kapatıldı.")
            break
        else:
            print("Geçersiz seçim!")

if __name__ == "__main__":
    main()