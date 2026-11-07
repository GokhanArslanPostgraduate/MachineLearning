# Gen İfadesi Mikrodizi Verilerinden Lösemi Alt Tipi Sınıflandırmasında Platformlar Arası Keşifsel Analiz

[![Python Version](https://shields.io)](https://python.org)
[![License: Public Domain](https://shields.io)](https://nih.gov)

Bu proje, yüksek boyutlu biyolojik verilerde (mikrodizi) makine öğrenmesi modellerinin sınırlarını, **veri sızıntısız (data leakage-free)** öznitelik seçimi süreçlerini ve farklı jenerasyon cihaz platformları arasındaki **teknik yığın etkisini (batch effect)** sistematik olarak sorgulayan uçtan uca tekrarlanabilir bir iş akışıdır.

## 📌 Proje Özeti & Motivasyon

Klinik pratikte löseminin **Akut Lenfoblastik Lösemi (ALL)** ve **Akut Miyeloid Lösemi (AML)** olarak doğru alt tiplendirilmesi, uygulanacak kemoterapi rejimlerinin tamamen farklı olması sebebiyle hayati öneme sahiptir. Bu çalışmada, makine öğrenmesi modellerinin tek bir kohorttaki içsel başarısına odaklanan yaygın yaklaşımların ötesine geçilerek, **platformlar arası bir genellenebilirlik krizi** deneysel olarak ispatlanmıştır.

* **İçsel Başarı (Golub 5-Fold CV):** Random Forest ve Lojistik Regresyon modelleri **%97.2 Doğruluk** ve **%96.0 F1-Skoru** ile kusursuz performans sergilemiştir.
* **Dış Test Genellenebilirlik Krizi:** Modeller bağımsız dış veri setlerine (GPL570 jenerasyonu) uygulandığında dış test F1-skorları tamamen **0.0**'a çakılmıştır.
* **Kök Neden Analizi:** Teknik cihaz platform farklarından dolayı ortak gen kesişim havuzunun kritik sınır olan **40 gene düşmesi** ve evrensel biyolojik sinyalin taşınamamasıdır.

---

## 🔬 Metodoloji ve Veri Sızıntısı Koruması

Çalışmanın bilimsel doğruluğu için eksik veri doldurma (`SimpleImputer`) ve Welch T-Test tabanlı öznitelik seçimi süreçleri asla tüm veriye toplu uygulanmamıştır. Bu ön işleme adımları, **Stratified 5-Fold Cross-Validation** döngüsünün sadece ilgili Fold'un eğitim (Train) kümesinde fit edilmiş, test bloklarına ise sadece transform edilerek tatbik edilmiştir. Sınıf dengesizliğine karşı modellere `class_weight='balanced'` cezalandırması entegre edilmiştir.

---

## 🔄 Tekrarlanabilirlik 

Bu projedeki analizlerin ve model sonuçlarının her çalıştırıldığında birebir aynı çıkması (tekrarlanabilir olması) için sabit bir random seed kullanılmıştır. 

- **Random Seed:** `42`

Projedeki veri bölme (train/test split) ve model eğitim süreçlerinde bu sabit değer kullanılarak sonuçların kararlılığı garanti altına alınmıştır.

## 📊 Veri Setleri Künyesi

| Veri Seti | Cihaz / Platform | Boyutlar | Sınıf Dağılımı (%) | Rolü |
| :--- | :--- | :--- | :--- | :--- |
| **Golub** | Affymetrix Hu6800 (GPL31) | 72 hasta, 4457 gen | %65.3 ALL / %34.7 AML | Eğitim & İç Doğrulama |
| **GSE6891_GSE26713** | Affymetrix HG U133 P2 (GPL570) | 654 hasta, 54675 gen | %17.9 ALL / %82.1 AML | Bağımsız Dış Test Seti 1 |
| **GSE13159 (MILE)** | Affymetrix HG U133 P2 (GPL570) | 2096 hasta, 54675 gen | %35.8 ALL / %64.2 AML | Bağımsız Dış Test Seti 2 |

---

## 📈 Deneysel Sonuçlar

### 1. İç Doğrulama Performansı (Golub Kohortu - 5-Fold CV)
* **Random Forest:** %97.2 Accuracy | %96.0 F1-Score | 1.000 AUC
* **Logistic Regression:** %97.2 Accuracy | %96.0 F1-Score | 0.984 AUC
* **Support Vector Machine:** %95.8 Accuracy | %93.7 F1-Score
* **Decision Tree (En Zayıf):** %87.2 Accuracy | %82.7 F1-Score *(Kök düğümde CD33 biyobelirteci doğrulanmıştır)*

### 2. Bağımsız Dış Test Performans Matrisi (Kriz Tablosu)

| Test Veri Seti | Model | Dış Test Doğruluğu (Accuracy) | Dış Test F1-Skoru | Durum |
| :--- | :--- | :---: | :---: | :---: |
| **GSE6891_GSE26713** | Tüm Modeller | %17.89 | **0.0** | ⚠️ Körleşme Anomali |
| **GSE13159** | Tüm Modeller | %35.78 | **0.0** | ⚠️ Körleşme Anomali |

> **Analiz Notu:** Doğruluk skorlarının tam olarak dış setlerdeki ALL hastalarının doğal dağılım oranına kilitlenmesi, modellerin ortak gen bulamayıp körleştiğini ve riske girmeyerek gelen tüm hastaları ALL olarak tahmin ettiğini ispatlamaktadır.

---

## 🛠️ Klasör Yapısı

```text
├── Datasets/          # NCBI GEO'dan alınan ham/işlenmiş veriler (Git'e push edilmez)
├── Outputs/           # PCA, t-SNE, Clustermap ve Karar Ağacı görsel çıktıları
├── References/        # Literatür taraması tam metin PDF arşivi (2021-2026)
├── Src/               # Modüler kaynak kodlar (Preprocessing, Feature Selection, vb.)
├── .gitignore         # Büyük veri dosyalarının repoyu kirletmesini engeller
├── README.md          # Proje ana dökümantasyonu
└── requirements.txt   # Proje kütüphane bağımlılıkları ve versiyonları
```

---

## 💻 Kurulum ve Çalıştırma

1. Projeyi yerelinize klonlayın:
   ```bash
   git clone https://github.com
   cd MachineLearning
   ```

2. Gerekli kütüphaneleri yükleyin (Önerilen: Sabit sonuçlar için Python 3.9+):
   ```bash
   pip install -r requirements.txt
   ```

3. Analiz boru hattını (pipeline) çalıştırın:
   ```bash
   python Src/main.py
   ```

---

## 🚀 Final Dönemi Yol Haritası (Gelecek Çalışmalar)

Ara raporda saptanan bu platform bariyerini aşmak ve dış test F1-skorlarını **%90'ın üzerine** taşımak amacıyla şu adımlar entegre edilmektedir:
1. **HUGO Gene Symbol Mapping:** NCBI GEO üzerinden `.soft` tabloları kullanılarak GPL31 ve GPL570 ham kodları evrensel gen sembollerine map edilecek, ortak gen havuzu 40'tan **5.000+** seviyesine çıkarılacaktır.
2. **ComBat (sva) Algoritması:** Laboratuvarlar arası varyans kaymalarını ve gizli teknik gürültüleri sıfırlamak amacıyla boru hattına dahil edilecektir.

---

## ✍️ Yazar & İletişim

* **Yasin Gökhan Arslan** - [GitHub](https://github.com/GokhanArslanPostgraduate)
* **Ders Bilgisi:** 503001508 – Makine Öğrenmesi ve Anomali Tespiti