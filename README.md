# 📊 Veri Madenciliği Yaklaşımlarıyla Şirket Hissesi Performansının Endeks Karşılaştırmalı Analizi ve Metin Verisi Entegrasyonu



> *Yapısal ve yapısal olmayan veriyi birleştirerek finansal analizi bir adım öteye taşıyan yenilikçi veri madenciliği projesi.*

## 📈 Proje Özeti

Bu proje, veri madenciliği tekniklerini kullanarak, belirlenen bir şirketin hisse senedi performansının ilgili borsa endeksine (örn. BIST 100) göre göreli durumunu analiz etmeyi ve bu analizi ilgili metin verilerinden (örn. haberler, finansal rapor özetleri) çıkarılan bilgilerle zenginleştirmeyi amaçlamaktadır. 

Finans piyasalarında şirket hisselerinin performansı genellikle genel piyasa endekslerine göre değerlendirilir. Ancak, bir şirketin hissesinin piyasa geneline göre neden daha iyi veya daha kötü performans gösterdiği karmaşık bir konudur ve sadece tarihsel fiyat hareketleriyle açıklanamaz. Şirkete özel olaylar, haberler, yönetim kararları gibi faktörler hisse değerini önemli ölçüde etkileyebilir.

Proje kapsamında:
1. Tarihsel hisse senedi ve endeks verileri ile metin verileri toplanarak
2. Veri ön işleme adımlarından geçirilerek
3. Şirket hissesi ile endeks arasındaki korelasyonlar ve desenler incelenerek
4. Metin verilerinden elde edilen duyarlılık (sentiment) veya konu bilgisi hisse performansı ile ilişkilendirilerek

şirkete özel faktörlerin hissenin genel piyasaya göre sapmasında ne kadar etkili olduğunu ortaya koymak hedeflenmektedir.

## 🔍 Araştırma Sorusu ve Önemi

### Ana Araştırma Sorusu
Belirlenen bir şirketin hisse senedinin borsa endeksine göre günlük göreli performansındaki değişimler, şirkete ilişkin metin verilerinden (haberler, duyurular vb.) çıkarılan duyarlılık, konu veya olay bilgileri ile nasıl bir korelasyon sergilemektedir? Metin verilerinden elde edilen bilgiler, bu göreli performansı tahmin etmek için kullanılabilir mi?

### Projenin Önemi
- **Yatırımcılar İçin:** Hem finansal analistler hem de bireysel yatırımcılar için, sadece hisse senedinin kendi geçmiş performansına değil, aynı zamanda piyasa geneline göre durumuna ve şirkete özel güncel gelişmelere dayalı daha bilinçli kararlar almalarına yardımcı olma potansiyeli taşımaktadır.
- **Bilimsel Açıdan:** Yapısal (hisse/endeks verileri) ve yapısal olmayan (metin verileri) heterojen veri kümelerinin veri madenciliği yaklaşımlarıyla entegre edilerek analiz edilmesine yönelik pratik bir uygulama sunmaktadır.
- **Yenilikçi Yön:** Veri bütünleştirme ve farklı veri madenciliği işlevlerinin bir arada kullanılması, projenin yenilikçi yönünü oluşturmaktadır.

## 🧮 Kullanılan Veri Madenciliği Teknikleri



Bu proje, çeşitli veri madenciliği tekniklerini birleştirerek kapsamlı bir analiz çerçevesi sunmaktadır:

1. **Keşifsel Veri Analizi:**
   - Temel istatistiksel analizler (ortalama, medyan, varyans, vs.)
   - Veri dağılımı incelemeleri ve görselleştirmeler
   - Korelasyon analizleri ve ısı haritaları

2. **Veri Ön İşleme ve Bütünleştirme:**
   - Eksik değerlerin tespiti ve doldurulması
   - Yapısal ve yapısal olmayan verinin entegrasyonu
   - Öznitelik mühendisliği (göreli performans hesabı, duyarlılık analizi)

3. **Desen Madenciliği:**
   - Birliktelik kuralları ve desenlerin keşfi
   - Güven, destek ve kaldıraç metrikleriyle kural değerlendirme
   - Yüksek kaldıraçlı kuralların görselleştirilmesi

4. **Sınıflandırma ve Tahmin:**
   - Göreli performans kategorilerini tahmin eden modeller
   - Rastgele orman sınıflandırıcısı kullanımı
   - Öznitelik önemliliği analizi

5. **Kümeleme Analizi:**
   - Benzer gün özelliklerinin tespit edilmesi
   - Optimal küme sayısının belirlenmesi
   - Küme özelliklerinin incelenmesi

6. **Metin Madenciliği Entegrasyonu:**
   - Duyarlılık analizi ile metnin duygusal tonunun çıkarılması
   - Konu modellemesi ile haber içeriklerinin kategorize edilmesi
   - Google Gemini API kullanılarak gelişmiş metin analizi

## 🛠️ Proje Mimarisi ve Yapısı

Proje, modüler ve genişletilebilir bir mimari ile tasarlanmıştır:

```
📦 Veri Madenciliği Projesi
 ┣ 📜 main.py                 # Ana program akışı
 ┣ 📜 data_acquisition.py     # Veri toplama işlemleri
 ┣ 📜 data_preprocessing.py   # Veri temizleme ve dönüştürme
 ┣ 📜 data_mining.py          # Analiz ve madencilik işlemleri
 ┣ 📜 evaluation.py           # Değerlendirme fonksiyonları
 ┣ 📜 visualization.py        # Görselleştirme fonksiyonları
 ┣ 📜 requirements.txt        # Gerekli Python kütüphaneleri
 ┣ 📜 processed_data.csv      # İşlenmiş veri (çıktı)
 ┣ 📊 fiyat_grafigi.png       # Görselleştirme çıktıları
 ┣ 📊 korelasyon_matrisi.png  # Görselleştirme çıktıları
 ┗ 📄 README.md               # Proje dokümantasyonu
```

### Ana Modüller ve İşlevleri

#### 🔍 Veri Toplama (Data Acquisition)
`data_acquisition.py`, Yahoo Finance API ve News API kullanarak tarihsel hisse senedi verilerini ve haber metinlerini toplama işlevini gerçekleştirir. API anahtarları olmadığında bile çalışabilmesi için akıllı bir tasarım kullanılmıştır - gerçek veri toplanamadığında gerçekçi örnek veriler üretir.

```python
# Hisse ve endeks verilerini toplama örneği
tickers = [stock_symbol, index_symbol]
data = yf.download(tickers, start=start_date, end=end_date)
```

#### 🧹 Veri Ön İşleme (Data Preprocessing)
`data_preprocessing.py`, toplanan ham verileri analiz için hazırlar:
- Eksik değerlerin tespiti ve doldurulması
- Yapısal ve yapısal olmayan verinin bütünleştirilmesi
- Günlük getiri, göreli performans hesaplamaları
- Google Gemini API ile duyarlılık analizi

```python
# Göreli performans hesaplama örneği
result_df['relative_performance'] = result_df['daily_return'] - result_df['index_daily_return']
```

#### 🔮 Veri Madenciliği ve Analiz (Data Mining)
`data_mining.py`, veri setinde keşifsel analiz, desen madenciliği, sınıflandırma ve kümeleme işlemlerini gerçekleştirir:

```python
# Desen madenciliği örneği
frequent_itemsets = apriori(binary_df, min_support=min_support, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
```

#### 📏 Değerlendirme (Evaluation)
`evaluation.py`, analiz sonuçlarını değerlendirerek anlam çıkarma işlemlerini gerçekleştirir:
- Sınıflandırma modelinin doğruluk, kesinlik, duyarlılık metrikleri
- Desen madenciliği sonuçlarının güven, destek, kaldıraç değerlendirmesi
- Korelasyon analizleri ve istatistiksel anlamlılık testleri

#### 📊 Görselleştirme (Visualization)
`visualization.py`, analiz sonuçlarını interaktif ve anlaşılır grafiklerle sunar:
- Zaman serisi görselleştirmeleri
- Korelasyon ısı haritaları
- Kural ve desen görselleştirmeleri
- Sınıflandırma sonuç grafikleri

## 📥 Kurulum ve Kullanım

### Sistem Gereksinimleri
- Python 3.7 veya üstü
- Gerekli kütüphaneleri kurmak için yeterli disk alanı (~500MB)
- API anahtarları (isteğe bağlı, olmadığında simüle edilmiş veri kullanılır)

### Kurulum Adımları

1. **Projeyi Klonlayın**
   ```bash
   git clone https://github.com/username/borsa-veri-madenciligi.git
   cd borsa-veri-madenciligi
   ```

2. **Gerekli Kütüphaneleri Yükleyin**
   ```bash
   pip install -r requirements.txt
   ```

3. **API Anahtarlarını Ayarlayın (İsteğe Bağlı)**
   - `.env` dosyası oluşturun:
   ```
   NEWS_API_KEY=sizin_news_api_anahtariniz
   GEMINI_API_KEY=sizin_gemini_api_anahtariniz
   ```

4. **Programı Çalıştırın**
   ```bash
   python main.py
   ```

### Analiz Parametrelerinin Özelleştirilmesi

Analizlerinizi özelleştirmek için `main.py` dosyasında şu değişkenleri ayarlayabilirsiniz:

```python
# Analiz edilecek hisse ve endeks
stock_symbol = "THYAO.IS"  # Örnek: Türk Hava Yolları
index_symbol = "XU100.IS"  # BIST 100

# Analiz süresi
start_date = "2022-01-01"
end_date = "2023-01-01"
```


Proje şu görselleri ve analiz çıktılarını üretir:

### 📈 Zaman Serisi Grafikleri
- **Fiyat Grafiği:** Hisse ve endeks kapanış fiyatlarının karşılaştırmalı gösterimi
- **Getiri Grafiği:** Günlük getiri oranlarının zaman içindeki değişimi
- **Göreli Performans Grafiği:** Hissenin endekse göre performansının zaman içindeki değişimi

### 🔍 Korelasyon Analizleri
- **Korelasyon Matrisi:** Tüm sayısal öznitelikler arasındaki ilişkileri gösteren ısı haritası
- **Duyarlılık-Performans Korelasyonu:** Metin duyarlılığı ile göreli performans arasındaki ilişki analizi

### 🔮 Desen Madenciliği Sonuçları
- **Kaldıraç Grafiği:** En ilginç birliktelik kurallarının kaldıraç değerlerine göre sıralaması
- **Kural Dağılımı:** Kuralların destek, güven ve kaldıraç değerlerine göre dağılımı

### 🤖 Sınıflandırma Model Sonuçları
- **Karmaşıklık Matrisi:** Model tahminlerinin doğruluk dağılımını gösteren matris
- **Sınıflandırma Metrikleri:** Kesinlik, duyarlılık ve F1-skor değerlendirmeleri
- **Öznitelik Önemliliği:** Modelde en etkili özniteliklerin görselleştirilmesi

### 📊 Kümeleme Analizi
- **Küme Profilleri:** Tespit edilen kümelerin özelliklerini gösteren grafikler
- **Silhouette Analizi:** Optimal küme sayısının tespiti için kullanılan analiz

## 💡 Literatür ve Yenilikçi Yaklaşım

Finansal zaman serisi analizi, tarihsel fiyat verilerini kullanarak borsa tahmini üzerine yoğunlaşmıştır. Bunun yanı sıra, korelasyon analizi ile hisse senedi ile endeks arasındaki doğrusal ilişki incelenmektedir. Son yıllarda, metin madenciliği ve duyarlılık analizi tekniklerini finansal piyasa verileriyle birleştiren çalışmalar artış göstermiştir.

Bu proje, literatürdeki yaklaşımlardan farklı olarak:

1. **Entegre Veri Analizi:** Yapısal (hisse/endeks fiyat verileri) ve yapısal olmayan (metin) verileri birleştirerek bütünsel bir analiz çerçevesi sunar.

2. **Göreli Performans Odaklı:** Mutlak fiyat tahmininden ziyade, hissenin endekse göre göreli performansını analiz ederek daha anlamlı içgörüler sağlar.

3. **Çoklu Veri Madenciliği Teknikleri:** Tek bir teknik yerine, keşifsel analiz, desen madenciliği, sınıflandırma ve kümeleme gibi çeşitli teknikleri birleştirerek çok boyutlu bir analiz sunar.

4. **Gelişmiş NLP Entegrasyonu:** Google Gemini API kullanarak ileri seviye metin analizi ve duyarlılık değerlendirmesi yapabilir.

## 🛡️ Riskler ve Çözüm Stratejileri

Projenin uygulanmasında karşılaşılabilecek zorluklar ve çözüm stratejileri:

| Risk | Çözüm Stratejisi |
|------|------------------|
| Metin verisi toplama ve işleme zorlukları | Belirli bir finans haber kaynağı ile sınırlı kalınabilir, temel duyarlılık analizi yöntemleri kullanılabilir |
| Borsa verilerinin volatilitesi ve tahmin zorluğu | Tahmin yerine korelasyon ve desen keşfine odaklanılabilir |
| Yüksek boyutlu veri | Veri azaltma teknikleri (PCA, özellik seçimi) kullanılacaktır |
| API kısıtlamaları | Örnek veri üretme mekanizması ile API anahtarı olmadığında bile çalışabilir |

## 📚 Kaynak Kodlar ve Notlar

- API anahtarlarının sağlanması, analiz sonuçlarının gerçek veri üzerinden elde edilmesini sağlar
- API anahtarları yoksa, sistem sahte/örnek veri üretecektir
- Proje parametreleri `main.py` dosyasında özelleştirilebilir
- Detaylı API kurulum talimatları için `ENV_KURULUM.md` dosyasını inceleyebilirsiniz

## 📫 İletişim ve Katkıda Bulunma

Bu proje açık kaynaklıdır ve katkılarınızı bekliyoruz! Önerileriniz, geri bildirimleriniz veya katkılarınız için:

- GitHub üzerinden bir issue açın
- Projeyi forklayıp pull request gönderin

## 📄 Lisans

Bu proje [MIT lisansı](LICENSE) altında dağıtılmaktadır. 