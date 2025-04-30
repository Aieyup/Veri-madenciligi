# API Anahtarları Kurulumu

Bu proje, haber verileri toplamak için News API ve metin analizi için Google Gemini API'yi kullanmaktadır. 
API anahtarları olmadan program örnek veriler üretecektir, ancak gerçek veri analizi için kendi anahtarlarınızı edinmeniz önerilir.

## .env Dosyası Oluşturma

Proje dizininde `.env` adında bir dosya oluşturun ve aşağıdaki içeriği ekleyin:

```
# News API anahtarı
NEWS_API_KEY=sizin_news_api_anahtariniz

# Google Gemini API anahtarı
GEMINI_API_KEY=sizin_gemini_api_anahtariniz
```

## API Anahtarlarını Edinme

1. **News API Anahtarı**:
   - [NewsAPI.org](https://newsapi.org/) adresine gidin
   - Hesap oluşturun ve oturum açın
   - Dashboard sayfasından API anahtarınızı alın
   - Ücretsiz plan günlük 100 istek ile sınırlıdır

2. **Google Gemini API Anahtarı**:
   - [Google AI Studio](https://makersuite.google.com/app/apikey) adresine gidin
   - Bir Google hesabı ile oturum açın
   - "Get API key" seçeneğini kullanarak yeni bir API anahtarı oluşturun
   - Ücretsiz kullanım kısıtlamaları vardır, güncel bilgiler için Google'ın dokümanlarını kontrol edin

## Notlar

- API anahtarlarını GitHub gibi halka açık platformlarda paylaşmayın
- API sağlayıcıların kullanım koşullarına ve sınırlamalarına dikkat edin
- Projeyi çalıştırmadan önce `.env` dosyasının doğru dizinde olduğundan emin olun 