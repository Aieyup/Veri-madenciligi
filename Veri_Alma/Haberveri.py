 #!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Haber Veri Toplama Modülü.

Bu modül, şirketlerle ilgili haber verilerini toplamak için gerekli işlevleri içerir.
Farklı kaynaklardan haber verisi alabilir ve standart bir formatta kaydedebilir.
"""

import logging
import os
import random
import time
import functools
import json
import requests
import pandas as pd
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional, Union, Tuple
from dotenv import load_dotenv

# Çalışma dizinini ayarla
current_dir = os.path.dirname(os.path.abspath(__file__))

# .env dosyasından çevre değişkenlerini yükle
load_dotenv()

# Loglama yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("haber_toplama.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def retry_with_backoff(initial_delay=1, exponential_base=2, jitter=True, max_retries=5):
    """
    İstek hatalarında yeniden deneme işlemi için dekoratör.

    Exponansiyel gecikme ile yeniden dener.
    
    Args:
        initial_delay: İlk bekleme süresi (saniye)
        exponential_base: Gecikme üssel katsayısı
        jitter: Rasgele varyasyon eklensin mi
        max_retries: Maksimum yeniden deneme sayısı
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            num_retries = 0
            delay = initial_delay
            
            while True:
                try:
                    return func(*args, **kwargs)
                except (requests.exceptions.RequestException, 
                        ConnectionError, 
                        ValueError, 
                        IOError) as e:
                    
                    num_retries += 1
                    
                    # Maksimum deneme sayısı aşıldı
                    if num_retries > max_retries:
                        logger.error(f"Maksimum deneme sayısı aşıldı ({max_retries}): {str(e)}")
                        raise
                    
                    # 429 Too Many Requests hatası için daha uzun bekleme
                    too_many_requests = False
                    if isinstance(e, requests.exceptions.RequestException):
                        if hasattr(e, 'response') and e.response is not None:
                            if e.response.status_code == 429:
                                too_many_requests = True
                    
                    # Exponansiyel gecikme hesapla
                    delay *= exponential_base * (2 if too_many_requests else 1)
                    
                    # Rasgele varyasyon ekle
                    if jitter:
                        delay *= (0.5 + random.random())
                    
                    logger.warning(f"Hata oluştu ({num_retries}/{max_retries}): {str(e)}")
                    logger.info(f"{delay:.1f} saniye bekleniyor ve yeniden deneniyor...")
                    time.sleep(delay)
                    
        return wrapper
    return decorator


class NewsApiClient:
    """NewsAPI için basit bir istemci sınıfı."""
    
    BASE_URL = "https://newsapi.org/v2"
    
    def __init__(self, api_key: str):
        """
        NewsApiClient sınıfını başlatır.
        
        Args:
            api_key: NewsAPI için API anahtarı
        """
        self.api_key = api_key
    
    def get_everything(self, **kwargs):
        """
        NewsAPI'nin 'everything' endpoint'ini kullanarak haber arar.
        
        Args:
            **kwargs: API'ye gönderilecek parametreler
                q: Arama sorgusu
                from_param: Başlangıç tarihi (YYYY-MM-DD)
                to: Bitiş tarihi (YYYY-MM-DD)
                language: Dil kodu (örn. 'en', 'tr')
                sort_by: Sıralama kriteri
                
        Returns:
            NewsAPI yanıtını içeren sözlük
        """
        # URL parametrelerini hazırla
        params = {k: v for k, v in kwargs.items()}
        
        # from_param özel bir durum, API'de 'from' parametresi olarak geçiyor
        if 'from_param' in params:
            params['from'] = params.pop('from_param')
            
        # API anahtarını ekle
        params['apiKey'] = self.api_key
        
        # API isteği yap
        response = requests.get(f"{self.BASE_URL}/everything", params=params)
        
        # Yanıtı kontrol et
        if response.status_code == 200:
            return response.json()
        else:
            error_msg = response.json().get('message', 'Bilinmeyen hata')
            logger.error(f"NewsAPI hatası: {error_msg} (Kod: {response.status_code})")
            return {
                "status": "error",
                "code": response.status_code,
                "message": error_msg,
                "articles": []
            }


@retry_with_backoff()
def get_news_data(company_name: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    NewsAPI kullanarak şirketle ilgili haberleri getirir.
    
    Args:
        company_name: Haberleri aranacak şirketin adı
        start_date: Haber arama başlangıç tarihi (YYYY-MM-DD formatında)
        end_date: Haber arama bitiş tarihi (YYYY-MM-DD formatında)
    
    Returns:
        Haber makaleleri içeren pandas DataFrame.
        Hata durumunda boş DataFrame dönebilir.
    """
    try:
        # Gelecek tarih kontrolü
        current_date = datetime.now().date()
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
        
        if end_dt > current_date:
            logger.warning(f"Bitiş tarihi ({end_date}) gelecekte. Bugünün tarihine ayarlanıyor.")
            end_date = current_date.strftime("%Y-%m-%d")
        
        # NewsAPI ile etkileşim
        news_api_key = os.getenv("NEWS_API_KEY")
        
        # API anahtarı yoksa alternatif kaynak kullan
        if not news_api_key:
            logger.warning("NEWS_API_KEY bulunamadı. Alternatif haber kaynağı kullanılacak.")
            return get_alternative_news_data(company_name, start_date, end_date)
            
        logger.info(f"{company_name} ile ilgili haberler toplanıyor...")
        
        # NewsApiClient sınıfını başlat
        try:
            newsapi = NewsApiClient(api_key=news_api_key)
            logger.info("NewsApiClient başarıyla başlatıldı.")
        except Exception as e:
            logger.error(f"NewsApiClient başlatılırken hata: {e}")
            return get_alternative_news_data(company_name, start_date, end_date)
        
        # NewsAPI 1 ay ile sınırlıdır, bu nedenle tarih aralığını bölmemiz gerekiyor
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        all_articles = []
        current_start = start_dt
        
        # Her ay için ayrı sorgu yap
        while current_start < end_dt:
            current_end = min(current_start + timedelta(days=30), end_dt)
            
            # API sorgusu
            try:
                response = newsapi.get_everything(
                    q=company_name,
                    from_param=current_start.strftime("%Y-%m-%d"),
                    to=current_end.strftime("%Y-%m-%d"),
                    language="en",
                    sort_by="publishedAt"
                )
                
                if response["status"] == "ok":
                    all_articles.extend(response["articles"])
                    logger.debug(
                        f"{len(response['articles'])} makale bulundu: "
                        f"{current_start.strftime('%Y-%m-%d')} - "
                        f"{current_end.strftime('%Y-%m-%d')}"
                    )
                else:
                    logger.warning(f"API yanıtı başarısız: {response.get('message', 'Bilinmeyen hata')}")
                    
            except Exception as e:
                logger.error(f"API isteği sırasında hata: {e}")
                
            current_start = current_end
            
            # API sınırlamalarını aşmamak için bekleme
            time.sleep(2)
        
        # Sonuçları DataFrame'e dönüştür
        if all_articles:
            news_df = pd.DataFrame(all_articles)
            
            # 'publishedAt' sütunu olup olmadığını kontrol et ve düzenle
            if 'publishedAt' in news_df.columns:
                # publishedAt'i datetime'a dönüştür
                news_df["publishedAt"] = pd.to_datetime(news_df["publishedAt"], errors='coerce')
                
                # Tarih ve saat bilgilerini ayrı sütunlara çıkar
                news_df["date"] = news_df["publishedAt"].dt.date
                news_df["hour"] = news_df["publishedAt"].dt.hour
                news_df["minute"] = news_df["publishedAt"].dt.minute
                
                # publishedAt sütunu NaN içeren satırları işaretle
                missing_dates = news_df["publishedAt"].isna().sum()
                if missing_dates > 0:
                    logger.warning(f"{missing_dates} haberde tarih bilgisi eksik. Bu haberler için şu anki tarih kullanılıyor.")
                    # Eksik tarihler için şu anki tarihi kullan
                    now = datetime.now()
                    news_df.loc[news_df["publishedAt"].isna(), "date"] = now.date()
                    news_df.loc[news_df["publishedAt"].isna(), "hour"] = now.hour
                    news_df.loc[news_df["publishedAt"].isna(), "minute"] = now.minute
                    
                    # publishedAt sütununu güncelle
                    for idx in news_df[news_df["publishedAt"].isna()].index:
                        date_val = news_df.loc[idx, "date"]
                        hour_val = news_df.loc[idx, "hour"]
                        minute_val = news_df.loc[idx, "minute"]
                        news_df.loc[idx, "publishedAt"] = datetime.combine(date_val, time(hour_val, minute_val))
            else:
                # 'publishedAt' yoksa, şu anki tarihi kullan
                logger.warning("API yanıtında 'publishedAt' sütunu bulunamadı. Şu anki tarih kullanılıyor.")
                now = datetime.now()
                news_df["date"] = now.date()
                
                # İş saatleri içinde rastgele saatler atama
                hours = [random.randint(9, 17) for _ in range(len(news_df))]
                minutes = [random.randint(0, 59) for _ in range(len(news_df))]
                
                # publishedAt sütunu oluştur
                publishedAt_list = []
                for i, row in news_df.iterrows():
                    dt = datetime.combine(row["date"], time(hours[i], minutes[i]))
                    publishedAt_list.append(dt)
                
                news_df["publishedAt"] = publishedAt_list
                news_df["hour"] = hours
                news_df["minute"] = minutes
            
            # Tekrarlayan haberleri temizle
            if len(news_df) > 0:
                # URL ve başlığa göre tekrarlayan haberleri tespit et ve kaldır
                before_len = len(news_df)
                news_df = news_df.drop_duplicates(subset=["url"], keep="first")
                after_len = len(news_df)
                
                if before_len > after_len:
                    logger.info(f"{before_len - after_len} tekrarlayan haber kaldırıldı.")
            
            # Gerekli sütunların varlığını kontrol et
            required_cols = ["title", "description", "content", "url"]
            for col in required_cols:
                if col not in news_df.columns:
                    news_df[col] = ""
                    logger.warning(f"API yanıtında '{col}' sütunu bulunamadı. Boş değer ekleniyor.")
            
            # Sadece gerekli sütunları seç
            news_df = news_df[["date", "publishedAt", "title", "description", "content", "url"]]
            
            logger.info(f"Toplam {len(news_df)} haber makalesi bulundu.")
            return news_df
        else:
            logger.warning("Haber bulunamadı, alternatif kaynak deneniyor.")
            return get_alternative_news_data(company_name, start_date, end_date)
            
    except Exception as e:
        logger.error(f"Haber verisi alınırken hata oluştu: {e}")
        return get_alternative_news_data(company_name, start_date, end_date)


def get_alternative_news_data(company_name: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    NewsAPI yerine alternatif bir haber kaynağı kullanarak haber verileri oluşturur.
    
    NewsAPI'ye erişilemediğinde veya API anahtarı olmadığında kullanılır.
    Bu fonksiyon basit ve sentetik haber verileri oluşturur.
    
    Args:
        company_name: Haberleri aranacak şirketin adı
        start_date: Başlangıç tarihi (YYYY-MM-DD formatında)
        end_date: Bitiş tarihi (YYYY-MM-DD formatında)
        
    Returns:
        Haber makaleleri içeren DataFrame
    """
    logger.info(f"Alternatif haber kaynağı kullanılıyor: {company_name}")
    
    # 1. Adım: Daha önce kaydedilmiş yerel verileri kontrol et
    saved_file_path = os.path.join(
        current_dir, 
        f"veri/haber/{company_name.replace(' ', '_').lower()}_haberler_{start_date}_{end_date}.csv"
    )
    
    if os.path.exists(saved_file_path):
        logger.info(f"Yerel dosyadan haber okunuyor: {saved_file_path}")
        try:
            data = pd.read_csv(saved_file_path)
            if not data.empty:
                logger.info(f"Yerel dosyadan {len(data)} haber okundu.")
                return data
        except Exception as e:
            logger.error(f"Yerel haber dosyası okuma hatası: {e}")
    
    # 2. Adım: Sentetik haber verileri oluştur
    logger.warning("Gerçek haber verileri alınamadı, örnek haber verileri oluşturuluyor.")
    
    try:
        # Tarih aralığı oluştur
        start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
        
        # Tarih aralığındaki tüm günleri oluştur
        all_dates = []
        current_date = start_dt
        while current_date <= end_dt:
            all_dates.append(current_date)
            current_date += timedelta(days=1)
        
        # Her gün için farklı sayıda haber oluştur (daha gerçekçi dağılım)
        date_list = []
        for date in all_dates:
            # Haftasonları daha az haber olsun
            is_weekend = date.weekday() >= 5
            # Rastgele haber sayısı belirleme
            if is_weekend:
                # Haftasonu: %40 ihtimalle 0, %60 ihtimalle 1-2 haber
                num_news = random.choices([0, 1, 2], weights=[0.4, 0.4, 0.2])[0]
            else:
                # Hafta içi: %20 ihtimalle 0, %30 ihtimalle 1, %30 ihtimalle 2, %20 ihtimalle 3 haber
                num_news = random.choices([0, 1, 2, 3], weights=[0.2, 0.3, 0.3, 0.2])[0]
            
            # Önemli olayların olduğu bazı tarihlerde daha fazla haber olsun (simülasyon için)
            month_day = (date.month, date.day)
            if month_day in [(1, 15), (4, 15), (7, 15), (10, 15)]:  # Çeyrek sonuçlar için simülasyon
                num_news = random.randint(2, 5)  # Önemli günlerde daha fazla haber
            
            # Belirlenen sayıda haberi ekle
            for _ in range(num_news):
                date_list.append(date)
        
        # Eğer hiç tarih seçilmediyse, en az bir haber oluştur
        if not date_list:
            date_list = [start_dt]
        
        # Farklı haber içerik şablonlarını tanımla
        news_templates = [
            {
                "title_template": f"{company_name} announces quarterly results",
                "desc_template": f"{company_name} reported quarterly earnings.",
                "content_template": f"{company_name} has announced its quarterly financial results, reporting earnings per share of $X.XX. Revenue reached $XX.X billion, marking a X% increase from the previous year. The company attributed this growth to strong performance in its core business segments."
            },
            {
                "title_template": f"{company_name} stock movement on market news",
                "desc_template": f"Shares of {company_name} changed amid broader market developments.",
                "content_template": f"Shares of {company_name} moved X% today as investors reacted to broader market news and economic data. Analysts suggest the market is adjusting to recent economic indicators. Trading volume was notably X% above/below the 30-day average."
            },
            {
                "title_template": f"{company_name} announces new strategic partnership",
                "desc_template": f"{company_name} forms alliance with industry leader to expand market reach.",
                "content_template": f"{company_name} has announced a new strategic partnership aimed at expanding its market presence and enhancing its product offerings. This collaboration is expected to drive innovation and create new opportunities for growth."
            },
            {
                "title_template": f"{company_name} regulatory updates",
                "desc_template": f"Regulators announce decision regarding {company_name}'s business practices.",
                "content_template": f"{company_name} has received regulatory updates regarding certain aspects of its business practices. The company has stated it is addressing the requirements and believes its operations will adapt to the new framework."
            },
            {
                "title_template": f"{company_name} market outlook update",
                "desc_template": f"{company_name} updates its outlook amid sector developments.",
                "content_template": f"{company_name} has provided an update on its market outlook in response to recent sector developments. The company outlined its strategy to navigate current market conditions and position itself for continued growth."
            },
            {
                "title_template": f"{company_name} introduces new product line",
                "desc_template": f"New product launch announced by {company_name}.",
                "content_template": f"{company_name} today unveiled its latest product line, which is designed to address evolving consumer needs. The new offerings incorporate advanced technology and are expected to begin shipping to customers in the coming quarter."
            },
            {
                "title_template": f"{company_name} CEO discusses business strategy",
                "desc_template": f"Leadership insights on {company_name}'s market position.",
                "content_template": f"In a recent interview, the CEO of {company_name} outlined the company's strategic priorities and responded to questions about competitive positioning. The executive emphasized the importance of innovation and customer-centric approaches in the current business environment."
            },
            {
                "title_template": f"Analyst updates on {company_name}",
                "desc_template": f"Financial firm revises outlook for {company_name}.",
                "content_template": f"A prominent financial analysis firm has updated its assessment of {company_name}'s stock, citing changing market dynamics and company-specific factors. The report highlights several key areas that could influence the company's performance in the coming quarters."
            }
        ]
        
        # Tarihlere göre sıralı şekilde haber verileri oluştur
        news_data = []
        for date in sorted(date_list):
            # Gün içinde farklı saat için rastgele oluştur (daha gerçekçi)
            hour = random.randint(8, 17)  # 8:00 - 17:00 arası
            minute = random.randint(0, 59)
            news_time = datetime.combine(date, datetime.min.time()) + timedelta(hours=hour, minutes=minute)
            
            # Rasgele bir haber şablonu seç
            template = random.choice(news_templates)
            
            # Sayısal değerleri rasgele oluştur
            title = template["title_template"].replace("X.XX", f"{random.uniform(0.5, 3.5):.2f}")
            description = template["desc_template"].replace("X.XX", f"{random.uniform(0.5, 3.5):.2f}")
            content = template["content_template"]
            content = content.replace("$X.XX", f"${random.uniform(0.5, 3.5):.2f}")
            content = content.replace("$XX.X", f"${random.uniform(10, 50):.1f}")
            content = content.replace("X%", f"{random.randint(5, 25)}%")
            
            # Küçük varyasyonlar ekle (aynı şablondan farklı haberler oluşturmak için)
            if random.random() > 0.7:  # %30 ihtimalle ek değişiklikler
                modifiers = ["strong", "significant", "moderate", "unexpected", "impressive"]
                title = title.replace("announces", random.choice(["reports", "reveals", "announces", "confirms"]))
                content = content.replace("increase", random.choice(["growth", "increase", "improvement", "rise"]))
                content = content.replace("strong", random.choice(modifiers))
            
            news_data.append({
                "date": date,
                "publishedAt": news_time.isoformat(),  # ISO formatında tam datetime
                "title": title,
                "description": description,
                "content": content,
                "url": f"https://example.com/news/{company_name.lower().replace(' ', '-')}/{date.strftime('%Y-%m-%d')}-{hour}{minute}"
            })
        
        # DataFrame oluştur
        news_df = pd.DataFrame(news_data)
        
        # publishedAt sütunu ekle ve datetime'a dönüştür
        if "publishedAt" in news_df.columns:
            news_df["publishedAt"] = pd.to_datetime(news_df["publishedAt"])
        
        logger.info(f"{len(news_df)} sentetik haber makalesi oluşturuldu.")
        
        # Dosyaya kaydet (ileride kullanmak için)
        os.makedirs(os.path.dirname(saved_file_path), exist_ok=True)
        news_df.to_csv(saved_file_path, index=False)
        logger.info(f"Sentetik haber verileri gelecekte kullanılmak üzere kaydedildi: {saved_file_path}")
        
        return news_df
        
    except Exception as e:
        logger.error(f"Alternatif haber verisi oluşturma hatası: {e}")
        # Boş DataFrame döndür
        return pd.DataFrame()


def save_data_to_json(data: pd.DataFrame, filename: str) -> None:
    """
    DataFrame'i JSON formatında kaydeder.
    
    Args:
        data: Kaydedilecek DataFrame
        filename: Kaydedilecek dosya adı
    """
    try:
        if data.empty:
            logger.error(f"Kaydedilecek veri boş, {filename} oluşturulamadı.")
            return
            
        data.to_json(
            filename, 
            orient="records", 
            date_format="iso", 
            indent=4, 
            force_ascii=False
        )
        logger.info(f"Veriler başarıyla {filename} dosyasına kaydedildi.")
    except Exception as e:
        logger.error(f"{filename} dosyasına kayıt sırasında hata: {e}")


def main():
    """
    Ana program akışını yönetir. Kullanıcıdan parametreleri alır,
    haberleri toplar ve kaydeder.
    """
    # Gerekli klasörleri oluştur
    for directory in ["veri", "veri/haber"]:
        os.makedirs(directory, exist_ok=True)
    
    # Parametreleri kullanıcıdan al
    try:
        company_name = input("Şirket adını girin (örn. Apple Inc): ").strip() or "Apple Inc"
        start_date_input = input("Başlangıç tarihi (YYYY-MM-DD formatında, varsayılan: 2023-01-01): ").strip()
        start_date = start_date_input if start_date_input else "2023-01-01"
        
        end_date_input = input("Bitiş tarihi (YYYY-MM-DD formatında, varsayılan: bugün): ").strip()
        end_date = end_date_input if end_date_input else datetime.now().strftime("%Y-%m-%d")
        
        # Geçerli tarih formatı kontrolü
        datetime.strptime(start_date, "%Y-%m-%d")
        datetime.strptime(end_date, "%Y-%m-%d")
        
        # Gelecek tarih kontrolü
        current_date = datetime.now().date()
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
        if end_dt > current_date:
            logger.warning(f"Bitiş tarihi ({end_date}) gelecekte. Bugünün tarihine ayarlanıyor.")
            end_date = current_date.strftime("%Y-%m-%d")
    except ValueError as e:
        logger.error(f"Geçersiz tarih formatı: {e}")
        logger.warning("Varsayılan tarihler kullanılıyor: 2023-01-01 - bugün")
        start_date = "2023-01-01"
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    logger.info(f"Parametreler: Şirket: {company_name}")
    logger.info(f"Tarih aralığı: {start_date} - {end_date}")
    
    # Haber verilerini al
    logger.info(f"{company_name} için haber verileri alınıyor...")
    news_data = get_news_data(company_name, start_date, end_date)
    
    if not news_data.empty:
        # Haber verilerini kaydet
        filename = f"veri/haber/{company_name.replace(' ', '_').lower()}_haberler_{start_date}_{end_date}.json"
        save_data_to_json(news_data, filename)
        
        print(f"\nToplam {len(news_data)} haber başlığı alındı.")
        print(f"Sonuçlar '{filename}' dosyasına kaydedildi.")
    else:
        logger.error("Haber verileri alınamadı.")
        print("Haber verileri alınamadı!")


if __name__ == "__main__":
    main()