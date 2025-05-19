#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Veri Toplama Modülü.

Bu modül, hisse senedi verilerini ve haberlerini toplamak için gerekli işlevleri içerir.
Haber verilerine opsiyonel sentiment (duygu) analizi skorları ekler.
"""

import logging
import os
import random
import sys
import time
import requests
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional, Union, Tuple

# Yeni importlar
import math
import functools
import warnings
import numpy as np

# Veri_Alma klasörünü path'e ekleyerek modül importunu düzelt
# Şu anki çalışma dizinini modül yoluna ekle
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import nltk
import nltk.corpus
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Göreli import kullanarak dosyadan import
try:
    from HisseVerisiAlma import get_stock_data
except ImportError:
    try:
        # Alternatif import yolu
        from Veri_Alma.HisseVerisiAlma import get_stock_data
    except ImportError:
        # Hiçbir şekilde import edilemezse logu bildir
        logging.error("HisseVerisiAlma modülü import edilemedi. get_stock_data fonksiyonu kullanılamayacak.")
        
        # Yedek olarak boş bir fonksiyon tanımla
        def get_stock_data(stock_symbol, index_symbol, start_date, end_date):
            """Yedek (dummy) hisse verisi alma fonksiyonu."""
            logging.error("HisseVerisiAlma modülü yüklenemedi, hisse verileri alınamıyor.")
            return pd.DataFrame()

# NLTK gerekli dosyaları indir (eğer yoksa)
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# .env dosyasını yükle
load_dotenv()

# Loglama yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("veri_toplama.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Yahoo Finance uyarılarını bastır
warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")
warnings.filterwarnings("ignore", category=UserWarning, module="yfinance")
yf.pdr_override()  # pandas_datareader'ı yfinance ile değiştir

# Yeniden deneme dekoratörü ekle
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


def format_stock_symbol(symbol: str, exchange: str = "") -> str:
    """
    Hisse senedi sembolünü doğru formata dönüştürür.
    
    Args:
        symbol: Dönüştürülecek sembol (ör. "AAPL", "THYAO")
        exchange: Borsa/Pazar bilgisi (ör. "NASDAQ", "BIST")
    
    Returns:
        Yahoo Finance için düzgün formatlı sembol
    """
    symbol = symbol.upper().strip()
    
    # Sembol zaten formatlıysa (örn. "AAPL.US", "THYAO.IS")
    if "." in symbol:
        return symbol
    
    # Borsa bilgisine göre sembol formatı
    if exchange:
        exchange = exchange.upper().strip()
        
        # Türkiye Borsası
        if exchange in ["BIST", "BORSA ISTANBUL", "BORSA İSTANBUL", "XU100"]:
            return f"{symbol}.IS"
        # ABD Borsaları (özel bir uzantı gerektirmez)
        elif exchange in ["NASDAQ", "NYSE", "US", "USA", "AMERICA"]:
            return symbol
        
    # Yaygın sembollerin varsayılan formatlamaları
    common_symbols = {
        "THYAO": "THYAO.IS",
        "ASELS": "ASELS.IS",
        "ISCTR": "ISCTR.IS",
        "GARAN": "GARAN.IS"
    }
    
    return common_symbols.get(symbol, symbol)


def format_index_symbol(index_name: str) -> str:
    """
    Endeks adını Yahoo Finance için doğru sembol formatına dönüştürür.
    
    Args:
        index_name: Endeks adı veya sembolü
    
    Returns:
        Yahoo Finance için düzgün formatlı endeks sembolü
    """
    index_name = index_name.upper().strip()
    
    # Endeks sembolleri sözlüğü
    index_map = {
        # ABD Endeksleri
        "NASDAQ": "^IXIC",
        "NASDQ": "^IXIC",  # Yaygın yazım hatası
        "S&P": "^GSPC",
        "S&P500": "^GSPC",
        "S&P 500": "^GSPC",
        "SP500": "^GSPC",
        "DOW": "^DJI",
        "DOWJONES": "^DJI",
        "DOW JONES": "^DJI",
        
        # Türkiye
        "BIST": "XU100.IS",
        "BIST100": "XU100.IS",
        "XU100": "XU100.IS",
        "BORSA ISTANBUL": "XU100.IS",
        
        # Almanya
        "DAX": "^GDAXI",
        
        # İngiltere
        "FTSE": "^FTSE",
        "FTSE100": "^FTSE",
        
        # Japonya
        "NIKKEI": "^N225",
        
        # Diğer
        "VIX": "^VIX"
    }
    
    return index_map.get(index_name, index_name)


class NewsApiClient:
    """
    NewsAPI için basit bir istemci sınıfı.
    
    Not: Bu sınıf, 'newsapi-python' paketi yüklenemediğinde veya sorun olduğunda
    kullanılır. Tam işlevsellik için 'pip install newsapi-python' komutunu
    çalıştırarak resmi kütüphaneyi yüklemeniz önerilir.
    """
    
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
def check_symbol_exists(symbol: str) -> Tuple[bool, str]:
    """
    Yahoo Finance üzerinde bir sembolün varlığını kontrol eder.
    Yeniden deneme mekanizması ile güçlendirilmiştir.
    
    Args:
        symbol: Kontrol edilecek sembol
    
    Returns:
        (var_mı, bilgi_mesajı) içeren tuple
    """
    try:
        logger.info(f"'{symbol}' sembolü Yahoo Finance'de kontrol ediliyor...")
        ticker = yf.Ticker(symbol)
        
        # İlk olarak geçerli bilgi var mı kontrol et
        try:
            # Daha az veri gerektiren basit bir sorgulama
            history = ticker.history(period="1d")
            if not history.empty:
                return True, f"'{symbol}' sembolü doğrulandı (veri var)"
        except Exception:
            pass

        # Tam bilgi almayı dene
        try:
            info = ticker.info
            if info and 'symbol' in info:
                return True, f"'{symbol}' sembolü doğrulandı: {info.get('shortName', 'Bilinmeyen Şirket')}"
        except Exception:
            pass
        
        return False, f"'{symbol}' sembolü bulunamadı veya geçerli değil."
    except Exception as e:
        return False, f"'{symbol}' sembolü doğrulanamadı: {str(e)}"


@retry_with_backoff(max_retries=7)
def get_direct_stock_data(
    stock_symbol: str, 
    index_symbol: str, 
    start_date: str, 
    end_date: str
) -> pd.DataFrame:
    """
    Yahoo Finance API'yi doğrudan kullanarak hisse senedi verilerini getirir.
    Rate limit sorunlarına karşı yeniden deneme mekanizması içerir.
    
    Args:
        stock_symbol: Hisse sembolü (ör. "AAPL")
        index_symbol: Endeks sembolü (ör. "^IXIC")
        start_date: Başlangıç tarihi (YYYY-MM-DD formatında)
        end_date: Bitiş tarihi (YYYY-MM-DD formatında)
    
    Returns:
        Hisse senedi ve endeks verilerini içeren DataFrame.
        Hata durumunda boş DataFrame döner.
    """
    logger.info(f"Hisse ve endeks verileri doğrudan alınıyor: {stock_symbol}, {index_symbol}")
    
    # Tarihleri parse et
    try:
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
    except ValueError as e:
        logger.error(f"Geçersiz tarih formatı: {e}")
        return pd.DataFrame()
    
    try:
        # Önce ayrı ayrı verileri almayı dene
        stock_data = None
        index_data = None
        
        # Hisse verisi al
        try:
            logger.info(f"{stock_symbol} için veri indiriliyor...")
            stock_data = yf.download(stock_symbol, start=start, end=end, progress=False)
            time.sleep(1)  # Rate limit için bekle
        except Exception as e:
            logger.error(f"{stock_symbol} verisi alınamadı: {e}")
        
        # Endeks verisi al
        try:
            logger.info(f"{index_symbol} için veri indiriliyor...")
            index_data = yf.download(index_symbol, start=start, end=end, progress=False)
            time.sleep(1)  # Rate limit için bekle
        except Exception as e:
            logger.error(f"{index_symbol} verisi alınamadı: {e}")
        
        # Her iki veri de boşsa boş DataFrame döndür
        if (stock_data is None or stock_data.empty) and (index_data is None or index_data.empty):
            logger.error("Hem hisse hem de endeks verileri alınamadı.")
            return pd.DataFrame()
        
        # Sonuçları birleştir
        result_dfs = []
        
        # Hisse verileri
        if stock_data is not None and not stock_data.empty:
            stock_df = stock_data.copy()
            stock_df["Symbol"] = stock_symbol
            stock_df = stock_df.reset_index()
            result_dfs.append(stock_df)
        
        # Endeks verileri
        if index_data is not None and not index_data.empty:
            index_df = index_data.copy()
            index_df["Symbol"] = index_symbol
            index_df = index_df.reset_index()
            result_dfs.append(index_df)
        
        # Sonuçları birleştir
        if result_dfs:
            final_df = pd.concat(result_dfs, ignore_index=True)
            logger.info(f"Toplam {len(final_df)} satır veri alındı.")
            return final_df
        else:
            return pd.DataFrame()
        
    except Exception as e:
        logger.error(f"Hisse verisi alınırken beklenmeyen hata: {e}")
        return pd.DataFrame()


@retry_with_backoff()
def get_news_data(
    company_name: str, start_date: str, end_date: str, analyze_sentiment: bool = True
) -> pd.DataFrame:
    """
    NewsAPI kullanarak şirketle ilgili haberleri getirir.
    Rate limit sorunlarına karşı yeniden deneme mekanizması içerir.
    
    Args:
        company_name: Haberleri aranacak şirketin adı (ör. "THYAO")
        start_date: Haber arama başlangıç tarihi (YYYY-MM-DD formatında)
        end_date: Haber arama bitiş tarihi (YYYY-MM-DD formatında)
        analyze_sentiment: Sentiment analizi yapılıp yapılmayacağı (varsayılan: True)
    
    Returns:
        Haber makaleleri ve sentiment skorlarını içeren pandas DataFrame.
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
        
        # API anahtarı yoksa kullanıcıdan iste
        if not news_api_key:
            logger.warning("NEWS_API_KEY bulunamadı. Alternatif haber kaynağı kullanılacak.")
            # Alternatif bir haber kaynağı kullanmayı dene
            return get_alternative_news_data(company_name, start_date, end_date, analyze_sentiment)
            
        logger.info(f"{company_name} ile ilgili haberler toplanıyor...")
        
        # NewsApiClient sınıfını başlat
        try:
            newsapi = NewsApiClient(api_key=news_api_key)
            logger.info("NewsApiClient başarıyla başlatıldı.")
        except Exception as e:
            logger.error(f"NewsApiClient başlatılırken hata: {e}")
            return get_alternative_news_data(company_name, start_date, end_date, analyze_sentiment)
        
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
            time.sleep(2)  # Daha uzun bekleme süresi
        
        # Sonuçları DataFrame'e dönüştür
        if all_articles:
            news_df = pd.DataFrame(all_articles)
            
            # 'publishedAt' sütunu olup olmadığını kontrol et ve düzenle
            if 'publishedAt' in news_df.columns:
                # publishedAt'i datetime'a dönüştür (API ISO format döndürür: 2023-05-01T14:30:00Z)
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
                
                # İş saatleri içinde rastgele saatler atama (daha gerçekçi)
                hours = np.random.randint(9, 18, size=len(news_df))  # 9:00 - 17:59
                minutes = np.random.randint(0, 60, size=len(news_df))
                
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
            
            # Sentiment analizi yap (istenirse)
            if analyze_sentiment:
                news_df = add_sentiment_scores(news_df)
                
            return news_df
        else:
            logger.warning("Haber bulunamadı, alternatif kaynak deneniyor.")
            return get_alternative_news_data(company_name, start_date, end_date, analyze_sentiment)
            
    except Exception as e:
        logger.error(f"Haber verisi alınırken hata oluştu: {e}")
        return get_alternative_news_data(company_name, start_date, end_date, analyze_sentiment)


def get_alternative_news_data(
    company_name: str, start_date: str, end_date: str, analyze_sentiment: bool = True
) -> pd.DataFrame:
    """
    NewsAPI yerine alternatif bir haber kaynağı kullanarak haber verileri oluşturur.
    NewsAPI'ye erişilemediğinde veya API anahtarı olmadığında kullanılır.
    
    Bu fonksiyon basit ve sentetik haber verileri oluşturur.
    Gerçek projede RSS beslemeleri, web scraping veya başka bir API'lar kullanılabilir.
    
    Args:
        company_name: Haberleri aranacak şirketin adı
        start_date: Başlangıç tarihi (YYYY-MM-DD formatında)
        end_date: Bitiş tarihi (YYYY-MM-DD formatında)
        analyze_sentiment: Sentiment analizi yapılıp yapılmayacağı
        
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
            # Örneğin, çeyrek finansal sonuçlar veya önemli duyurular
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
                "title_template": f"{company_name} announces positive quarterly results",
                "desc_template": f"{company_name} reported quarterly earnings that exceeded analyst expectations.",
                "content_template": f"{company_name} has announced its quarterly financial results, reporting earnings per share of $X.XX, which exceeded analyst expectations of $X.XX. Revenue reached $XX.X billion, marking a X% increase from the previous year. The company attributed this growth to strong performance in its core business segments. CEO commented, 'We are pleased with our performance this quarter and remain optimistic about our future prospects.'",
                "sentiment": "positive"
            },
            {
                "title_template": f"{company_name} stock falls on market uncertainty",
                "desc_template": f"Shares of {company_name} declined amid broader market concerns.",
                "content_template": f"Shares of {company_name} fell X% today as investors reacted to broader market uncertainties and economic concerns. Analysts suggest this decline may be temporary, with some viewing it as a potential buying opportunity. Market volatility continues to impact the entire sector.",
                "sentiment": "negative"
            },
            {
                "title_template": f"{company_name} announces new strategic partnership",
                "desc_template": f"{company_name} forms alliance with industry leader to expand market reach.",
                "content_template": f"{company_name} has announced a new strategic partnership aimed at expanding its market presence and enhancing its product offerings. This collaboration is expected to drive innovation and create new opportunities for growth. The company's CEO stated, 'This partnership represents a significant milestone in our strategy to deliver greater value to our customers and shareholders.'",
                "sentiment": "positive"
            },
            {
                "title_template": f"{company_name} faces regulatory scrutiny",
                "desc_template": f"Regulators announce investigation into {company_name}'s business practices.",
                "content_template": f"{company_name} is facing increased regulatory scrutiny as authorities announce an investigation into certain aspects of its business practices. The company has stated it is cooperating fully with investigators and believes its operations comply with all applicable regulations. Some analysts suggest this development could create uncertainty for investors in the short term.",
                "sentiment": "negative"
            },
            {
                "title_template": f"{company_name} maintains stable outlook despite industry challenges",
                "desc_template": f"{company_name} continues to perform steadily amid sector-wide pressures.",
                "content_template": f"Despite facing industry-wide challenges, {company_name} maintains a stable outlook, according to recent analyst reports. The company has implemented strategic initiatives to navigate current market conditions and position itself for long-term growth. Management remains focused on operational efficiency and innovation.",
                "sentiment": "neutral"
            },
            # Ek haber şablonları
            {
                "title_template": f"{company_name} introduces innovative new product line",
                "desc_template": f"New product launch expected to drive future growth for {company_name}.",
                "content_template": f"{company_name} today unveiled its latest product line, which analysts say could significantly impact the company's revenue stream in coming quarters. The new offerings are designed to meet evolving consumer demands and leverage cutting-edge technology. Market reaction has been generally positive, with shares rising X% following the announcement.",
                "sentiment": "positive"
            },
            {
                "title_template": f"{company_name} CEO discusses future direction in industry interview",
                "desc_template": f"Leadership insights reveal strategic vision for {company_name}'s market position.",
                "content_template": f"In a recent industry interview, the CEO of {company_name} outlined the company's strategic priorities and responded to questions about competitive challenges. The executive emphasized the importance of innovation and operational excellence, while acknowledging the dynamic nature of the market. Analyst reactions were mixed, with some praising the clarity of vision while others questioned aspects of the execution strategy.",
                "sentiment": "neutral"
            },
            {
                "title_template": f"Analyst downgrades {company_name} citing sector headwinds",
                "desc_template": f"Major firm revises outlook for {company_name} stock amid industry challenges.",
                "content_template": f"A prominent Wall Street firm has downgraded {company_name}'s stock from 'buy' to 'hold', citing concerns about industry-wide challenges and potential margin pressure. The analyst report highlighted several risk factors, including supply chain constraints and increasing competition. The stock declined X% following the publication of the analysis.",
                "sentiment": "negative"
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
        
        # Sentiment analizi yap (istenirse)
        if analyze_sentiment:
            news_df = add_sentiment_scores(news_df)
        else:
            # Şablonlardan sentiment bilgisini ekle
            sentiment_map = {"positive": 0.5, "negative": -0.5, "neutral": 0.0}
            for template in news_templates:
                mask = news_df["title"].str.contains(template["title_template"].split()[0])
                news_df.loc[mask, "sentiment_compound"] = sentiment_map[template["sentiment"]]
            
            # Eksik sütunları ekle
            if "sentiment_compound" not in news_df.columns:
                news_df["sentiment_compound"] = 0.0
            news_df["sentiment_pos"] = news_df["sentiment_compound"].apply(lambda x: max(0, x))
            news_df["sentiment_neg"] = news_df["sentiment_compound"].apply(lambda x: max(0, -x))
            news_df["sentiment_neu"] = 1.0 - news_df["sentiment_pos"] - news_df["sentiment_neg"]
            news_df["sentiment_label"] = news_df["sentiment_compound"].apply(
                lambda x: "pozitif" if x > 0.05 else ("negatif" if x < -0.05 else "nötr")
            )
        
        # Dosyaya kaydet (ileride kullanmak için)
        os.makedirs(os.path.dirname(saved_file_path), exist_ok=True)
        news_df.to_csv(saved_file_path, index=False)
        logger.info(f"Sentetik haber verileri gelecekte kullanılmak üzere kaydedildi: {saved_file_path}")
        
        return news_df
        
    except Exception as e:
        logger.error(f"Alternatif haber verisi oluşturma hatası: {e}")
        # Boş DataFrame döndür
        return pd.DataFrame()


def add_sentiment_scores(news_df: pd.DataFrame) -> pd.DataFrame:
    """
    Haber verilerine VADER sentiment analizi skorları ekler.
    
    NLTK ve VADER kullanarak haber metinlerindeki duygu durumunu analiz eder.
    
    Args:
        news_df: Haber verilerini içeren DataFrame.
            'title' ve 'description' sütunları olmalıdır.
    
    Returns:
        Sentiment skorları eklenmiş DataFrame.
    """
    logger.info("Haberlere sentiment analizi uygulanıyor...")
    try:
        # VADER analiz aracını başlat
        analyzer = SentimentIntensityAnalyzer()
        
        # Başlık ve açıklama metinlerini birleştir (birisi boşsa diğeri kullanılır)
        def combine_text(row: pd.Series) -> str:
            title = row["title"] or ""
            desc = row["description"] or ""
            if title and desc:
                return f"{title}. {desc}"
            return title or desc or ""
        
        news_df["combined_text"] = news_df.apply(combine_text, axis=1)
        
        # Sentiment skorlarını hesapla
        def get_sentiment(text: str) -> Dict[str, float]:
            if not text:
                return {"pos": 0.0, "neu": 0.0, "neg": 0.0, "compound": 0.0}
            return analyzer.polarity_scores(text)
        
        # Skorları DataFrame'e ekle
        sentiments = news_df["combined_text"].apply(get_sentiment)
        news_df["sentiment_pos"] = sentiments.apply(lambda x: x["pos"])
        news_df["sentiment_neu"] = sentiments.apply(lambda x: x["neu"])
        news_df["sentiment_neg"] = sentiments.apply(lambda x: x["neg"])
        news_df["sentiment_compound"] = sentiments.apply(lambda x: x["compound"])
        
        # Sentiment etiketlerini ekle
        def get_sentiment_label(compound: float) -> str:
            if compound >= 0.05:
                return "pozitif"
            elif compound <= -0.05:
                return "negatif"
            else:
                return "nötr"
        
        news_df["sentiment_label"] = news_df["sentiment_compound"].apply(get_sentiment_label)
        
        # Geçici sütunları temizle
        news_df.drop(columns=["combined_text"], inplace=True)
        
        logger.info("Sentiment analizi tamamlandı.")
        return news_df
        
    except Exception as e:
        logger.error(f"Sentiment analizi sırasında hata oluştu: {e}")
        return news_df  # Sentiment eklenemezse orijinal DataFrame'i döndür


def merge_stock_and_news_data(
    stock_data: pd.DataFrame, news_data: pd.DataFrame, stock_symbol: str
) -> pd.DataFrame:
    """
    Hisse senedi verileri ile haber verilerini birleştirir.
    
    Hisse verilerini ve haberleri tarih bazında eşleştirir, günlük haber
    sentiment skorlarını hesaplar.
    
    Args:
        stock_data: Hisse senedi verilerini içeren DataFrame
        news_data: Haber verilerini içeren DataFrame
        stock_symbol: Filtrelenecek hisse senedi sembolü
    
    Returns:
        Birleştirilmiş veri seti
    """
    logger.info(f"Hisse ve haber verileri birleştiriliyor: {stock_symbol}")
    
    try:
        # Hisse verisi boş mu kontrol et
        if stock_data.empty:
            logger.error("Hisse verileri boş, birleştirme işlemi yapılamadı.")
            return pd.DataFrame()
            
        # Sadece ilgili hisse senedinin verilerini filtrele
        stock_df = stock_data[stock_data["Symbol"] == stock_symbol].copy()
        
        if stock_df.empty:
            logger.warning(f"'{stock_symbol}' sembolü için veri bulunamadı.")
            # Tüm veriyi kullan
            stock_df = stock_data.copy()
        
        # Tarih sütununu birleştirme için hazırla
        if "Date" in stock_df.columns:
            stock_df["date"] = pd.to_datetime(stock_df["Date"]).dt.date
        else:
            logger.error("Hisse verisinde 'Date' sütunu bulunamadı.")
            return pd.DataFrame()
        
        # Haberlerin daha detaylı zaman dilimlerine göre gruplandırılması
        if not news_data.empty:
            logger.info("Haberleri zaman dilimlerine göre gruplama...")
            
            # Daha iyi zaman kontrolü için publishedAt kullan (varsa)
            news_with_time = news_data.copy()
            
            # publishedAt sütunu oluştur (yoksa)
            if "publishedAt" not in news_with_time.columns:
                # Tarihler zaten parse edilmiş olabilir, kontrol et
                if isinstance(news_with_time["date"].iloc[0], str):
                    news_with_time["date"] = pd.to_datetime(news_with_time["date"]).dt.date
                
                # Güniçi saatleri ekle (yoksa)
                business_hours = range(9, 17)  # 9:00 - 16:59
                
                # Haberlere tarih ve saat ata - daha gerçekçi dağılım
                datetime_list = []
                for _, row in news_with_time.iterrows():
                    date = row["date"]
                    hour = random.choice(business_hours)
                    minute = random.randint(0, 59)
                    datetime_val = datetime.combine(date, time(hour, minute))
                    datetime_list.append(datetime_val)
                
                news_with_time["publishedAt"] = datetime_list
            elif isinstance(news_with_time["publishedAt"].iloc[0], str):
                # Eğer publishedAt string ise datetime'a çevir
                news_with_time["publishedAt"] = pd.to_datetime(news_with_time["publishedAt"])
            
            # Tarih bilgisini güncelle
            news_with_time["date"] = news_with_time["publishedAt"].dt.date
            
            # Günün farklı zaman dilimleri için sentiment skorları hesapla
            # Sabah (9:00-12:00), Öğle (12:00-14:00), Öğleden sonra (14:00-17:00)
            def get_time_of_day(hour):
                if 9 <= hour < 12:
                    return "sabah"
                elif 12 <= hour < 14:
                    return "öğle"
                elif 14 <= hour < 17:
                    return "akşam"
                else:
                    return "diğer"
            
            # Zaman dilimi ekle
            news_with_time["time_of_day"] = news_with_time["publishedAt"].dt.hour.apply(get_time_of_day)
            
            # Günlük ve zaman dilimi bazlı sentiment ortalaması al
            if "sentiment_compound" in news_with_time.columns:
                # 1. Günlük genel sentiment ortalaması
                daily_sentiment = news_with_time.groupby("date").agg({
                    "sentiment_compound": "mean",
                    "sentiment_pos": "mean",
                    "sentiment_neg": "mean",
                    "sentiment_neu": "mean",
                    "title": lambda x: "; ".join(x.astype(str))[:500]  # Uzunluğu sınırla
                }).reset_index()
                
                daily_sentiment.rename(columns={
                    "sentiment_compound": "daily_sentiment",
                    "sentiment_pos": "daily_pos",
                    "sentiment_neg": "daily_neg",
                    "sentiment_neu": "daily_neu",
                    "title": "daily_news_titles"
                }, inplace=True)
                
                # 2. Zaman dilimi bazlı sentiment
                time_sentiment = news_with_time.groupby(["date", "time_of_day"]).agg({
                    "sentiment_compound": "mean",
                    "title": lambda x: "; ".join(x.astype(str)[:2])  # Her zaman dilimi için ilk 2 başlık
                }).reset_index()
                
                # Pivot ile günün zaman dilimlerine göre ayrı sütunlar oluştur
                time_pivot = time_sentiment.pivot(
                    index="date", 
                    columns="time_of_day", 
                    values=["sentiment_compound", "title"]
                ).reset_index()
                
                # Sütun isimlerini düzenle
                time_pivot.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in time_pivot.columns]
                
                # Her zaman dilimi için sentiment değeri garanti etmek amacıyla NaN değerleri doldur
                for col in time_pivot.columns:
                    if 'sentiment_compound' in col and col != 'date':
                        time_pivot[col] = time_pivot[col].fillna(0)
                
                # Önce günlük sentiment ile birleştir
                sentiment_df = pd.merge(daily_sentiment, time_pivot, on="date", how="left")
                
                # Sonra hisse verileri ile birleştir
                merged_df = pd.merge(stock_df, sentiment_df, on="date", how="left")
                
            else:
                # Sentiment yoksa sadece haber başlıklarını birleştir
                daily_news = news_with_time.groupby("date").agg({
                    "title": lambda x: "; ".join(x.astype(str))
                }).reset_index()
                
                daily_news.rename(columns={"title": "daily_news_titles"}, inplace=True)
                
                # Hisse verileri ile haberleri birleştir
                merged_df = pd.merge(stock_df, daily_news, on="date", how="left")
        else:
            # Haber yoksa sadece hisse verilerini döndür
            merged_df = stock_df
            logger.warning("Haber verisi bulunamadı. Sadece hisse verileri döndürülüyor.")
        
        # NaN değerleri 0 olarak doldur (sentiment skorları için)
        for col in merged_df.columns:
            if 'sentiment' in col:
                merged_df[col] = merged_df[col].fillna(0)
        
        # Boş haber başlıkları için "Haber yok" değeri ekle
        if 'daily_news_titles' in merged_df.columns:
            merged_df['daily_news_titles'] = merged_df['daily_news_titles'].fillna("Haber yok")
        
        logger.info(f"Veri birleştirme tamamlandı. Sonuç: {len(merged_df)} satır.")
        
        # Sütun sayısını kontrol et ve bildir
        column_count = len(merged_df.columns)
        logger.info(f"Birleştirilmiş veri {column_count} sütun içeriyor.")
        
        return merged_df
        
    except Exception as e:
        logger.error(f"Veri birleştirme sırasında hata: {e}")
        # Hata detaylarını yazdır
        import traceback
        logger.error(traceback.format_exc())
        return stock_data  # Hata durumunda orijinal hisse verilerini döndür


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


def get_stock_data_alternative(
    stock_symbol: str, 
    index_symbol: str, 
    start_date: str, 
    end_date: str
) -> pd.DataFrame:
    """
    Yahoo Finance yerine alternatif kaynaklardan veri almayı dener.
    Yahoo Finance rate-limit aşıldığında kullanılabilir.
    
    Bu örnek implementasyonda, veriyi CSV'den okuma, 
    yerel cache kullanma veya başka bir API'ye yönlendirme yapılabilir.
    
    Args:
        stock_symbol: Hisse sembolü (ör. "AAPL")
        index_symbol: Endeks sembolü (ör. "^IXIC")
        start_date: Başlangıç tarihi (YYYY-MM-DD formatında)
        end_date: Bitiş tarihi (YYYY-MM-DD formatında)
        
    Returns:
        Hisse senedi verilerini içeren DataFrame veya boş DataFrame
    """
    logger.info(f"Alternatif veri kaynakları kullanılıyor: {stock_symbol}, {index_symbol}")
    
    # 1. Adım: Daha önce kaydedilmiş yerel verileri kontrol et
    saved_file_path = os.path.join(
        current_dir, 
        f"veri/hisse/{stock_symbol.replace('^', '').replace('.', '_')}_{start_date}_{end_date}.csv"
    )
    
    if os.path.exists(saved_file_path):
        logger.info(f"Yerel dosyadan veri okunuyor: {saved_file_path}")
        try:
            data = pd.read_csv(saved_file_path)
            if not data.empty:
                logger.info(f"Yerel dosyadan {len(data)} satır veri okundu.")
                return data
        except Exception as e:
            logger.error(f"Yerel dosya okuma hatası: {e}")
    
    # 2. Adım: Örnek veri oluşturma (simüle edilmiş veri)
    # Not: Bu kısım gerçek veri yerine geçici dolgu verisi oluşturur
    # Gerçek projede Alpha Vantage, IEX Cloud vb. başka bir API kullanılabilir
    
    logger.warning("Gerçek veri alınamadı, örnek veri oluşturuluyor.")
    
    try:
        # Simüle edilmiş veri için tarih aralığı oluştur
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        date_range = pd.date_range(start=start, end=end, freq='B')  # İş günleri
        
        # Örnek fiyat verisi oluştur
        stock_price = 150.0  # Örnek bir başlangıç fiyatı
        index_price = 15000.0  # Örnek bir endeks başlangıç değeri
        
        # Fiyat hareketleri için değişimler oluştur
        np.random.seed(42)  # Tekrarlanabilirlik için
        stock_changes = np.random.normal(0.0005, 0.015, len(date_range))  # Ortalama ve std
        index_changes = np.random.normal(0.0003, 0.01, len(date_range))  # Ortalama ve std
        
        # Kümülatif değişim hesapla
        stock_cumulative = np.cumprod(1 + stock_changes)
        index_cumulative = np.cumprod(1 + index_changes)
        
        # Fiyat serileri oluştur
        stock_prices = stock_price * stock_cumulative
        index_prices = index_price * index_cumulative
        
        # Hisse verisi DataFrame'i
        stock_df = pd.DataFrame({
            'Date': date_range,
            'Open': stock_prices * (1 - np.random.uniform(0, 0.005, len(date_range))),
            'High': stock_prices * (1 + np.random.uniform(0, 0.01, len(date_range))),
            'Low': stock_prices * (1 - np.random.uniform(0, 0.01, len(date_range))),
            'Close': stock_prices,
            'Adj Close': stock_prices,
            'Volume': np.random.randint(1000000, 10000000, len(date_range)),
            'Symbol': stock_symbol
        })
        
        # Endeks verisi DataFrame'i
        index_df = pd.DataFrame({
            'Date': date_range,
            'Open': index_prices * (1 - np.random.uniform(0, 0.003, len(date_range))),
            'High': index_prices * (1 + np.random.uniform(0, 0.007, len(date_range))),
            'Low': index_prices * (1 - np.random.uniform(0, 0.007, len(date_range))),
            'Close': index_prices,
            'Adj Close': index_prices,
            'Volume': np.random.randint(500000000, 1000000000, len(date_range)),
            'Symbol': index_symbol
        })
        
        # Verileri birleştir
        combined_df = pd.concat([stock_df, index_df], ignore_index=True)
        
        # Uyarı mesajı
        logger.warning(
            f"DİKKAT: Bu veriler simülasyon verileridir ve gerçek piyasa verilerini yansıtmaz! "
            f"Toplam {len(combined_df)} satır simülasyon verisi oluşturuldu."
        )
        
        # Veriyi kaydet (ileride kullanmak için)
        os.makedirs(os.path.dirname(saved_file_path), exist_ok=True)
        combined_df.to_csv(saved_file_path, index=False)
        logger.info(f"Simülasyon verisi gelecekte kullanılmak üzere kaydedildi: {saved_file_path}")
        
        return combined_df
    
    except Exception as e:
        logger.error(f"Alternatif veri oluşturma hatası: {e}")
        # Boş DataFrame döndür
        return pd.DataFrame()


if __name__ == "__main__":
    try:
        # Gerekli paketleri yükleme kontrolü
        logger.info("Gerekli paketler kontrol ediliyor...")
        try:
            import pkg_resources
            pkg_resources.require('newsapi-python')
            logger.info("newsapi-python paketi yüklü.")
        except (pkg_resources.DistributionNotFound, ImportError):
            logger.warning("newsapi-python paketi yüklü değil.")
            installation_prompt = input("NewsAPI için resmi kütüphaneyi yüklemek ister misiniz? (e/h): ")
            if installation_prompt.lower() in ['e', 'evet', 'y', 'yes']:
                import subprocess
                logger.info("newsapi-python paketi yükleniyor...")
                result = subprocess.run(['pip', 'install', 'newsapi-python'], capture_output=True, text=True)
                if result.returncode == 0:
                    logger.info("newsapi-python paketi başarıyla yüklendi.")
                else:
                    logger.error(f"Paket yükleme hatası: {result.stderr}")
                    logger.info("Dahili NewsApiClient implementasyonu kullanılacak.")
            else:
                logger.info("Dahili NewsApiClient implementasyonu kullanılacak.")
    except Exception as e:
        logger.error(f"Paket kontrolü sırasında hata: {e}")
        logger.info("Dahili NewsApiClient implementasyonu kullanılacak.")
    
    # Gerekli klasörleri oluştur
    for directory in ["veri", "veri/hisse", "veri/haber", "veri/birlestir"]:
        os.makedirs(directory, exist_ok=True)
    
    # Örnek kullanım
    logger.info("Veri toplama işlemi başlıyor...")
    
    # Parametreler - Kullanıcıdan alınabilir veya sabit olabilir
    stock_symbol = "AAPL"  # Apple Inc.
    index_symbol = "^IXIC"  # NASDAQ Composite Index
    company_name = "Apple Inc"  # Haber aramada kullanılacak (İngilizce)
    
    # Parametreleri değiştirmek için kullanıcıdan girdi al (opsiyonel)
    try:
        use_default = input("Varsayılan parametreleri kullanmak ister misiniz? (AAPL, ^IXIC) (e/h): ")
        if use_default.lower() not in ['e', 'evet', 'y', 'yes', '']:
            stock_symbol = input("Hisse sembolünü girin (örn. AAPL, MSFT): ").strip().upper() or "AAPL"
            index_symbol = input("Endeks sembolünü girin (örn. ^IXIC, ^GSPC): ").strip() or "^IXIC"
            company_name = input(f"Şirket adını girin (haber aramada kullanılacak): ").strip() or "Apple Inc"
    except Exception as e:
        logger.error(f"Kullanıcı girdisi alınırken hata: {e}")
    
    # Tarihler - Gelecek tarih kullanımını engelle
    current_date = datetime.now().date()
    
    try:
        start_date_input = input("Başlangıç tarihi (YYYY-MM-DD formatında, varsayılan: 2023-01-01): ").strip()
        start_date = start_date_input if start_date_input else "2023-01-01"
        
        end_date_input = input("Bitiş tarihi (YYYY-MM-DD formatında, varsayılan: 2023-12-31): ").strip()
        end_date_default = "2023-12-31"
        end_date = end_date_input if end_date_input else end_date_default
        
        # Geçerli tarih formatı kontrolü
        datetime.strptime(start_date, "%Y-%m-%d")
        datetime.strptime(end_date, "%Y-%m-%d")
        
        # Gelecek tarih kontrolü
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
        if end_dt > current_date:
            logger.warning(f"Bitiş tarihi ({end_date}) gelecekte. Bugünün tarihine ayarlanıyor.")
            end_date = current_date.strftime("%Y-%m-%d")
    except ValueError as e:
        logger.error(f"Geçersiz tarih formatı: {e}")
        logger.warning("Varsayılan tarihler kullanılıyor: 2023-01-01 - 2023-12-31")
        start_date = "2023-01-01"
        end_date = min(datetime.strptime("2023-12-31", "%Y-%m-%d").date(), current_date).strftime("%Y-%m-%d")
    
    logger.info(f"Kullanılacak parametreler: Hisse: {stock_symbol}, Endeks: {index_symbol}")
    logger.info(f"Tarih aralığı: {start_date} - {end_date}")
    
    # Önce sembolleri doğrula
    stock_valid, stock_msg = check_symbol_exists(stock_symbol)
    index_valid, index_msg = check_symbol_exists(index_symbol)
    
    if not stock_valid or not index_valid:
        logger.warning("Sembol format düzeltmesi uygulanıyor...")
        # Sembol formatlarını düzelt
        original_stock = stock_symbol
        original_index = index_symbol
        
        stock_symbol = format_stock_symbol(stock_symbol, "NASDAQ")
        index_symbol = format_index_symbol(index_symbol)
        
        logger.info(f"Sembol düzeltmesi: {original_stock} -> {stock_symbol}")
        logger.info(f"Endeks düzeltmesi: {original_index} -> {index_symbol}")
        
        # Tekrar doğrula
        stock_valid, stock_msg = check_symbol_exists(stock_symbol)
        index_valid, index_msg = check_symbol_exists(index_symbol)
        
        logger.info(stock_msg)
        logger.info(index_msg)
    
    # Hisse senedi verilerini alma
    logger.info("Hisse senedi verileri alınıyor...")
    
    # Önce HisseVerisiAlma modülü ile deneyin
    try:
        logger.info("HisseVerisiAlma modülü kullanılarak veriler alınıyor...")
        stock_data = get_stock_data(stock_symbol, index_symbol, start_date, end_date)
        
        # Veriler boş ise doğrudan yedek fonksiyonu kullan
        if stock_data.empty:
            logger.warning("HisseVerisiAlma modülü ile veri alınamadı, doğrudan yöntem deneniyor...")
            stock_data = get_direct_stock_data(stock_symbol, index_symbol, start_date, end_date)
    except Exception as e:
        logger.error(f"HisseVerisiAlma modülü ile hata: {e}")
        logger.info("Doğrudan veri alma yöntemi deneniyor...")
        stock_data = get_direct_stock_data(stock_symbol, index_symbol, start_date, end_date)
    
    # Hala veri alınamadıysa alternatif kaynak dene
    if stock_data.empty:
        logger.warning("Doğrudan yöntemle de veri alınamadı, alternatif kaynak deneniyor...")
        stock_data = get_stock_data_alternative(stock_symbol, index_symbol, start_date, end_date)
    
    if stock_data.empty:
        logger.error("Hisse senedi verileri alınamadı. İşlem sonlandırılıyor.")
    else:
        # Hisse verilerini kaydet
        stock_filename = f"veri/hisse/{stock_symbol.replace('^', '').replace('.', '_')}_{start_date}_{end_date}.json"
        save_data_to_json(stock_data, stock_filename)
        
        # Haber verilerini al (sentiment analizi ile)
        logger.info(f"{company_name} için haber verileri alınıyor...")
        news_data = get_news_data(company_name, start_date, end_date, analyze_sentiment=True)
        
        if not news_data.empty:
            # Haber verilerini kaydet
            news_filename = f"veri/haber/{stock_symbol.replace('^', '').replace('.', '_')}_haberler_{start_date}_{end_date}.json"
            save_data_to_json(news_data, news_filename)
            
            # Hisse ve haber verilerini birleştir
            merged_data = merge_stock_and_news_data(stock_data, news_data, stock_symbol)
            
            if not merged_data.empty:
                # Birleştirilmiş veriyi kaydet
                merged_filename = f"veri/birlestir/{stock_symbol.replace('^', '').replace('.', '_')}_birlesik_{start_date}_{end_date}.json"
                save_data_to_json(merged_data, merged_filename)
                
                logger.info("Veri toplama ve işleme tamamlandı.")
                logger.info(f"Sonuçlar şu dosyalara kaydedildi:")
                logger.info(f"  - Hisse verileri: {stock_filename}")
                logger.info(f"  - Haber verileri: {news_filename}")
                logger.info(f"  - Birleştirilmiş veriler: {merged_filename}")
                
                print("\nİşlem başarıyla tamamlandı!")
                print(f"Toplam {len(stock_data)} satır hisse verisi, {len(news_data)} haber verisi alındı.")
                print(f"Sonuçlar 'veri' klasörüne kaydedildi.")
            else:
                logger.error("Veriler birleştirilemedi.")
        else:
            logger.error("Haber verileri alınamadı.")





