#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Veri Toplama Modülü
Bu modül, hisse senedi verilerini ve haberlerini toplamak için gerekli işlevleri içerir.
"""

import os
import time
import pandas as pd
import yfinance as yf
from newsapi import NewsApiClient
from datetime import datetime, timedelta

def get_stock_data(stock_symbol, index_symbol, start_date, end_date):
    """
    Yahoo Finance API'yi kullanarak hisse senedi ve endeks verilerini getirir.
    
    Args:
        stock_symbol (str): Hisse sembolü (ör. "THYAO.IS")
        index_symbol (str): Endeks sembolü (ör. "XU100.IS")
        start_date (str): Başlangıç tarihi (YYYY-MM-DD formatında)
        end_date (str): Bitiş tarihi (YYYY-MM-DD formatında)
    
    Returns:
        pd.DataFrame: Hisse senedi ve endeks verilerini içeren DataFrame
    """
    try:
        print(f"Hisse ve endeks verileri alınıyor: {stock_symbol}, {index_symbol}")
        
        # Veriyi indir
        tickers = [stock_symbol, index_symbol]
        data = yf.download(tickers, start=start_date, end=end_date)
        
        # Çoklu sütunları düzleştir
        data = data.stack(level=0).reset_index()
        data.rename(columns={'level_1': 'Symbol'}, inplace=True)
        
        print(f"Toplam {len(data)} satır veri alındı.")
        return data
        
    except Exception as e:
        print(f"Hisse verisi alınırken hata oluştu: {e}")
        return pd.DataFrame()  # Boş DataFrame dön

def get_news_data(company_name, start_date, end_date):
    """
    NewsAPI kullanarak şirketle ilgili haberleri getirir.
    
    Args:
        company_name (str): Şirket adı (ör. "THYAO")
        start_date (str): Başlangıç tarihi (YYYY-MM-DD formatında)
        end_date (str): Bitiş tarihi (YYYY-MM-DD formatında)
    
    Returns:
        pd.DataFrame: Haber verilerini içeren DataFrame
    """
    try:
        # NewsAPI ile etkileşim
        news_api_key = os.getenv("NEWS_API_KEY")
        
        if not news_api_key:
            print("UYARI: NEWS_API_KEY bulunamadı. Sahte haber verileri oluşturuluyor.")
            return generate_sample_news_data(company_name, start_date, end_date)
            
        print(f"{company_name} ile ilgili haberler toplanıyor...")
        
        newsapi = NewsApiClient(api_key=news_api_key)
        
        # NewsAPI 1 ay ile sınırlıdır, bu nedenle tarih aralığını bölmemiz gerekiyor
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        all_articles = []
        current_start = start_dt
        
        # Her ay için ayrı sorgu yap
        while current_start < end_dt:
            current_end = min(current_start + timedelta(days=30), end_dt)
            
            # API sorgusu
            response = newsapi.get_everything(
                q=company_name,
                from_param=current_start.strftime("%Y-%m-%d"),
                to=current_end.strftime("%Y-%m-%d"),
                language='tr',
                sort_by='publishedAt'
            )
            
            if response['status'] == 'ok':
                all_articles.extend(response['articles'])
                
            current_start = current_end
            
            # API sınırlamalarını aşmamak için bekleme
            time.sleep(1)
        
        # Sonuçları DataFrame'e dönüştür
        if all_articles:
            news_df = pd.DataFrame(all_articles)
            news_df['date'] = pd.to_datetime(news_df['publishedAt']).dt.date
            news_df = news_df[['date', 'title', 'description', 'content', 'url']]
            
            print(f"Toplam {len(news_df)} haber makalesi bulundu.")
            return news_df
        else:
            print("Haber bulunamadı, örnek veri oluşturuluyor.")
            return generate_sample_news_data(company_name, start_date, end_date)
            
    except Exception as e:
        print(f"Haber verisi alınırken hata oluştu: {e}")
        print("Örnek haber verileri oluşturuluyor.")
        return generate_sample_news_data(company_name, start_date, end_date)

def generate_sample_news_data(company_name, start_date, end_date):
    """
    Örnek haber verileri oluşturur (API anahtarı yoksa veya hata durumunda).
    
    Args:
        company_name (str): Şirket adı
        start_date (str): Başlangıç tarihi
        end_date (str): Bitiş tarihi
    
    Returns:
        pd.DataFrame: Örnek haber verilerini içeren DataFrame
    """
    print(f"Örnek haber verileri oluşturuluyor: {company_name}")
    
    # Tarih aralığını oluştur
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')  # İş günleri
    
    # Pozitif haber örnekleri
    positive_templates = [
        f"{company_name} güçlü finansal sonuçlar açıkladı",
        f"{company_name} yeni bir pazara giriş yapıyor",
        f"{company_name} hisseleri yükselişe geçti",
        f"Analistler {company_name} için alım tavsiyesi verdi",
        f"{company_name} yeni bir stratejik ortaklık duyurdu"
    ]
    
    # Negatif haber örnekleri
    negative_templates = [
        f"{company_name} kar beklentilerini karşılayamadı",
        f"{company_name} bazı operasyonlarını azaltıyor",
        f"{company_name} hisseleri düşüşe geçti", 
        f"Analistler {company_name} için satış tavsiyesi verdi",
        f"{company_name} ile ilgili düzenleyici soruşturma başlatıldı"
    ]
    
    # Nötr haber örnekleri
    neutral_templates = [
        f"{company_name} yıllık genel kurul toplantısını gerçekleştirdi",
        f"{company_name} yeni bir yönetim kurulu üyesi atadı", 
        f"{company_name} şirket yapısında değişikliğe gidiyor",
        f"{company_name} sektördeki gelişmeleri değerlendiriyor",
        f"{company_name} ile ilgili yeni bir haber yok"
    ]
    
    # Tüm şablonları birleştir
    all_templates = positive_templates + negative_templates + neutral_templates
    
    # Rastgele veri oluştur
    import random
    
    data = []
    for date in date_range:
        # Her gün için 0-3 arası haber oluştur
        num_news = random.randint(0, 3)
        
        for _ in range(num_news):
            headline = random.choice(all_templates)
            description = f"Bu bir örnek haber açıklamasıdır: {headline}"
            content = f"Bu bir örnek haber içeriğidir. {company_name} ile ilgili detaylı bilgi..."
            
            data.append({
                'date': date.date(),
                'title': headline,
                'description': description,
                'content': content,
                'url': 'https://example.com/sample-news'
            })
    
    df = pd.DataFrame(data)
    print(f"Toplam {len(df)} örnek haber oluşturuldu.")
    return df 