#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Veri Ön İşleme Modülü
Bu modül, hisse senedi, endeks ve haber verilerinin temizlenmesi, bütünleştirilmesi ve dönüştürülmesi
için gerekli işlevleri içerir.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import time
import google.generativeai as genai

def clean_data(df, is_text_data=False):
    """
    Veri setini temizler.
    
    Args:
        df (pd.DataFrame): Temizlenecek veri seti
        is_text_data (bool): Metin verisi mi yoksa hisse verisi mi işleniyor
    
    Returns:
        pd.DataFrame: Temizlenmiş veri seti
    """
    if df.empty:
        print("Temizlenecek veri yok.")
        return df
    
    print("Veri temizleme işlemi yapılıyor...")
    
    if is_text_data:
        # Metin verisi temizleme
        # Eksik metin verilerini temizle
        df.dropna(subset=['title'], inplace=True)
        
        # Çok kısa/alakasız metinleri temizle (ör. 5 karakterden kısa başlıklar)
        df = df[df['title'].str.len() > 5]
        
        # Description veya content sütunlarında eksik değerler varsa title ile doldur
        df['description'] = df['description'].fillna(df['title'])
        df['content'] = df['content'].fillna(df['description'])
        
    else:
        # Hisse/endeks verisi temizleme
        # Eksik değerleri tespit et ve doldur (ileri yönde doldurma)
        df = df.fillna(method='ffill')
        
        # Yine de kalan eksik değerler varsa, bunları ortalama ile doldur
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].mean())
    
    return df

def integrate_data(stock_df, news_df):
    """
    Hisse senedi ve haber verilerini tarih bazında birleştirir.
    
    Args:
        stock_df (pd.DataFrame): Hisse senedi verileri
        news_df (pd.DataFrame): Haber verileri
    
    Returns:
        pd.DataFrame: Birleştirilmiş veri seti
    """
    if stock_df.empty or news_df.empty:
        print("Birleştirilecek veri setlerinden en az biri boş.")
        return stock_df
    
    print("Veri bütünleştirme işlemi yapılıyor...")
    
    # Tarih sütununun formatını kontrol et ve düzenle
    if 'Date' in stock_df.columns:
        stock_df['date'] = pd.to_datetime(stock_df['Date']).dt.date
    
    # Tarih sütununu dizine çevir
    stock_data_grouped = stock_df.copy()
    
    # Her tarih için haber verilerini birleştir
    news_grouped = news_df.groupby('date').agg({
        'title': lambda x: ' | '.join(x),
        'description': lambda x: ' | '.join(x),
        'content': lambda x: ' | '.join(x),
        'url': lambda x: ' | '.join(x)
    }).reset_index()
    
    # İki veri setini birleştir
    combined_df = pd.merge(stock_data_grouped, news_grouped, on='date', how='left')
    
    # Haber verisi olmayanlar için NaN değerleri boş dize ile değiştir
    text_cols = ['title', 'description', 'content', 'url']
    for col in text_cols:
        if col in combined_df.columns:
            combined_df[col] = combined_df[col].fillna('')
    
    return combined_df

def transform_data(df, stock_symbol, index_symbol):
    """
    Veri setinden yeni öznitelikler oluşturur.
    
    Args:
        df (pd.DataFrame): Dönüştürülecek veri seti
        stock_symbol (str): Hisse sembolü
        index_symbol (str): Endeks sembolü
    
    Returns:
        pd.DataFrame: Dönüştürülmüş veri seti
    """
    if df.empty:
        print("Dönüştürülecek veri yok.")
        return df
    
    print("Veri dönüştürme işlemi yapılıyor...")
    
    # Sembole ve Açık/Kapanış değerlerine göre verileri filtrele
    stock_df = df[df['Symbol'] == stock_symbol].reset_index(drop=True)
    index_df = df[df['Symbol'] == index_symbol].reset_index(drop=True)
    
    # Her sembol için günlük getiriyi hesapla
    for symbol_df in [stock_df, index_df]:
        symbol_df['daily_return'] = (symbol_df['Close'] - symbol_df['Open']) / symbol_df['Open']
    
    # İlk veri setini temel al ve diğerlerini katıştır
    result_df = stock_df.copy()
    
    # Endeks verilerini ekle
    result_df['index_open'] = index_df['Open'].values
    result_df['index_high'] = index_df['High'].values
    result_df['index_low'] = index_df['Low'].values
    result_df['index_close'] = index_df['Close'].values
    result_df['index_volume'] = index_df['Volume'].values
    result_df['index_daily_return'] = index_df['daily_return'].values
    
    # Göreli performansı hesapla
    result_df['relative_performance'] = result_df['daily_return'] - result_df['index_daily_return']
    
    # Göreli performans kategorisini oluştur
    result_df['relative_perf_category'] = pd.cut(
        result_df['relative_performance'],
        bins=[-float('inf'), -0.01, 0.01, float('inf')],
        labels=['Negatif Sapma', 'Nötr', 'Pozitif Sapma']
    )
    
    return result_df

def process_text_data(df):
    """
    Metin verilerini Google Gemini API kullanarak işler.
    
    Args:
        df (pd.DataFrame): İşlenecek veri seti
    
    Returns:
        pd.DataFrame: Metin analizi yapılmış veri seti
    """
    if df.empty:
        print("İşlenecek metin verisi yok.")
        return df
    
    print("Metin verisi işleniyor (Google Gemini API)...")
    
    # Google Gemini API'yi yapılandır
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("UYARI: GEMINI_API_KEY bulunamadı. Metin analizi yapılamıyor.")
        # Örnek duyarlılık skorları oluştur
        df['sentiment_score'] = np.random.uniform(-1, 1, size=len(df))
        df['sentiment_category'] = pd.cut(
            df['sentiment_score'],
            bins=[-1, -0.33, 0.33, 1],
            labels=['Negatif', 'Nötr', 'Pozitif']
        )
        return df
    
    # Gemini API'yi yapılandır
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-pro')
    
    sentiment_scores = []
    main_topics = []
    
    for idx, row in df.iterrows():
        news_text = ' '.join(filter(None, [row.get('title', ''), row.get('description', '')]))
        
        if not news_text or len(news_text.strip()) < 10:
            # Metin yoksa veya çok kısaysa nötr skor (0) ekle
            sentiment_scores.append(0)
            main_topics.append("")
            continue
        
        #TODO: Gemini API'nin duyarlılık analizi için prompti geliştirlecek alternatif bir yaklaşım olarak ilgili analizleri yapmak için AGENT yapısı kurularak raporlaştırma süreci geliştirilecek.    
        # Duyarlılık analizi için prompt
        sentiment_prompt = f"""
        Aşağıdaki finansal haber metninin genel duyarlılığını (sentiment) analiz et.
        Metni -1 (çok negatif) ile 1 (çok pozitif) arasında bir skor olarak değerlendir.
        Sadece sayısal skoru dön.
        
        Haber Metni: {news_text}
        """
        
        try:
            # API'ye istek gönder
            sentiment_response = model.generate_content(sentiment_prompt)
            sentiment_text = sentiment_response.text.strip()
            
            # Yanıttan sayısal değeri çıkarmaya çalış
            import re
            sentiment_match = re.search(r'(-?\d+(\.\d+)?)', sentiment_text)
            
            if sentiment_match:
                sentiment_score = float(sentiment_match.group(1))
                # Skorun -1 ile 1 arasında olduğundan emin ol
                sentiment_score = max(-1, min(1, sentiment_score))
            else:
                sentiment_score = 0  # Eşleşme bulunamazsa nötr
                
            sentiment_scores.append(sentiment_score)
            
            # Ana konuları çıkarmak için prompt (opsiyonel)
            topics_prompt = f"""
            Aşağıdaki finansal haber metnindeki ana konuları kısa anahtar kelimelerle belirt.
            En fazla 3 konu döndür, virgülle ayır.
            
            Haber Metni: {news_text}
            """
            
            topics_response = model.generate_content(topics_prompt)
            topics_text = topics_response.text.strip()
            main_topics.append(topics_text)
            
            # API rate limitini aşmamak için kısa bekle
            time.sleep(1)
            
        except Exception as e:
            print(f"API hatası (satır {idx}): {e}")
            sentiment_scores.append(0)  # Hata durumunda nötr skor
            main_topics.append("")
    
    # Duyarlılık skorlarını ve konuları DataFrame'e ekle
    df['sentiment_score'] = sentiment_scores
    df['main_topics'] = main_topics
    
    # Duyarlılık kategorisini oluştur
    df['sentiment_category'] = pd.cut(
        df['sentiment_score'],
        bins=[-1, -0.33, 0.33, 1],
        labels=['Negatif', 'Nötr', 'Pozitif']
    )
    
    return df 