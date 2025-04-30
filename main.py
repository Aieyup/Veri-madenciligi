#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Hisse Senedi ve Borsa Endeksi Analizi Projesi
Bu program, belirlenen bir şirketin hisse senedi performansının ilgili borsa endeksine göre
göreli durumunu analiz eder ve bu analizi haberler gibi metin verilerinden çıkarılan bilgilerle
zenginleştirir.
"""

import os
import pandas as pd
from dotenv import load_dotenv

from data_acquisition import get_stock_data, get_news_data
from data_preprocessing import clean_data, integrate_data, transform_data, process_text_data
from data_mining import exploratory_analysis, pattern_mining, classification_analysis, clustering_analysis
from evaluation import evaluate_classification, evaluate_patterns, evaluate_correlation
from visualization import plot_timeseries, plot_correlation, plot_patterns, plot_classification_results

def main():
    """Ana program akışı."""
    
    # Çevre değişkenlerini yükle (.env dosyasından)
    load_dotenv()
    
    # Parametreleri ayarla
    stock_symbol = "THYAO.IS"  # Örnek: Türk Hava Yolları
    index_symbol = "XU100.IS"  # BIST 100
    start_date = "2022-01-01"
    end_date = "2023-01-01"
    
    print(f"Veri Madenciliği Analiz Projesi: {stock_symbol} vs {index_symbol}")
    print(f"Tarih Aralığı: {start_date} - {end_date}")
    
    # 1. Veri Toplama
    print("\n1. Veri Toplama Aşaması")
    # Hisse senedi ve endeks verileri
    stock_index_df = get_stock_data(stock_symbol, index_symbol, start_date, end_date)
    # Haber verileri
    news_df = get_news_data(stock_symbol.split('.')[0], start_date, end_date)
    
    # 2. Veri Ön İşleme
    print("\n2. Veri Ön İşleme Aşaması")
    # Veri temizleme
    stock_index_df = clean_data(stock_index_df)
    news_df = clean_data(news_df, is_text_data=True)
    
    # Veri bütünleştirme
    combined_df = integrate_data(stock_index_df, news_df)
    
    # Veri dönüştürme (getiri hesaplama, göreli performans vb.)
    transformed_df = transform_data(combined_df, stock_symbol, index_symbol)
    
    # Metin verisi işleme (Google Gemini API ile duyarlılık analizi)
    if os.getenv("GEMINI_API_KEY"):
        final_df = process_text_data(transformed_df)
    else:
        print("UYARI: GEMINI_API_KEY bulunamadı. Metin verisi işleme adımı atlanıyor.")
        final_df = transformed_df
    
    # Veri setini kaydet
    final_df.to_csv('processed_data.csv', index=False)
    print("İşlenmiş veri seti 'processed_data.csv' olarak kaydedildi.")
    
    # 3. Veri Madenciliği ve Analiz
    print("\n3. Veri Madenciliği ve Analiz Aşaması")
    # Keşifsel analiz
    exploratory_results = exploratory_analysis(final_df)
    
    # Desen madenciliği (Birliktelik kuralları)
    pattern_results = pattern_mining(final_df)
    
    # Sınıflandırma
    classification_results = classification_analysis(final_df)
    
    # Kümeleme (opsiyonel)
    clustering_results = clustering_analysis(final_df)
    
    # 4. Değerlendirme
    print("\n4. Değerlendirme Aşaması")
    # Sınıflandırma modelini değerlendir
    classification_evaluation = evaluate_classification(classification_results)
    
    # Desenleri değerlendir
    pattern_evaluation = evaluate_patterns(pattern_results)
    
    # Korelasyonları değerlendir
    correlation_evaluation = evaluate_correlation(exploratory_results)
    
    # 5. Görselleştirme ve Sonuçların Sunumu
    print("\n5. Görselleştirme ve Sonuçların Sunumu")
    # Zaman serisi grafikleri
    plot_timeseries(final_df, stock_symbol, index_symbol)
    
    # Korelasyon matrisini görselleştir
    plot_correlation(exploratory_results)
    
    # Desenleri görselleştir
    plot_patterns(pattern_results)
    
    # Sınıflandırma sonuçlarını görselleştir
    plot_classification_results(classification_evaluation)
    
    print("\nAnaliz tamamlandı.")

if __name__ == "__main__":
    main() 