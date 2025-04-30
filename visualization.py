#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Görselleştirme Modülü
Bu modül, veri madenciliği sonuçlarını görselleştirmek için işlevleri içerir.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator

def plot_timeseries(df, stock_symbol, index_symbol):
    """
    Zaman serisi verilerini görselleştirir.
    
    Args:
        df (pd.DataFrame): Görselleştirilecek veri seti
        stock_symbol (str): Hisse sembolü
        index_symbol (str): Endeks sembolü
    """
    if df.empty:
        print("Görselleştirilecek veri yok.")
        return
    
    print("Zaman serisi grafikleri oluşturuluyor...")
    
    # Tarih sütununu doğru formata dönüştür
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    elif 'Date' in df.columns:
        df['date'] = pd.to_datetime(df['Date'])
    
    # Düzenli tarih dizini olmayan DataFrame
    df_sorted = df.sort_values('date')
    
    # Hisse ve Endeks Kapanış Fiyatları
    plt.figure(figsize=(12, 6))
    plt.plot(df_sorted['date'], df_sorted['Close'], label=f'{stock_symbol} Kapanış')
    plt.plot(df_sorted['date'], df_sorted['index_close'], label=f'{index_symbol} Kapanış')
    plt.title(f'{stock_symbol} ve {index_symbol} Kapanış Fiyatları')
    plt.xlabel('Tarih')
    plt.ylabel('Fiyat')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Tarih formatını ayarla
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gcf().autofmt_xdate()
    
    plt.tight_layout()
    plt.savefig('fiyat_grafigi.png')
    print("Fiyat grafiği 'fiyat_grafigi.png' olarak kaydedildi.")
    
    # Günlük Getiriler
    plt.figure(figsize=(12, 6))
    plt.plot(df_sorted['date'], df_sorted['daily_return'], label=f'{stock_symbol} Getiri')
    plt.plot(df_sorted['date'], df_sorted['index_daily_return'], label=f'{index_symbol} Getiri')
    plt.title(f'{stock_symbol} ve {index_symbol} Günlük Getiriler')
    plt.xlabel('Tarih')
    plt.ylabel('Getiri')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Tarih formatını ayarla
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gcf().autofmt_xdate()
    
    plt.tight_layout()
    plt.savefig('getiri_grafigi.png')
    print("Getiri grafiği 'getiri_grafigi.png' olarak kaydedildi.")
    
    # Göreli Performans
    plt.figure(figsize=(12, 6))
    plt.plot(df_sorted['date'], df_sorted['relative_performance'])
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.title(f'{stock_symbol} Göreli Performans ({index_symbol}\'e göre)')
    plt.xlabel('Tarih')
    plt.ylabel('Göreli Performans')
    plt.grid(True, alpha=0.3)
    
    # Tarih formatını ayarla
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gcf().autofmt_xdate()
    
    plt.tight_layout()
    plt.savefig('goreli_performans_grafigi.png')
    print("Göreli performans grafiği 'goreli_performans_grafigi.png' olarak kaydedildi.")
    
    # Duyarlılık Skoru vs Göreli Performans
    if 'sentiment_score' in df.columns:
        plt.figure(figsize=(12, 6))
        plt.scatter(df_sorted['sentiment_score'], df_sorted['relative_performance'])
        plt.title('Duyarlılık Skoru vs Göreli Performans')
        plt.xlabel('Duyarlılık Skoru')
        plt.ylabel('Göreli Performans')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='r', linestyle='-', alpha=0.3)
        
        # Trend çizgisi ekle
        z = np.polyfit(df_sorted['sentiment_score'], df_sorted['relative_performance'], 1)
        p = np.poly1d(z)
        plt.plot(df_sorted['sentiment_score'], p(df_sorted['sentiment_score']), "r--", alpha=0.8)
        
        plt.tight_layout()
        plt.savefig('duyarlilik_vs_performans.png')
        print("Duyarlılık vs performans grafiği 'duyarlilik_vs_performans.png' olarak kaydedildi.")
    
def plot_correlation(exploratory_results):
    """
    Korelasyon matrisini ısı haritası olarak görselleştirir.
    
    Args:
        exploratory_results (dict): Keşifsel analiz sonuçları
    """
    if not exploratory_results:
        print("Görselleştirilecek korelasyon sonucu yok.")
        return
    
    correlation_matrix = exploratory_results.get('correlation_matrix', None)
    
    if correlation_matrix is None or correlation_matrix.empty:
        print("Korelasyon matrisi bulunamadı.")
        return
    
    print("Korelasyon matrisi görselleştiriliyor...")
    
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', mask=mask, 
                vmin=-1, vmax=1, center=0, square=True, linewidths=.5)
    plt.title('Öznitelikler Arasındaki Korelasyon Matrisi')
    plt.tight_layout()
    plt.savefig('korelasyon_matrisi.png')
    print("Korelasyon matrisi 'korelasyon_matrisi.png' olarak kaydedildi.")
    
    # Duyarlılık skoru ile göreli performans arasındaki korelasyon
    sentiment_rel_perf_corr = exploratory_results.get('sentiment_rel_perf_corr', None)
    
    if sentiment_rel_perf_corr is not None:
        plt.figure(figsize=(8, 6))
        plt.bar(['Duyarlılık - Göreli Performans'], [sentiment_rel_perf_corr], 
                color='skyblue' if sentiment_rel_perf_corr >= 0 else 'salmon')
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.title('Duyarlılık Skoru ve Göreli Performans Korelasyonu')
        plt.ylabel('Korelasyon Katsayısı')
        plt.ylim(-1, 1)
        plt.grid(axis='y', alpha=0.3)
        
        # Korelasyon değerini göster
        plt.text(0, sentiment_rel_perf_corr + (0.05 if sentiment_rel_perf_corr >= 0 else -0.05), 
                 f'{sentiment_rel_perf_corr:.4f}', 
                 ha='center', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('duyarlilik_korelasyonu.png')
        print("Duyarlılık korelasyon grafiği 'duyarlilik_korelasyonu.png' olarak kaydedildi.")

def plot_patterns(pattern_results):
    """
    Desen madenciliği sonuçlarını görselleştirir.
    
    Args:
        pattern_results (dict): Desen madenciliği sonuçları
    """
    if not pattern_results:
        print("Görselleştirilecek desen sonucu yok.")
        return
    
    best_rules = pattern_results.get('best_rules', None)
    
    if best_rules is None or best_rules.empty:
        print("Görselleştirilecek kural bulunamadı.")
        return
    
    print("Desen sonuçları görselleştiriliyor...")
    
    # Kuralları görselleştir
    plt.figure(figsize=(12, 8))
    
    # En iyi 5 kuralı seç
    top_rules = best_rules.head(5)
    
    # Kaldıraç grafiği
    rule_names = [f"{list(row['antecedents'])} => {list(row['consequents'])}" for idx, row in top_rules.iterrows()]
    plt.barh(rule_names, top_rules['lift'], color='skyblue')
    plt.title('En Yüksek Kaldıraç Değerine Sahip Kurallar')
    plt.xlabel('Kaldıraç (Lift) Değeri')
    plt.ylabel('Kurallar')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('desen_kurallari.png')
    print("Desen kuralları grafiği 'desen_kurallari.png' olarak kaydedildi.")
    
    # Güven ve Destek Dağılımı
    plt.figure(figsize=(10, 6))
    rules = pattern_results.get('rules', None)
    
    if rules is not None and not rules.empty:
        plt.scatter(rules['support'], rules['confidence'], alpha=0.5, 
                   c=rules['lift'], cmap='viridis', s=rules['lift']*50)
        plt.colorbar(label='Kaldıraç (Lift)')
        plt.title('Kuralların Destek, Güven ve Kaldıraç Dağılımı')
        plt.xlabel('Destek (Support)')
        plt.ylabel('Güven (Confidence)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('kural_dagilimi.png')
        print("Kural dağılım grafiği 'kural_dagilimi.png' olarak kaydedildi.")

def plot_classification_results(classification_evaluation):
    """
    Sınıflandırma sonuçlarını görselleştirir.
    
    Args:
        classification_evaluation (dict): Sınıflandırma değerlendirme sonuçları
    """
    if not classification_evaluation:
        print("Görselleştirilecek sınıflandırma sonucu yok.")
        return
    
    print("Sınıflandırma sonuçları görselleştiriliyor...")
    
    # Karmaşıklık matrisi
    conf_matrix = classification_evaluation.get('confusion_matrix', None)
    
    if conf_matrix is not None:
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Karmaşıklık Matrisi')
        plt.ylabel('Gerçek Sınıf')
        plt.xlabel('Tahmin Edilen Sınıf')
        plt.tight_layout()
        plt.savefig('karmasiklik_matrisi.png')
        print("Karmaşıklık matrisi 'karmasiklik_matrisi.png' olarak kaydedildi.")
    
    # Sınıflandırma raporu
    class_report = classification_evaluation.get('classification_report', None)
    
    if class_report is not None:
        # Rapordan gerekli bilgileri çıkar
        class_df = pd.DataFrame(class_report).transpose()
        
        # Sınıf bazında metrikler (precision, recall, f1-score)
        plt.figure(figsize=(12, 8))
        
        # Sadece sınıf adlarını içeren satırları seç (mikro, makro, ağırlıklı ortalamaları hariç tut)
        class_metrics = class_df.loc[~class_df.index.isin(['accuracy', 'macro avg', 'weighted avg', 'micro avg'])]
        
        bar_width = 0.25
        positions = np.arange(len(class_metrics))
        
        # Çubuk grafikleri
        plt.bar(positions - bar_width, class_metrics['precision'], bar_width, label='Precision', color='skyblue')
        plt.bar(positions, class_metrics['recall'], bar_width, label='Recall', color='salmon')
        plt.bar(positions + bar_width, class_metrics['f1-score'], bar_width, label='F1-score', color='lightgreen')
        
        plt.title('Sınıflandırma Metrikleri (Sınıf Bazında)')
        plt.xticks(positions, class_metrics.index)
        plt.ylabel('Skor')
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig('siniflandirma_metrikleri.png')
        print("Sınıflandırma metrikleri 'siniflandirma_metrikleri.png' olarak kaydedildi.")
    
    # Öznitelik önemliliği
    feature_importance = classification_evaluation.get('feature_importance', None)
    
    if feature_importance is not None and not feature_importance.empty:
        plt.figure(figsize=(10, 8))
        
        # En önemli 10 özniteliği göster
        top_features = feature_importance.head(10)
        
        # Öznitelik önemliliği grafikleri
        sns.barplot(x='importance', y='feature', data=top_features, palette='viridis')
        plt.title('Öznitelik Önemliliği')
        plt.xlabel('Önem Derecesi')
        plt.ylabel('Öznitelik')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig('oznitelik_onemliligi.png')
        print("Öznitelik önemliliği grafiği 'oznitelik_onemliligi.png' olarak kaydedildi.") 