#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Duygu Skorlarının Zaman Dilimine Göre Görselleştirilmesi.

Bu script, birleştirilmiş veri setindeki duygu skorlarını sabah, öğle ve akşam olarak
zaman dilimlerine bağlı olarak görselleştirir.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import seaborn as sns

# Varsayılan ayarları ayarla
plt.style.use('ggplot')
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.family'] = 'DejaVu Sans'  # Türkçe karakter desteği için
plt.rcParams['axes.unicode_minus'] = False   # Unicode eksi işareti desteği
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14

def load_json_data(file_path):
    """
    JSON veri dosyasını yükler ve DataFrame'e dönüştürür
    
    Args:
        file_path: JSON dosyasının yolu
        
    Returns:
        pandas DataFrame
    """
    try:
        # Dosyanın varlığını kontrol et
        if not os.path.exists(file_path):
            print(f"Hata: Dosya bulunamadı - {file_path}")
            return None
            
        # JSON dosyasını oku
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            
        # DataFrame'e dönüştür
        df = pd.DataFrame(data)
        
        # Date sütunları datetime'a çevir
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        elif 'Date' in df.columns:
            df['date'] = pd.to_datetime(df['Date'])
            
        print(f"Veri başarıyla yüklendi. Toplam {len(df)} satır.")
        return df
        
    except Exception as e:
        print(f"Veri yüklenirken hata oluştu: {e}")
        return None

def prepare_sentiment_data(df):
    """
    Duygu skorlarını zaman dilimine göre hazırlar
    
    Args:
        df: Birleştirilmiş veri içeren DataFrame
        
    Returns:
        Hazırlanmış DataFrame
    """
    # Yeni düzenlememizden sonra veri setinde olması beklenen sütunlar
    time_columns = {
        'Sabah': 'sentiment_compound_sabah', 
        'Öğle': 'sentiment_compound_öğle', 
        'Akşam': 'sentiment_compound_akşam'
    }
    
    # Veri tiplerini kontrol et ve title sütunlarını hariç tut
    title_columns = []
    for col in df.columns:
        if 'title_' in col or col == 'daily_news_titles':
            title_columns.append(col)
            
    # Eski format için uyumluluk (yoksa eski format olabilir)
    if not all(col in df.columns for col in time_columns.values()):
        print("Yeni format sütunlar bulunamadı, alternatif sütunlar kontrol ediliyor...")
        
        # Alternatif sütun isimleri
        alt_columns = {
            'Sabah': 'sentiment_compound_sabah',
            'Öğle': 'sentiment_compound_ogle',
            'Akşam': 'sentiment_compound_aksam'
        }
        
        # İngilizce format
        eng_columns = {
            'Sabah': 'sentiment_compound_morning',
            'Öğle': 'sentiment_compound_noon',
            'Akşam': 'sentiment_compound_evening'
        }
        
        # Kontrol et ve kullan
        if all(col in df.columns for col in alt_columns.values()):
            time_columns = alt_columns
            print("Alternatif Türkçe format kullanılıyor.")
        elif all(col in df.columns for col in eng_columns.values()):
            time_columns = eng_columns
            print("İngilizce format kullanılıyor.")
        else:
            # Hiçbir zaman dilimi sütunu yoksa, sadece genel sentiment kullan
            print("Zaman dilimi sütunları bulunamadı. Günlük sentiment verisi kullanılıyor.")
            
            if 'sentiment_compound' in df.columns:
                # Sadece günlük sentiment varsa, üç zaman dilimi için eşit dağıt
                df['sentiment_compound_sabah'] = df['sentiment_compound']
                df['sentiment_compound_öğle'] = df['sentiment_compound']
                df['sentiment_compound_akşam'] = df['sentiment_compound']
                
                time_columns = {
                    'Sabah': 'sentiment_compound_sabah',
                    'Öğle': 'sentiment_compound_öğle',
                    'Akşam': 'sentiment_compound_akşam'
                }
            elif 'daily_sentiment' in df.columns:
                # Yeni format günlük sentiment
                df['sentiment_compound_sabah'] = df['daily_sentiment']
                df['sentiment_compound_öğle'] = df['daily_sentiment']
                df['sentiment_compound_akşam'] = df['daily_sentiment']
                
                time_columns = {
                    'Sabah': 'sentiment_compound_sabah',
                    'Öğle': 'sentiment_compound_öğle',
                    'Akşam': 'sentiment_compound_akşam'
                }
            else:
                print("Duygu skoru sütunları bulunamadı! İşlem yapılamıyor.")
                return None
    
    # Tüm sütunları görüntüle (debug için)
    print("Mevcut veri sütunları:", ", ".join(df.columns))
    print("Kullanılacak zaman dilimi sütunları:", time_columns)
    
    # NaN değerleri 0 ile doldur
    for col in time_columns.values():
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    # Metin içeren sütunları listele
    print(f"Metin sütunları (ortalama hesaplamasına dahil edilmeyecek): {title_columns}")
    
    return df, time_columns, title_columns

def plot_sentiment_over_time(df, time_columns, title_columns, output_path=None):
    """
    Duygu skorlarını zaman dilimlerine göre çizdirir
    
    Args:
        df: Veri içeren DataFrame
        time_columns: Zaman dilimi sütunlarını içeren sözlük
        title_columns: Metin içeren sütunların listesi
        output_path: Kaydedilecek dosya yolu (opsiyonel)
    """
    try:
        fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
        fig.suptitle('AAPL Hissesi için Duygu Skorlarının Zamana Bağlı Dağılımı', fontsize=18)
        
        colors = ['#3498db', '#2ecc71', '#e74c3c']  # Mavi, Yeşil, Kırmızı
        
        # Date sütununu temizle ve doğru formata getir
        if 'date' in df.columns:
            # Eğer 'date' sütunu varsa ve düzgün datetime formatında değilse düzelt
            if not pd.api.types.is_datetime64_dtype(df['date']):
                print("'date' sütunu datetime formatına dönüştürülüyor...")
                try:
                    df['date'] = pd.to_datetime(df['date'])
                except Exception as e:
                    print(f"Tarih dönüşümü hatası: {e}, alternatif yöntem uygulanıyor")
                    # Date ve Date sütunlarını kontrol et
                    if 'Date' in df.columns:
                        df['date'] = pd.to_datetime(df['Date'])
        
        # Haftalık örnekleme fonksiyonu
        def weekly_resample(data):
            try:
                # Sadece sayısal sütunları seç
                numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                numeric_data = data[['date'] + numeric_cols].copy()
                
                # Indeks olarak tarih kullan
                numeric_data = numeric_data.set_index('date')
                
                # Haftalık örnekleme
                weekly_data = numeric_data.resample('W').mean().reset_index()
                print(f"Haftalık örnekleme başarılı: {len(weekly_data)} hafta")
                return weekly_data
            except Exception as e:
                print(f"Haftalık örnekleme başarısız: {e}")
                return None
        
        # Haftalık veriler
        df_resampled = weekly_resample(df)
        
        # Her zaman dilimi için çizdir
        for i, (time_period, column) in enumerate(time_columns.items()):
            if column in df.columns:
                # Orijinal günlük veri
                axes[i].plot(df['date'], df[column], 
                          alpha=0.4, color=colors[i], label='Günlük')
                
                # Haftalık ortalama (eğer başarılıysa)
                if df_resampled is not None and column in df_resampled.columns:
                    axes[i].plot(df_resampled['date'], df_resampled[column], 
                            linewidth=2.5, color=colors[i], label='Haftalık Ortalama')
                
                # Sıfır çizgisi
                axes[i].axhline(y=0, color='gray', linestyle='--', alpha=0.7)
                
                # Pozitif ve negatif bölgeler
                axes[i].fill_between(df['date'], df[column], 0, 
                                where=(df[column] >= 0), alpha=0.3, color='green', interpolate=True)
                axes[i].fill_between(df['date'], df[column], 0, 
                                where=(df[column] < 0), alpha=0.3, color='red', interpolate=True)
                
                axes[i].set_title(f'{time_period} Duygu Skoru')
                axes[i].set_ylabel('Sentiment Değeri')
                axes[i].legend(loc='upper right')
                
                # Eksen düzenlemeleri
                axes[i].xaxis.set_major_locator(mdates.MonthLocator())
                axes[i].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
                axes[i].set_ylim(-1.1, 1.1)  # Sentiment değerleri -1 ile 1 arasında
        
        # x-ekseni ayarları
        plt.xlabel('Tarih')
        plt.xticks(rotation=45)
        plt.tight_layout(rect=[0, 0, 1, 0.97])  # Suptitle için alan bırak
        
        # Eğer çıktı yolu belirtildiyse kaydet
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Görsel başarıyla kaydedildi: {output_path}")
        
        plt.show()
    except Exception as e:
        print(f"Görselleştirme sırasında hata: {e}")
        import traceback
        traceback.print_exc()

def plot_sentiment_heatmap(df, time_columns, title_columns, output_path=None):
    """
    Duygu skorlarını ısı haritası olarak görselleştirir
    
    Args:
        df: Veri içeren DataFrame
        time_columns: Zaman dilimi sütunlarını içeren sözlük
        title_columns: Metin içeren sütunların listesi
        output_path: Kaydedilecek dosya yolu (opsiyonel)
    """
    try:
        # Date sütununu temizle ve doğru formata getir
        if 'date' in df.columns:
            # Eğer 'date' sütunu varsa ve düzgün datetime formatında değilse düzelt
            if not pd.api.types.is_datetime64_dtype(df['date']):
                try:
                    df['date'] = pd.to_datetime(df['date'])
                except Exception as e:
                    print(f"Tarih dönüşümü hatası: {e}")
                    if 'Date' in df.columns:
                        df['date'] = pd.to_datetime(df['Date'])
        
        # Sadece ilgilendiğimiz sütunları seçelim (tarih ve sentiment sütunları)
        sentiment_cols = [col for col in time_columns.values() if col in df.columns]
        plot_df = df[['date'] + sentiment_cols].copy()
        
        # Tarihi ay formatına dönüştür
        plot_df['month'] = plot_df['date'].dt.strftime('%Y-%m')
        
        # Aylık ortalama hesapla (manuel olarak groupby kullanarak)
        monthly_avg = plot_df.groupby('month')[sentiment_cols].mean().reset_index()
        print(f"Aylık ortalamaları hesaplandı: {len(monthly_avg)} ay")
        
        # Isı haritası için veriyi yeniden şekillendir
        # MultiIndex kullanma, bunun yerine pivot_table kullan
        try:
            plot_data = monthly_avg.pivot_table(
                index='month', 
                values=sentiment_cols
            )
            
            # Sütun isimlerini değiştir
            column_mapping = {v: k for k, v in time_columns.items() if v in sentiment_cols}
            plot_data.columns = [column_mapping.get(col, col) for col in plot_data.columns]
            
            # Tarihleri sırala
            plot_data = plot_data.sort_index()
            
            # Isı haritası çizimi
            plt.figure(figsize=(14, 8))
            ax = sns.heatmap(plot_data.T, cmap="RdBu_r", vmin=-1, vmax=1, center=0,
                            annot=True, fmt=".2f", linewidths=.5,
                            cbar_kws={'label': 'Duygu Skoru'})
            
            # Eksenleri ayarla
            ax.set_title('AAPL Hissesi için Aylık Duygu Skoru Isı Haritası', fontsize=16)
            ax.set_xlabel('Tarih', fontsize=14)
            ax.set_ylabel('Zaman Dilimi', fontsize=14)
            
            # X ekseni formatını ayarla
            plt.tight_layout()
            
            # Eğer çıktı yolu belirtildiyse kaydet
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                print(f"Isı haritası başarıyla kaydedildi: {output_path}")
            
            plt.show()
        except Exception as e:
            print(f"Isı haritası çizimi sırasında hata: {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"Isı haritası oluşturma sırasında hata: {e}")
        import traceback
        traceback.print_exc()

def main():
    """
    Ana işlev
    """
    # Veri dosyası yolu
    file_path = 'veri/birlestir/AAPL_birlesik_2024-05-05_2025-05-05.json'
    
    # Çalışma dizinini kontrol et
    current_dir = os.getcwd()
    if current_dir.endswith('Veri_Alma'):
        # Zaten Veri_Alma klasöründeyiz
        pass
    elif os.path.exists(os.path.join(current_dir, 'Veri_Alma')):
        # Ana proje dizinindeyiz, Veri_Alma alt klasörüne geç
        os.chdir(os.path.join(current_dir, 'Veri_Alma'))
    else:
        # Başka bir yerdeyiz, dosya yolunu tam yolla değiştirmeye çalış
        possible_paths = [
            file_path,
            os.path.join('Veri_Alma', file_path),
            os.path.join('Project C', 'Veri_Alma', file_path),
            os.path.join('Fınans', 'Project C', 'Veri_Alma', file_path)
        ]
        
        # Olası yolları dene
        found = False
        for path in possible_paths:
            if os.path.exists(path):
                file_path = path
                found = True
                break
        
        if not found:
            print(f"Hata: Veri dosyası bulunamadı. Lütfen tam dosya yolunu belirtin.")
            return
    
    # Tam dosya yolunu görüntüle
    print(f"Veri dosyası yolu: {os.path.abspath(file_path)}")
    
    # Veriyi yükle
    df = load_json_data(file_path)
    if df is None:
        return
    
    # Duygu verilerini hazırla
    sentiment_data = prepare_sentiment_data(df)
    if sentiment_data is None:
        return
        
    # Üç değer döndürülüyor, üç değişkene atanmalı
    df, time_columns, title_columns = sentiment_data
    
    # Çıktı klasörünü oluştur
    output_dir = 'goruntuler'
    os.makedirs(output_dir, exist_ok=True)
    
    # Zaman serisi grafiği
    plot_sentiment_over_time(
        df, 
        time_columns,
        title_columns,  # title_columns parametresini geçir
        output_path=os.path.join(output_dir, 'AAPL_sentiment_zaman_serisi.png')
    )
    
    # Isı haritası
    plot_sentiment_heatmap(
        df,
        time_columns,
        title_columns,  # title_columns parametresini geçir
        output_path=os.path.join(output_dir, 'AAPL_sentiment_isi_haritasi.png')
    )
    
    print("Görselleştirme tamamlandı!")
    print(f"Görseller '{os.path.abspath(output_dir)}' klasörüne kaydedildi.")

if __name__ == "__main__":
    main() 