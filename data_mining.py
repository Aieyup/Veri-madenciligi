#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Veri Madenciliği ve Analiz Modülü
Bu modül, keşifsel analiz, desen madenciliği, sınıflandırma ve kümeleme algoritmalarını içerir.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from mlxtend.frequent_patterns import apriori, association_rules
import scipy.stats as stats

def exploratory_analysis(df):
    """
    Veri seti üzerinde keşifsel analiz yapar.
    
    Args:
        df (pd.DataFrame): Analiz edilecek veri seti
    
    Returns:
        dict: Analiz sonuçlarını içeren sözlük
    """
    if df.empty:
        print("Analiz edilecek veri yok.")
        return {}
    
    print("Keşifsel analiz yapılıyor...")
    
    # Temel istatistikler
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    basic_stats = df[numeric_cols].describe()
    
    # Korelasyon matrisi
    correlation_matrix = df[numeric_cols].corr()
    
    # P-değeri hesaplamaları için
    correlation_pvalues = pd.DataFrame(index=correlation_matrix.index, columns=correlation_matrix.columns)
    
    for i in correlation_matrix.columns:
        for j in correlation_matrix.columns:
            if i != j:
                stat, p_value = stats.pearsonr(df[i].values, df[j].values)
                correlation_pvalues.loc[i, j] = p_value
            else:
                correlation_pvalues.loc[i, j] = 0  # Kendisiyle korelasyonun p-değeri 0
    
    # Duyarlılık skoru ile göreli performans arasındaki korelasyon
    if 'sentiment_score' in df.columns and 'relative_performance' in df.columns:
        sentiment_rel_perf_corr = correlation_matrix.loc['sentiment_score', 'relative_performance']
        sentiment_rel_perf_pval = correlation_pvalues.loc['sentiment_score', 'relative_performance']
    else:
        sentiment_rel_perf_corr = None
        sentiment_rel_perf_pval = None
    
    # Kategori sayılarını inceleme
    categorical_counts = {}
    categorical_cols = df.select_dtypes(include=['category', 'object']).columns
    for col in categorical_cols:
        if col in ['relative_perf_category', 'sentiment_category']:
            categorical_counts[col] = df[col].value_counts().to_dict()
    
    results = {
        'basic_stats': basic_stats,
        'correlation_matrix': correlation_matrix,
        'correlation_pvalues': correlation_pvalues,
        'sentiment_rel_perf_corr': sentiment_rel_perf_corr,
        'sentiment_rel_perf_pval': sentiment_rel_perf_pval,
        'categorical_counts': categorical_counts
    }
    
    print("Keşifsel analiz tamamlandı.")
    return results

def pattern_mining(df):
    """
    Veri setinde desenler arar, birliktelik kuralları çıkarır.
    
    Args:
        df (pd.DataFrame): Desen aranacak veri seti
    
    Returns:
        dict: Desen analiz sonuçlarını içeren sözlük
    """
    if df.empty:
        print("Desen aranacak veri yok.")
        return {}
    
    print("Desen madenciliği yapılıyor...")
    
    # İlgili kategorik değişkenleri seç
    if 'relative_perf_category' in df.columns and 'sentiment_category' in df.columns:
        # Kategorik verilerle çalışmak için one-hot encoding uygula
        binary_df = pd.get_dummies(df[['date', 'relative_perf_category', 'sentiment_category']])
        
        # Tarih sütununu kaldır
        if 'date' in binary_df.columns:
            binary_df = binary_df.drop('date', axis=1)
        
        # Minimum destek değeri ile sık öğe kümeleri bul
        min_support = 0.1
        frequent_itemsets = apriori(binary_df, min_support=min_support, use_colnames=True)
        
        # Minimum güven değeri ile kuralları çıkar
        min_confidence = 0.5
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
        
        # En iyi kuralları filtrele (kaldıraç değerine göre)
        best_rules = rules.sort_values('lift', ascending=False).head(10)
        
        results = {
            'frequent_itemsets': frequent_itemsets,
            'rules': rules,
            'best_rules': best_rules
        }
    else:
        print("Kategorik sütunlar bulunamadı, desen analizi yapılamıyor.")
        results = {}
    
    print("Desen madenciliği tamamlandı.")
    return results

def classification_analysis(df):
    """
    Veri setini kullanarak sınıflandırma analizi yapar.
    
    Args:
        df (pd.DataFrame): Sınıflandırma için kullanılacak veri seti
    
    Returns:
        dict: Sınıflandırma sonuçlarını içeren sözlük
    """
    if df.empty or 'relative_perf_category' not in df.columns:
        print("Sınıflandırma için gerekli veri yok.")
        return {}
    
    print("Sınıflandırma analizi yapılıyor...")
    
    # Eksik değerleri doldur
    df_filled = df.copy()
    numeric_cols = df_filled.select_dtypes(include=['float64', 'int64']).columns
    df_filled[numeric_cols] = df_filled[numeric_cols].fillna(df_filled[numeric_cols].mean())
    
    # Hedef değişkeni ve öznitelikleri belirle
    target = 'relative_perf_category'
    
    # NaN değerler yüzünden çıkabilecek anahtar sütunları kontrol et ve uygun şekilde doldur
    if 'sentiment_score' in df_filled.columns:
        df_filled['sentiment_score'] = df_filled['sentiment_score'].fillna(0)
    
    # Gecikmeli öznitelikler oluştur (1 gün önce)
    for col in ['Open', 'Close', 'daily_return', 'index_daily_return', 'relative_performance']:
        if col in df_filled.columns:
            df_filled[f'{col}_lag1'] = df_filled[col].shift(1)
    
    # Metin konularını sayısallaştırmak (özellik çıkarımı için)
    if 'main_topics' in df_filled.columns:
        # Konuları sayaç olarak dönüştürmek (basit yaklaşım)
        df_filled['topics_count'] = df_filled['main_topics'].str.count(',') + 1
        df_filled.loc[df_filled['main_topics'] == '', 'topics_count'] = 0
    
    # İlk satırı gecikmeli değerler NaN olduğu için atla
    df_filled = df_filled.dropna()
    
    # Sınıflandırma için kullanılacak öznitelikleri seç
    feature_cols = [
        'Open', 'Close', 'High', 'Low', 'Volume', 
        'daily_return', 'index_daily_return', 'relative_performance',
        'Open_lag1', 'Close_lag1', 'daily_return_lag1', 
        'index_daily_return_lag1', 'relative_performance_lag1'
    ]
    
    # Duyarlılık skorunu ekle (varsa)
    if 'sentiment_score' in df_filled.columns:
        feature_cols.append('sentiment_score')
    
    # Konu sayısını ekle (varsa)
    if 'topics_count' in df_filled.columns:
        feature_cols.append('topics_count')
    
    # Özniteliklerde eksik olan sütunları kaldır
    feature_cols = [col for col in feature_cols if col in df_filled.columns]
    
    # Öznitelikler ve hedef
    X = df_filled[feature_cols]
    y = df_filled[target]
    
    # Veri setini eğitim ve test olarak böl
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Öznitelikleri ölçeklendir
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Rastgele Orman sınıflandırıcısı oluştur
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    
    # Test seti üzerinde tahmin yap
    y_pred = rf_model.predict(X_test_scaled)
    
    # Öznitelik önemliliği
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    results = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred': y_pred,
        'model': rf_model,
        'feature_importance': feature_importance
    }
    
    print("Sınıflandırma analizi tamamlandı.")
    return results

def clustering_analysis(df):
    """
    Veri setini kullanarak kümeleme analizi yapar.
    
    Args:
        df (pd.DataFrame): Kümeleme için kullanılacak veri seti
    
    Returns:
        dict: Kümeleme sonuçlarını içeren sözlük
    """
    if df.empty:
        print("Kümeleme için gerekli veri yok.")
        return {}
    
    print("Kümeleme analizi yapılıyor...")
    
    # Eksik değerleri doldur
    df_filled = df.copy()
    numeric_cols = df_filled.select_dtypes(include=['float64', 'int64']).columns
    df_filled[numeric_cols] = df_filled[numeric_cols].fillna(df_filled[numeric_cols].mean())
    
    # Kümeleme için öznitelikleri seç
    features = [
        'daily_return', 'index_daily_return', 'relative_performance'
    ]
    
    # Duyarlılık skorunu ekle (varsa)
    if 'sentiment_score' in df_filled.columns:
        features.append('sentiment_score')
    
    # Konu sayısını ekle (varsa)
    if 'topics_count' in df_filled.columns and df_filled['topics_count'].sum() > 0:
        features.append('topics_count')
    
    # Özniteliklerde eksik olan sütunları kaldır
    features = [col for col in features if col in df_filled.columns]
    
    if not features:
        print("Kümeleme için yeterli öznitelik yok.")
        return {}
    
    # Veriyi al ve NaN değerleri temizle
    X = df_filled[features].dropna()
    
    if len(X) < 10:
        print("Kümeleme için yeterli veri yok.")
        return {}
    
    # Veriyi ölçeklendir
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Optimal küme sayısını belirle (Silhouette Yöntemi)
    silhouette_scores = []
    K_range = range(2, min(6, len(X) // 5 + 1))
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        silhouette_avg = silhouette_score(X_scaled, cluster_labels)
        silhouette_scores.append(silhouette_avg)
    
    # En iyi silhouette skoru olan küme sayısını seç
    if silhouette_scores:
        optimal_k = K_range[np.argmax(silhouette_scores)]
    else:
        optimal_k = 3  # Varsayılan değer
    
    # Optimal küme sayısına göre KMeans uygula
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    df_filled['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Her küme için istatistikler hesapla
    cluster_stats = df_filled.groupby('cluster')[features].mean()
    
    # Küme büyüklüklerini hesapla
    cluster_sizes = df_filled['cluster'].value_counts().sort_index()
    
    results = {
        'X_scaled': X_scaled,
        'optimal_k': optimal_k,
        'kmeans_model': kmeans,
        'silhouette_scores': silhouette_scores,
        'cluster_stats': cluster_stats,
        'cluster_sizes': cluster_sizes,
        'df_with_clusters': df_filled
    }
    
    print("Kümeleme analizi tamamlandı.")
    return results 