#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Değerlendirme Modülü
Bu modül, analiz sonuçlarını değerlendirme işlevlerini içerir.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def evaluate_classification(classification_results):
    """
    Sınıflandırma modelini değerlendirir.
    
    Args:
        classification_results (dict): Sınıflandırma analizi sonuçları
    
    Returns:
        dict: Değerlendirme sonuçlarını içeren sözlük
    """
    if not classification_results:
        print("Değerlendirilecek sınıflandırma sonucu yok.")
        return {}
    
    print("Sınıflandırma modeli değerlendiriliyor...")
    
    try:
        # Gerekli değişkenleri çıkar
        y_test = classification_results.get('y_test', None)
        y_pred = classification_results.get('y_pred', None)
        model = classification_results.get('model', None)
        feature_importance = classification_results.get('feature_importance', None)
        
        if y_test is None or y_pred is None or model is None:
            print("Eksik değerlendirme verileri.")
            return {}
        
        # Sınıflandırma raporu oluştur
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Karmaşıklık matrisi oluştur
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Doğruluk oranı hesapla
        accuracy = accuracy_score(y_test, y_pred)
        
        results = {
            'classification_report': report,
            'confusion_matrix': conf_matrix,
            'accuracy': accuracy,
            'feature_importance': feature_importance
        }
        
        # Sonuçları yazdır
        print(f"Model doğruluk oranı: {accuracy:.4f}")
        print("En önemli öznitelikler:")
        if feature_importance is not None and not feature_importance.empty:
            for idx, row in feature_importance.head(5).iterrows():
                print(f"  - {row['feature']}: {row['importance']:.4f}")
                
        return results
    
    except Exception as e:
        print(f"Sınıflandırma değerlendirmesinde hata: {e}")
        return {}

def evaluate_patterns(pattern_results):
    """
    Desen madenciliği sonuçlarını değerlendirir.
    
    Args:
        pattern_results (dict): Desen madenciliği sonuçları
    
    Returns:
        dict: Değerlendirme sonuçlarını içeren sözlük
    """
    if not pattern_results:
        print("Değerlendirilecek desen sonucu yok.")
        return {}
    
    print("Desen sonuçları değerlendiriliyor...")
    
    try:
        # En iyi kuralları al
        best_rules = pattern_results.get('best_rules', None)
        rules = pattern_results.get('rules', None)
        
        if best_rules is None or rules is None or len(rules) == 0:
            print("Değerlendirilecek kural yok.")
            return {}
        
        # Desen istatistikleri
        num_rules = len(rules)
        avg_confidence = rules['confidence'].mean()
        avg_lift = rules['lift'].mean()
        max_lift = rules['lift'].max()
        
        # İlginçlik ölçütlerine göre en iyi kuralları sırala
        top_by_lift = rules.sort_values('lift', ascending=False).head(5)
        top_by_confidence = rules.sort_values('confidence', ascending=False).head(5)
        
        results = {
            'num_rules': num_rules,
            'avg_confidence': avg_confidence,
            'avg_lift': avg_lift,
            'max_lift': max_lift,
            'top_by_lift': top_by_lift,
            'top_by_confidence': top_by_confidence
        }
        
        # Sonuçları yazdır
        print(f"Toplam {num_rules} kural bulundu.")
        print(f"Ortalama güven (confidence): {avg_confidence:.4f}")
        print(f"Ortalama kaldıraç (lift): {avg_lift:.4f}")
        print(f"Maksimum kaldıraç (lift): {max_lift:.4f}")
        
        print("\nKaldıraç değerine göre en iyi kurallar:")
        if not top_by_lift.empty:
            for idx, row in top_by_lift.iterrows():
                antecedents = list(row['antecedents'])
                consequents = list(row['consequents'])
                print(f"  - {antecedents} => {consequents} (Lift: {row['lift']:.4f}, Confidence: {row['confidence']:.4f})")
                
        return results
        
    except Exception as e:
        print(f"Desen değerlendirmesinde hata: {e}")
        return {}

def evaluate_correlation(exploratory_results):
    """
    Korelasyon sonuçlarını değerlendirir.
    
    Args:
        exploratory_results (dict): Keşifsel analiz sonuçları
    
    Returns:
        dict: Değerlendirme sonuçlarını içeren sözlük
    """
    if not exploratory_results:
        print("Değerlendirilecek korelasyon sonucu yok.")
        return {}
    
    print("Korelasyon sonuçları değerlendiriliyor...")
    
    try:
        # Korelasyon matrisi ve p-değerlerini al
        correlation_matrix = exploratory_results.get('correlation_matrix', None)
        correlation_pvalues = exploratory_results.get('correlation_pvalues', None)
        
        if correlation_matrix is None:
            print("Değerlendirilecek korelasyon matrisi yok.")
            return {}
        
        # Duyarlılık skoru - göreli performans korelasyonu
        sentiment_rel_perf_corr = exploratory_results.get('sentiment_rel_perf_corr', None)
        sentiment_rel_perf_pval = exploratory_results.get('sentiment_rel_perf_pval', None)
        
        # En güçlü korelasyonları bul (kendisiyle korelasyonları hariç tut)
        corr_df = correlation_matrix.unstack().reset_index()
        corr_df.columns = ['Var1', 'Var2', 'Correlation']
        corr_df = corr_df[corr_df['Var1'] != corr_df['Var2']]  # Kendisiyle korelasyonları çıkar
        
        # Mutlak değere göre sırala
        corr_df['Abs_Correlation'] = corr_df['Correlation'].abs()
        strongest_correlations = corr_df.sort_values('Abs_Correlation', ascending=False).head(10)
        
        # P-değerlerini ekle (varsa)
        if correlation_pvalues is not None:
            pval_df = correlation_pvalues.unstack().reset_index()
            pval_df.columns = ['Var1', 'Var2', 'P_Value']
            
            # Korelasyon DataFrame'ine p-değerlerini ekle
            strongest_correlations = strongest_correlations.merge(
                pval_df, on=['Var1', 'Var2'], how='left'
            )
        
        # İstatistiksel anlamlılığı değerlendir
        significant_correlations = []
        if correlation_pvalues is not None:
            significant_mask = (correlation_pvalues < 0.05) & (correlation_matrix.abs() > 0.3)
            
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    var1 = correlation_matrix.columns[i]
                    var2 = correlation_matrix.columns[j]
                    
                    if significant_mask.iloc[i, j]:
                        corr = correlation_matrix.iloc[i, j]
                        pval = correlation_pvalues.iloc[i, j]
                        significant_correlations.append({
                            'var1': var1,
                            'var2': var2,
                            'correlation': corr,
                            'p_value': pval
                        })
        
        results = {
            'strongest_correlations': strongest_correlations,
            'significant_correlations': pd.DataFrame(significant_correlations) if significant_correlations else None,
            'sentiment_rel_perf_corr': sentiment_rel_perf_corr,
            'sentiment_rel_perf_pval': sentiment_rel_perf_pval
        }
        
        # Sonuçları yazdır
        print("En güçlü korelasyonlar:")
        if not strongest_correlations.empty:
            for idx, row in strongest_correlations.head(5).iterrows():
                corr_str = f"{row['Correlation']:.4f}"
                pval_str = f", p-değeri: {row['P_Value']:.4f}" if 'P_Value' in row else ""
                print(f"  - {row['Var1']} ve {row['Var2']}: {corr_str}{pval_str}")
        
        # Duyarlılık skoru - göreli performans korelasyonunu ayrıca vurgula
        if sentiment_rel_perf_corr is not None:
            is_significant = sentiment_rel_perf_pval < 0.05 if sentiment_rel_perf_pval is not None else "bilinmiyor"
            print(f"\nDuyarlılık skoru ve göreli performans korelasyonu: {sentiment_rel_perf_corr:.4f}")
            print(f"İstatistiksel anlamlılık: {is_significant}")
            
        return results
        
    except Exception as e:
        print(f"Korelasyon değerlendirmesinde hata: {e}")
        return {} 