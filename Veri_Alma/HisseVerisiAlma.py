#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Hisse Senedi Veri Alma Modülü.

Bu modül, çeşitli kaynaklardan (tvDatafeed, yfinance) hisse senedi fiyat verilerini alır.
Farklı zaman dilimlerinde (günlük, saatlik vb.) veri indirme ve farklı formatlarda kaydetme işlevselliği sunar.

PEP8 ve PEP257 standartlarına uygun şekilde yazılmıştır.
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Dict, Optional, Union, List, Tuple

import pandas as pd
import yfinance as yf
try:
    from tvDatafeed import TvDatafeed, Interval
    TVDATAFEED_AVAILABLE = True
except ImportError:
    TVDATAFEED_AVAILABLE = False
    logging.warning("tvDatafeed kütüphanesi yüklü değil. TradingView verileri kullanılamayacak.")

# Loglama yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("veri_toplama.log", mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def get_stock_data(
    stock_symbol: str, 
    index_symbol: str, 
    start_date: str, 
    end_date: str
) -> pd.DataFrame:
    """
    Yahoo Finance API'yi kullanarak hisse senedi ve endeks verilerini getirir.
    
    Args:
        stock_symbol: Hisse sembolü (ör. "THYAO.IS")
        index_symbol: Endeks sembolü (ör. "XU100.IS")
        start_date: Başlangıç tarihi (YYYY-MM-DD formatında)
        end_date: Bitiş tarihi (YYYY-MM-DD formatında)
    
    Returns:
        Hisse senedi ve endeks verilerini içeren DataFrame.
        Hata durumunda boş DataFrame döner.
    
    Raises:
        ValueError: Geçersiz sembol veya tarih formatı.
        ConnectionError: Yahoo Finance'e bağlanırken bir sorun oluştuğunda.
    """
    try:
        logger.info(
            f"Hisse ve endeks verileri alınıyor: {stock_symbol}, {index_symbol}"
        )
        
        # Veriyi indir
        tickers = [stock_symbol, index_symbol]
        data = yf.download(tickers, start=start_date, end=end_date)
        
        # Çoklu sütunları düzleştir
        data = data.stack(level=0).reset_index()
        data.rename(columns={"level_1": "Symbol"}, inplace=True)
        
        logger.info(f"Toplam {len(data)} satır veri alındı.")
        return data
        
    except ValueError as e:
        logger.error(f"Geçersiz sembol veya tarih formatı: {e}")
        return pd.DataFrame()
    except ConnectionError as e:
        logger.error(f"Yahoo Finance bağlantı hatası: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Hisse verisi alınırken beklenmeyen hata oluştu: {e}")
        return pd.DataFrame()


def get_tv_data(
    symbol: str, 
    exchange: str, 
    interval: Union[str, object] = 'D', 
    days: int = 365, 
    username: Optional[str] = None, 
    password: Optional[str] = None
) -> pd.DataFrame:
    """
    TradingView platformundan hisse senedi verilerini indirir.
    
    Args:
        symbol: Hisse senedi sembolü (ör: 'AAPL').
        exchange: Borsa adı (ör: 'NASDAQ').
        interval: Verilerin zaman aralığı ('D' günlük, 'H' saatlik, '1h' 1 saatlik, vb.).
        days: Kaç günlük veri indirileceği (varsayılan: 365 gün).
        username: TradingView kullanıcı adı (isteğe bağlı).
        password: TradingView şifresi (isteğe bağlı).
        
    Returns:
        Hisse senedi verilerini içeren DataFrame.
        Hata durumunda boş DataFrame döner.
        
    Raises:
        ImportError: tvDatafeed kütüphanesi yüklü değilse.
        ValueError: Geçersiz sembol, borsa veya interval.
    """
    if not TVDATAFEED_AVAILABLE:
        logger.error("tvDatafeed kütüphanesi yüklü değil. 'pip install --upgrade tvdatafeed' komutu ile yükleyin.")
        return pd.DataFrame()
        
    try:
        logger.info(f"{symbol} sembolü için {exchange} borsasından veri alınıyor...")
        
        # TvDatafeed zaman aralığını çevir
        if isinstance(interval, str):
            interval_map = {
                'D': Interval.in_daily,
                'W': Interval.in_weekly,
                'M': Interval.in_monthly,
                'H': Interval.in_1_hour,
                '1h': Interval.in_1_hour,
                '4h': Interval.in_4_hour,
                '15m': Interval.in_15_minute,
            }
            tv_interval = interval_map.get(interval, Interval.in_daily)
        else:
            tv_interval = interval
            
        # tvDatafeed nesnesi oluştur
        tv = TvDatafeed(username=username, password=password)

        # Veri indirme (saatlik veri ise gün_sayısı * 24 bar alınır)
        n_bars = days * 24 if tv_interval in [Interval.in_1_hour, Interval.in_4_hour, 
                                           Interval.in_15_minute, Interval.in_5_minute] else days
                                           
        data = tv.get_hist(symbol=symbol, exchange=exchange, interval=tv_interval, n_bars=n_bars)
        
        if data.empty:
            logger.warning(f"{symbol} için veri bulunamadı.")
            return pd.DataFrame()
            
        logger.info(f"Toplam {len(data)} satır veri alındı.")
        return data
        
    except Exception as e:
        logger.error(f"TradingView verisi alınırken hata oluştu: {e}")
        return pd.DataFrame()
        

def veri_indir_ve_kaydet(
    sembol: str, 
    borsa: str, 
    zaman_araligi: Union[str, object], 
    gun_sayisi: int, 
    cikti_dosyasi: str,
    username: Optional[str] = None,
    password: Optional[str] = None
) -> None:
    """
    Belirtilen sembol için hisse senedi verilerini indirir ve CSV dosyasına kaydeder.

    Args:
        sembol: Hisse senedi sembolü (ör: 'AAPL').
        borsa: Borsa adı (ör: 'NASDAQ').
        zaman_araligi: Verilerin zaman aralığı (ör: Interval.in_1_hour veya '1h').
        gun_sayisi: Kaç günlük veri indirileceği.
        cikti_dosyasi: Verilerin kaydedileceği CSV dosyasının yolu.
        username: TradingView kullanıcı adı (isteğe bağlı).
        password: TradingView şifresi (isteğe bağlı).

    Returns:
        None
    """
    try:
        # Veri indirme
        veri = get_tv_data(
            symbol=sembol, 
            exchange=borsa, 
            interval=zaman_araligi, 
            days=gun_sayisi,
            username=username,
            password=password
        )
        
        if veri.empty:
            logger.error(f"{sembol} için veri alınamadı, dosya kaydedilemedi.")
            return
            
        # Dizin yoksa oluştur
        os.makedirs(os.path.dirname(os.path.abspath(cikti_dosyasi)) if os.path.dirname(cikti_dosyasi) else '.', exist_ok=True)
        
        # CSV dosyasına kaydetme
        logger.info(f"Veriler {cikti_dosyasi} dosyasına kaydediliyor...")
        veri.to_csv(cikti_dosyasi, index=True)
        logger.info(f"Veriler başarıyla {cikti_dosyasi} dosyasına kaydedildi.")
    except Exception as e:
        logger.error(f"Veri indirme ve kaydetme sırasında hata: {e}")


if __name__ == "__main__":
    # Temel parametreler
    SEMBOL = "AAPL"  # Hisse sembolü
    BORSA = "NASDAQ"  # Borsa adı
    GUN_SAYISI = 365  # Yaklaşık bir yıl
    CIKTI_KLASORU = "veri/hisse"
    
    # Çıktı klasörünü oluştur
    os.makedirs(CIKTI_KLASORU, exist_ok=True)
    
    # Örnek 1: TradingView'dan saatlik veri
    try:
        if TVDATAFEED_AVAILABLE:
            logger.info("TradingView'dan saatlik veri alınıyor...")
            ZAMAN_ARALIGI = Interval.in_1_hour  # Saatlik veri
            CIKTI_DOSYASI = os.path.join(CIKTI_KLASORU, f"{SEMBOL}_saatlik_tv.csv")
            veri_indir_ve_kaydet(SEMBOL, BORSA, ZAMAN_ARALIGI, GUN_SAYISI, CIKTI_DOSYASI)
        else:
            logger.warning("tvDatafeed kütüphanesi yüklü olmadığı için TradingView örneği atlanıyor.")
    except Exception as e:
        logger.error(f"TradingView örneği çalıştırılırken hata: {e}")
    
    # Örnek 2: Yahoo Finance'den günlük veri
    try:
        logger.info("Yahoo Finance'den günlük veri alınıyor...")
        stock_data = get_stock_data(
            SEMBOL, 
            f"^{'GSPC' if BORSA == 'NASDAQ' else 'IXIC'}", 
            (datetime.now() - timedelta(days=GUN_SAYISI)).strftime("%Y-%m-%d"),
            datetime.now().strftime("%Y-%m-%d")
        )
        
        if not stock_data.empty:
            CIKTI_DOSYASI = os.path.join(CIKTI_KLASORU, f"{SEMBOL}_gunluk_yf.csv")
            stock_data.to_csv(CIKTI_DOSYASI, index=False)
            logger.info(f"Yahoo Finance verileri {CIKTI_DOSYASI} dosyasına kaydedildi.")
    except Exception as e:
        logger.error(f"Yahoo Finance örneği çalıştırılırken hata: {e}")
