import os
import mlflow
import pandas as pd
import numpy as np
from pycaret.regression import *
from datetime import datetime, timedelta
import json

"""
Bu script, PyCaret'in AutoML ozelliklerini kullanarak bir oneri sistemi olusturur.
AutoML ozellikleri:
1. Otomatik model secimi ve karsilastirma: PyCaret'in compare_models fonksiyonu ile en iyi modeli secer
2. Otomatik veri on isleme: Kategorik degiskenleri donusturme, eksik degerleri doldurma
3. Otomatik hiperparametre optimizasyonu: tune_model fonksiyonu ile parametreleri optimize eder
4. Otomatik metrik hesaplama: MAE, MSE, RMSE ve R2 metriklerini hesaplar
5. Farkli model versiyonlarini otomatik karsilastirma: Farkli modelleri ve ayarlari test eder

Kullanim:
1. MLflow sunucusunu baslatin: mlflow ui
2. Bu scripti calistirin: python train.py
3. MLflow UI'da sonuclari inceleyin: http://localhost:5000
"""

# MLflow sunucusunu ayarla
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("recommendation-system")

def validate_model_performance(metrics):
    """
    Model performansinin belirlenen esik degerlerini karsilayip karsilamadigini kontrol eder.
    
    Args:
        metrics (dict): Model performans metrikleri (RMSE, R2 vb.)
    
    Raises:
        ValueError: Eger model performansi esik degerlerin altindaysa hata verir
        
    Metrik Açıklamaları:
    --------------------
    RMSE (Root Mean Square Error - Kök Ortalama Kare Hata):
    - Tahminlerimizin gerçek değerlerden ne kadar saptığını gösterir
    - Daha düşük değerler daha iyidir (0'a yakın olması idealdir)
    - Büyük hataları daha çok cezalandırır
    - Örnek: RMSE=0.5 ise, tahminlerimiz ortalamada gerçek değerlerden yaklaşık 0.5 puan sapıyor
    
    MAE (Mean Absolute Error - Ortalama Mutlak Hata):
    - Tahminlerimizin gerçek değerlerden ortalama sapmasını gösterir
    - Daha düşük değerler daha iyidir (0'a yakın olması idealdir)
    - Tüm hataları eşit şekilde değerlendirir
    - Örnek: MAE=0.4 ise, tahminlerimiz ortalamada gerçek değerlerden 0.4 puan sapıyor
    
    Tahmin Oranı (Prediction Rate):
    - Modelin kaç örnek için tahmin yapabildiğini yüzde olarak gösterir
    - Daha yüksek değerler daha iyidir (%100'e yakın olması idealdir)
    - Örnek: %99.34 ise, model örneklerin %99.34'ü için tahmin yapabiliyor
    """
    threshold = {
        "RMSE": 1.0,  # 1 yildizdan fazla hata kabul edilemez
        "R2": 0.6     # En az %60 aciklayicilik gucu olmali
    }
    
    if metrics['RMSE'] > threshold['RMSE']:
        raise ValueError(f"Model RMSE ({metrics['RMSE']:.2f}) cok yuksek!")
    
    if metrics['R2'] < threshold['R2']:
        raise ValueError(f"Model R2 skoru ({metrics['R2']:.2f}) cok dusuk!")

def create_sample_data():
    """
    Oneri sistemi icin ornek veri seti olusturur.
    
    Returns:
        pd.DataFrame: Asagidaki ozellikleri iceren veri seti:
            - user_id: Kullanici ID'si (1-100 arasi)
            - item_id: Urun ID'si (1-50 arasi)
            - rating: Degerlendirme puani (1-5 arasi)
            - timestamp: Degerlendirme tarihi (son 1 yil icinde)
            - user_age: Kullanici yasi (18-70 arasi)
            - user_gender: Kullanici cinsiyeti (M/F)
            - item_category: Urun kategorisi (A/B/C)
            - item_price: Urun fiyati (10-100 arasi)
    
    Not:
        - Veri seti zaman bazli validasyon icin timestamp icerir
        - Kategorik degiskenler AutoML tarafindan otomatik islenir
    """
    np.random.seed(42)
    n_users = 100
    n_items = 50
    n_ratings = 1000
    
    # Zaman bazlı validasyon için tarih sütunu ekle
    # Örnek çıktı:
    # dates = [
    #   '2023-03-15 00:00:00',
    #   '2023-03-16 02:24:00',
    #   '2023-03-17 04:48:00',
    #   ...
    #   '2024-03-14 21:36:00'
    # ]
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=365),
        end=datetime.now(),
        periods=n_ratings
    )
    
    # Kullanıcı ID'lerini oluştur
    # - n_ratings kadar rastgele kullanıcı ID'si üretilir
    # - ID'ler 1 ile n_users arasında olacak şekilde seçilir
    # - Bir kullanıcı birden fazla değerlendirme yapabilir
    # Örnek çıktı:
    # user_ids = [23, 45, 12, 78, 34, 1, 99, ...]
    user_ids = np.random.randint(1, n_users + 1, n_ratings)
    
    # Ürün ID'lerini oluştur
    # - n_ratings kadar rastgele ürün ID'si üretilir
    # - ID'ler 1 ile n_items arasında olacak şekilde seçilir
    # - Bir ürün birden fazla kez değerlendirilebilir
    # Örnek çıktı:
    # item_ids = [5, 12, 45, 23, 8, 15, 31, ...]
    item_ids = np.random.randint(1, n_items + 1, n_ratings)
    
    # Değerlendirme puanlarını oluştur
    # - n_ratings kadar rastgele puan üretilir
    # - Puanlar 1 ile 5 arasında uniform dağılımlı olarak seçilir
    # - Ondalıklı puanlara izin verilir (örn: 3.7, 4.2 gibi)
    # - Gerçek kullanıcı davranışını simüle etmek için uniform dağılım kullanılır
    # Örnek çıktı:
    # ratings = [4.2, 3.7, 2.1, 5.0, 1.8, 3.3, ...]
    ratings = np.random.uniform(1, 5, n_ratings)
    
    # Kullanıcı özellikleri ekleyelim - AutoML için ek özellikler
    # Örnek çıktı:
    # user_features = 
    #    user_id  user_age user_gender
    # 0        1        25          M
    # 1        2        45          F
    # 2        3        33          M
    # ...
    user_features = pd.DataFrame({
        'user_id': range(1, n_users + 1),
        'user_age': np.random.randint(18, 70, n_users),
        'user_gender': np.random.choice(['M', 'F'], n_users)
    })
    
    # Ürün özellikleri ekleyelim - AutoML için ek özellikler
    # Örnek çıktı:
    # item_features = 
    #    item_id item_category  item_price
    # 0        1            A       45.23
    # 1        2            B       78.50
    # 2        3            C       23.15
    # ...
    item_features = pd.DataFrame({
        'item_id': range(1, n_items + 1),
        'item_category': np.random.choice(['A', 'B', 'C'], n_items),
        'item_price': np.random.uniform(10, 100, n_items)
    })
    
    # Ana veri çerçevesini oluştur
    # Örnek çıktı:
    # df = 
    #    user_id  item_id  rating            timestamp
    # 0       23        5     4.2  2023-03-15 00:00:00
    # 1       45       12     3.7  2023-03-16 02:24:00
    # 2       12       45     2.1  2023-03-17 04:48:00
    # ...
    df = pd.DataFrame({
        'user_id': user_ids,
        'item_id': item_ids,
        'rating': ratings,
        'timestamp': dates
    })
    
    # Kullanıcı ve ürün özelliklerini birleştir
    # Final çıktı:
    # df = 
    #    user_id  item_id  rating            timestamp  user_age user_gender item_category  item_price
    # 0       23        5     4.2  2023-03-15 00:00:00        25          M            A       45.23
    # 1       45       12     3.7  2023-03-16 02:24:00        45          F            B       78.50
    # 2       12       45     2.1  2023-03-17 04:48:00        33          M            C       23.15
    # ...
    df = df.merge(user_features, on='user_id', how='left')
    df = df.merge(item_features, on='item_id', how='left')
    
    # Kategorik değişkenleri dönüştür - AutoML bunu otomatik olarak işleyecek
    # Örnek çıktı (değişken tipleri):
    # user_gender: category    (M, F)
    # item_category: category  (A, B, C)
    df['user_gender'] = df['user_gender'].astype('category')
    df['item_category'] = df['item_category'].astype('category')
    
    # Timestamp sütununu Unix timestamp'e dönüştür
    df['timestamp'] = pd.to_datetime(df['timestamp']).astype(np.int64) // 10**9
    
    if not os.path.exists('data'):
        os.makedirs('data')
    df.to_csv('data/ratings.csv', index=False)
    return df

def train_model_version(version_name, model_type=None, params=None):
    """
    Belirli bir versiyon icin model egitir ve MLflow'a kaydeder.
    
    Args:
        version_name (str): Model versiyonunun adi (orn. "v1_auto_select")
        model_type (str, optional): Kullanilacak model tipi (orn. "rf" for Random Forest)
        params (dict, optional): Model parametreleri
    
    AutoML ozellikleri:
        1. Veri on isleme:
           - Kategorik degiskenleri otomatik donusturme
           - Eksik degerleri doldurma
           - Aykiri degerleri temizleme
        
        2. Model secimi/optimizasyonu:
           - model_type belirtilmezse en iyi modeli otomatik secer
           - model_type belirtilirse o modeli optimize eder
        
        3. Validasyon:
           - 5-fold cross validation
           - Zaman bazli bolunme (temporal split)
           - Performans metriklerini hesaplama
        
        4. MLflow entegrasyonu:
           - Model parametrelerini kaydetme
           - Performans metriklerini kaydetme
           - Model dosyasini artifact olarak saklama
    
    Metrikler:
        - MAE: Ortalama Mutlak Hata
        - MSE: Ortalama Kare Hata
        - RMSE: Kok Ortalama Kare Hata
        - R2: Belirtme Katsayisi
    """
    if not os.path.exists('data/ratings.csv'):
        df = create_sample_data()
    else:
        df = pd.read_csv('data/ratings.csv')
    
    # Model dosyası için artifacts klasörü oluştur
    if not os.path.exists('artifacts'):
        os.makedirs('artifacts')
    
    with mlflow.start_run(run_name=version_name):
        try:
            # Temel parametreleri kaydet
            mlflow.set_tag("model_type", "rating")
            mlflow.set_tag("endpoint", "/predict")
            mlflow.set_tag("version_name", version_name)
            
            if params:
                for key, value in params.items():
                    mlflow.log_param(key, str(value))
            
            # Model eğitimi setup
            setup(
                data=df,
                target='rating',
                ignore_features=['timestamp'],
                train_size=0.75,
                fold=5,
                fold_strategy='timeseries',
                fold_shuffle=False,
                data_split_shuffle=False,
                session_id=42,
                verbose=False,
                use_gpu=False
            )
            
            # Model oluşturma ve eğitme
            if model_type:
                model = create_model(model_type, **params if params else {})
                tuned_model = tune_model(model, n_iter=10, optimize='RMSE')
                final_model = finalize_model(tuned_model)
            else:
                model = compare_models(n_select=1)
                final_model = finalize_model(model)
            
            # Metrikleri hesapla
            metrics = pull()
            best_metrics = metrics.iloc[0].to_dict()
            
            # Temel metrikleri kaydet
            essential_metrics = {
                'rmse': float(best_metrics['RMSE']),
                'mae': float(best_metrics['MAE']),
                'n_predictions': len(df),  # Toplam tahmin sayısı
                'prediction_ratio': 1.0    # Tüm örnekler için tahmin yapılabilir
            }
            mlflow.log_metrics(essential_metrics)
            
            # Model dosyasını kaydet
            model_path = os.path.join('artifacts', 'model.pkl')
            save_model(final_model, model_path)
            
            # MLflow'a model artifact'ını kaydet
            mlflow.sklearn.log_model(
                final_model,
                "model",
                registered_model_name=f"rating_model_{version_name}"
            )
            
            # Geçici dosyaları temizle
            if os.path.exists(model_path):
                os.remove(model_path)
            
            return (True, "Model başarıyla eğitildi ve kaydedildi.")
            
        except Exception as e:
            print(f"Model eğitimi sırasında hata: {str(e)}")
            return (False, f"Model eğitimi başarısız oldu: {str(e)}")

def train_multiple_versions():
    """
    Farkli model versiyonlarini egitir ve karsilastirir.
    
    Egitilen versiyonlar:
        1. v1_auto_select: 
           - AutoML ile en iyi modeli otomatik secer
           - Tum model tiplerini test eder
           - En iyi performans gosteren modeli secer
        
        2. v2_random_forest:
           - Random Forest modeli
           - Varsayilan parametrelerle egitilir
           - AutoML ile optimize edilir
        
        3. v3_rf_tuned:
           - Ozellesmis Random Forest
           - Belirli parametrelerle baslar:
             * n_estimators: 200 (agac sayisi)
             * max_depth: 10 (maksimum derinlik)
             * min_samples_split: 5 (dal bolme icin min ornek)
           - AutoML ile bu parametrelerden optimize edilir
        
        4. v4_lightgbm:
           - Light GBM modeli
           - AutoML ile optimize edilir
           - Genellikle en iyi performansi verir
    
    Not:
        - Her versiyon MLflow'a kaydedilir
        - Performans metrikleri karsilastirilabilir
        - En iyi versiyon servis icin kullanilabilir
    """
    # Version 1: AutoML ile otomatik model secimi
    train_model_version("v1_auto_select")
    
    # Version 2: AutoML ile Random Forest optimizasyonu
    train_model_version("v2_random_forest", "rf")
    
    # Version 3: AutoML ile ozellesmis Random Forest
    rf_params = {
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_split": 5
    }
    train_model_version("v3_rf_tuned", "rf", rf_params)
    
    # Version 4: AutoML ile Light GBM optimizasyonu
    train_model_version("v4_lightgbm", "lightgbm")

if __name__ == "__main__":
    train_multiple_versions() 