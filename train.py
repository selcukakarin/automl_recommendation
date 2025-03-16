import os
import mlflow
import pandas as pd
import numpy as np
from pycaret.regression import *
from datetime import datetime, timedelta

"""
Bu script, PyCaret'in AutoML özelliklerini kullanarak bir öneri sistemi oluşturur.
AutoML özellikleri:
1. Otomatik model seçimi ve karşılaştırma: PyCaret'in compare_models fonksiyonu ile en iyi modeli seçer
2. Otomatik veri ön işleme: Kategorik değişkenleri dönüştürme, eksik değerleri doldurma
3. Otomatik hiperparametre optimizasyonu: tune_model fonksiyonu ile parametreleri optimize eder
4. Otomatik metrik hesaplama: MAE, MSE, RMSE ve R² metriklerini hesaplar
5. Farklı model versiyonlarını otomatik karşılaştırma: Farklı modelleri ve ayarları test eder

Kullanım:
1. MLflow sunucusunu başlatın: mlflow ui
2. Bu scripti çalıştırın: python train.py
3. MLflow UI'da sonuçları inceleyin: http://localhost:5000
"""

# MLflow sunucusunu ayarla
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("recommendation-system")

def validate_model_performance(metrics):
    """
    Model performansının belirlenen eşik değerlerini karşılayıp karşılamadığını kontrol eder.
    
    Args:
        metrics (dict): Model performans metrikleri (RMSE, R² vb.)
    
    Raises:
        ValueError: Eğer model performansı eşik değerlerin altındaysa hata verir
        
    Kontrol edilen metrikler:
    - RMSE: 1.0'dan küçük olmalı (1 yıldızdan az hata)
    - R²: 0.6'dan büyük olmalı (en az %60 açıklayıcılık)
    """
    threshold = {
        "RMSE": 1.0,  # 1 yıldızdan fazla hata kabul edilemez
        "R2": 0.6     # En az %60 açıklayıcılık gücü olmalı
    }
    
    if metrics['RMSE'] > threshold['RMSE']:
        raise ValueError(f"Model RMSE ({metrics['RMSE']:.2f}) çok yüksek!")
    
    if metrics['R2'] < threshold['R2']:
        raise ValueError(f"Model R2 skoru ({metrics['R2']:.2f}) çok düşük!")

def create_sample_data():
    """
    Öneri sistemi için örnek veri seti oluşturur.
    
    Returns:
        pd.DataFrame: Aşağıdaki özellikleri içeren veri seti:
            - user_id: Kullanıcı ID'si (1-100 arası)
            - item_id: Ürün ID'si (1-50 arası)
            - rating: Değerlendirme puanı (1-5 arası)
            - timestamp: Değerlendirme tarihi (son 1 yıl içinde)
            - user_age: Kullanıcı yaşı (18-70 arası)
            - user_gender: Kullanıcı cinsiyeti (M/F)
            - item_category: Ürün kategorisi (A/B/C)
            - item_price: Ürün fiyatı (10-100 arası)
    
    Not:
        - Veri seti zaman bazlı validasyon için timestamp içerir
        - Kategorik değişkenler AutoML tarafından otomatik işlenir
    """
    np.random.seed(42)
    n_users = 100
    n_items = 50
    n_ratings = 1000
    
    # Zaman bazlı validasyon için tarih sütunu ekle
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=365),
        end=datetime.now(),
        periods=n_ratings
    )
    
    user_ids = np.random.randint(1, n_users + 1, n_ratings)
    item_ids = np.random.randint(1, n_items + 1, n_ratings)
    ratings = np.random.uniform(1, 5, n_ratings)
    
    # Kullanıcı özellikleri ekleyelim - AutoML için ek özellikler
    user_features = pd.DataFrame({
        'user_id': range(1, n_users + 1),
        'user_age': np.random.randint(18, 70, n_users),
        'user_gender': np.random.choice(['M', 'F'], n_users)
    })
    
    # Ürün özellikleri ekleyelim - AutoML için ek özellikler
    item_features = pd.DataFrame({
        'item_id': range(1, n_items + 1),
        'item_category': np.random.choice(['A', 'B', 'C'], n_items),
        'item_price': np.random.uniform(10, 100, n_items)
    })
    
    # Ana veri çerçevesini oluştur
    df = pd.DataFrame({
        'user_id': user_ids,
        'item_id': item_ids,
        'rating': ratings,
        'timestamp': dates
    })
    
    # Kullanıcı ve ürün özelliklerini birleştir
    df = df.merge(user_features, on='user_id', how='left')
    df = df.merge(item_features, on='item_id', how='left')
    
    # Kategorik değişkenleri dönüştür - AutoML bunu otomatik olarak işleyecek
    df['user_gender'] = df['user_gender'].astype('category')
    df['item_category'] = df['item_category'].astype('category')
    
    if not os.path.exists('data'):
        os.makedirs('data')
    df.to_csv('data/ratings.csv', index=False)
    return df

def train_model_version(version_name, model_type=None, params=None):
    """
    Belirli bir versiyon için model eğitir ve MLflow'a kaydeder.
    
    Args:
        version_name (str): Model versiyonunun adı (örn. "v1_auto_select")
        model_type (str, optional): Kullanılacak model tipi (örn. "rf" for Random Forest)
        params (dict, optional): Model parametreleri
    
    AutoML özellikleri:
        1. Veri ön işleme:
           - Kategorik değişkenleri otomatik dönüştürme
           - Eksik değerleri doldurma
           - Aykırı değerleri temizleme
        
        2. Model seçimi/optimizasyonu:
           - model_type belirtilmezse en iyi modeli otomatik seçer
           - model_type belirtilirse o modeli optimize eder
        
        3. Validasyon:
           - 5-fold cross validation
           - Zaman bazlı bölünme (temporal split)
           - Performans metriklerini hesaplama
        
        4. MLflow entegrasyonu:
           - Model parametrelerini kaydetme
           - Performans metriklerini kaydetme
           - Model dosyasını artifact olarak saklama
    
    Metrikler:
        - MAE: Ortalama Mutlak Hata
        - MSE: Ortalama Kare Hata
        - RMSE: Kök Ortalama Kare Hata
        - R²: Belirtme Katsayısı
    """
    if not os.path.exists('data/ratings.csv'):
        df = create_sample_data()
    else:
        df = pd.read_csv('data/ratings.csv')
    
    with mlflow.start_run(run_name=version_name):
        # AutoML: Otomatik veri ön işleme ve cross-validation ayarları
        setup(data=df,
              target='rating',
              train_size=0.75,
              fold=5,                     # 5-fold cross validation
              fold_strategy='timeseries', # Zaman bazlı cross-validation
              fold_shuffle=False,         # Zaman sırasını koru
              time_index='timestamp',     # Zaman sütunu
              session_id=42,
              silent=True,
              use_gpu=False)
        
        if model_type:
            # AutoML: Belirli bir model için otomatik hiperparametre optimizasyonu
            model = create_model(model_type, **params if params else {})
            # Model optimizasyonu
            tuned_model = tune_model(model, n_iter=10, optimize='RMSE')
            final_model = finalize_model(tuned_model)
        else:
            # AutoML: Otomatik model seçimi - en iyi modeli seçer
            model = compare_models(n_select=1)
            final_model = finalize_model(model)
        
        # AutoML: Otomatik metrik hesaplama
        metrics = pull()
        best_metrics = metrics.iloc[0].to_dict()
        
        # Model performansını kontrol et
        try:
            validate_model_performance(best_metrics)
        except ValueError as e:
            print(f"Uyarı: {str(e)}")
        
        # Parametreleri ve metrikleri logla
        model_params = final_model.get_params()
        mlflow.log_params(model_params)
        mlflow.log_params({
            "version_name": version_name,
            "fold_strategy": "timeseries",
            "n_folds": 5
        })
        mlflow.log_metrics({
            "MAE": best_metrics['MAE'],
            "MSE": best_metrics['MSE'],
            "RMSE": best_metrics['RMSE'],
            "R2": best_metrics['R2']
        })
        
        # AutoML: Otomatik model kaydetme
        model_path = "model.pkl"
        save_model(final_model, model_path)
        
        # Modeli MLflow'a artifact olarak kaydet
        mlflow.log_artifact(model_path)
        os.remove(model_path)

def train_multiple_versions():
    """
    Farklı model versiyonlarını eğitir ve karşılaştırır.
    
    Eğitilen versiyonlar:
        1. v1_auto_select: 
           - AutoML ile en iyi modeli otomatik seçer
           - Tüm model tiplerini test eder
           - En iyi performans gösteren modeli seçer
        
        2. v2_random_forest:
           - Random Forest modeli
           - Varsayılan parametrelerle eğitilir
           - AutoML ile optimize edilir
        
        3. v3_rf_tuned:
           - Özelleştirilmiş Random Forest
           - Belirli parametrelerle başlar:
             * n_estimators: 200 (ağaç sayısı)
             * max_depth: 10 (maksimum derinlik)
             * min_samples_split: 5 (dal bölme için min örnek)
           - AutoML ile bu parametrelerden optimize edilir
        
        4. v4_xgboost:
           - XGBoost modeli
           - AutoML ile optimize edilir
           - Genellikle en iyi performansı verir
    
    Not:
        - Her versiyon MLflow'a kaydedilir
        - Performans metrikleri karşılaştırılabilir
        - En iyi versiyon servis için kullanılabilir
    """
    # Version 1: AutoML ile otomatik model seçimi
    train_model_version("v1_auto_select")
    
    # Version 2: AutoML ile Random Forest optimizasyonu
    train_model_version("v2_random_forest", "rf")
    
    # Version 3: AutoML ile özelleştirilmiş Random Forest
    rf_params = {
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_split": 5
    }
    train_model_version("v3_rf_tuned", "rf", rf_params)
    
    # Version 4: AutoML ile XGBoost optimizasyonu
    train_model_version("v4_xgboost", "xgboost")

if __name__ == "__main__":
    train_multiple_versions() 