import os
import mlflow
import pandas as pd
import numpy as np
from pycaret.regression import *

"""
Bu script, PyCaret'in AutoML özelliklerini kullanarak bir öneri sistemi oluşturur.
AutoML özellikleri:
1. Otomatik model seçimi ve karşılaştırma
2. Otomatik veri ön işleme
3. Otomatik hiperparametre optimizasyonu
4. Otomatik metrik hesaplama
5. Farklı model versiyonlarını otomatik karşılaştırma
"""

# MLflow sunucusunu ayarla
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("recommendation-system")

def create_sample_data():
    """Örnek veri seti oluşturur. Bu veri seti, AutoML sürecinde kullanılacak
    kullanıcı ve ürün özelliklerini içerir."""
    np.random.seed(42)
    n_users = 100
    n_items = 50
    n_ratings = 1000
    
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
        'rating': ratings
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
    Belirli bir versiyon için model eğitir. PyCaret'in AutoML özellikleri burada devreye girer:
    1. Otomatik veri ön işleme
    2. Otomatik model seçimi veya belirli bir modelin optimizasyonu
    3. Otomatik metrik hesaplama ve logging
    
    Args:
        version_name: Model versiyonunun adı
        model_type: Kullanılacak model tipi (örn. 'rf' for Random Forest)
        params: Model parametreleri
    """
    if not os.path.exists('data/ratings.csv'):
        df = create_sample_data()
    else:
        df = pd.read_csv('data/ratings.csv')
    
    with mlflow.start_run(run_name=version_name):
        # AutoML: Otomatik veri ön işleme
        # - Eksik değerleri doldurma
        # - Kategorik değişkenleri sayısallaştırma
        # - Aykırı değerleri tespit etme
        # - Değişken ölçeklendirme
        setup(data=df,
              target='rating',
              train_size=0.75,
              session_id=42,
              silent=True,
              use_gpu=False)
        
        if model_type:
            # AutoML: Belirli bir model için otomatik hiperparametre optimizasyonu
            model = create_model(model_type, **params if params else {})
        else:
            # AutoML: Otomatik model seçimi - en iyi modeli seçer
            model = compare_models(n_select=1)
        
        # AutoML: Model finalizasyonu ve son optimizasyonlar
        final_model = finalize_model(model)
        
        # AutoML: Otomatik metrik hesaplama
        metrics = pull()
        best_metrics = metrics.iloc[0].to_dict()
        
        # Parametreleri ve metrikleri logla
        model_params = final_model.get_params()
        mlflow.log_params(model_params)
        mlflow.log_params({"version_name": version_name})
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
    Farklı versiyonlarda modeller eğitir. Bu fonksiyon AutoML'in farklı yeteneklerini gösterir:
    1. Otomatik model seçimi
    2. Belirli modellerin otomatik optimizasyonu
    3. Farklı hiperparametrelerle otomatik model eğitimi
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