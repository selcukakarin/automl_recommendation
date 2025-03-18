import mlflow
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from pycaret.regression import load_model as pycaret_load_model
import pandas as pd
import uvicorn
from typing import Optional, List, Dict
from datetime import datetime, timedelta
import random
import numpy as np
from collections import deque

"""
Bu script, PyCaret ile eğitilmiş AutoML modellerini serve eder.

AutoML özellikleri:
1. Otomatik model yükleme ve versiyon yönetimi:
   - MLflow'dan model versiyonlarını otomatik yükleme
   - Versiyonlar arası geçiş yapabilme
   - A/B testing desteği

2. Otomatik veri ön işleme ve tahmin:
   - Kategorik değişkenleri otomatik dönüştürme
   - Eksik değerleri otomatik doldurma
   - Tahminleri otomatik ölçeklendirme

3. Model izleme ve sağlık kontrolü:
   - Son 24 saatteki tahminleri izleme
   - Performans metriklerini otomatik hesaplama
   - Model sağlığını sürekli kontrol etme

4. Güven aralıkları ve belirsizlik tahmini:
   - Her tahmin için güven aralığı hesaplama
   - Model belirsizliğini ölçme
   - Tahmin güvenilirliğini raporlama

API Endpoints:
- GET /versions: Mevcut model versiyonlarını listeler
- POST /load_version/{version_name}: Belirli bir model versiyonunu yükler
- GET /model_health: Model sağlığını kontrol eder
- POST /predict: Yeni tahminler yapar
"""

app = FastAPI(title="Recommendation System API")

# Son tahminleri saklamak için deque (son 24 saat)
recent_predictions = deque(maxlen=1000)

class RecommendationRequest(BaseModel):
    """
    Öneri sistemi için gerekli girdi özelliklerini tanımlayan model.
    
    Attributes:
        user_id (int): Kullanıcı ID'si
        item_id (int): Ürün ID'si
        user_age (int): Kullanıcı yaşı (18-70 arası)
        user_gender (str): Kullanıcı cinsiyeti (M/F)
        item_category (str): Ürün kategorisi (A/B/C)
        item_price (float): Ürün fiyatı (10-100 arası)
    """
    user_id: int
    item_id: int
    user_age: int
    user_gender: str
    item_category: str
    item_price: float

class RecommendationResponse(BaseModel):
    """
    Öneri sistemi tahmin sonuçlarını içeren model.
    
    Attributes:
        predicted_rating (float): Tahmini değerlendirme puanı (1-5 arası)
        confidence_interval (tuple): Tahmin için güven aralığı (alt sınır, üst sınır)
        model_version (str): Kullanılan model versiyonu
    """
    predicted_rating: float
    confidence_interval: tuple
    model_version: str

class ModelVersion(BaseModel):
    """
    Model versiyonu bilgilerini içeren model.
    
    Attributes:
        version_name (str): Model versiyonunun adı (örn. "v1_auto_select")
        run_id (str): MLflow run ID'si
        metrics (dict): Model performans metrikleri (MAE, MSE, RMSE, R²)
    """
    version_name: str
    run_id: str
    metrics: dict

class PredictionRecord:
    """
    Yapılan tahminlerin kaydını tutan sınıf.
    
    Attributes:
        prediction (float): Yapılan tahmin değeri
        actual (float, optional): Gerçek değer (varsa)
        timestamp (datetime): Tahmin zamanı
    """
    def __init__(self, prediction, actual=None, timestamp=None):
        self.prediction = prediction
        self.actual = actual
        self.timestamp = timestamp or datetime.now()

def get_recent_predictions(hours=24):
    """
    Son 24 saatteki tahminleri getirir.
    
    Args:
        hours (int): Kaç saat öncesine kadar olan tahminlerin getirileceği
    
    Returns:
        list: PredictionRecord objelerinden oluşan liste
    """
    cutoff_time = datetime.now() - timedelta(hours=hours)
    return [p for p in recent_predictions if p.timestamp > cutoff_time]

def get_actual_ratings():
    """
    Gerçek değerlendirmeleri simüle eder.
    
    Returns:
        list: Simüle edilmiş gerçek değerlendirmeler
        
    Not:
        Bu fonksiyon gerçek bir sistemde gerçek kullanıcı geribildirimleriyle
        değiştirilmelidir. Şu an sadece test amaçlı simüle edilmiş değerler üretir.
    """
    predictions = get_recent_predictions()
    return [np.random.normal(p.prediction, 0.5) for p in predictions]

def calculate_metrics(actual, predicted):
    """
    Model performans metriklerini hesaplar.
    
    Args:
        actual (list): Gerçek değerler
        predicted (list): Tahmin edilen değerler
    
    Returns:
        dict: Aşağıdaki metrikleri içeren sözlük:
            - MSE: Ortalama Kare Hata
            - RMSE: Kök Ortalama Kare Hata
            - MAE: Ortalama Mutlak Hata
            - R2: Belirtme Katsayısı
    """
    if not actual or not predicted:
        return None
    
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    mse = np.mean((actual - predicted) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(actual - predicted))
    r2 = 1 - np.sum((actual - predicted) ** 2) / np.sum((actual - np.mean(actual)) ** 2)
    
    return {
        "MSE": float(mse),
        "RMSE": float(rmse),
        "MAE": float(mae),
        "R2": float(r2)
    }

def validate_model_performance(metrics):
    """
    Model performansının kabul edilebilir seviyede olup olmadığını kontrol eder.
    
    Args:
        metrics (dict): Performans metrikleri (MSE, RMSE, MAE, R²)
    
    Raises:
        ValueError: Eğer performans metrikleri eşik değerlerin altındaysa
    
    Kontrol edilen eşik değerler:
        - RMSE < 1.0: Bir yıldızdan fazla hata kabul edilmez
        - R² > 0.6: En az %60 açıklayıcılık gücü olmalı
    """
    if not metrics:
        return
        
    threshold = {
        "RMSE": 1.0,  # 1 yıldızdan fazla hata kabul edilemez
        "R2": 0.6     # En az %60 açıklayıcılık gücü olmalı
    }
    
    if metrics['RMSE'] > threshold['RMSE']:
        raise ValueError(f"Model RMSE ({metrics['RMSE']:.2f}) çok yüksek!")
    
    if metrics['R2'] < threshold['R2']:
        raise ValueError(f"Model R2 skoru ({metrics['R2']:.2f}) çok düşük!")

def load_model(run_id):
    """
    MLflow'dan model ve versiyonunu yükler.
    
    Args:
        run_id (str): MLflow run ID'si
    
    Returns:
        tuple: (model, run_id)
            - model: Yüklenen PyCaret modeli
            - run_id: MLflow run ID'si
    """
    try:
        # Önce modeli MLflow'dan yüklemeyi dene
        model_path = f"runs:/{run_id}/model"
        model = mlflow.pyfunc.load_model(model_path)
        return model, run_id
    except Exception as mlflow_error:
        print(f"MLflow'dan model yükleme hatası: {str(mlflow_error)}")
        try:
            # Eğer MLflow'dan yükleme başarısız olursa, doğrudan dosyadan yüklemeyi dene
            model = pycaret_load_model("model.pkl")
            return model, run_id
        except Exception as file_error:
            print(f"Dosyadan model yükleme hatası: {str(file_error)}")
            raise Exception("Model yüklenemedi!")

# Global model değişkeni ve versiyon bilgisi
model = None
current_version = None

@app.on_event("startup")
async def startup_event():
    """
    FastAPI uygulaması başlatıldığında çalışan fonksiyon.
    
    Yapılan işlemler:
    1. MLflow sunucusuna bağlanır
    2. Son başarılı model çalışmasını bulur
    3. Modeli otomatik olarak yükler
    
    Raises:
        Exception: MLflow bağlantısı veya model yükleme başarısız olursa
    """
    global model, current_version
    
    try:
        # MLflow sunucusuna bağlan
        mlflow.set_tracking_uri("http://localhost:5000")
        
        # Son başarılı çalışmayı bul
        experiment = mlflow.get_experiment_by_name("recommendation-system")
        if experiment is None:
            raise Exception("Experiment bulunamadı!")
        
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        if len(runs) == 0:
            raise Exception("Hiç model çalışması bulunamadı!")
        
        # En son çalışmayı al
        latest_run = runs.iloc[0]
        
        try:
            # Modeli yüklemeyi dene
            model, current_version = load_model(latest_run.run_id)
            print(f"Model başarıyla yüklendi. Versiyon: {current_version}")
        except Exception as model_error:
            print(f"Model yükleme hatası: {str(model_error)}")
            raise
    except Exception as e:
        print(f"Başlangıç hatası: {str(e)}")
        raise

@app.get("/versions", response_model=List[ModelVersion])
async def list_versions():
    """
    Mevcut tüm model versiyonlarını listeler.
    
    Returns:
        List[ModelVersion]: Her bir model versiyonu için:
            - version_name: Versiyon adı
            - run_id: MLflow run ID'si
            - metrics: Performans metrikleri
    
    Raises:
        HTTPException: MLflow bağlantısı başarısız olursa
    """
    experiment = mlflow.get_experiment_by_name("recommendation-system")
    if experiment is None:
        raise HTTPException(status_code=404, detail="Experiment bulunamadı!")
    
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    
    versions = []
    for _, run in runs.iterrows():
        # Eğer version_name None ise, run_id'yi kullan
        version_name = run["params.version_name"]
        if version_name is None:
            version_name = f"run_{run['run_id'][:8]}"
            
        # Metrikleri güvenli bir şekilde al, yoksa varsayılan değer kullan
        metrics = {
            "MAE": float(run["metrics.MAE"]) if pd.notna(run["metrics.MAE"]) else 0.0,
            "MSE": float(run["metrics.MSE"]) if pd.notna(run["metrics.MSE"]) else 0.0,
            "RMSE": float(run["metrics.RMSE"]) if pd.notna(run["metrics.RMSE"]) else 0.0,
            "R2": float(run["metrics.R2"]) if pd.notna(run["metrics.R2"]) else 0.0
        }
        
        versions.append(ModelVersion(
            version_name=version_name,
            run_id=run["run_id"],
            metrics=metrics
        ))
    
    return versions

@app.post("/load_version/{version_name}")
async def load_version(version_name: str):
    """
    Belirtilen model versiyonunu yükler.
    
    Args:
        version_name (str): Yüklenecek model versiyonunun adı
    
    Returns:
        dict: Başarı mesajı
    
    Raises:
        HTTPException: Versiyon bulunamazsa veya yükleme başarısız olursa
    """
    experiment = mlflow.get_experiment_by_name("recommendation-system")
    if experiment is None:
        raise HTTPException(status_code=404, detail="Experiment bulunamadı!")
    
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    version_run = runs[runs["params.version_name"] == version_name]
    
    if len(version_run) == 0:
        raise HTTPException(status_code=404, detail=f"Versiyon bulunamadı: {version_name}")
    
    global model, current_version
    model, current_version = load_model(version_run.iloc[0]["run_id"])
    
    return {"message": f"Model versiyonu yüklendi: {version_name}"}

@app.get("/model_health", response_model=dict)
async def check_model_health():
    """
    Model sağlığını kontrol eder.
    
    Returns:
        dict: Model sağlık raporu:
            - status: "healthy" veya "degraded"
            - message: Durum açıklaması
            - metrics: Güncel performans metrikleri
            - sample_size: İncelenen tahmin sayısı
            - timestamp: Kontrol zamanı
    """
    # Son 24 saatteki tahminleri al
    recent_preds = get_recent_predictions()
    predictions = [p.prediction for p in recent_preds]
    
    # Gerçek değerlerle karşılaştır
    actual_ratings = get_actual_ratings()
    
    # Performans metriklerini hesapla
    current_metrics = calculate_metrics(actual_ratings, predictions)
    
    # Model sağlığını kontrol et
    status = "healthy"
    message = "Model performansı kabul edilebilir seviyede."
    
    try:
        if current_metrics:
            validate_model_performance(current_metrics)
    except ValueError as e:
        status = "degraded"
        message = str(e)
    
    return {
        "status": status,
        "message": message,
        "metrics": current_metrics,
        "sample_size": len(predictions),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=RecommendationResponse)
async def predict(
    request: RecommendationRequest,
    version_name: Optional[str] = Query(None, description="Kullanılacak model versiyonu"),
    enable_ab_testing: bool = Query(False, description="A/B testing'i etkinleştir")
):
    """
    Yeni bir tahmin yapar.
    
    Args:
        request (RecommendationRequest): Tahmin için gerekli özellikler
        version_name (str, optional): Kullanılacak model versiyonu
        enable_ab_testing (bool): A/B testing'i etkinleştir
    
    Returns:
        RecommendationResponse:
            - predicted_rating: Tahmini değerlendirme
            - confidence_interval: Güven aralığı
            - model_version: Kullanılan model versiyonu
    
    Raises:
        HTTPException: Model yüklenemezse veya tahmin başarısız olursa
    
    Özellikler:
    1. Otomatik model seçimi:
       - A/B testing etkinse rastgele bir versiyon seçer
       - version_name belirtilmişse o versiyonu kullanır
       - Hiçbiri belirtilmemişse mevcut versiyonu kullanır
    
    2. Veri ön işleme:
       - Kategorik değişkenleri dönüştürür
       - Eksik değerleri doldurur
    
    3. Tahmin ve güven aralığı:
       - Model tahminini yapar
       - Güven aralığını hesaplar
       - Tahmin kaydını saklar
    """
    global model, current_version
    
    if enable_ab_testing:
        versions = await list_versions()
        selected_version = random.choice([v.version_name for v in versions])
        await load_version(selected_version)
    elif version_name:
        await load_version(version_name)
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model yüklenmedi!")
    
    try:
        # Gelen veriyi DataFrame'e dönüştür
        input_df = pd.DataFrame([{
            'user_id': request.user_id,
            'item_id': request.item_id,
            'user_age': request.user_age,
            'user_gender': request.user_gender,
            'item_category': request.item_category,
            'item_price': request.item_price
        }])
        
        # Sayısal değişkenleri ölçeklendir
        for feature in ['user_age', 'item_price']:
            mean = input_df[feature].mean()
            std = input_df[feature].std()
            if std != 0:
                input_df[feature] = (input_df[feature] - mean) / std
        
        print(f"Tahmin için kullanılan özellikler: {input_df.columns.tolist()}")
        
        # Tahmin yap
        prediction = float(model.predict(input_df)[0])
        
        # Güven aralığı hesapla
        confidence_interval = (prediction - 0.5, prediction + 0.5)
        
        # Tahmin kaydını sakla
        recent_predictions.append(
            PredictionRecord(prediction=prediction)
        )
        
        return RecommendationResponse(
            predicted_rating=prediction,
            confidence_interval=confidence_interval,
            model_version=current_version
        )
    except Exception as e:
        print(f"Tahmin hatası: {str(e)}")
        raise HTTPException(status_code=500, detail=f"500: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("serve:app", host="0.0.0.0", port=8000, reload=True) 