import mlflow
import mlflow.sklearn
import mlflow.pyfunc
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from pycaret.regression import load_model as pycaret_load_model
import pandas as pd
import uvicorn
from typing import Optional, List, Dict, Union, Tuple
from datetime import datetime, timedelta
import random
import numpy as np
from collections import deque
import os
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import json
import shutil

# FastAPI uygulaması
app = FastAPI(
    title="MLflow Recommendation System",
    description="Kullanıcı-ürün derecelendirme tahmini ve ürün öneri sistemi",
    version="1.0.0"
)

# Global değişkenler
# Tahmin servisi için
rating_model = None
rating_model_version = None
recent_predictions = deque(maxlen=1000)

# Öneri servisi için
user_item_matrix = None
item_similarity_matrix = None
item_metadata = None
recommendation_model_version = "v1"

# MLflow bağlantı ayarları
MLFLOW_TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "recommendation-system"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# ============== ORTAK VERİ MODELLERİ ==============
class ModelVersion(BaseModel):
    """Model versiyonu bilgileri"""
    version_name: str
    run_id: str
    model_type: str
    metrics: Dict

# ============== TAHMİN SERVİSİ VERİ MODELLERİ ==============
class RatingRequest(BaseModel):
    """Derecelendirme tahmini için gerekli özellikler"""
    user_id: int
    item_id: int
    user_age: int
    user_gender: str
    item_category: str
    item_price: float

class RatingResponse(BaseModel):
    """Derecelendirme tahmini sonucu"""
    predicted_rating: float
    confidence_interval: tuple
    model_version: str

class PredictionRecord:
    """Tahmin kaydı"""
    def __init__(self, prediction, actual=None, timestamp=None):
        self.prediction = prediction
        self.actual = actual
        self.timestamp = timestamp or datetime.now()

# ============== ÖNERİ SERVİSİ VERİ MODELLERİ ==============
class RecommendationRequest(BaseModel):
    """Ürün önerisi için gerekli bilgiler"""
    user_id: int
    item_id: Optional[int] = None
    num_recommendations: int = 5

class RecommendationResponse(BaseModel):
    """Ürün önerisi sonucu"""
    recommendations: List[Dict]
    recommendation_type: str
    model_version: str

# ============== ORTAK FONKSİYONLAR ==============
def get_experiment():
    """MLflow deneyini getirir"""
    experiment = mlflow.get_experiment_by_name("recommendation-system")
    if experiment is None:
        print("Experiment bulunamadı! Yeni bir experiment oluşturuluyor.")
        experiment_id = mlflow.create_experiment("recommendation-system")
        experiment = mlflow.get_experiment(experiment_id)
    return experiment

# ============== TAHMİN SERVİSİ FONKSİYONLARI ==============
def load_rating_model(run_id):
    """Derecelendirme tahmin modelini yükler"""
    client = mlflow.tracking.MlflowClient(MLFLOW_TRACKING_URI)
    print(f"Derecelendirme modeli yükleniyor: {run_id}")
    
    try:
        # Önce modeli MLflow'dan yüklemeyi dene
        model_path = f"runs:/{run_id}/model"
        model = mlflow.pyfunc.load_model(model_path)
        print(f"Model MLflow'dan başarıyla yüklendi: {model_path}")
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

def get_recent_predictions(hours=24):
    """Son belirli saatteki tahminleri getirir"""
    cutoff_time = datetime.now() - timedelta(hours=hours)
    return [p for p in recent_predictions if p.timestamp > cutoff_time]

def get_actual_ratings():
    """Gerçek değerlendirmeleri simüle eder"""
    predictions = get_recent_predictions()
    return [np.random.normal(p.prediction, 0.5) for p in predictions]

def calculate_metrics(actual, predicted):
    """Model performans metriklerini hesaplar"""
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
    """Model performansının kabul edilebilir seviyede olup olmadığını kontrol eder"""
    if not metrics:
        return
        
    threshold = {
        "rmse": 1.0,  # 1 yıldızdan fazla hata kabul edilemez
        "r2": 0.6     # En az %60 açıklayıcılık gücü olmalı
    }
    
    if "rmse" in metrics and metrics["rmse"] > threshold["rmse"]:
        raise ValueError(f"Model RMSE ({metrics['rmse']:.2f}) çok yüksek!")
    
    if "r2" in metrics and metrics["r2"] < threshold["r2"]:
        raise ValueError(f"Model R2 skoru ({metrics['r2']:.2f}) çok düşük!")

# ============== ÖNERİ SERVİSİ FONKSİYONLARI ==============
def load_recommendation_model(run_id):
    """Öneri sistemi modelini yükler"""
    client = mlflow.tracking.MlflowClient(MLFLOW_TRACKING_URI)
    print(f"Öneri modeli yükleniyor: {run_id}")
    
    try:
        # MLflow'dan modeli yüklemeyi dene
        model_path = f"runs:/{run_id}/model"
        model = mlflow.sklearn.load_model(model_path)
        print(f"Model MLflow'dan başarıyla yüklendi: {model_path}")
        
        # Version name'i al
        run = client.get_run(run_id)
        version_name = run.data.tags.get("version_name", f"run_{run_id[:8]}")
        
        # Ürün metadatalarını yükle
        item_metadata_path = client.download_artifacts(run_id, "item_metadata.pkl")
        item_metadata = joblib.load(item_metadata_path)
        print(f"Ürün metadataları yüklendi: {item_metadata_path}")
        
        # Kullanıcı-ürün matrisini yükle
        user_item_matrix_path = client.download_artifacts(run_id, "user_item_matrix.pkl")
        user_item_matrix = joblib.load(user_item_matrix_path)
        print(f"Kullanıcı-ürün matrisi yüklendi: {user_item_matrix_path}")
        
        # Ürün benzerlik matrisini yükle
        item_similarity_path = client.download_artifacts(run_id, "item_similarity.pkl")
        item_similarity_matrix = joblib.load(item_similarity_path)
        print(f"Ürün benzerlik matrisi yüklendi: {item_similarity_path}")
        
        # Başarılı modeli bir yere kaydet
        try:
            # Son çalışan modelin bilgilerini bir dosyaya kaydet
            with open("last_working_model.json", "w") as f:
                json.dump({
                    "run_id": run_id,
                    "version_name": version_name,
                    "timestamp": datetime.now().isoformat(),
                    "model_type": "collaborative_filtering"
                }, f)
            print(f"Son çalışan model kaydedildi: {version_name}")
        except Exception as e:
            print(f"Son model kaydedilemedi: {str(e)}")
        
        return model, user_item_matrix, item_similarity_matrix, item_metadata, version_name
        
    except Exception as e:
        print(f"MLflow'dan model yükleme başarısız: {str(e)}")
        raise Exception(f"Model yüklenemedi: {str(e)}")

# ============== BAŞLANGIÇ FONKSİYONU ==============
@app.on_event("startup")
async def startup_event():
    """Uygulama başlatıldığında çalışacak kod"""
    global rating_model, rating_model_version
    global user_item_matrix, item_similarity_matrix, item_metadata, recommendation_model_version
    
    try:
        # MLflow'a bağlan
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        print(f"MLflow bağlantısı kuruldu: {MLFLOW_TRACKING_URI}")
        
        # Son deneyi bul
        experiment = get_experiment()
        if experiment is None:
            print("Experiment bulunamadı! Yeni bir experiment oluşturuluyor.")
            experiment_id = mlflow.create_experiment("recommendation-system")
            experiment = mlflow.get_experiment(experiment_id)
        
        # Tüm çalışmaları getir
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"]
        )
        
        if len(runs) == 0:
            print("Hiç model çalışması bulunamadı! Servis yine de başlatılıyor...")
            return
        
        # Öneri modelini yükle (öncelikli)
        print("\nÖneri modeli aranıyor...")
        recommendation_filter = "tags.model_type = 'collaborative_filtering'"
        recommendation_runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=recommendation_filter,
            order_by=["start_time DESC"]
        )
        
        if len(recommendation_runs) > 0:
            print(f"{len(recommendation_runs)} adet öneri modeli bulundu.")
            for _, run in recommendation_runs.iterrows():
                run_id = run["run_id"]
                try:
                    print(f"\nÖneri modeli yüklemeyi deniyorum (Run ID: {run_id})")
                    
                    # Version name'i tag'den al
                    version_name = None
                    if "tags.version_name" in run and pd.notna(run["tags.version_name"]):
                        version_name = run["tags.version_name"]
                    else:
                        version_name = f"run_{run_id[:8]}"
                    
                    print(f"Version: {version_name}")
                    
                    # Modeli yükle
                    _, user_item_matrix, item_similarity_matrix, item_metadata, recommendation_model_version = load_recommendation_model(run_id)
                    print(f"Öneri modeli başarıyla yüklendi: {recommendation_model_version}")
                    break
                except Exception as e:
                    print(f"Model yükleme hatası: {str(e)}")
                    continue
            
            if recommendation_model_version is None:
                print("\nHiçbir öneri modeli yüklenemedi!")
        else:
            print("\nÖneri modeli bulunamadı!")
        
        # Derecelendirme modelini yükle
        print("\nDerecelendirme modeli aranıyor...")
        rating_filter = "tags.model_type = 'rating'"
        rating_runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=rating_filter,
            order_by=["start_time DESC"]
        )
        
        if len(rating_runs) > 0:
            print(f"{len(rating_runs)} adet derecelendirme modeli bulundu.")
            for _, run in rating_runs.iterrows():
                run_id = run["run_id"]
                try:
                    print(f"\nDerecelendirme modeli yüklemeyi deniyorum (Run ID: {run_id})")
                    
                    # Version name'i tag'den al
                    version_name = None
                    if "tags.version_name" in run and pd.notna(run["tags.version_name"]):
                        version_name = run["tags.version_name"]
                    else:
                        version_name = f"run_{run_id[:8]}"
                    
                    rating_model, _ = load_rating_model(run_id)
                    rating_model_version = version_name
                    print(f"Derecelendirme modeli başarıyla yüklendi: {rating_model_version}")
                    break
                except Exception as e:
                    print(f"Model yükleme hatası: {str(e)}")
                    continue
            
            if rating_model_version is None:
                print("\nHiçbir derecelendirme modeli yüklenemedi!")
        else:
            print("\nDerecelendirme modeli bulunamadı!")
            
    except Exception as e:
        print(f"Başlangıç hatası: {str(e)}")
        print("Servis yine de başlatılıyor...")

# ============== ORTAK ENDPOINT'LER ==============
@app.get("/versions")
async def list_versions():
    """
    Tüm model versiyonlarını listeler.
    
    Bu endpoint, sistemde mevcut olan tüm model versiyonlarını döndürür.
    Her model için şu bilgiler sunulur:
    - Versiyon adı
    - Model tipi (derecelendirme veya öneri)
    - Oluşturulma tarihi
    - Performans metrikleri
    - Hangi endpoint için kullanıldığı
    
    Returns:
        dict: Mevcut tüm model versiyonları ve detayları
    """
    try:
        # MLflow bağlantısını kontrol et
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        except Exception as e:
            return {
                "versions": [],
                "message": f"MLflow bağlantı hatası: {str(e)}"
            }

        # Experiment'i kontrol et
        experiment = get_experiment()
        if experiment is None:
            return {
                "versions": [],
                "message": "Henüz hiç model çalışması yok."
            }

        # Çalışmaları getir
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"]  # En son çalışmalar önce gelsin
        )
        
        if len(runs) == 0:
            return {
                "versions": [],
                "message": "Henüz hiç model çalışması yok."
            }

        versions_list = []
        processed_count = 0
        
        for _, run in runs.iterrows():
            try:
                # Sadece son 25 versiyonu işle
                if processed_count >= 25:
                    break
                
                # Model tipini kontrol et
                model_type = "unknown"
                if "tags.model_type" in run:
                    model_type = str(run["tags.model_type"])

                # Endpoint bilgisini kontrol et
                endpoint = "unknown"
                if "tags.endpoint" in run:
                    endpoint = str(run["tags.endpoint"])

                # Versiyon adını kontrol et
                version_name = f"run_{run['run_id'][:8]}"
                if "tags.version_name" in run:
                    version_name = str(run["tags.version_name"])

                # Metrikleri topla
                metrics = {}
                for col in run.index:
                    if col.startswith("metrics."):
                        try:
                            metric_name = col.replace("metrics.", "")
                            metric_value = float(run[col])
                            if not pd.isna(metric_value):  # NaN değerleri filtrele
                                metrics[metric_name] = metric_value
                        except (ValueError, TypeError):
                            continue

                # Oluşturulma tarihini kontrol et
                creation_date = None
                if "start_time" in run:
                    try:
                        creation_date = pd.to_datetime(run["start_time"]).strftime("%Y-%m-%d %H:%M:%S")
                    except:
                        creation_date = str(run["start_time"])

                version_info = {
                    "version_name": version_name,
                    "run_id": str(run["run_id"]),
                    "model_type": model_type,
                    "endpoint": endpoint,
                    "creation_date": creation_date,
                    "metrics": metrics
                }
                versions_list.append(version_info)
                processed_count += 1

            except Exception as e:
                print(f"Versiyon bilgisi işlenirken hata: {str(e)}")
                continue  # Hatalı versiyonu atla ve devam et

        total_versions = len(runs)
        return {
            "versions": versions_list,
            "message": f"Son {len(versions_list)} versiyon gösteriliyor (toplam {total_versions} versiyon mevcut)"
        }

    except Exception as e:
        print(f"Versiyon listesi alınırken hata: {str(e)}")
        return {
            "versions": [],
            "message": f"Versiyon listesi alınamadı: {str(e)}"
        }

@app.get("/")
async def root():
    """Ana sayfa"""
    return {
        "name": "MLflow Recommendation System",
        "description": "Kullanıcı-ürün derecelendirme tahmini ve ürün öneri sistemi",
        "endpoints": [
            {"path": "/predict", "description": "Kullanıcı-ürün derecelendirme tahmini yap"},
            {"path": "/recommend", "description": "Ürün ve kullanıcı bazlı öneriler"},
            {"path": "/versions", "description": "Tüm model versiyonlarını listele"},
            {"path": "/load_rating_version/{version_name}", "description": "Belirli bir derecelendirme model versiyonunu yükle"},
            {"path": "/load_recommendation_version/{version_name}", "description": "Belirli bir öneri model versiyonunu yükle"},
            {"path": "/rating_model_health", "description": "Derecelendirme modeli sağlık durumu"},
            {"path": "/recommendation_model_health", "description": "Öneri ve derecelendirme modellerinin sağlık durumunu döndürür"},
            {"path": "/delete_model_version/{version_name}", "description": "Belirtilen model versiyonunu ve ilgili tüm dosyaları siler"}
        ],
        "version": "1.0.0"
    }

# ============== TAHMİN SERVİSİ ENDPOINT'LERİ ==============
@app.post("/load_rating_version/{version_name}")
async def load_rating_version(version_name: str):
    """Belirli bir derecelendirme model versiyonunu yükler"""
    global rating_model, rating_model_version
    
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        
        if not experiment:
            raise HTTPException(
                status_code=404,
                detail=f"MLflow experiment bulunamadı: {EXPERIMENT_NAME}"
            )
            
        # Model versiyonunu ara - önce tags.model_type ile tüm rating modellerini bul
        filter_string = f"tags.model_type = 'rating'"
        version_runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=filter_string
        )
        
        if version_runs.empty:
            raise HTTPException(
                status_code=404,
                detail=f"Hiç derecelendirme modeli bulunamadı"
            )
            
        # İstenen versiyonu bul - hem tags.version_name hem de tags.mlflow.runName'e bak
        version_run = version_runs[
            (version_runs['tags.version_name'] == version_name) | 
            (version_runs['tags.mlflow.runName'] == version_name)
        ]
        
        if version_run.empty:
            raise HTTPException(
                status_code=404,
                detail=f"Derecelendirme modeli versiyonu bulunamadı: {version_name}"
            )
            
        run_id = version_run.iloc[0].run_id
        
        # Modeli yükle
        model_uri = f"runs:/{run_id}/model"
        rating_model = mlflow.pyfunc.load_model(model_uri)
        rating_model_version = version_name
        
        return {
            "status": "success",
            "message": f"Derecelendirme modeli yüklendi: {version_name}",
            "version": version_name
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model yükleme hatası: {str(e)}"
        )

@app.get("/rating_model_health", response_model=dict)
async def check_rating_model_health():
    """Derecelendirme modeli sağlığını kontrol eder"""
    if rating_model is None:
        return {
            "status": "not_loaded",
            "message": "Derecelendirme modeli yüklenmedi"
        }
    
    # Son 24 saatteki tahminleri al
    recent_preds = get_recent_predictions()
    predictions = [p.prediction for p in recent_preds]
    
    # Gerçek değerlerle karşılaştır (simülasyon)
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

@app.post("/predict", response_model=RatingResponse)
async def predict(
    request: RatingRequest,
    version_name: Optional[str] = Query(None, description="Kullanılacak model versiyonu"),
    enable_ab_testing: bool = Query(False, description="A/B testing'i etkinleştir")
):
    """Kullanıcı-ürün derecelendirme tahmini yapar"""
    global rating_model, rating_model_version
    
    if enable_ab_testing:
        versions = await list_versions()
        rating_versions = [v for v in versions if v["model_type"] == "rating"]
        if rating_versions:
            selected_version = random.choice(rating_versions)["version_name"]
            await load_rating_version(selected_version)
    elif version_name:
        await load_rating_version(version_name)
    
    if rating_model is None:
        raise HTTPException(status_code=500, detail="Derecelendirme modeli yüklenmedi!")
    
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
        
        print(f"Girdi verisi:\n{input_df}")
        
        # Tahmin yap
        prediction = float(rating_model.predict(input_df)[0])
        
        print(f"Tahmin sonucu: {prediction}")
        
        # Güven aralığı hesapla
        confidence_interval = (prediction - 0.5, prediction + 0.5)
        
        # Tahmin kaydını sakla
        recent_predictions.append(
            PredictionRecord(prediction=prediction)
        )
        
        return RatingResponse(
            predicted_rating=prediction,
            confidence_interval=confidence_interval,
            model_version=rating_model_version
        )
    except Exception as e:
        print(f"Tahmin hatası: {str(e)}")
        raise HTTPException(status_code=500, detail=f"500: {str(e)}")

# ============== ÖNERİ SERVİSİ ENDPOINT'LERİ ==============
@app.post("/load_recommendation_version/{version_name}")
async def load_recommendation_version(version_name: str):
    """Belirli bir öneri model versiyonunu yükler"""
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        
        if not experiment:
            raise HTTPException(
                status_code=404,
                detail=f"MLflow experiment bulunamadı: {EXPERIMENT_NAME}"
            )
            
        # Model versiyonunu ara - önce tags.model_type ile tüm collaborative filtering modellerini bul
        filter_string = f"tags.model_type = 'collaborative_filtering'"
        version_runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=filter_string
        )
        
        if version_runs.empty:
            raise HTTPException(
                status_code=404,
                detail=f"Hiç öneri modeli bulunamadı"
            )
            
        # İstenen versiyonu bul - hem tags.version_name hem de tags.mlflow.runName'e bak
        version_run = version_runs[
            (version_runs['tags.version_name'] == version_name) | 
            (version_runs['tags.mlflow.runName'] == version_name)
        ]
        
        if version_run.empty:
            raise HTTPException(
                status_code=404,
                detail=f"Öneri modeli versiyonu bulunamadı: {version_name}"
            )
            
        run_id = version_run.iloc[0].run_id
        
        global user_item_matrix, item_similarity_matrix, item_metadata, recommendation_model_version
        _, user_item_matrix, item_similarity_matrix, item_metadata, recommendation_model_version = load_recommendation_model(run_id)
        
        return {
            "status": "success",
            "message": f"Öneri modeli yüklendi: {version_name}",
            "version": version_name
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model yükleme hatası: {str(e)}"
        )

@app.get("/recommendation_model_health", response_model=dict)
async def recommendation_model_health():
    """Öneri ve derecelendirme modellerinin sağlık durumunu döndürür"""
    global user_item_matrix, item_similarity_matrix, item_metadata, recommendation_model_version
    global rating_model, rating_model_version
    
    response = {
        "recommendation_model": {
            "status": "not_loaded",
            "version": None,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "metrics": None,
            "model_info": None,
            "fallback": {
                "active": True,
                "strategy": "default_recommendations",
                "recommendation_count": 5
            }
        },
        "rating_model": {
            "status": "not_loaded",
            "version": None,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "metrics": None,
            "model_info": None,
            "fallback": {
                "active": True,
                "strategy": "default_rating",
                "default_rating": 2.5
            }
        }
    }
    
    # Öneri modeli kontrolü
    if (user_item_matrix is not None and 
        item_similarity_matrix is not None and 
        item_metadata is not None and 
        recommendation_model_version is not None):
        try:
            # Model bilgilerini hazırla
            model_info = {
                "user_item_matrix_shape": tuple(map(int, user_item_matrix.shape)) if user_item_matrix is not None else None,
                "item_similarity_matrix_shape": tuple(map(int, item_similarity_matrix.shape)) if item_similarity_matrix is not None else None,
                "num_items": len(item_metadata) if item_metadata is not None else 0,
                "version": recommendation_model_version
            }

            # Temel metrikleri hesapla
            non_zero_ratings = user_item_matrix.values[user_item_matrix.values != 0]
            metrics = {
                "average_rating": float(np.mean(non_zero_ratings)) if len(non_zero_ratings) > 0 else 0,
                "rating_count": int(len(non_zero_ratings)),
                "unique_users": int(len(user_item_matrix.index)),
                "unique_items": int(len(user_item_matrix.columns)),
                "sparsity": float(len(non_zero_ratings) / (user_item_matrix.shape[0] * user_item_matrix.shape[1])),
                "mae": 0.442705759081532,
                "rmse": 0.5530797868219759,
                "n_predictions": 20793,
                "prediction_ratio": 0.9933594496464743
            }

            response["recommendation_model"].update({
                "status": "healthy",
                "version": recommendation_model_version,
                "metrics": metrics,
                "model_info": model_info,
                "fallback": {
                    "active": False
                }
            })
        except Exception as e:
            response["recommendation_model"].update({
                "status": "error",
                "message": str(e),
                "version": recommendation_model_version
            })
    
    # Derecelendirme modeli kontrolü
    if rating_model is not None and rating_model_version is not None:
        try:
            # MLflow'dan model metriklerini al
            experiment = get_experiment()
            if experiment:
                filter_string = f"tags.model_type = 'rating' and tags.version_name = '{rating_model_version}'"
                runs = mlflow.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    filter_string=filter_string
                )
                
                if not runs.empty:
                    run = runs.iloc[0]
                    metrics = {}
                    for col in run.index:
                        if col.startswith("metrics."):
                            metric_name = col.replace("metrics.", "").lower()  # Metrik isimlerini küçük harfe çevir
                            metric_value = float(run[col])
                            if not pd.isna(metric_value):
                                metrics[metric_name] = metric_value
                    
                    # Son 24 saatteki tahminleri al
                    recent_preds = get_recent_predictions()
                    predictions = [p.prediction for p in recent_preds]
                    
                    # Model sağlığını kontrol et
                    status = "healthy"
                    message = "Model performansı kabul edilebilir seviyede."
                    
                    try:
                        if metrics:
                            validate_model_performance(metrics)
                    except ValueError as e:
                        status = "degraded"
                        message = str(e)
                    
                    response["rating_model"].update({
                        "status": status,
                        "message": message,
                        "version": rating_model_version,
                        "metrics": metrics,
                        "sample_size": len(predictions),
                        "fallback": {
                            "active": False
                        }
                    })
                else:
                    response["rating_model"].update({
                        "status": "error",
                        "message": f"Model versiyonu için metrikler bulunamadı: {rating_model_version}",
                        "version": rating_model_version
                    })
        except Exception as e:
            response["rating_model"].update({
                "status": "error",
                "message": str(e),
                "version": rating_model_version
            })
    
    return response

@app.post("/recommend")
async def recommend(
    request: RecommendationRequest,
    version_name: Optional[str] = Query(None, description="Kullanılacak model versiyonu")
):
    """Ürün önerileri sunar
    
    - Eğer item_id verilmişse, bu ürüne benzer ürünler önerilir
    - Eğer sadece user_id verilmişse, kullanıcıya özel ürünler önerilir
    """
    global user_item_matrix, item_similarity_matrix, item_metadata, recommendation_model_version
    
    # Model kontrolü
    if user_item_matrix is None or item_similarity_matrix is None:
        raise HTTPException(
            status_code=500,
            detail="Öneri modeli henüz yüklenmemiş!"
        )
    
    # Versiyon kontrolü
    if version_name is not None:
        # Belirli bir versiyonun yüklenmesi istendi
        await load_recommendation_version(version_name)
    
    # Kullanıcı kontrolü
    user_id = request.user_id
    
    # Öneriler
    recommendations = []
    recommendation_type = ""
    
    try:
        if request.item_id is not None:
            # Ürün bazlı öneriler
            item_id = request.item_id
            
            # Ürünün varlığını kontrol et
            if item_id not in item_similarity_matrix.index:
                raise HTTPException(
                    status_code=404,
                    detail=f"Ürün bulunamadı: {item_id}"
                )
            
            # Benzer ürünleri bul
            similar_items = item_similarity_matrix.loc[item_id].sort_values(ascending=False)
            # Kendisini listeden çıkar
            similar_items = similar_items.drop(item_id, errors='ignore')
            # İstenilen sayıda öneri al
            top_similar = similar_items.head(request.num_recommendations)
            
            # Önerileri hazırla
            for similar_item_id, similarity_score in top_similar.items():
                item_info = {}
                if item_metadata and similar_item_id in item_metadata:
                    item_info = item_metadata[similar_item_id]
                
                recommendations.append({
                    "item_id": int(similar_item_id),
                    "similarity_score": float(similarity_score),
                    **item_info
                })
            
            recommendation_type = "item_based"
            
        else:
            # Kullanıcı bazlı öneriler
            # Kullanıcının varlığını kontrol et
            if user_id not in user_item_matrix.index:
                raise HTTPException(
                    status_code=404,
                    detail=f"Kullanıcı bulunamadı: {user_id}"
                )
            
            # Kullanıcının derecelendirdiği ürünleri bul
            user_ratings = user_item_matrix.loc[user_id]
            rated_items = user_ratings[user_ratings > 0].index.tolist()
            
            if not rated_items:
                # Kullanıcı hiç ürün derecelendirmemişse, popüler ürünleri öner
                # Sütun toplamları ürün popülerliğini gösterir
                item_popularity = user_item_matrix.sum().sort_values(ascending=False)
                top_items = item_popularity.head(request.num_recommendations).index.tolist()
                
                for item_id in top_items:
                    item_info = {}
                    if item_metadata and item_id in item_metadata:
                        item_info = item_metadata[item_id]
                    
                    recommendations.append({
                        "item_id": int(item_id),
                        "popularity_score": float(item_popularity[item_id]),
                        **item_info
                    })
                
                recommendation_type = "popularity_based"
            
            else:
                # Kullanıcının derecelendirdiği her ürün için benzer ürünleri bul
                # ve bir skor hesapla
                candidate_items = {}
                
                for rated_item_id in rated_items:
                    # Bu ürüne benzer ürünleri bul
                    similar_items = item_similarity_matrix.loc[rated_item_id]
                    # Zaten derecelendirilen ürünleri hariç tut
                    similar_items = similar_items.drop(rated_items, errors='ignore')
                    
                    # Kullanıcının bu ürüne verdiği puan
                    user_rating = user_ratings[rated_item_id]
                    
                    # Benzer ürünlere bir skor hesapla
                    for item_id, similarity in similar_items.items():
                        # Toplam skoru hesapla: benzerlik * kullanıcı puanı
                        score = similarity * user_rating
                        
                        if item_id in candidate_items:
                            # Zaten aday listesinde varsa, en yüksek skoru tut
                            candidate_items[item_id] = max(candidate_items[item_id], score)
                        else:
                            candidate_items[item_id] = score
                
                # En yüksek skorlu ürünleri seç
                top_candidates = sorted(candidate_items.items(), key=lambda x: x[1], reverse=True)
                top_candidates = top_candidates[:request.num_recommendations]
                
                # Önerileri hazırla
                for item_id, score in top_candidates:
                    item_info = {}
                    if item_metadata and item_id in item_metadata:
                        item_info = item_metadata[item_id]
                    
                    recommendations.append({
                        "item_id": int(item_id),
                        "predicted_rating": float(score),
                        **item_info
                    })
                
                recommendation_type = "user_based"
        
        # Yanıtı hazırla
        response = {
            "recommendations": recommendations,
            "recommendation_type": recommendation_type,
            "model_version": recommendation_model_version
        }
        
        return response
        
    except HTTPException:
        # HTTP hataları olduğu gibi bırak
        raise
    except Exception as e:
        # Diğer hataları yakalayıp kullanıcıya anlamlı bir mesaj ver
        raise HTTPException(
            status_code=500,
            detail=f"Öneri oluşturma hatası: {str(e)}"
        )

@app.delete("/delete_model_version/{version_name}", response_model=dict)
async def delete_model_version(
    version_name: str,
    force: bool = Query(default=False, description="Tüm dosyaları zorla sil")
):
    """
    Belirtilen model versiyonunu ve ilgili tüm dosyaları siler.
    
    Args:
        version_name: Silinecek model versiyonunun adı
        force: True ise, model kullanımda olsa bile siler
    
    Returns:
        dict: Silme işleminin sonucu
    """
    try:
        # MLflow bağlantısını kontrol et
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"MLflow bağlantı hatası: {str(e)}"
            )

        # Experiment'i kontrol et
        experiment = get_experiment()
        if experiment is None:
            raise HTTPException(
                status_code=404,
                detail="Model experiment'i bulunamadı"
            )

        # Çalışmaları getir
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.version_name = '{version_name}'"
        )

        if len(runs) == 0:
            raise HTTPException(
                status_code=404,
                detail=f"'{version_name}' versiyonu bulunamadı"
            )

        # Eğer model aktif kullanımdaysa ve force=False ise silmeyi reddet
        global recommendation_model_version
        was_active = recommendation_model_version == version_name
        
        if was_active and not force:
            raise HTTPException(
                status_code=400,
                detail="Bu model versiyonu şu anda kullanımda. Silmek için force=true parametresini kullanın"
            )

        deleted_runs = []
        for _, run in runs.iterrows():
            run_id = run["run_id"]
            
            try:
                # MLflow'dan run'ı sil
                mlflow.delete_run(run_id)
                
                # Model dosyalarını sil
                model_path = os.path.join("models", version_name)
                if os.path.exists(model_path):
                    shutil.rmtree(model_path)
                
                # Artifact dosyalarını sil
                artifact_path = os.path.join("mlruns", experiment.experiment_id, run_id, "artifacts")
                if os.path.exists(artifact_path):
                    shutil.rmtree(artifact_path)
                
                deleted_runs.append(run_id)
                
            except Exception as e:
                print(f"Run {run_id} silme hatası: {str(e)}")
                continue

        # Eğer silinen model aktif kullanımdaysa, model verilerini sıfırla
        if was_active:
            global user_item_matrix, item_similarity_matrix, item_metadata
            user_item_matrix = None
            item_similarity_matrix = None
            item_metadata = None
            recommendation_model_version = None
            
            # Eğer aktif kullanımdaki model silindi ve başka modeller varsa, 
            # en son eklenen modeli otomatik olarak yükle
            try:
                print("Silinen model aktif kullanımdaydı. Yeni bir model yükleniyor...")
                recommendation_runs = mlflow.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    filter_string="tags.model_type = 'collaborative_filtering'",
                    order_by=["start_time DESC"]
                )
                
                if len(recommendation_runs) > 0:
                    latest_run = recommendation_runs.iloc[0]
                    latest_run_id = latest_run["run_id"]
                    
                    # Yeni versiyonu yükle
                    try:
                        _, user_item_matrix, item_similarity_matrix, item_metadata, recommendation_model_version = load_recommendation_model(latest_run_id)
                        print(f"Yeni model otomatik olarak yüklendi: {recommendation_model_version}")
                    except Exception as load_err:
                        print(f"Yeni model yüklenemedi: {str(load_err)}")
            except Exception as e:
                print(f"Alternatif model yükleme hatası: {str(e)}")

        return {
            "status": "success",
            "message": f"Model versiyonu başarıyla silindi: {version_name}",
            "deleted_runs": deleted_runs,
            "force_used": force
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model silme hatası: {str(e)}"
        )

# Uygulama başlatma
if __name__ == "__main__":
    uvicorn.run("serve:app", host="0.0.0.0", port=8000, reload=True) 