import mlflow
import mlflow.sklearn
import mlflow.pyfunc
from fastapi import FastAPI, HTTPException, Query, Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response
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
import logging
from logging.handlers import RotatingFileHandler
import sys
import time

# Log yapılandırması
def setup_logger(name, log_dir="logs", level=logging.INFO, log_type=None):
    """Logger'ı yapılandırır ve döndürür"""
    # Log dizinini oluştur
    if log_type:
        log_dir = os.path.join(log_dir, datetime.now().strftime("%Y%m"), log_type)
    os.makedirs(log_dir, exist_ok=True)
    
    # Logger'ı oluştur
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Handler'ları temizle
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Dosya handler'ı
    log_file = os.path.join(log_dir, f"{name}.log")
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    
    # Konsol handler'ı
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    
    # Handler'ları ekle
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# API Logger ayarları
api_logger = setup_logger(
    name="api_service",
    level=logging.INFO,
    log_dir="logs",
    log_type="api"
)

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # İstek başlangıç zamanı
        start_time = time.time()
        
        # İstek detaylarını al
        method = request.method
        url = str(request.url)
        client_host = request.client.host if request.client else "unknown"
        
        try:
            # İstek gövdesini oku (eğer varsa)
            body = None
            if method in ["POST", "PUT", "PATCH"]:
                try:
                    body = await request.body()
                    body = body.decode() if body else None
                except Exception as e:
                    api_logger.warning(f"Request body okunamadı: {str(e)}")
            
            # İsteği logla
            api_logger.info(f"Request: {method} {url} from {client_host}")
            if body:
                api_logger.debug(f"Request Body: {body}")
            
            # İsteği işle
            response = await call_next(request)
            
            # İşlem süresini hesapla
            duration = time.time() - start_time
            
            # Yanıtı logla
            api_logger.info(
                f"Response: {method} {url} - Status: {response.status_code} - Duration: {duration:.4f}s"
            )
            
            return response
            
        except Exception as e:
            # Hata durumunu logla
            api_logger.error(f"Error processing {method} {url}: {str(e)}")
            raise

# FastAPI uygulaması
app = FastAPI(
    title="MLflow Recommendation System",
    description="Kullanıcı-ürün derecelendirme tahmini ve ürün öneri sistemi",
    version="1.0.0"
)

# Middleware'i ekle
app.add_middleware(RequestLoggingMiddleware)

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

# Ürün verilerini yükle
try:
    products_df = pd.read_csv("data/products.csv")
    api_logger.info(f"Ürün verileri yüklendi: {len(products_df)} ürün")
except Exception as e:
    api_logger.error(f"Ürün verileri yüklenemedi: {str(e)}")
    products_df = pd.DataFrame()

# Etkileşim verilerini yükle
try:
    interactions_df = pd.read_csv("data/interactions.csv")
    api_logger.info(f"Etkileşim verileri yüklendi: {len(interactions_df)} etkileşim")
except Exception as e:
    api_logger.error(f"Etkileşim verileri yüklenemedi: {str(e)}")
    interactions_df = pd.DataFrame()

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

# Yeni veri modeli
class ItemIdsRequest(BaseModel):
    """Toplu ürün detayları için istek modeli"""
    item_ids: List[int]

# ============== ORTAK FONKSİYONLAR ==============
def get_experiment():
    """MLflow deneyini getirir"""
    experiment = mlflow.get_experiment_by_name("recommendation-system")
    if experiment is None:
        api_logger.warning("Experiment bulunamadı! Yeni bir experiment oluşturuluyor.")
        experiment_id = mlflow.create_experiment("recommendation-system")
        experiment = mlflow.get_experiment(experiment_id)
        api_logger.info(f"Yeni experiment oluşturuldu. ID: {experiment_id}")
    return experiment

# ============== TAHMİN SERVİSİ FONKSİYONLARI ==============
def load_rating_model(run_id):
    """Derecelendirme tahmin modelini yükler"""
    client = mlflow.tracking.MlflowClient(MLFLOW_TRACKING_URI)
    api_logger.info(f"Derecelendirme modeli yükleniyor: {run_id}")
    
    try:
        # Önce modeli MLflow'dan yüklemeyi dene
        model_path = f"runs:/{run_id}/model"
        model = mlflow.pyfunc.load_model(model_path)
        api_logger.info(f"Model MLflow'dan başarıyla yüklendi: {model_path}")
        
        # Version name'i ve metrikleri al
        run = client.get_run(run_id)
        version_name = run.data.tags.get("version_name", f"run_{run_id[:8]}")
        
        # Son çalışan tahmin modelinin bilgilerini kaydet
        try:
            # Metrikleri al
            metrics = {}
            api_logger.debug("Metrikleri alıyorum...")
            try:
                run_metrics = run.data.metrics
                api_logger.debug(f"MLflow'dan alınan metrikler: {run_metrics}")
                metrics = {key: float(value) for key, value in run_metrics.items()}
                api_logger.debug(f"İşlenmiş metrikler: {metrics}")
            except Exception as metric_error:
                api_logger.error(f"Metrikler alınırken hata: {str(metric_error)}")
            
            last_working_model = {
                "run_id": run_id,
                "version_name": version_name,
                "timestamp": datetime.now().isoformat(),
                "model_type": "rating",
                "metrics": metrics
            }
            api_logger.debug(f"Kaydedilecek model bilgileri: {json.dumps(last_working_model, indent=2)}")
            
            # Dosya yolunu oluştur
            model_info_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "last_working_prediction_model.json")
            api_logger.debug(f"Dosya yolu: {model_info_path}")
            
            # JSON dosyasını kaydet
            with open(model_info_path, "w", encoding="utf-8") as f:
                json.dump(last_working_model, f, indent=4, ensure_ascii=False)
            api_logger.info(f"Son çalışan tahmin modeli bilgileri başarıyla kaydedildi: {model_info_path}")
            
            # Dosyanın içeriğini kontrol et
            try:
                with open(model_info_path, "r", encoding="utf-8") as f:
                    saved_content = json.load(f)
                api_logger.debug(f"Kaydedilen dosya içeriği: {json.dumps(saved_content, indent=2)}")
            except Exception as read_error:
                api_logger.error(f"Kaydedilen dosya kontrol edilirken hata: {str(read_error)}")
            
        except Exception as e:
            api_logger.error(f"Son model bilgileri kaydedilemedi: {str(e)}")
            api_logger.error(f"Hata detayı: {type(e).__name__}")
            import traceback
            api_logger.error(f"Hata stack trace: {traceback.format_exc()}")
        
        return model, version_name
        
    except Exception as mlflow_error:
        api_logger.error(f"MLflow'dan model yükleme hatası: {str(mlflow_error)}")
        try:
            # Eğer MLflow'dan yükleme başarısız olursa, doğrudan dosyadan yüklemeyi dene
            api_logger.info("Dosyadan model yükleme deneniyor...")
            model = pycaret_load_model("model.pkl")
            return model, run_id
        except Exception as file_error:
            api_logger.error(f"Dosyadan model yükleme hatası: {str(file_error)}")
            raise Exception("Model yüklenemedi!")

def get_recent_predictions(hours=24):
    """Son belirli saatteki tahminleri getirir"""
    cutoff_time = datetime.now() - timedelta(hours=hours)
    predictions = [p for p in recent_predictions if p.timestamp > cutoff_time]
    api_logger.debug(f"Son {hours} saatteki tahmin sayısı: {len(predictions)}")
    return predictions

def get_actual_ratings():
    """Gerçek değerlendirmeleri simüle eder"""
    predictions = get_recent_predictions()
    actual_ratings = [np.random.normal(p.prediction, 0.5) for p in predictions]
    api_logger.debug(f"Simüle edilen gerçek değerlendirme sayısı: {len(actual_ratings)}")
    return actual_ratings

def calculate_metrics(actual, predicted):
    """Model performans metriklerini hesaplar"""
    if not actual or not predicted:
        api_logger.warning("Metrik hesaplaması için yeterli veri yok")
        return None
    
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    mse = np.mean((actual - predicted) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(actual - predicted))
    r2 = 1 - np.sum((actual - predicted) ** 2) / np.sum((actual - np.mean(actual)) ** 2)
    
    metrics = {
        "MSE": float(mse),
        "RMSE": float(rmse),
        "MAE": float(mae),
        "R2": float(r2)
    }
    
    api_logger.info(f"Performans metrikleri hesaplandı: MSE={mse:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
    return metrics

def validate_model_performance(metrics):
    """Model performansının kabul edilebilir seviyede olup olmadığını kontrol eder"""
    if not metrics:
        api_logger.warning("Performans validasyonu için metrik bulunamadı")
        return
        
    threshold = {
        "rmse": 1.0,  # 1 yıldızdan fazla hata kabul edilemez
        "r2": 0.6     # En az %60 açıklayıcılık gücü olmalı
    }
    
    api_logger.debug(f"Performans eşik değerleri: {threshold}")
    
    if "rmse" in metrics and metrics["rmse"] > threshold["rmse"]:
        error_msg = f"Model RMSE ({metrics['rmse']:.2f}) çok yüksek!"
        api_logger.error(error_msg)
        raise ValueError(error_msg)
    
    if "r2" in metrics and metrics["r2"] < threshold["r2"]:
        error_msg = f"Model R2 skoru ({metrics['r2']:.2f}) çok düşük!"
        api_logger.error(error_msg)
        raise ValueError(error_msg)
    
    api_logger.info("Model performansı kabul edilebilir seviyede")

# ============== ÖNERİ SERVİSİ FONKSİYONLARI ==============
def load_recommendation_model(run_id):
    """Öneri sistemi modelini yükler"""
    client = mlflow.tracking.MlflowClient(MLFLOW_TRACKING_URI)
    api_logger.info(f"Öneri modeli yükleniyor: {run_id}")
    
    try:
        # MLflow'dan modeli yüklemeyi dene
        model_path = f"runs:/{run_id}/model"
        model = mlflow.sklearn.load_model(model_path)
        api_logger.info(f"Model MLflow'dan başarıyla yüklendi: {model_path}")
        
        # Version name'i al
        run = client.get_run(run_id)
        version_name = run.data.tags.get("version_name", f"run_{run_id[:8]}")
        
        # Ürün metadatalarını yükle
        item_metadata_path = client.download_artifacts(run_id, "item_metadata.pkl")
        item_metadata = joblib.load(item_metadata_path)
        api_logger.info(f"Ürün metadataları yüklendi: {item_metadata_path}")
        
        # Kullanıcı-ürün matrisini yükle
        user_item_matrix_path = client.download_artifacts(run_id, "user_item_matrix.pkl")
        user_item_matrix = joblib.load(user_item_matrix_path)
        api_logger.info(f"Kullanıcı-ürün matrisi yüklendi: {user_item_matrix_path}")
        
        # Ürün benzerlik matrisini yükle
        item_similarity_path = client.download_artifacts(run_id, "item_similarity.pkl")
        item_similarity_matrix = joblib.load(item_similarity_path)
        api_logger.info(f"Ürün benzerlik matrisi yüklendi: {item_similarity_path}")
        
        # Son çalışan modelin bilgilerini kaydet
        try:
            # Metrikleri al
            metrics = {}
            api_logger.debug("Metrikleri alıyorum...")
            try:
                run_metrics = run.data.metrics
                api_logger.debug(f"MLflow'dan alınan metrikler: {run_metrics}")
                metrics = {key: float(value) for key, value in run_metrics.items()}
                api_logger.debug(f"İşlenmiş metrikler: {metrics}")
            except Exception as metric_error:
                api_logger.error(f"Metrikler alınırken hata: {str(metric_error)}")
            
            last_working_model = {
                "run_id": run_id,
                "version_name": version_name,
                "timestamp": datetime.now().isoformat(),
                "model_type": "collaborative_filtering",
                "metrics": metrics
            }
            api_logger.debug(f"Kaydedilecek model bilgileri: {json.dumps(last_working_model, indent=2)}")
            
            # Dosya yolunu oluştur
            model_info_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "last_working_recommendation_model.json")
            api_logger.debug(f"Dosya yolu: {model_info_path}")
            
            # JSON dosyasını kaydet
            with open(model_info_path, "w", encoding="utf-8") as f:
                json.dump(last_working_model, f, indent=4, ensure_ascii=False)
            api_logger.info(f"Son çalışan öneri modeli bilgileri başarıyla kaydedildi: {model_info_path}")
            
            # Dosyanın içeriğini kontrol et
            try:
                with open(model_info_path, "r", encoding="utf-8") as f:
                    saved_content = json.load(f)
                api_logger.debug(f"Kaydedilen dosya içeriği: {json.dumps(saved_content, indent=2)}")
            except Exception as read_error:
                api_logger.error(f"Kaydedilen dosya kontrol edilirken hata: {str(read_error)}")
            
        except Exception as e:
            api_logger.error(f"Son model bilgileri kaydedilemedi: {str(e)}")
            api_logger.error(f"Hata detayı: {type(e).__name__}")
            import traceback
            api_logger.error(f"Hata stack trace: {traceback.format_exc()}")
        
        return model, user_item_matrix, item_similarity_matrix, item_metadata, version_name
        
    except Exception as e:
        api_logger.error(f"MLflow'dan model yükleme başarısız: {str(e)}")
        raise Exception(f"Model yüklenemedi: {str(e)}")

# ============== BAŞLANGIÇ FONKSİYONU ==============
@app.on_event("startup")
async def startup_event():
    """Uygulama başlatıldığında çalışacak kod"""
    global rating_model, rating_model_version
    global user_item_matrix, item_similarity_matrix, item_metadata, recommendation_model_version
    
    api_logger.info("Servis başlatılıyor...")
    
    try:
        # MLflow'a bağlan
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        api_logger.info(f"MLflow bağlantısı kuruldu: {MLFLOW_TRACKING_URI}")
        
        # Son deneyi bul
        experiment = get_experiment()
        if experiment is None:
            api_logger.warning("Experiment bulunamadı! Yeni bir experiment oluşturuluyor.")
            experiment_id = mlflow.create_experiment("recommendation-system")
            experiment = mlflow.get_experiment(experiment_id)
            api_logger.info(f"Yeni experiment oluşturuldu. ID: {experiment_id}")
        
        # Tüm çalışmaları getir
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"]
        )
        
        if len(runs) == 0:
            api_logger.warning("Hiç model çalışması bulunamadı! Servis yine de başlatılıyor...")
            return
        
        # Öneri modelini yükle (öncelikli)
        api_logger.info("Öneri modeli aranıyor...")
        recommendation_filter = "tags.model_type = 'collaborative_filtering'"
        recommendation_runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=recommendation_filter,
            order_by=["start_time DESC"]
        )
        
        if len(recommendation_runs) > 0:
            api_logger.info(f"{len(recommendation_runs)} adet öneri modeli bulundu.")
            for _, run in recommendation_runs.iterrows():
                run_id = run["run_id"]
                try:
                    api_logger.info(f"Öneri modeli yüklemeyi deniyorum (Run ID: {run_id})")
                    
                    # Version name'i tag'den al
                    version_name = None
                    if "tags.version_name" in run and pd.notna(run["tags.version_name"]):
                        version_name = run["tags.version_name"]
                    else:
                        version_name = f"run_{run_id[:8]}"
                    
                    api_logger.debug(f"Version: {version_name}")
                    
                    # Modeli yükle
                    _, user_item_matrix, item_similarity_matrix, item_metadata, recommendation_model_version = load_recommendation_model(run_id)
                    api_logger.info(f"Öneri modeli başarıyla yüklendi: {recommendation_model_version}")
                    break
                except Exception as e:
                    api_logger.error(f"Model yükleme hatası: {str(e)}")
                    continue
            
            if recommendation_model_version is None:
                api_logger.warning("Hiçbir öneri modeli yüklenemedi!")
        else:
            api_logger.warning("Öneri modeli bulunamadı!")
        
        # Derecelendirme modelini yükle
        api_logger.info("Derecelendirme modeli aranıyor...")
        rating_filter = "tags.model_type = 'rating'"
        rating_runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=rating_filter,
            order_by=["start_time DESC"]
        )
        
        if len(rating_runs) > 0:
            api_logger.info(f"{len(rating_runs)} adet derecelendirme modeli bulundu.")
            for _, run in rating_runs.iterrows():
                run_id = run["run_id"]
                try:
                    api_logger.info(f"Derecelendirme modeli yüklemeyi deniyorum (Run ID: {run_id})")
                    
                    # Version name'i tag'den al
                    version_name = None
                    if "tags.version_name" in run and pd.notna(run["tags.version_name"]):
                        version_name = run["tags.version_name"]
                    else:
                        version_name = f"run_{run_id[:8]}"
                    
                    rating_model, _ = load_rating_model(run_id)
                    rating_model_version = version_name
                    api_logger.info(f"Derecelendirme modeli başarıyla yüklendi: {rating_model_version}")
                    break
                except Exception as e:
                    api_logger.error(f"Model yükleme hatası: {str(e)}")
                    continue
            
            if rating_model_version is None:
                api_logger.warning("Hiçbir derecelendirme modeli yüklenemedi!")
        else:
            api_logger.warning("Derecelendirme modeli bulunamadı!")
            
    except Exception as e:
        api_logger.error(f"Başlangıç hatası: {str(e)}")
        api_logger.warning("Servis yine de başlatılıyor...")
    
    api_logger.info("Servis başlatma işlemi tamamlandı.")

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
            {"path": "/delete_model_version/{version_name}", "description": "Belirtilen model versiyonunu ve ilgili tüm dosyaları siler"},
            {"path": "/item/{item_id}", "description": "Ürün detaylarını getir"},
            {"path": "/items", "description": "Birden fazla ürünün detaylarını getir"},
            {"path": "/user_interactions/{user_id}", "description": "Kullanıcının etkileşimlerini getir"},
            {"path": "/popular_items", "description": "En popüler ürünleri getir"},
            {"path": "/metrics", "description": "Tüm sistem metriklerini getir"}
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
        
        # Modeli yükle ve bilgileri kaydet
        loaded_model, loaded_version = load_rating_model(run_id)
        rating_model = loaded_model
        rating_model_version = loaded_version
        
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
    
    api_logger.info(f"Tahmin isteği alındı - User ID: {request.user_id}, Item ID: {request.item_id}")
    
    if enable_ab_testing:
        api_logger.info("A/B testing aktif - Rastgele model versiyonu seçiliyor")
        versions = await list_versions()
        rating_versions = [v for v in versions if v["model_type"] == "rating"]
        if rating_versions:
            selected_version = random.choice(rating_versions)["version_name"]
            api_logger.info(f"A/B testing için seçilen versiyon: {selected_version}")
            await load_rating_version(selected_version)
    elif version_name:
        api_logger.info(f"Belirtilen model versiyonu yükleniyor: {version_name}")
        await load_rating_version(version_name)
    
    if rating_model is None:
        api_logger.error("Derecelendirme modeli yüklenmedi!")
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
        
        api_logger.debug(f"Girdi verisi:\n{input_df}")
        
        # Tahmin yap
        start_time = datetime.now()
        prediction = float(rating_model.predict(input_df)[0])
        response_time = (datetime.now() - start_time).total_seconds()
        
        api_logger.info(f"Tahmin sonucu: {prediction:.2f} (yanıt süresi: {response_time:.3f}s)")
        
        # Güven aralığı hesapla
        confidence_interval = (prediction - 0.5, prediction + 0.5)
        
        # Tahmin kaydını sakla
        recent_predictions.append(
            PredictionRecord(prediction=prediction)
        )
        
        api_logger.debug(f"Güven aralığı: {confidence_interval}")
        
        return RatingResponse(
            predicted_rating=prediction,
            confidence_interval=confidence_interval,
            model_version=rating_model_version
        )
    except Exception as e:
        api_logger.error(f"Tahmin hatası: {str(e)}")
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
            "model_info": None
        },
        "rating_model": {
            "status": "not_loaded",
            "version": None,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "metrics": None,
            "model_info": None
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
                "sparsity": float(len(non_zero_ratings) / (user_item_matrix.shape[0] * user_item_matrix.shape[1]))
            }

            response["recommendation_model"].update({
                "status": "healthy",
                "version": recommendation_model_version,
                "metrics": metrics,
                "model_info": model_info
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
                    
                    # Rating model için model bilgilerini hazırla
                    model_info = {
                        "version": rating_model_version,
                        "features": ["user_id", "item_id", "user_age", "user_gender", "item_category", "item_price"],
                        "target": "rating",
                        "model_type": "rating"
                    }
                    
                    response["rating_model"].update({
                        "status": status,
                        "message": message,
                        "version": rating_model_version,
                        "metrics": metrics,
                        "model_info": model_info,
                        "sample_size": len(predictions)
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
    
    api_logger.info(f"Öneri isteği alındı - User ID: {request.user_id}, Item ID: {request.item_id if request.item_id else 'None'}")
    
    # Model kontrolü
    if user_item_matrix is None or item_similarity_matrix is None:
        api_logger.error("Öneri modeli henüz yüklenmemiş!")
        raise HTTPException(
            status_code=500,
            detail="Öneri modeli henüz yüklenmemiş!"
        )
    
    # Versiyon kontrolü
    if version_name is not None:
        api_logger.info(f"Belirtilen model versiyonu yükleniyor: {version_name}")
        await load_recommendation_version(version_name)
    
    # Kullanıcı kontrolü
    user_id = request.user_id
    
    # Öneriler
    recommendations = []
    recommendation_type = ""
    
    try:
        start_time = datetime.now()
        
        if request.item_id is not None:
            # Ürün bazlı öneriler
            item_id = request.item_id
            api_logger.info(f"Ürün bazlı öneri yapılıyor - Item ID: {item_id}")
            
            # Ürünün varlığını kontrol et
            if item_id not in item_similarity_matrix.index:
                api_logger.error(f"Ürün bulunamadı: {item_id}")
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
            
            api_logger.debug(f"En benzer {len(top_similar)} ürün bulundu")
            
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
            api_logger.info(f"Kullanıcı bazlı öneri yapılıyor - User ID: {user_id}")
            
            # Kullanıcının varlığını kontrol et
            if user_id not in user_item_matrix.index:
                api_logger.error(f"Kullanıcı bulunamadı: {user_id}")
                raise HTTPException(
                    status_code=404,
                    detail=f"Kullanıcı bulunamadı: {user_id}"
                )
            
            # Kullanıcının derecelendirdiği ürünleri bul
            user_ratings = user_item_matrix.loc[user_id]
            rated_items = user_ratings[user_ratings > 0].index.tolist()
            
            if not rated_items:
                api_logger.info(f"Kullanıcı henüz hiç ürün derecelendirmemiş - Popüler ürünler önerilecek")
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
                api_logger.info(f"Kullanıcının {len(rated_items)} adet derecelendirmesi bulundu")
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
                
                api_logger.debug(f"En yüksek skorlu {len(top_candidates)} ürün seçildi")
                
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
        
        response_time = (datetime.now() - start_time).total_seconds()
        api_logger.info(f"Öneri tamamlandı - {len(recommendations)} öneri, {recommendation_type} yöntemi (yanıt süresi: {response_time:.3f}s)")
        
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
        api_logger.error(f"Öneri oluşturma hatası: {str(e)}")
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

@app.get("/item/{item_id}")
async def get_item_details(item_id: int):
    """Ürün detaylarını getirir"""
    try:
        # Ürünü bul
        item = products_df[products_df['item_id'] == item_id]
        
        if item.empty:
            raise HTTPException(
                status_code=404,
                detail=f"Ürün bulunamadı: {item_id}"
            )
        
        # Ürün bilgilerini hazırla
        item_info = item.iloc[0].to_dict()
        
        # Eğer öneri modeli yüklüyse, ürünün ortalama puanını ve kaç kişinin puanladığını ekle
        if user_item_matrix is not None:
            item_ratings = user_item_matrix[item_id]
            non_zero_ratings = item_ratings[item_ratings > 0]
            
            item_info.update({
                "average_rating": float(non_zero_ratings.mean()) if len(non_zero_ratings) > 0 else None,
                "rating_count": int(len(non_zero_ratings)),
                "rating_distribution": {
                    "1": int(len(non_zero_ratings[non_zero_ratings == 1])),
                    "2": int(len(non_zero_ratings[non_zero_ratings == 2])),
                    "3": int(len(non_zero_ratings[non_zero_ratings == 3])),
                    "4": int(len(non_zero_ratings[non_zero_ratings == 4])),
                    "5": int(len(non_zero_ratings[non_zero_ratings == 5]))
                }
            })
        
        return item_info
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ürün detayları alınırken hata oluştu: {str(e)}"
        )

@app.get("/items")
async def get_items(ids: str = Query(..., description="Virgülle ayrılmış ürün ID'leri (örn: 456,789,123)")):
    """Birden fazla ürünün detaylarını getirir"""
    try:
        # Ürün verilerinin yüklü olup olmadığını kontrol et
        if products_df.empty:
            raise HTTPException(
                status_code=500,
                detail="Ürün verileri yüklenemedi"
            )
        
        # ID'leri parse et
        try:
            item_ids = [int(id.strip()) for id in ids.split(",")]
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Geçersiz ID formatı. Örnek format: 456,789,123"
            )
        
        # Ürünleri bul
        items = products_df[products_df['item_id'].isin(item_ids)]
        
        # Bulunamayan ID'leri tespit et
        missing_ids = set(item_ids) - set(items['item_id'])
        if missing_ids:
            raise HTTPException(
                status_code=404,
                detail=f"Bu ID'lere sahip ürünler bulunamadı: {list(missing_ids)}"
            )
        
        # Ürün bilgilerini hazırla
        items_info = []
        for _, item in items.iterrows():
            item_info = item.to_dict()
            item_id = item['item_id']
            
            # Sayısal değerleri float'a çevir
            for key in ['item_price', 'item_rating_avg']:
                if key in item_info and pd.notna(item_info[key]):
                    item_info[key] = float(item_info[key])
            
            # Integer değerleri int'e çevir
            for key in ['item_id', 'stock_quantity']:
                if key in item_info and pd.notna(item_info[key]):
                    item_info[key] = int(item_info[key])
            
            # Eğer öneri modeli yüklüyse, ürünün ortalama puanını ve kaç kişinin puanladığını ekle
            if user_item_matrix is not None:
                item_ratings = user_item_matrix[item_id]
                non_zero_ratings = item_ratings[item_ratings > 0]
                
                item_info.update({
                    "rating_stats": {
                        "average_rating": float(non_zero_ratings.mean()) if len(non_zero_ratings) > 0 else None,
                        "rating_count": int(len(non_zero_ratings)),
                        "rating_distribution": {
                            "1": int(len(non_zero_ratings[non_zero_ratings == 1])),
                            "2": int(len(non_zero_ratings[non_zero_ratings == 2])),
                            "3": int(len(non_zero_ratings[non_zero_ratings == 3])),
                            "4": int(len(non_zero_ratings[non_zero_ratings == 4])),
                            "5": int(len(non_zero_ratings[non_zero_ratings == 5]))
                        }
                    }
                })
            
            items_info.append(item_info)
        
        return {
            "items": items_info,
            "total": len(items_info)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ürün detayları alınırken hata oluştu: {str(e)}"
        )

@app.get("/user_interactions/{user_id}")
async def get_user_interactions(
    user_id: int,
    limit: Optional[int] = Query(None, description="Maksimum etkileşim sayısı"),
    sort_by: str = Query("timestamp", description="Sıralama kriteri (timestamp, rating, item_id)"),
    order: str = Query("desc", description="Sıralama yönü (asc, desc)")
):
    """Kullanıcının etkileşimlerini getirir"""
    try:
        # Etkileşim verilerinin yüklü olup olmadığını kontrol et
        if interactions_df.empty:
            raise HTTPException(
                status_code=500,
                detail="Etkileşim verileri yüklenemedi"
            )
        
        # Kullanıcının etkileşimlerini bul
        user_interactions = interactions_df[interactions_df['user_id'] == user_id].copy()
        
        if user_interactions.empty:
            raise HTTPException(
                status_code=404,
                detail=f"Kullanıcı bulunamadı veya hiç etkileşimi yok: {user_id}"
            )
        
        # Sıralama yönünü kontrol et
        if order.lower() not in ['asc', 'desc']:
            raise HTTPException(
                status_code=400,
                detail="Geçersiz sıralama yönü. 'asc' veya 'desc' kullanın"
            )
        
        # Sıralama kriterini kontrol et
        if sort_by not in user_interactions.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Geçersiz sıralama kriteri. Kullanılabilir kriterler: {list(user_interactions.columns)}"
            )
        
        # Sırala
        ascending = order.lower() == 'asc'
        user_interactions = user_interactions.sort_values(by=sort_by, ascending=ascending)
        
        # Limit uygula
        if limit is not None:
            user_interactions = user_interactions.head(limit)
        
        # Etkileşimleri hazırla
        interactions = []
        for _, interaction in user_interactions.iterrows():
            interaction_info = interaction.to_dict()
            
            # Sayısal değerleri düzelt
            for key, value in interaction_info.items():
                if pd.isna(value):
                    interaction_info[key] = None
                elif isinstance(value, (np.int64, np.int32)):
                    interaction_info[key] = int(value)
                elif isinstance(value, (np.float64, np.float32)):
                    interaction_info[key] = float(value)
            
            # Ürün bilgilerini ekle
            item_id = interaction_info.get('item_id')
            if item_id is not None and not products_df.empty:
                item = products_df[products_df['item_id'] == item_id]
                if not item.empty:
                    item_info = item.iloc[0].to_dict()
                    # Ürün bilgilerini düzelt
                    for key in ['item_price', 'item_rating_avg']:
                        if key in item_info and pd.notna(item_info[key]):
                            item_info[key] = float(item_info[key])
                    for key in ['item_id', 'stock_quantity']:
                        if key in item_info and pd.notna(item_info[key]):
                            item_info[key] = int(item_info[key])
                    interaction_info['item'] = item_info
            
            interactions.append(interaction_info)
        
        return {
            "user_id": user_id,
            "interactions": interactions,
            "total": len(interactions)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Kullanıcı etkileşimleri alınırken hata oluştu: {str(e)}"
        )

@app.get("/popular_items")
async def get_popular_items(
    limit: Optional[int] = Query(10, description="Maksimum ürün sayısı", ge=1, le=100)
):
    """En popüler ürünleri getirir"""
    try:
        # Öneri modelinin yüklü olup olmadığını kontrol et
        if user_item_matrix is None:
            raise HTTPException(
                status_code=500,
                detail="Öneri modeli henüz yüklenmemiş"
            )
        
        # Ürün verilerinin yüklü olup olmadığını kontrol et
        if products_df.empty:
            raise HTTPException(
                status_code=500,
                detail="Ürün verileri yüklenemedi"
            )
        
        # Her ürünün toplam derecelendirme sayısını hesapla
        item_popularity = user_item_matrix[user_item_matrix > 0].count()
        
        # En popüler ürünleri al
        top_items = item_popularity.sort_values(ascending=False).head(limit)
        
        # Popüler ürünlerin detaylarını hazırla
        popular_items = []
        for item_id, rating_count in top_items.items():
            # Ürün bilgilerini al
            item = products_df[products_df['item_id'] == item_id]
            if item.empty:
                continue
                
            item_info = item.iloc[0].to_dict()
            
            # Sayısal değerleri düzelt
            for key in ['item_price', 'item_rating_avg']:
                if key in item_info and pd.notna(item_info[key]):
                    item_info[key] = float(item_info[key])
            for key in ['item_id', 'stock_quantity']:
                if key in item_info and pd.notna(item_info[key]):
                    item_info[key] = int(item_info[key])
            
            # Derecelendirme istatistiklerini ekle
            item_ratings = user_item_matrix[item_id]
            non_zero_ratings = item_ratings[item_ratings > 0]
            
            item_info.update({
                "rating_stats": {
                    "average_rating": float(non_zero_ratings.mean()) if len(non_zero_ratings) > 0 else None,
                    "rating_count": int(rating_count),
                    "rating_distribution": {
                        "1": int(len(non_zero_ratings[non_zero_ratings == 1])),
                        "2": int(len(non_zero_ratings[non_zero_ratings == 2])),
                        "3": int(len(non_zero_ratings[non_zero_ratings == 3])),
                        "4": int(len(non_zero_ratings[non_zero_ratings == 4])),
                        "5": int(len(non_zero_ratings[non_zero_ratings == 5]))
                    }
                }
            })
            
            popular_items.append(item_info)
        
        return {
            "items": popular_items,
            "total": len(popular_items)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Popüler ürünler alınırken hata oluştu: {str(e)}"
        )

@app.get("/metrics")
async def get_metrics(version_name: Optional[str] = Query(None, description="Model versiyonu")):
    """Sistem metriklerini döndürür"""
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

        # Eğer versiyon belirtilmişse, o versiyonun run'ını bul
        if version_name:
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=f"tags.mlflow.runName = '{version_name}'"
            )

            if runs.empty:
                raise HTTPException(
                    status_code=404,
                    detail=f"Model versiyonu bulunamadı: {version_name}"
                )

            run = runs.iloc[0]
            metrics = {}
            for col in run.index:
                if col.startswith("metrics."):
                    metric_name = col.replace("metrics.", "").lower()
                    metric_value = float(run[col])
                    if not pd.isna(metric_value):
                        metrics[metric_name] = metric_value

            return {
                "model_version": version_name,
                "run_id": run["run_id"],
                "metrics": metrics,
                "tags": {k.replace("tags.", ""): v for k, v in run.items() if k.startswith("tags.") and pd.notna(v)},
                "status": "active" if version_name in [recommendation_model_version, rating_model_version] else "inactive"
            }

        # Versiyon belirtilmemişse tüm sistem metriklerini getir
        response = {
            "system_status": "healthy",
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "models": {
                "recommendation": {
                    "status": "not_loaded",
                    "version": None,
                    "metrics": None
                },
                "rating": {
                    "status": "not_loaded",
                    "version": None,
                    "metrics": None
                }
            },
            "data_stats": {
                "products": None,
                "interactions": None
            }
        }

        # Öneri modeli metrikleri
        if (user_item_matrix is not None and 
            item_similarity_matrix is not None and 
            item_metadata is not None):
            
            non_zero_ratings = user_item_matrix.values[user_item_matrix.values != 0]
            rec_metrics = {
                "average_rating": float(np.mean(non_zero_ratings)) if len(non_zero_ratings) > 0 else 0,
                "rating_count": int(len(non_zero_ratings)),
                "unique_users": int(len(user_item_matrix.index)),
                "unique_items": int(len(user_item_matrix.columns)),
                "sparsity": float(len(non_zero_ratings) / (user_item_matrix.shape[0] * user_item_matrix.shape[1]))
            }
            
            response["models"]["recommendation"].update({
                "status": "healthy",
                "version": recommendation_model_version,
                "metrics": rec_metrics
            })
        
        # Rating modeli metrikleri
        if rating_model is not None:
            experiment = get_experiment()
            if experiment:
                filter_string = f"tags.model_type = 'rating'"
                if rating_model_version:
                    filter_string += f" and tags.mlflow.runName = '{rating_model_version}'"
                
                runs = mlflow.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    filter_string=filter_string
                )
                
                if not runs.empty:
                    run = runs.iloc[0]
                    rating_metrics = {}
                    for col in run.index:
                        if col.startswith("metrics."):
                            metric_name = col.replace("metrics.", "").lower()
                            metric_value = float(run[col])
                            if not pd.isna(metric_value):
                                rating_metrics[metric_name] = metric_value
                    
                    response["models"]["rating"].update({
                        "status": "healthy",
                        "version": rating_model_version,
                        "metrics": rating_metrics
                    })
        
        # Veri istatistikleri
        try:
            products_df = pd.read_csv("data/products.csv")
            interactions_df = pd.read_csv("data/interactions.csv")
            
            response["data_stats"].update({
                "products": {
                    "total_count": len(products_df),
                    "categories": len(products_df["item_category"].unique())
                },
                "interactions": {
                    "total_count": len(interactions_df),
                    "unique_users": len(interactions_df["user_id"].unique()),
                    "unique_items": len(interactions_df["item_id"].unique()),
                    "average_rating": float(interactions_df["rating"].mean())
                }
            })
        except Exception as e:
            response["data_stats"].update({
                "error": str(e)
            })
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Metrik hesaplama hatası: {str(e)}"
        )

# Uygulama başlatma
if __name__ == "__main__":
    uvicorn.run("serve:app", host="0.0.0.0", port=8000, reload=True) 
