import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Dict, Optional
import mlflow
import mlflow.sklearn
from sklearn.metrics.pairwise import cosine_similarity
import os
import joblib
from datetime import datetime
import random

# FastAPI uygulaması
app = FastAPI(
    title="E-Ticaret Ürün Öneri Sistemi",
    description="Kullanıcının satın alma geçmişine göre benzer ürünler öneren servis",
    version="1.0.0"
)

# Veri modelleri
class RecommendationRequest(BaseModel):
    user_id: int
    item_id: Optional[int] = None
    num_recommendations: int = 5

class RecommendationResponse(BaseModel):
    recommended_items: List[Dict]
    recommendation_type: str
    model_version: str

class VersionInfo(BaseModel):
    version_name: str
    model_type: str
    timestamp: str
    metrics: Dict

# Global değişkenler
user_item_matrix = None
item_similarity_matrix = None
item_features = None
current_version = "v1"
item_metadata = None

# MLflow bağlantı ayarları
MLFLOW_TRACKING_URI = "http://localhost:5000"

# Model yükleme fonksiyonu
def load_model(run_id):
    client = mlflow.tracking.MlflowClient(MLFLOW_TRACKING_URI)
    print(f"Model yükleniyor: {run_id}")
    
    try:
        # MLflow'dan modeli yüklemeyi dene
        model_path = f"runs:/{run_id}/model"
        model = mlflow.sklearn.load_model(model_path)
        print(f"Model MLflow'dan başarıyla yüklendi: {model_path}")
        
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
        
        return model, user_item_matrix, item_similarity_matrix, item_metadata, run_id
        
    except Exception as e:
        print(f"MLflow'dan model yükleme başarısız: {str(e)}")
        raise Exception(f"Model yüklenemedi: {str(e)}")

# Başlangıçta çalışacak fonksiyon
@app.on_event("startup")
async def startup_event():
    global user_item_matrix, item_similarity_matrix, item_metadata, current_version
    
    try:
        # MLflow'a bağlan
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        print(f"MLflow bağlantısı kuruldu: {MLFLOW_TRACKING_URI}")
        
        # Son deneyi bul
        experiment = mlflow.get_experiment_by_name("recommendation-system")
        if experiment is None:
            print("Experiment bulunamadı! Yeni bir experiment oluşturuluyor.")
            experiment_id = mlflow.create_experiment("recommendation-system")
            experiment = mlflow.get_experiment(experiment_id)
        
        # Son çalışmayı bul
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        if len(runs) == 0:
            raise Exception("Hiç model çalışması bulunamadı!")
        
        latest_run = runs.sort_values("start_time", ascending=False).iloc[0]
        run_id = latest_run["run_id"]
        
        # Modeli yükle
        model, user_item_matrix, item_similarity_matrix, item_metadata, current_version = load_model(run_id)
        print(f"Başlangıç modeli yüklendi: {current_version}")
        
    except Exception as e:
        print(f"Başlangıç hatası: {str(e)}")
        print("Servis yine de başlatılıyor...")

# Tüm versiyonları listele
@app.get("/versions", response_model=List[VersionInfo])
async def list_versions():
    experiment = mlflow.get_experiment_by_name("recommendation-system")
    if experiment is None:
        raise HTTPException(status_code=404, detail="Experiment bulunamadı!")
    
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    
    versions = []
    for _, run in runs.iterrows():
        version_info = VersionInfo(
            version_name=run["params.version_name"] if "params.version_name" in run else "unnamed",
            model_type=run["params.model_type"] if "params.model_type" in run else "unknown",
            timestamp=run["start_time"],
            metrics={
                "accuracy": run["metrics.accuracy"] if "metrics.accuracy" in run else 0,
                "precision": run["metrics.precision"] if "metrics.precision" in run else 0,
                "recall": run["metrics.recall"] if "metrics.recall" in run else 0
            }
        )
        versions.append(version_info)
    
    return versions

# Belirli bir versiyonu yükle
@app.post("/load_version/{version_name}")
async def load_version(version_name: str):
    experiment = mlflow.get_experiment_by_name("recommendation-system")
    if experiment is None:
        raise HTTPException(status_code=404, detail="Experiment bulunamadı!")
    
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    version_run = runs[runs["params.version_name"] == version_name]
    
    if len(version_run) == 0:
        raise HTTPException(status_code=404, detail=f"Versiyon bulunamadı: {version_name}")
    
    global user_item_matrix, item_similarity_matrix, item_metadata, current_version
    model, user_item_matrix, item_similarity_matrix, item_metadata, current_version = load_model(version_run.iloc[0]["run_id"])
    
    return {"message": f"Model versiyonu yüklendi: {version_name}"}

# Öneriler
@app.post("/recommend", response_model=RecommendationResponse)
async def recommend(
    request: RecommendationRequest,
    version_name: Optional[str] = Query(None, description="Kullanılacak model versiyonu")
):
    global user_item_matrix, item_similarity_matrix, item_metadata, current_version
    
    if version_name:
        await load_version(version_name)
    
    if user_item_matrix is None or item_similarity_matrix is None:
        raise HTTPException(status_code=500, detail="Model yüklenmedi!")
    
    try:
        user_id = request.user_id
        item_id = request.item_id
        num_recommendations = request.num_recommendations
        
        # Önerilen ürünleri tut
        recommended_items = []
        recommendation_type = ""
        
        if item_id is not None:
            # "Bu ürünü alanlar şunları da aldı" modeli
            recommendation_type = "item_based"
            
            # Ürün ID'si item_similarity_matrix'te var mı kontrol et
            if item_id not in item_similarity_matrix.index:
                raise HTTPException(status_code=404, detail=f"Ürün bulunamadı: {item_id}")
            
            # Benzer ürünleri bul
            similar_items = item_similarity_matrix.loc[item_id].sort_values(ascending=False)
            similar_items = similar_items[similar_items.index != item_id]  # Kendisini çıkar
            
            # En benzer ürünleri seç
            top_similar_items = similar_items.head(num_recommendations)
            
            # Sonuçları formatla
            for item_id, similarity in top_similar_items.items():
                item_info = {
                    "item_id": int(item_id),
                    "similarity": float(similarity)
                }
                
                # Ürün metadataları varsa ekle
                if item_metadata is not None and item_id in item_metadata:
                    item_info.update(item_metadata[item_id])
                
                recommended_items.append(item_info)
                
        else:
            # Kullanıcı bazlı öneriler
            recommendation_type = "user_based"
            
            # Kullanıcı ID'si user_item_matrix'te var mı kontrol et
            if user_id not in user_item_matrix.index:
                raise HTTPException(status_code=404, detail=f"Kullanıcı bulunamadı: {user_id}")
            
            # Kullanıcının daha önce satın aldığı ürünleri bul
            user_items = user_item_matrix.loc[user_id]
            purchased_items = user_items[user_items > 0].index.tolist()
            
            # Kullanıcının satın aldığı ürünlere benzer ürünleri bul
            candidate_items = {}
            
            for item in purchased_items:
                if item in item_similarity_matrix.index:
                    similar_items = item_similarity_matrix.loc[item].sort_values(ascending=False)
                    similar_items = similar_items[~similar_items.index.isin(purchased_items)]  # Satın alınanları çıkar
                    
                    for similar_item, similarity in similar_items.items():
                        if similar_item in candidate_items:
                            # Daha yüksek benzerlik skoru varsa güncelle
                            candidate_items[similar_item] = max(candidate_items[similar_item], similarity)
                        else:
                            candidate_items[similar_item] = similarity
            
            # En iyi ürünleri seç
            top_items = sorted(candidate_items.items(), key=lambda x: x[1], reverse=True)[:num_recommendations]
            
            # Sonuçları formatla
            for item_id, similarity in top_items:
                item_info = {
                    "item_id": int(item_id),
                    "similarity": float(similarity)
                }
                
                # Ürün metadataları varsa ekle
                if item_metadata is not None and item_id in item_metadata:
                    item_info.update(item_metadata[item_id])
                
                recommended_items.append(item_info)
        
        return RecommendationResponse(
            recommended_items=recommended_items,
            recommendation_type=recommendation_type,
            model_version=current_version
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Model sağlık durumu
@app.get("/model_health", response_model=dict)
async def model_health():
    global user_item_matrix, item_similarity_matrix
    
    if user_item_matrix is None or item_similarity_matrix is None:
        return {
            "status": "not_loaded",
            "message": "Model yüklenmedi"
        }
    
    # Rastgele örnek seçerek test et
    try:
        sample_size = min(10, len(user_item_matrix))
        sample_users = random.sample(list(user_item_matrix.index), sample_size)
        
        recommendations = []
        for user_id in sample_users:
            request = RecommendationRequest(user_id=user_id, num_recommendations=3)
            response = await recommend(request)
            recommendations.append(response)
        
        return {
            "status": "healthy",
            "sample_size": sample_size,
            "message": "Model sağlıklı çalışıyor",
            "version": current_version,
            "user_item_matrix_shape": user_item_matrix.shape,
            "item_similarity_matrix_shape": item_similarity_matrix.shape
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Model test edilirken hata oluştu: {str(e)}"
        }

# Ana sayfa
@app.get("/")
async def root():
    return {
        "name": "E-Ticaret Ürün Öneri Sistemi",
        "description": "Kullanıcının satın alma geçmişine göre benzer ürünler öneren ve 'Bu ürünü alanlar şunları da aldı' modelini uygulayan servis",
        "endpoints": [
            {"path": "/recommend", "description": "Ürün ve kullanıcı bazlı öneriler"},
            {"path": "/versions", "description": "Tüm model versiyonlarını listele"},
            {"path": "/load_version/{version_name}", "description": "Belirli bir model versiyonunu yükle"},
            {"path": "/model_health", "description": "Model sağlık durumu"}
        ],
        "version": "1.0.0"
    }

# Uygulama başlatma
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) 