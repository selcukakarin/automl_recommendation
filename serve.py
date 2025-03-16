import mlflow
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from pycaret.regression import load_model as pycaret_load_model
import pandas as pd
import uvicorn
from typing import Optional, List

"""
Bu script, PyCaret ile eğitilmiş AutoML modellerini serve eder.
AutoML özellikleri:
1. Otomatik model yükleme ve versiyon yönetimi
2. Otomatik veri ön işleme ve tahmin
3. Farklı model versiyonları arasında geçiş yapabilme
4. Model metriklerinin otomatik raporlanması
"""

app = FastAPI(title="Recommendation System API")

class RecommendationRequest(BaseModel):
    """AutoML modeli için gerekli özellikleri içeren request modeli"""
    user_id: int
    item_id: int
    user_age: int
    user_gender: str
    item_category: str
    item_price: float

class RecommendationResponse(BaseModel):
    """AutoML model tahmini ve versiyon bilgisi"""
    predicted_rating: float
    model_version: str

class ModelVersion(BaseModel):
    """AutoML model versiyonu ve performans metrikleri"""
    version_name: str
    run_id: str
    metrics: dict

# Model yükleme fonksiyonu
def load_model(run_id):
    """
    AutoML modelini MLflow'dan yükler
    PyCaret'in otomatik model yükleme özelliğini kullanır
    """
    client = mlflow.tracking.MlflowClient()
    artifacts_path = client.download_artifacts(run_id, "model.pkl")
    model = pycaret_load_model(artifacts_path)
    return model, run_id

# Global model değişkeni ve versiyon bilgisi
model = None
current_version = None

@app.on_event("startup")
async def startup_event():
    """
    Uygulama başlangıcında AutoML modelinin en son versiyonunu otomatik olarak yükler
    """
    global model, current_version
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
    model, current_version = load_model(latest_run.run_id)

@app.get("/versions", response_model=List[ModelVersion])
async def list_versions():
    """
    Mevcut tüm AutoML model versiyonlarını ve performans metriklerini listeler
    Her versiyonun MAE, MSE, RMSE ve R2 skorlarını gösterir
    """
    experiment = mlflow.get_experiment_by_name("recommendation-system")
    if experiment is None:
        raise HTTPException(status_code=404, detail="Experiment bulunamadı!")
    
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    
    versions = []
    for _, run in runs.iterrows():
        metrics = {
            "MAE": run["metrics.MAE"],
            "MSE": run["metrics.MSE"],
            "RMSE": run["metrics.RMSE"],
            "R2": run["metrics.R2"]
        }
        versions.append(ModelVersion(
            version_name=run["params.version_name"],
            run_id=run["run_id"],
            metrics=metrics
        ))
    
    return versions

@app.post("/load_version/{version_name}")
async def load_version(version_name: str):
    """
    Belirli bir AutoML model versiyonunu yükler
    Bu sayede farklı model tipleri ve hiperparametreler arasında geçiş yapılabilir
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

@app.post("/predict", response_model=RecommendationResponse)
async def predict(
    request: RecommendationRequest,
    version_name: Optional[str] = Query(None, description="Kullanılacak model versiyonu")
):
    """
    AutoML modeli ile tahmin yapar
    - Otomatik veri ön işleme
    - Otomatik model seçimi (versiyon bazlı)
    - Otomatik tahmin
    """
    global model, current_version
    
    if version_name:
        # Belirtilen versiyonu yükle
        await load_version(version_name)
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model yüklenmedi!")
    
    try:
        # Gelen veriyi DataFrame'e dönüştür
        input_data = pd.DataFrame([{
            'user_id': request.user_id,
            'item_id': request.item_id,
            'user_age': request.user_age,
            'user_gender': request.user_gender,
            'item_category': request.item_category,
            'item_price': request.item_price
        }])
        
        # AutoML: Kategorik değişkenleri otomatik dönüştür
        input_data['user_gender'] = input_data['user_gender'].astype('category')
        input_data['item_category'] = input_data['item_category'].astype('category')
        
        # AutoML: Otomatik tahmin
        prediction = model.predict(input_data)
        return RecommendationResponse(
            predicted_rating=float(prediction[0]),
            model_version=current_version
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("serve:app", host="0.0.0.0", port=8000, reload=True) 