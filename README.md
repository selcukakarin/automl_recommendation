# PyCaret MLflow Öneri Sistemi

Bu proje, PyCaret ve MLflow kullanarak bir öneri sistemi oluşturur, modeli kaydeder ve FastAPI ile serve eder. PyCaret'in otomatik makine öğrenimi özellikleri kullanılarak farklı model versiyonları eğitilir ve karşılaştırılır.

## Özellikler

- PyCaret ile otomatik model seçimi ve optimizasyon
- Kullanıcı ve ürün özelliklerini içeren zengin veri seti
- MLflow ile model versiyonlama ve experiment tracking
- FastAPI ile model serving
- Farklı model versiyonları arasında geçiş yapabilme

## Kurulum

1. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

2. MLflow sunucusunu başlatın:
```bash
mlflow server --host 0.0.0.0 --port 5000
```

## Kullanım

1. Model versiyonlarını eğitmek için:
```bash
python train.py
```
Bu komut 4 farklı model versiyonu eğitecektir:
- v1_auto_select: PyCaret'in otomatik seçtiği en iyi model
- v2_random_forest: Varsayılan parametrelerle Random Forest
- v3_rf_tuned: Özelleştirilmiş parametrelerle Random Forest
- v4_xgboost: XGBoost modeli

2. Model serving için:
```bash
python serve.py
```

3. Mevcut model versiyonlarını listelemek için:
```bash
curl http://localhost:8000/versions
```

4. Belirli bir model versiyonunu yüklemek için:
```bash
curl -X POST http://localhost:8000/load_version/v2_random_forest
```

5. Tahmin yapmak için (belirli bir versiyon kullanarak):
```bash
curl -X POST "http://localhost:8000/predict?version_name=v2_random_forest" \
     -H "Content-Type: application/json" \
     -d '{
           "user_id": 1,
           "item_id": 1,
           "user_age": 25,
           "user_gender": "M",
           "item_category": "A",
           "item_price": 50.0
         }'
```

## Proje Yapısı

- `train.py`: PyCaret ile model eğitimi ve MLflow logging
- `serve.py`: FastAPI ile model serving ve versiyon yönetimi
- `data/`: Veri seti dizini
- `requirements.txt`: Gerekli Python paketleri

## API Endpoints

### GET /versions
Mevcut tüm model versiyonlarını listeler.

### POST /load_version/{version_name}
Belirli bir model versiyonunu yükler.

### POST /predict
Bir kullanıcı-ürün çifti için tahmin yapar.

Query Parameters:
- version_name (optional): Kullanılacak model versiyonu

Request body:
```json
{
    "user_id": int,
    "item_id": int,
    "user_age": int,
    "user_gender": str,
    "item_category": str,
    "item_price": float
}
```

Response:
```json
{
    "predicted_rating": float,
    "model_version": str
}
```

## Model Özellikleri

PyCaret, aşağıdaki özellikleri kullanarak modelleri eğitir:

- Kullanıcı özellikleri:
  - user_id
  - user_age
  - user_gender

- Ürün özellikleri:
  - item_id
  - item_category
  - item_price

## MLflow Tracking

MLflow UI'da (http://localhost:5000) şunları görüntüleyebilirsiniz:
- Farklı model versiyonları
- Her versiyonun performans metrikleri (MAE, MSE, RMSE, R2)
- Model parametreleri
- Model artifactleri 