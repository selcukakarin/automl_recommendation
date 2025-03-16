# PyCaret MLflow Öneri Sistemi

Bu proje, modern makine öğrenimi teknolojilerini kullanarak gelişmiş bir öneri sistemi oluşturur. PyCaret'in otomatik makine öğrenimi özellikleri kullanılarak farklı model versiyonları eğitilir, MLflow ile versiyonlanır ve FastAPI ile serve edilir.

## 1. Proje Genel Bakış

Projede kullanılan temel teknolojiler:

- **PyCaret**: Otomatik makine öğrenimi (AutoML) için
- **MLflow**: Model versiyonlama ve deney takibi için
- **FastAPI**: Model servis etme ve API yönetimi için

### 1.1 Temel Özellikler

1. **Otomatik Model Geliştirme**
   - Farklı algoritmaların otomatik karşılaştırılması
   - Hiperparametre optimizasyonu
   - Cross-validation ve model değerlendirme

2. **Model Versiyonlama**
   - Farklı model versiyonlarını saklama
   - Versiyonlar arası geçiş yapabilme
   - A/B testing desteği

3. **Gerçek Zamanlı İzleme**
   - Model performans metrikleri
   - Sağlık kontrolleri
   - Tahmin güvenilirliği analizi

## 2. Kurulum

### 2.1 Sistem Gereksinimleri
- Python 3.8 veya üzeri
- pip paket yöneticisi
- Git (opsiyonel)

### 2.2 Kurulum Adımları

1. Projeyi klonlayın (veya ZIP olarak indirin):
```bash
git clone <proje-url>
cd mlflow_recommender
```

2. Sanal ortam oluşturun:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac için
venv\Scripts\activate     # Windows için
```

3. Gereksinimleri yükleyin:
```bash
pip install -r requirements.txt
```

## 3. Kullanım

### 3.1 Model Eğitimi

1. MLflow sunucusunu başlatın:
```bash
mlflow server --host 0.0.0.0 --port 5000
```
- MLflow UI: http://localhost:5000
- Tüm model versiyonlarını ve metrikleri buradan izleyebilirsiniz

2. Model versiyonlarını eğitin:
```bash
python train.py
```

Bu komut 4 farklı model versiyonu eğitir:

1. **v1_auto_select**
   - PyCaret'in otomatik seçtiği en iyi model
   - Tüm algoritmaları karşılaştırır
   - En iyi performans gösteren modeli seçer

2. **v2_random_forest**
   - Random Forest algoritması
   - Varsayılan parametrelerle eğitim
   - Temel performans karşılaştırması için

3. **v3_rf_tuned**
   - Optimize edilmiş Random Forest
   - Özel parametreler:
     * n_estimators: 200 (ağaç sayısı)
     * max_depth: 10 (maksimum derinlik)
     * min_samples_split: 5 (dal bölme için min örnek)

4. **v4_xgboost**
   - XGBoost algoritması
   - Otomatik parametre optimizasyonu
   - Genellikle en iyi performansı verir

### 3.2 Model Serving

1. API servisini başlatın:
```bash
python serve.py
```
- API servisi: http://localhost:8000
- API dokümantasyonu: http://localhost:8000/docs

## 4. API Endpoints

### 4.1 Model Versiyonlarını Listeleme
```bash
GET /versions

curl http://localhost:8000/versions
```
- Tüm mevcut model versiyonlarını listeler
- Her versiyon için performans metriklerini gösterir

### 4.2 Model Versiyonu Yükleme
```bash
POST /load_version/{version_name}

curl -X POST http://localhost:8000/load_version/v2_random_forest
```
- Belirli bir model versiyonunu aktif hale getirir
- Versiyonlar arası geçiş yapmak için kullanılır

### 4.3 Model Sağlığını Kontrol Etme
```bash
GET /model_health

curl http://localhost:8000/model_health
```
- Son 24 saatteki performansı kontrol eder
- RMSE ve R² metriklerini hesaplar
- Model durumunu raporlar

### 4.4 Tahmin Yapma
```bash
POST /predict

curl -X POST "http://localhost:8000/predict" \
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

Opsiyonel parametreler:
- `version_name`: Belirli bir versiyon kullanmak için
- `enable_ab_testing`: A/B testing için

## 5. Özellik Detayları

### 5.1 Veri Özellikleri

1. **Kullanıcı Özellikleri**
   - user_id: Kullanıcı kimliği (1-100)
   - user_age: Kullanıcı yaşı (18-70)
   - user_gender: Cinsiyet (M/F)

2. **Ürün Özellikleri**
   - item_id: Ürün kimliği (1-50)
   - item_category: Kategori (A/B/C)
   - item_price: Fiyat (10-100)

### 5.2 Model Metrikleri

1. **Temel Metrikler**
   - MAE (Mean Absolute Error)
   - MSE (Mean Squared Error)
   - RMSE (Root Mean Squared Error)
   - R² (R-squared)

2. **Sağlık Kontrolleri**
   - RMSE < 1.0 (1 yıldızdan az hata)
   - R² > 0.6 (en az %60 açıklayıcılık)

### 5.3 Güven Aralıkları
- Her tahmin için güven aralığı hesaplanır
- Model belirsizliği ölçülür
- Tahmin güvenilirliği raporlanır

## 6. İleri Seviye Özellikler

### 6.1 A/B Testing
1. `/predict` endpoint'inde `enable_ab_testing=true` parametresi kullanın
2. Sistem otomatik olarak farklı versiyonları test eder
3. Performans metriklerini karşılaştırın

### 6.2 Model İzleme
1. MLflow UI'da model performansını takip edin
2. `/model_health` endpoint'i ile canlı izleme yapın
3. Metrikler düşükse otomatik uyarı alın

### 6.3 Otomatik Model Güncelleme
1. Yeni veri geldiğinde `train.py` ile modeli güncelleyin
2. MLflow'da yeni versiyon otomatik kaydedilir
3. API üzerinden yeni versiyona geçiş yapın

## 7. Hata Giderme ve Bakım

### 7.1 Sık Karşılaşılan Hatalar

1. **MLflow Bağlantı Hatası**
   ```
   Solution: MLflow sunucusunun çalıştığını kontrol edin
   ```

2. **Model Yükleme Hatası**
   ```
   Solution: Model versiyonunun doğru olduğunu kontrol edin
   ```

3. **API Hataları**
   ```
   Solution: Input formatını kontrol edin
   ```

### 7.2 Performans İyileştirme
1. GPU kullanımını etkinleştirin
2. Batch prediction kullanın
3. Model parametrelerini optimize edin

### 7.3 Güvenlik Önlemleri
1. API anahtarı kullanın
2. Rate limiting uygulayın
3. Input validasyonu yapın

### 7.4 Düzenli Bakım
1. Modeli periyodik olarak güncelleyin
2. Performans metriklerini kontrol edin
3. Sistem loglarını temizleyin

## 8. Proje Yapısı

- `train.py`: Model eğitimi ve MLflow entegrasyonu
- `serve.py`: FastAPI ile model serving
- `data/`: Veri seti dizini
- `requirements.txt`: Gerekli Python paketleri 