# E-ticaret Ürün Öneri Sistemi 🛒

## Proje Özeti

Bu proje, MLflow kullanarak bir e-ticaret ürün öneri sistemi geliştirmeyi, deployment süreçlerini yönetmeyi ve REST API aracılığıyla servis etmeyi göstermektedir. Sistem, kullanıcı-ürün etkileşimlerini analiz ederek kişiselleştirilmiş öneriler sunmaktadır.

## 🔧 Kurulum

### Gereksinimler

```bash
pip install -r requirements.txt
```

Başlıca gereksinimler:
- mlflow==2.10.0
- fastapi==0.104.1
- uvicorn==0.24.0
- pandas==2.1.0
- numpy==1.25.2
- scikit-learn==1.2.2
- joblib==1.3.2

### Veri Oluşturma

```bash
python generate_data.py
```

Bu komut, `data/` klasörü altında şu dosyaları oluşturur:
- `users.csv`: Kullanıcı profilleri
- `products.csv`: Ürün bilgileri
- `interactions.csv`: Kullanıcı-ürün etkileşimleri

### MLflow Sunucusunu Başlatma

```bash
mlflow server --host 0.0.0.0 --port 5000
```

Bu komut, MLflow sunucusunu `http://localhost:5000` adresinde başlatır.

### Model Eğitimi

```bash
python train_recommendation.py
```

Bu script:
1. Sentetik veriyi yükler
2. Kullanıcı-ürün matrisini oluşturur
3. Ürün benzerlik matrislerini hesaplar
4. Modeli değerlendirir (RMSE, MAE)
5. Model ve artifactları MLflow'a kaydeder

### Öneri API Servisini Başlatma

```bash
python serve.py
```

Servis şu adreste çalışacaktır: `http://localhost:8000`

## 📋 Kullanım Kılavuzu

### API Endpoint'leri

| Endpoint | Yöntem | Açıklama |
|----------|--------|----------|
| `/` | GET | API bilgileri |
| `/versions` | GET | Mevcut tüm model versiyonlarını ve hangi endpoint için kullanıldığını listeler |
| `/load_recommendation_version/{version_name}` | POST | Belirli bir model versiyonunu yükler |
| `/recommend` | POST | Ürün önerileri sunar |
| `/recommendation_model_health` | GET | Model sağlık durumunu kontrol eder |

### Ürün-tabanlı Öneri İsteği

```bash
curl -X 'POST' \
  'http://localhost:8000/recommend' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
       "user_id": 123,
       "item_id": 42,
       "num_recommendations": 5
   }'
```

### Kullanıcı-tabanlı Öneri İsteği

```bash
curl -X 'POST' \
  'http://localhost:8000/recommend' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
       "user_id": 123,
       "num_recommendations": 5
   }'
```

### Model Versiyonları Arası Geçiş

```bash
curl -X 'POST' \
  'http://localhost:8000/load_recommendation_version/v1_20250318_122143' \
  -H 'accept: application/json'
```

Versiyon adını MLflow arayüzünden (`http://localhost:5000`) bulabilirsiniz.

## 🧪 Test Etme

```bash
python test_recommendation.py
```

Bu script şunları test eder:
- Model sağlık durumu
- Mevcut model versiyonları ve endpoint bilgileri
- Ürün-tabanlı öneriler
- Kullanıcı-tabanlı öneriler

## 🛠️ Hata Giderme

### "Ürün bulunamadı" Hatası

Bu hata, olmayan bir ürün ID'si ile istek yapıldığında görülür. Geçerli bir ürün ID'si kullanın:

```bash
# Hatalı istek (995 numaralı ürün yok)
curl -X 'POST' \
  'http://localhost:8000/recommend' \
  -d '{
       "user_id": 123,
       "item_id": 995,
       "num_recommendations": 5
   }'

# Cevap
{
  "detail": "Ürün bulunamadı: 995"
}

# Doğru istek (1-200 arası ürün ID'leri kullanın)
curl -X 'POST' \
  'http://localhost:8000/recommend' \
  -d '{
       "user_id": 123,
       "item_id": 42,
       "num_recommendations": 5
   }'
```


## 📝 Proje Dosyaları

```
mlflow_recommender/
│
├── data/                  # Veri dosyaları
│   ├── users.csv         # Kullanıcı bilgileri
│   ├── products.csv      # Ürün bilgileri
│   └── interactions.csv  # Kullanıcı-ürün etkileşimleri
│
├── artifacts/            # Lokal model artifact'leri
│
├── serve.py              # API servisi (FastAPI)
├── train_recommendation.py  # Model eğitim kodu
├── test_recommendation.py  # Test scripti
├── generate_data.py      # Veri oluşturma kodu
├── requirements.txt      # Bağımlılıklar
├── README.md             # Dokümantasyon
├── sunum.md              # Sunum notları
│
├── last_working_model.json         # Son çalışan öneri modeli bilgileri
└── last_working_rating_model.json  # Son çalışan rating modeli bilgileri
```

## ⚙️ Geliştirilecek Yönler

- **Hibrit Öneri Algoritmaları**: Collaborative filtering + content-based yaklaşımların birleştirilmesi
- **Derin Öğrenme Entegrasyonu**: Neural Collaborative Filtering modelleri
- **A/B Test Mekanizması**: Farklı model versiyonlarını karşılaştırma
- **Gerçek Zamanlı Güncelleme**: Kullanıcı davranışlarına göre sürekli iyileştirme
- **Aykırı Değer Tespiti**: Anormal kullanıcı davranışlarını filtreleme
- **Model Sağlık İzleme**: Düzenli performans kontrolü ve otomatik iyileştirme

## 📊 Model Performans Metrikleri ve Açıklamaları

### Metrik Nedir ve Ne İşe Yarar?

Model performans metrikleri, önerilerin ne kadar doğru ve güvenilir olduğunu gösterir. Aşağıdaki metrikler, modelin başarısını ölçmek için kullanılır:

#### RMSE (Root Mean Square Error - Kök Ortalama Kare Hata)
- 💡 Ne Anlama Gelir?: Tahminlerimizin gerçek değerlerden ne kadar saptığını gösterir
- 📉 İyi Değer: 0'a yakın değerler (Örn: 0.5531 çok iyi bir değerdir)
- ⚖️ Özellik: Büyük hataları daha çok cezalandırır
- 🎯 Örnek: RMSE=0.5 ise, tahminlerimiz ortalamada gerçek değerlerden yaklaşık 0.5 puan sapıyor

#### MAE (Mean Absolute Error - Ortalama Mutlak Hata)
- 💡 Ne Anlama Gelir?: Tahminlerimizin gerçek değerlerden ortalama sapmasını gösterir
- 📉 İyi Değer: 0'a yakın değerler (Örn: 0.4427 çok iyi bir değerdir)
- ⚖️ Özellik: Tüm hataları eşit şekilde değerlendirir
- 🎯 Örnek: MAE=0.4 ise, tahminlerimiz ortalamada gerçek değerlerden 0.4 puan sapıyor

#### Tahmin Oranı (Prediction Rate)
- 💡 Ne Anlama Gelir?: Modelin kaç örnek için tahmin yapabildiğini yüzde olarak gösterir
- 📈 İyi Değer: %100'e yakın değerler (Örn: %99.34 çok iyi bir değerdir)
- ⚖️ Özellik: Modelin kapsama alanını gösterir
- 🎯 Örnek: %99.34 ise, model örneklerin %99.34'ü için tahmin yapabiliyor

## Model Metrikleri

Model performansı ve veri kalitesi aşağıdaki metriklerle ölçülmektedir:

### Performans Metrikleri
- **RMSE (Root Mean Square Error)**: Tahminlerin gerçek değerlerden ne kadar saptığını gösteren hata metriği. Düşük olması iyidir.
- **MAE (Mean Absolute Error)**: Tahminlerin gerçek değerlerden ortalama sapmasını gösteren hata metriği. Düşük olması iyidir.
- **N_Predictions**: Modelin başarılı bir şekilde yaptığı tahmin sayısı.
- **Prediction_Ratio**: Başarılı tahminlerin tüm test örneklerine oranı (0-1 arası).

### Veri Metrikleri
- **Average Rating**: Tüm derecelendirmelerin ortalaması. Kullanıcı memnuniyetini gösterir.
- **Rating Count**: Toplam derecelendirme sayısı. Veri setinin büyüklüğünü gösterir.
- **Unique Users**: Sistemdeki benzersiz kullanıcı sayısı.
- **Unique Items**: Sistemdeki benzersiz ürün sayısı.
- **Sparsity**: Veri matrisinin seyrekliği. Düşük değer, az etkileşim olduğunu gösterir.

# MLflow Recommendation System

Bu proje, MLflow kullanarak eğitilen öneri ve derecelendirme modellerini yöneten bir API servisidir.

## Özellikler

- Kullanıcı-ürün derecelendirme tahmini
- Ürün bazlı ve kullanıcı bazlı öneriler
- Model versiyonlama ve yönetimi
- Model sağlık kontrolü
- A/B testing desteği

## API Endpoint'leri

### Model Versiyon Yönetimi

#### `/load_recommendation_version/{version_name}`
- Belirtilen versiyondaki öneri modelini yükler
- Bu endpoint ile yüklenen model, `/recommend` endpoint'inde kullanılır
- Yani ürün önerileri, en son yüklenen model versiyonu kullanılarak yapılır
- Örnek: `POST /load_recommendation_version/v1_20250319_102242`

#### `/load_rating_version/{version_name}`
- Belirtilen versiyondaki derecelendirme modelini yükler
- Bu endpoint ile yüklenen model, `/predict` endpoint'inde kullanılır
- Yani derecelendirme tahminleri, en son yüklenen model versiyonu kullanılarak yapılır
- Örnek: `POST /load_rating_version/v3_rf_tuned`

### Tahmin ve Öneri Endpoint'leri

#### `/predict`
- Kullanıcı-ürün derecelendirme tahmini yapar
- En son yüklenen rating model versiyonunu kullanır
- Model versiyonunu değiştirmek için önce `/load_rating_version` endpoint'ini kullanın

#### `/recommend`
- Ürün önerileri sunar (ürün bazlı veya kullanıcı bazlı)
- En son yüklenen recommendation model versiyonunu kullanır
- Model versiyonunu değiştirmek için önce `/load_recommendation_version` endpoint'ini kullanın

### Model Sağlık Kontrolü

#### `/recommendation_model_health`
- Yüklü olan öneri ve derecelendirme modellerinin sağlık durumunu kontrol eder
- Her iki model için de versiyon bilgisi, metrikler ve durum bilgisi döndürür

#### `/rating_model_health`
- Yüklü olan derecelendirme modelinin sağlık durumunu kontrol eder
- Model metrikleri ve performans bilgisi döndürür

