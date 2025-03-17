import os
import mlflow
import pandas as pd
import numpy as np
from pycaret.regression import *
from datetime import datetime, timedelta
import json

"""
Bu script, PyCaret'in AutoML ozelliklerini kullanarak bir oneri sistemi olusturur.
AutoML ozellikleri:
1. Otomatik model secimi ve karsilastirma: PyCaret'in compare_models fonksiyonu ile en iyi modeli secer
2. Otomatik veri on isleme: Kategorik degiskenleri donusturme, eksik degerleri doldurma
3. Otomatik hiperparametre optimizasyonu: tune_model fonksiyonu ile parametreleri optimize eder
4. Otomatik metrik hesaplama: MAE, MSE, RMSE ve R2 metriklerini hesaplar
5. Farkli model versiyonlarini otomatik karsilastirma: Farkli modelleri ve ayarlari test eder

Kullanim:
1. MLflow sunucusunu baslatin: mlflow ui
2. Bu scripti calistirin: python train.py
3. MLflow UI'da sonuclari inceleyin: http://localhost:5000
"""

# MLflow sunucusunu ayarla
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("recommendation-system")

def validate_model_performance(metrics):
    """
    Model performansinin belirlenen esik degerlerini karsilayip karsilamadigini kontrol eder.
    
    Args:
        metrics (dict): Model performans metrikleri (RMSE, R2 vb.)
    
    Raises:
        ValueError: Eger model performansi esik degerlerin altindaysa hata verir
        
    Kontrol edilen metrikler:
    - RMSE: 1.0'dan kucuk olmali (1 yildizdan az hata)
    - R2: 0.6'dan buyuk olmali (en az %60 aciklayicilik)
    """
    threshold = {
        "RMSE": 1.0,  # 1 yildizdan fazla hata kabul edilemez
        "R2": 0.6     # En az %60 aciklayicilik gucu olmali
    }
    
    if metrics['RMSE'] > threshold['RMSE']:
        raise ValueError(f"Model RMSE ({metrics['RMSE']:.2f}) cok yuksek!")
    
    if metrics['R2'] < threshold['R2']:
        raise ValueError(f"Model R2 skoru ({metrics['R2']:.2f}) cok dusuk!")

def create_sample_data():
    """
    Oneri sistemi icin ornek veri seti olusturur.
    
    Returns:
        pd.DataFrame: Asagidaki ozellikleri iceren veri seti:
            - user_id: Kullanici ID'si (1-100 arasi)
            - item_id: Urun ID'si (1-50 arasi)
            - rating: Degerlendirme puani (1-5 arasi)
            - timestamp: Degerlendirme tarihi (son 1 yil icinde)
            - user_age: Kullanici yasi (18-70 arasi)
            - user_gender: Kullanici cinsiyeti (M/F)
            - item_category: Urun kategorisi (A/B/C)
            - item_price: Urun fiyati (10-100 arasi)
    
    Not:
        - Veri seti zaman bazli validasyon icin timestamp icerir
        - Kategorik degiskenler AutoML tarafindan otomatik islenir
    """
    np.random.seed(42)
    n_users = 100
    n_items = 50
    n_ratings = 1000
    
    # Zaman bazlı validasyon için tarih sütunu ekle
    # Örnek çıktı:
    # dates = [
    #   '2023-03-15 00:00:00',
    #   '2023-03-16 02:24:00',
    #   '2023-03-17 04:48:00',
    #   ...
    #   '2024-03-14 21:36:00'
    # ]
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=365),
        end=datetime.now(),
        periods=n_ratings
    )
    
    # Kullanıcı ID'lerini oluştur
    # - n_ratings kadar rastgele kullanıcı ID'si üretilir
    # - ID'ler 1 ile n_users arasında olacak şekilde seçilir
    # - Bir kullanıcı birden fazla değerlendirme yapabilir
    # Örnek çıktı:
    # user_ids = [23, 45, 12, 78, 34, 1, 99, ...]
    user_ids = np.random.randint(1, n_users + 1, n_ratings)
    
    # Ürün ID'lerini oluştur
    # - n_ratings kadar rastgele ürün ID'si üretilir
    # - ID'ler 1 ile n_items arasında olacak şekilde seçilir
    # - Bir ürün birden fazla kez değerlendirilebilir
    # Örnek çıktı:
    # item_ids = [5, 12, 45, 23, 8, 15, 31, ...]
    item_ids = np.random.randint(1, n_items + 1, n_ratings)
    
    # Değerlendirme puanlarını oluştur
    # - n_ratings kadar rastgele puan üretilir
    # - Puanlar 1 ile 5 arasında uniform dağılımlı olarak seçilir
    # - Ondalıklı puanlara izin verilir (örn: 3.7, 4.2 gibi)
    # - Gerçek kullanıcı davranışını simüle etmek için uniform dağılım kullanılır
    # Örnek çıktı:
    # ratings = [4.2, 3.7, 2.1, 5.0, 1.8, 3.3, ...]
    ratings = np.random.uniform(1, 5, n_ratings)
    
    # Kullanıcı özellikleri ekleyelim - AutoML için ek özellikler
    # Örnek çıktı:
    # user_features = 
    #    user_id  user_age user_gender
    # 0        1        25          M
    # 1        2        45          F
    # 2        3        33          M
    # ...
    user_features = pd.DataFrame({
        'user_id': range(1, n_users + 1),
        'user_age': np.random.randint(18, 70, n_users),
        'user_gender': np.random.choice(['M', 'F'], n_users)
    })
    
    # Ürün özellikleri ekleyelim - AutoML için ek özellikler
    # Örnek çıktı:
    # item_features = 
    #    item_id item_category  item_price
    # 0        1            A       45.23
    # 1        2            B       78.50
    # 2        3            C       23.15
    # ...
    item_features = pd.DataFrame({
        'item_id': range(1, n_items + 1),
        'item_category': np.random.choice(['A', 'B', 'C'], n_items),
        'item_price': np.random.uniform(10, 100, n_items)
    })
    
    # Ana veri çerçevesini oluştur
    # Örnek çıktı:
    # df = 
    #    user_id  item_id  rating            timestamp
    # 0       23        5     4.2  2023-03-15 00:00:00
    # 1       45       12     3.7  2023-03-16 02:24:00
    # 2       12       45     2.1  2023-03-17 04:48:00
    # ...
    df = pd.DataFrame({
        'user_id': user_ids,
        'item_id': item_ids,
        'rating': ratings,
        'timestamp': dates
    })
    
    # Kullanıcı ve ürün özelliklerini birleştir
    # Final çıktı:
    # df = 
    #    user_id  item_id  rating            timestamp  user_age user_gender item_category  item_price
    # 0       23        5     4.2  2023-03-15 00:00:00        25          M            A       45.23
    # 1       45       12     3.7  2023-03-16 02:24:00        45          F            B       78.50
    # 2       12       45     2.1  2023-03-17 04:48:00        33          M            C       23.15
    # ...
    df = df.merge(user_features, on='user_id', how='left')
    df = df.merge(item_features, on='item_id', how='left')
    
    # Kategorik değişkenleri dönüştür - AutoML bunu otomatik olarak işleyecek
    # Örnek çıktı (değişken tipleri):
    # user_gender: category    (M, F)
    # item_category: category  (A, B, C)
    df['user_gender'] = df['user_gender'].astype('category')
    df['item_category'] = df['item_category'].astype('category')
    
    if not os.path.exists('data'):
        os.makedirs('data')
    df.to_csv('data/ratings.csv', index=False)
    return df

def train_model_version(version_name, model_type=None, params=None):
    """
    Belirli bir versiyon icin model egitir ve MLflow'a kaydeder.
    
    Args:
        version_name (str): Model versiyonunun adi (orn. "v1_auto_select")
        model_type (str, optional): Kullanilacak model tipi (orn. "rf" for Random Forest)
        params (dict, optional): Model parametreleri
    
    AutoML ozellikleri:
        1. Veri on isleme:
           - Kategorik degiskenleri otomatik donusturme
           - Eksik degerleri doldurma
           - Aykiri degerleri temizleme
        
        2. Model secimi/optimizasyonu:
           - model_type belirtilmezse en iyi modeli otomatik secer
           - model_type belirtilirse o modeli optimize eder
        
        3. Validasyon:
           - 5-fold cross validation
           - Zaman bazli bolunme (temporal split)
           - Performans metriklerini hesaplama
        
        4. MLflow entegrasyonu:
           - Model parametrelerini kaydetme
           - Performans metriklerini kaydetme
           - Model dosyasini artifact olarak saklama
    
    Metrikler:
        - MAE: Ortalama Mutlak Hata
        - MSE: Ortalama Kare Hata
        - RMSE: Kok Ortalama Kare Hata
        - R2: Belirtme Katsayisi
    """
    if not os.path.exists('data/ratings.csv'):
        df = create_sample_data()
    else:
        df = pd.read_csv('data/ratings.csv')
    
    with mlflow.start_run(run_name=version_name):
        # AutoML: Otomatik veri on isleme ve cross-validation ayarları
        # Setup fonksiyonu, model egitimi oncesi tum veri ve validasyon ayarlarını yapılandırır
        
        # Örnek veri yapısı:
        # df = 
        #    user_id  item_id  rating  timestamp  user_age  user_gender  item_category  item_price
        # 0      23       5     4.2    2023...        25          M            A          45.23
        # 1      45      12     3.7    2023...        45          F            B          78.50
        # 2       12       45     2.1    2023...        33          M            C          23.15
        # ...
        setup(
              # Ana veri çerçevesi - tum ozellikler ve hedef degisken
              data=df,
              
              # Tahmin edilecek hedef degisken
              # Örnek: rating = [4.2, 3.7, 2.1, 5.0, ...] (1-5 arası puanlar)
              target='rating',
              
              # Egitim seti oranı: %75 egitim, %25 test
              # Örnek: 1000 satırlık veri için
              # - İlk 750 satır: Egitim seti
              # - Son 250 satır: Test seti
              train_size=0.75,
              
              # 5-fold cross validation: Veriyi 5 parçaya böler
              # Örnek fold yapısı (750 satırlık egitim verisi için):
              # Fold 1: [0-150] test, [151-750] train
              # Fold 2: [151-300] test, [0-150, 301-750] train
              # Fold 3: [301-450] test, [0-300, 451-750] train
              # Fold 4: [451-600] test, [0-450, 601-750] train
              # Fold 5: [601-750] test, [0-600] train
              fold=5,
              
              # Zaman bazlı cross-validation
              # Normal CV'den farkı: Gelecekteki veriyi tahmin etmek için geçmis veriyi kullanır
              # Örnek: Mart 2024 verilerini tahmin etmek için Ocak-Şubat 2024 verilerini kullanır
              fold_strategy='timeseries',
              
              # Zaman sırasını korur, karıştırma yapmaz
              # False: 1 Ocak -> 2 Ocak -> 3 Ocak ... şeklinde sıralı gider
              # True olsaydı: 5 Ocak -> 1 Ocak -> 7 Ocak ... şeklinde karışık olurdu
              fold_shuffle=False,
              
              # Veri bölme işleminde karıştırma yapma
              # False: Zaman sırasını koru
              # True: Verileri rastgele karıştır
              data_split_shuffle=False,
              
              # Rastgele sayı üreteci için sabit değer
              # Aynı sonuçları tekrar üretebilmek için kullanılır
              # Her çalıştırmada aynı bölünmeleri ve sonuçları elde ederiz
              session_id=42,
              
              # Setup işlemi sırasında ekrana çıktı yazmaz
              # True: Detaylı çıktı gösterir
              # False: Minimum çıktı gösterir
              verbose=False,
              
              # GPU kullanımını devre dışı bırakır
              # False: CPU kullanır (daha yavaş ama her sistemde çalışır)
              # True: GPU kullanır (daha hızlı ama GPU gerektirir)
              use_gpu=False)
        
        if model_type:
            # AutoML: Belirli bir model için otomatik hiperparametre optimizasyonu
            # Örnek: Random Forest modeli için hiperparametreler
            # rf_params = {
            #    'n_estimators': [100, 200, 300],     # Ağaç sayısı
            #    'max_depth': [5, 10, 15, 20],        # Maksimum derinlik
            #    'min_samples_split': [2, 5, 10],     # Bölünme için minimum örnek sayısı
            #    'min_samples_leaf': [1, 2, 4]        # Yaprak için minimum örnek sayısı
            #  }

            # PyCaret bu kombinasyonları otomatik dener ve en iyisini bulur
            model = create_model(model_type, **params if params else {})
            # Model optimizasyonu
            
            # Perde arkasında yapılanlar:
            # Deneme 1: n_estimators=100, max_depth=5  -> RMSE: 0.85
            # Deneme 2: n_estimators=200, max_depth=10 -> RMSE: 0.82
            # Deneme 3: n_estimators=300, max_depth=15 -> RMSE: 0.83
            # ...
            # En iyi sonuç veren parametreler seçilir
            tuned_model = tune_model(model, n_iter=10, optimize='RMSE')
            final_model = finalize_model(tuned_model)
            # Bu otomatik optimizasyon süreci:
            # Manuel parametre ayarlamaya göre çok daha hızlıdır
            # Daha iyi sonuçlar verir
            # Zaman tasarrufu sağlar
            # İnsan hatasını minimize eder
            # Tutarlı ve tekrarlanabilir sonuçlar üretir
        else:
            # AutoML: Otomatik model secimi - en iyi modeli secer
            # Otomatik model secimi süreci:
            # 1. compare_models: Tüm model tiplerini (rf, xgb, lightgbm vb.) test eder
            #    - Her model için cross-validation ile performans hesaplar
            #    - RMSE, MAE, R2 gibi metrikleri değerlendirir
            #    - Egitim süresini ve kaynak kullanımını kontrol eder
            #    - En iyi performans/kaynak oranına sahip modeli secer
            # 
            # 2. finalize_model: Seçilen en iyi modeli son haline getirir
            #    - Tüm veri setiyle tekrar egitir
            #    - Modeli üretim için hazırlar
            model = compare_models(n_select=1)
            final_model = finalize_model(model)
        
        # AutoML: Otomatik metrik hesaplama
        # pull(): Sonuçları görüntüleme
        #    - Tüm modellere ait performans metriklerini görüntüler
        #    - En iyi modeli secmek için kullanılır
        metrics = pull()
        best_metrics = metrics.iloc[0].to_dict()
        
        # Tüm model ve egitim bilgilerini hazırla
        # Bu sözlük, modelin tüm önemli bilgilerini içerir ve MLflow'a kaydedilir
        base_info = {
            # Model bilgileri
            # - version_name: Model versiyonunun benzersiz adı (orn: "v1_auto_select")
            # - model_type: Kullanılan model tipi (rf, xgb, vb. veya otomatik secilmis)
            # - custom_params: Özel model parametreleri (varsa)
            "version_name": version_name,
            "model_type": model_type if model_type else "auto_selected",
            "custom_params": str(params) if params else "default",
            
            # Egitim konfigürasyonu
            # - fold_strategy: Cross-validation stratejisi (timeseries: zaman bazlı bölünme)
            # - n_folds: Kaç parçaya bölüneceği (5-fold cross validation)
            # - train_size: Egitim seti oranı (0.75 = %75 egitim, %25 test)
            # - fold_shuffle: Veri karıştırma durumu (False: zaman sırasını koru)
            # - data_split_shuffle: Veri bölme işleminde karıştırma durumu (False: zaman sırasını koru)
            # - session_id: Tekrar üretilebilirlik için sabit değer
            # - verbose: Çıktı detay seviyesi
            # - use_gpu: GPU kullanım durumu
            "fold_strategy": "timeseries",
            "n_folds": 5,
            "train_size": 0.75,
            "fold_shuffle": False,
            "data_split_shuffle": False,
            "session_id": 42,
            "verbose": False,
            "use_gpu": False,
            
            # Veri bilgileri
            # - data_shape: Veri setinin boyutu (satır x sütun)
            # - features: Kullanılan özellikler listesi
            # - target_variable: Tahmin edilecek degisken
            # - timestamp_column: Zaman serisi için kullanılan sütun
            "data_shape": f"{df.shape[0]}x{df.shape[1]}",
            "features": list(df.columns),
            "target_variable": "rating",
            "timestamp_column": "timestamp",
            
            # Egitim zamanı
            # Model egitiminin başlangıç zamanı (debug ve izleme için)
            "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            
            # Model parametreleri
            # final_model'in tüm parametrelerini sözlüğe ekle
            **final_model.get_params()
        }
        
        # Performans metrikleri sözlüğü
        # Modelin performansını ölçen tüm metrikler burada toplanır
        all_metrics = {
            "MAE": best_metrics['MAE'],      # Ortalama Mutlak Hata
            "MSE": best_metrics['MSE'],      # Ortalama Kare Hata
            "RMSE": best_metrics['RMSE'],    # Kok Ortalama Kare Hata
            "R2": best_metrics['R2'],        # Belirtme Katsayisi
            "Training_Time": best_metrics.get('TT', 0),    # Egitim süresi (saniye)
            "Memory_Usage": best_metrics.get('Memory', 0)  # Kullanılan bellek (MB)
        }
        
        # Model performans kontrolü
        # Belirlenen eşik değerlere göre model başarılı mı kontrol et
        validation_error = None
        try:
            # Model performans kontrolü
            validate_model_performance(best_metrics)
            
            # Başarılı durumda ek bilgiler
            base_info.update({
                "validation_status": "passed",  # Validasyon durumu
                "validation_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Validasyon zamanı
            })
            
            # Başarılı modelin performans detayları
            all_metrics.update({
                "threshold_rmse": 1.0,  # RMSE için kabul edilen maksimum değer
                "threshold_r2": 0.6,    # R2 için kabul edilen minimum değer
                "rmse_margin": 1.0 - best_metrics['RMSE'],  # RMSE'nin eşik değerden ne kadar iyi olduğu
                "r2_margin": best_metrics['R2'] - 0.6       # R2'nin eşik değerden ne kadar iyi olduğu
            })
            
        except ValueError as e:
            validation_error = str(e)
            # Hata durumunda kayıt
            base_info.update({
                "validation_status": "failed",        # Başarısız durumu
                "failure_reason": validation_error,   # Başarısızlık nedeni
                "failure_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # Hata zamanı
                "threshold_rmse": 1.0,                # Hedeflenen RMSE değeri
                "threshold_r2": 0.6                   # Hedeflenen R2 değeri
            })
        
        # MLflow'a kayıt
        # Tüm model bilgileri ve metrikleri MLflow'a kaydedilir
        mlflow.log_params(base_info)      # Model parametreleri ve konfigürasyonu
        mlflow.log_metrics(all_metrics)   # Performans metrikleri
        
        try:
            # Model dosyası kaydetme
            # Başarılı ve başarısız modeller farklı isimlerle kaydedilir
            model_filename = "model.pkl" if base_info["validation_status"] == "passed" else "failed_model.pkl"
            save_model(final_model, model_filename)
            
            # Model artifactlarını MLflow'a kaydet
            mlflow.log_artifact(model_filename)
            
            # Veri özeti kaydetme
            # Veri seti hakkında detaylı istatistikler ve bilgiler
            data_summary = {
                "data_statistics": df.describe().to_dict(),     # Sayısal istatistikler
                "missing_values": df.isnull().sum().to_dict(),  # Eksik değer sayıları
                "feature_correlations": df.corr()['rating'].to_dict(),  # Hedef degiskenle korelasyonlar
                "data_types": df.dtypes.to_dict(),             # Sütun veri tipleri
                "categorical_features": df.select_dtypes(include=['category', 'object']).columns.tolist(),  # Kategorik özellikler
                "numerical_features": df.select_dtypes(include=['int64', 'float64']).columns.tolist()      # Sayısal özellikler
            }
            
            # Veri özetini JSON formatında kaydet
            with open("data_summary.json", "w") as f:
                json.dump(data_summary, f)
            mlflow.log_artifact("data_summary.json")  # Veri özetini MLflow'a kaydet
            
        except Exception as e:
            print(f"Model ve veri özeti kaydedilirken hata oluştu: {str(e)}")
        
        finally:
            # Geçici dosyaları temizle
            # Hata olsa da olmasa da dosyaları temizlemeye çalış
            for file in [model_filename, "data_summary.json"]:
                if os.path.exists(file):
                    try:
                        os.remove(file)
                    except:
                        pass
        
        # Sonuç döndür
        return (True, "Model başarıyla eğitildi ve kaydedildi.") if base_info["validation_status"] == "passed" \
               else (False, f"Model performans kriterleri karşılanamadı: {validation_error}")

def train_multiple_versions():
    """
    Farkli model versiyonlarini egitir ve karsilastirir.
    
    Egitilen versiyonlar:
        1. v1_auto_select: 
           - AutoML ile en iyi modeli otomatik secer
           - Tum model tiplerini test eder
           - En iyi performans gosteren modeli secer
        
        2. v2_random_forest:
           - Random Forest modeli
           - Varsayilan parametrelerle egitilir
           - AutoML ile optimize edilir
        
        3. v3_rf_tuned:
           - Ozellesmis Random Forest
           - Belirli parametrelerle baslar:
             * n_estimators: 200 (agac sayisi)
             * max_depth: 10 (maksimum derinlik)
             * min_samples_split: 5 (dal bolme icin min ornek)
           - AutoML ile bu parametrelerden optimize edilir
        
        4. v4_lightgbm:
           - Light GBM modeli
           - AutoML ile optimize edilir
           - Genellikle en iyi performansi verir
    
    Not:
        - Her versiyon MLflow'a kaydedilir
        - Performans metrikleri karsilastirilabilir
        - En iyi versiyon servis icin kullanilabilir
    """
    # Version 1: AutoML ile otomatik model secimi
    train_model_version("v1_auto_select")
    
    # Version 2: AutoML ile Random Forest optimizasyonu
    train_model_version("v2_random_forest", "rf")
    
    # Version 3: AutoML ile ozellesmis Random Forest
    rf_params = {
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_split": 5
    }
    train_model_version("v3_rf_tuned", "rf", rf_params)
    
    # Version 4: AutoML ile Light GBM optimizasyonu
    train_model_version("v4_lightgbm", "lightgbm")

if __name__ == "__main__":
    train_multiple_versions() 