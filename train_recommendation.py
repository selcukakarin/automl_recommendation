import pandas as pd
import numpy as np
import os
import joblib
import mlflow
import mlflow.sklearn
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, RegressorMixin

# MLflow ayarları
MLFLOW_TRACKING_URI = "http://localhost:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
experiment_name = "recommendation-system"

# Deney oluştur veya mevcut bir deneyi al
experiment = mlflow.get_experiment_by_name(experiment_name)
if experiment is None:
    experiment_id = mlflow.create_experiment(experiment_name)
    experiment = mlflow.get_experiment(experiment_id)
else:
    experiment_id = experiment.experiment_id

print(f"Deney ID: {experiment_id}, Deney Adı: {experiment_name}")

# Önerici model sınıfı
class RecommenderModel(BaseEstimator, RegressorMixin):
    def __init__(self, user_item_matrix=None, item_similarity_matrix=None, item_metadata=None):
        self.user_item_matrix = user_item_matrix
        self.item_similarity_matrix = item_similarity_matrix
        self.item_metadata = item_metadata
    
    def fit(self, X, y=None):
        return self
    
    def predict(self, X):
        """
        Verilen kullanıcı-ürün çiftleri için tahmin yapar
        
        Args:
            X: DataFrame, ['user_id', 'item_id'] sütunlarını içermeli
            
        Returns:
            numpy.array: Tahmin edilen puanlar
        """
        # Veri tiplerini float64'e çevir
        X = X.astype({
            'user_id': 'float64',
            'item_id': 'float64'
        })
        
        predictions = []
        
        for _, row in X.iterrows():
            user_id = int(row['user_id'])  # float64'ten int'e çevir
            item_id = int(row['item_id'])  # float64'ten int'e çevir
            
            try:
                if user_id in self.user_item_matrix.index and item_id in self.item_similarity_matrix.index:
                    user_ratings = self.user_item_matrix.loc[user_id]
                    rated_items = user_ratings[user_ratings > 0].index
                    
                    if len(rated_items) > 0:
                        similar_items = self.item_similarity_matrix.loc[item_id, rated_items]
                        
                        if (similar_items > 0).any():
                            positive_similarities = similar_items[similar_items > 0]
                            positive_items = positive_similarities.index
                            predicted_rating = np.average(
                                user_ratings[positive_items],
                                weights=positive_similarities
                            )
                        else:
                            predicted_rating = user_ratings[user_ratings > 0].mean()
                    else:
                        predicted_rating = 2.5  # Varsayılan değer
                else:
                    predicted_rating = 2.5  # Varsayılan değer
            except Exception as e:
                predicted_rating = 2.5  # Hata durumunda varsayılan değer
                
            predictions.append(predicted_rating)
        
        return np.array(predictions)

def load_data():
    """CSV dosyalarından verileri yükler"""
    print("Veriler yükleniyor...")
    
    # Dosya yolları
    users_path = 'data/users.csv'
    products_path = 'data/products.csv'
    interactions_path = 'data/interactions.csv'
    
    # Dosyaların varlığını kontrol et
    if not all(os.path.exists(f) for f in [users_path, products_path, interactions_path]):
        raise FileNotFoundError("Veri dosyaları bulunamadı. Önce generate_data.py'yi çalıştırın.")
    
    # Verileri yükle
    users_df = pd.read_csv(users_path)
    products_df = pd.read_csv(products_path)
    interactions_df = pd.read_csv(interactions_path)
    
    print(f"Yüklenen veriler:")
    print(f"Kullanıcılar: {len(users_df)} kayıt")
    print(f"Ürünler: {len(products_df)} kayıt")
    print(f"Etkileşimler: {len(interactions_df)} kayıt")
    
    # Ürün metadatasını sözlük formatına dönüştür
    item_metadata = products_df.set_index('item_id').to_dict('index')
    
    return interactions_df, item_metadata

def create_user_item_matrix(interactions_df):
    """Kullanıcı-ürün etkileşim matrisini oluşturur"""
    # Tekrarlanan kullanıcı-ürün çiftlerinin ortalamasını al
    interactions_df = interactions_df.groupby(['user_id', 'item_id'])['rating'].mean().reset_index()
    
    # Pivot tablosunu oluştur
    return interactions_df.pivot(
        index='user_id',
        columns='item_id',
        values='rating'
    ).fillna(0)

def calculate_item_similarity(user_item_matrix):
    """Ürünler arası benzerlik matrisini hesaplar"""
    return pd.DataFrame(
        cosine_similarity(user_item_matrix.T),
        index=user_item_matrix.columns,
        columns=user_item_matrix.columns
    )

def evaluate_model(user_item_matrix, item_similarity_matrix, test_interactions):
    """Model performansını değerlendirir
    
    Bu fonksiyon, önerilen modelin performansını değerlendirir. İki farklı tahmin yöntemi kullanır:
    
    1. BENZERLİK AĞIRLIKLI ORTALAMA YÖNTEMİ (Birincil yöntem):
       - Hedef ürün ile kullanıcının daha önce etkileşimde bulunduğu ürünler arasındaki benzerliği kullanır
       - Yalnızca pozitif benzerlik değerlerine sahip ürünler dikkate alınır
       - Benzerlik değerleri, ağırlık olarak kullanılarak ağırlıklı bir ortalama hesaplanır
       
    2. KULLANICI ORTALAMA YÖNTEMİ (Yedek yöntem/Fallback):
       - Eğer birincil yöntem başarısız olursa (örn. pozitif benzerlik bulunamazsa) kullanılır
       - Kullanıcının mevcut etkileşimlerinin ortalaması alınır
       - Bu, özellikle "soğuk başlangıç" problemini hafifletmeye yardımcı olur
    
    Args:
        user_item_matrix: Kullanıcı-ürün etkileşim matrisi (satırlar: kullanıcılar, sütunlar: ürünler)
        item_similarity_matrix: Ürünler arası benzerlik matrisi
        test_interactions: Test veri seti (user_id, item_id, rating kolonları)
    
    Returns:
        dict: Model performans metrikleri. Şunları içerir:
            - rmse: Kök ortalama kare hata (düşük olması iyidir)
            - mae: Ortalama mutlak hata (düşük olması iyidir)
            - n_predictions: Başarılı tahmin sayısı
            - prediction_ratio: Tahmin yapılabilen örneklerin oranı (0-1 arası)
    
    Hatalar:
        Try-except bloğu, tek bir tahmin başarısız olduğunda tüm değerlendirme sürecinin
        durmasını önlemek için kullanılır. Bu, veri kalitesi sorunlarına karşı dayanıklılık sağlar.
    """
    predictions = []  # Tahmin edilen puanları saklamak için liste
    actuals = []     # Gerçek puanları saklamak için liste
    
    # Her test örneği için tahmin yap
    for _, row in test_interactions.iterrows():
        user_id = row['user_id']
        item_id = row['item_id']
        actual_rating = row['rating']
        
        try:
            # Kullanıcı ve ürünün veri setinde olup olmadığını kontrol et
            if user_id in user_item_matrix.index and item_id in item_similarity_matrix.index:
                # Kullanıcının daha önce puan verdiği ürünleri bul
                # Sıfırdan büyük puanlar, kullanıcının etkileşimde bulunduğu ürünleri gösterir
                user_ratings = user_item_matrix.loc[user_id]
                rated_items = user_ratings[user_ratings > 0].index
                
                # Kullanıcı daha önce en az bir ürüne puan verdiyse
                if len(rated_items) > 0:
                    # Hedef ürün ile kullanıcının daha önce puan verdiği ürünler arasındaki benzerliği al
                    similar_items = item_similarity_matrix.loc[item_id, rated_items]
                    
                    # Pozitif benzerlik değeri olan ürünleri kontrol et
                    # Not: Negatif veya sıfır benzerlik değerleri, tahmin için güvenilir değildir
                    if (similar_items > 0).any():
                        # YÖNTEM 1: Benzerlik-bazlı tahmin
                        # Sadece pozitif benzerlik değerlerine sahip ürünleri kullan
                        positive_similarities = similar_items[similar_items > 0]
                        positive_items = positive_similarities.index
                        
                        # Benzer ürünlerin puanlarının ağırlıklı ortalamasını al
                        # Ağırlıklar olarak benzerlik değerlerini kullan
                        predicted_rating = np.average(
                            user_ratings[positive_items],  # Benzer ürünlere verilen puanlar
                            weights=positive_similarities  # Benzerlik değerleri
                        )
                        
                        predictions.append(predicted_rating)
                        actuals.append(actual_rating)
                    else:
                        # YÖNTEM 2: Ortalama-bazlı tahmin (Fallback mekanizması)
                        # Eğer hiç pozitif benzerlik yoksa, kullanıcının ortalama puanını kullan
                        # Bu, soğuk başlama problemini hafifletmeye yardımcı olur
                        user_mean = user_ratings[user_ratings > 0].mean()
                        if not np.isnan(user_mean):
                            predictions.append(user_mean)
                            actuals.append(actual_rating)
        except Exception as e:
            # Hata durumunda işlemi atla ve bir sonraki örneğe geç
            # Bu, tek bir başarısız tahmin nedeniyle tüm değerlendirme sürecinin durmasını önler
            print(f"Değerlendirme hatası: {e} (user_id: {user_id}, item_id: {item_id})")
            continue
    
    # En az bir tahmin yapılabildiyse metrikleri hesapla
    if len(predictions) > 0:
        # Model performans metriklerini hesapla
        rmse = np.sqrt(np.mean((np.array(actuals) - np.array(predictions)) ** 2))  # Kök ortalama kare hata
        mae = np.mean(np.abs(np.array(actuals) - np.array(predictions)))          # Ortalama mutlak hata
        
        return {
            'rmse': rmse,                    # Kök ortalama kare hata (düşük olması iyidir)
            'mae': mae,                      # Ortalama mutlak hata (düşük olması iyidir)
            'n_predictions': len(predictions),  # Başarılı tahmin sayısı
            'prediction_ratio': len(predictions) / len(test_interactions)  # Tahmin yapılabilen örneklerin oranı
        }
    else:
        # Hiç tahmin yapılamadıysa None döndür
        return None

def main():
    print("E-Ticaret Ürün Öneri Sistemi Eğitimi Başlatılıyor...")
    
    # Model versiyonu
    version_name = f"v1_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"Model Versiyonu: {version_name}")
    
    # Verileri yükle
    interactions_df, item_metadata = load_data()
    
    # Eğitim ve test verilerini ayır
    train_df, test_df = train_test_split(
        interactions_df,
        test_size=0.2,
        random_state=42
    )
    
    # Kullanıcı-ürün matrisini oluştur
    print("Kullanıcı-ürün matrisi oluşturuluyor...")
    user_item_matrix = create_user_item_matrix(train_df)
    print(f"Kullanıcı-ürün matrisinin boyutu: {user_item_matrix.shape}")
    
    # Ürün benzerliklerini hesapla
    print("Ürün benzerlik matrisi hesaplanıyor...")
    item_similarity_matrix = calculate_item_similarity(user_item_matrix)
    print(f"Ürün benzerlik matrisinin boyutu: {item_similarity_matrix.shape}")
    
    # Recommender model oluştur
    model = RecommenderModel(
        user_item_matrix=user_item_matrix,
        item_similarity_matrix=item_similarity_matrix,
        item_metadata=item_metadata
    )
    
    # Modelin performansını değerlendir
    print("Model değerlendiriliyor...")
    metrics = evaluate_model(user_item_matrix, item_similarity_matrix, test_df)
    
    if metrics:
        print(f"Model metrikleri:")
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"MAE: {metrics['mae']:.4f}")
        print(f"Tahmin sayısı: {metrics['n_predictions']}")
        
        # MLflow ile çalışmayı kaydet
        with mlflow.start_run(experiment_id=experiment_id, run_name=version_name):
            print("MLflow çalışması başlatıldı...")
            
            # Parametreleri kaydet
            mlflow.log_param("version_name", version_name)
            mlflow.log_param("model_type", "collaborative_filtering")
            mlflow.log_param("endpoint", "/recommend")
            mlflow.log_param("num_users", len(user_item_matrix))
            mlflow.log_param("num_items", len(user_item_matrix.columns))
            mlflow.log_param("num_interactions", len(interactions_df))
            
            # Metrikleri kaydet
            mlflow.log_metrics(metrics)
            
            # Model giriş örneği oluştur
            input_example = pd.DataFrame({
                'user_id': [1.0],  # float64 olarak tanımla
                'item_id': [1.0]   # float64 olarak tanımla
            })
            
            # Veri tiplerini açıkça belirt
            input_example = input_example.astype({
                'user_id': 'float64',
                'item_id': 'float64'
            })
            
            # Model imzası oluştur
            signature = mlflow.models.signature.infer_signature(
                input_example,
                model.predict(input_example)
            )
            
            # Artifact'leri kaydet
            print("Artifact'ler kaydediliyor...")
            os.makedirs("artifacts", exist_ok=True)
            
            # Kullanıcı-ürün matrisini kaydet
            user_item_matrix_path = os.path.join("artifacts", "user_item_matrix.pkl")
            joblib.dump(user_item_matrix, user_item_matrix_path)
            
            # Ürün benzerlik matrisini kaydet
            item_similarity_path = os.path.join("artifacts", "item_similarity.pkl")
            joblib.dump(item_similarity_matrix, item_similarity_path)
            
            # Ürün metadatasını kaydet
            item_metadata_path = os.path.join("artifacts", "item_metadata.pkl")
            joblib.dump(item_metadata, item_metadata_path)
            
            # Modeli kaydet
            mlflow.sklearn.log_model(
                model,
                "model",
                signature=signature,
                input_example=input_example
            )
            
            # Artifact'leri MLflow'a kaydet
            mlflow.log_artifact(user_item_matrix_path)
            mlflow.log_artifact(item_similarity_path)
            mlflow.log_artifact(item_metadata_path)
            
            print(f"MLflow çalışması tamamlandı. Run ID: {mlflow.active_run().info.run_id}")
            
            # Geçici dosyaları temizle
            for file_path in [user_item_matrix_path, item_similarity_path, item_metadata_path]:
                if os.path.exists(file_path):
                    os.remove(file_path)
    else:
        print("Model değerlendirilemedi!")

if __name__ == "__main__":
    main() 