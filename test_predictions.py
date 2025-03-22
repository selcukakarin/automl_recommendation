import requests
import random
import time
from datetime import datetime
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import logging
from logger import setup_logger, log_api_request, log_test_start, log_test_end, log_performance_metrics
import traceback

# Logger ayarları
logger = setup_logger(
    name="prediction_test",
    level="debug",
    log_file="test_predictions_log.log",  # Doğrudan bu isimli dosyaya yazılacak
    console_output=True,
    json_output=False
)

# Kök logger'ın seviyesini DEBUG olarak ayarla ve propagation'ı engelle
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)
logger.propagate = False  # Mesajların üst loggerlara iletilmesini engelle

def generate_random_request():
    """Rastgele tahmin isteği oluşturur"""
    user_ages = [25, 30, 35, 40, 45, 50, 55]  # Farklı yaş grupları
    user_genders = ["E", "K"]
    item_categories = ["Roman", "Bilim", "Tarih", "Sanat", "Felsefe", "Çocuk", "Kişisel Gelişim"]
    item_prices = [15.99, 24.99, 29.99, 34.99, 39.99, 44.99, 49.99, 54.99]
    
    return {
        "user_id": random.randint(1, 1000),
        "item_id": random.randint(1, 500),
        "user_age": random.choice(user_ages),
        "user_gender": random.choice(user_genders),
        "item_category": random.choice(item_categories),
        "item_price": random.choice(item_prices)
    }

def send_prediction_request(data):
    """Tahmin isteği gönderir ve sonucu yazdırır"""
    url = "http://localhost:8000/predict"
    try:
        logger.debug(f"Sending prediction request with data: {data}")
        start_time = time.time()
        
        response = requests.post(url, json=data)
        
        duration = time.time() - start_time
        logger.debug(f"Received response in {duration:.4f} seconds")
        
        if response.status_code == 200:
            result = response.json()
            
            # API isteğini logla
            log_api_request(
                logger, 
                method="POST", 
                url=url, 
                data=data, 
                response=result, 
                duration=duration
            )
            
            logger.info(f"Prediction: {result['predicted_rating']:.5f} "
                  f"(Confidence: [{result['confidence_interval'][0]:.5f}, {result['confidence_interval'][1]:.5f}])")
            return result
        else:
            error_message = f"API error: {response.status_code} - {response.text}"
            logger.error(error_message)
            
            # Hatalı API isteğini logla
            log_api_request(
                logger, 
                method="POST", 
                url=url, 
                data=data, 
                error=error_message, 
                duration=duration
            )
            
            return None
    except Exception as e:
        error_message = f"Request error: {str(e)}"
        logger.error(error_message)
        logger.error(traceback.format_exc())
        
        # Hatalı API isteğini logla
        log_api_request(
            logger, 
            method="POST", 
            url=url, 
            data=data, 
            error=str(e)
        )
        
        return None

def check_model_health():
    """Model sağlığını kontrol eder ve sonuçları yazdırır"""
    url = "http://localhost:8000/recommendation_model_health"
    try:
        logger.info("Checking model health...")
        start_time = time.time()
        
        response = requests.get(url)
        
        duration = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            rating_model = result.get('rating_model', {})
            
            # Sağlık kontrolü sonucunu logla
            log_api_request(
                logger, 
                method="GET", 
                url=url, 
                response=result, 
                duration=duration
            )
            
            health_info = {
                "status": rating_model.get("status", "Bilinmiyor"),
                "model_version": rating_model.get("version", "Bilinmiyor"),
                "last_trained": rating_model.get("last_updated", "Bilinmiyor"),
                "metrics": rating_model.get("metrics", {
                    "rmse": 0.0,
                    "mae": 0.0,
                    "prediction_ratio": 0.0,
                    "n_predictions": 0.0
                }),
                "model_info": rating_model.get("model_info", {}),
                "message": rating_model.get("message", ""),
                "sample_size": rating_model.get("sample_size", 0)
            }
            
            logger.info(f"Model health status: {health_info['status']}")
            logger.info(f"Model version: {health_info['model_version']}")
            
            # Metrikleri logla
            log_performance_metrics(
                logger,
                health_info['metrics'],
                "Model Health Check"
            )
            
            return health_info
        
        error_message = f"Health check failed: {response.status_code} - {response.text}"
        logger.error(error_message)
        return None
    except Exception as e:
        error_message = f"Health check error: {str(e)}"
        logger.error(error_message)
        logger.error(traceback.format_exc())
        return None

def create_report_dir():
    """Test raporları için klasör oluşturur"""
    # Ana rapor dizini
    base_dir = "test_reports"
    
    # Tahmin testleri için alt dizin
    test_dir = os.path.join(base_dir, "prediction_tests")
    
    # Her çalıştırma için yeni zaman damgalı klasör
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(test_dir, f"run_{timestamp}")
    
    # Klasörleri oluştur
    os.makedirs(run_dir, exist_ok=True)
    
    logger.info(f"Created report directory: {run_dir}")
    return run_dir, timestamp

def main():
    """Ana fonksiyon"""
    start_time = time.time()
    
    # Test başlangıcını logla
    log_test_start(
        logger, 
        "Prediction System Test", 
        {"test_type": "automated", "prediction_count": 10}
    )
    
    logger.info("Starting automated prediction test...")
    
    # Test raporları için klasör oluştur
    report_dir, timestamp = create_report_dir()
    
    # Test raporu dosyası oluştur
    report_path = os.path.join(report_dir, "prediction_test_report.txt")
    logger.info(f"Test report will be saved to: {report_path}")
    
    # Test sonuçlarını saklamak için listeler
    predictions = []
    response_times = []
    
    # Kaç tahmin yapılacak
    num_predictions = 10
    
    successful_predictions = 0
    
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=== Tahmin Sistemi Test Raporu ===\n")
            f.write(f"Test Tarihi: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Planlanan Tahmin Sayısı: {num_predictions}\n\n")
            
            for i in range(num_predictions):
                logger.info(f"Running prediction {i+1}/{num_predictions}")
                
                # Rastgele veri oluştur
                data = generate_random_request()
                f.write(f"\nTahmin {i+1}:\n")
                f.write(f"Girdi: {json.dumps(data, indent=2, ensure_ascii=False)}\n")
                
                # Tahmin zamanını kaydet
                start_time = time.time()
                
                # Tahmin yap
                result = send_prediction_request(data)
                
                # Yanıt süresini hesapla
                response_time = time.time() - start_time
                response_times.append(response_time)
                
                if result:
                    successful_predictions += 1
                    predictions.append(result['predicted_rating'])
                    f.write(f"Tahmin: {result['predicted_rating']:.5f}\n")
                    f.write(f"Güven Aralığı: [{result['confidence_interval'][0]:.5f}, {result['confidence_interval'][1]:.5f}]\n")
                    f.write(f"Yanıt Süresi: {response_time:.4f} saniye\n")
                else:
                    f.write("Tahmin başarısız!\n")
            
            # Test sonuçlarını özetle
            f.write("\n=== Test Özeti ===\n")
            f.write(f"Toplam Tahmin: {num_predictions}\n")
            f.write(f"Başarılı Tahmin: {successful_predictions}\n")
            f.write(f"Başarı Oranı: %{(successful_predictions/num_predictions)*100:.1f}\n")
            
            if predictions:
                f.write(f"\nTahmin İstatistikleri:\n")
                f.write(f"Ortalama Tahmin: {np.mean(predictions):.5f}\n")
                f.write(f"Standart Sapma: {np.std(predictions):.5f}\n")
                f.write(f"Min Tahmin: {min(predictions):.5f}\n")
                f.write(f"Max Tahmin: {max(predictions):.5f}\n")
            
            f.write(f"\nPerformans İstatistikleri:\n")
            f.write(f"Ortalama Yanıt Süresi: {np.mean(response_times):.4f} saniye\n")
            f.write(f"Standart Sapma: {np.std(response_times):.4f} saniye\n")
            f.write(f"Min Yanıt Süresi: {min(response_times):.4f} saniye\n")
            f.write(f"Max Yanıt Süresi: {max(response_times):.4f} saniye\n")
        
        # Grafikleri oluştur
        logger.info("Creating test result graphs...")
        
        if predictions:
            # Tahmin dağılımı grafiği
            logger.debug("Creating prediction distribution graph")
            plt.figure(figsize=(10, 6))
            plt.hist(predictions, bins=20, edgecolor='black')
            plt.title('Tahmin Dağılımı')
            plt.xlabel('Tahmin Değeri')
            plt.ylabel('Frekans')
            plt.grid(True)
            plt.savefig(os.path.join(report_dir, f"prediction_distribution_{timestamp}.png"))
            plt.savefig(os.path.join(report_dir, f"prediction_distribution_{timestamp}.pdf"))
            plt.close()
            
            # Yanıt süresi grafiği
            logger.debug("Creating response times graph")
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(response_times) + 1), response_times, marker='o')
            plt.title('Yanıt Süreleri')
            plt.xlabel('Tahmin Sırası')
            plt.ylabel('Yanıt Süresi (saniye)')
            plt.grid(True)
            plt.savefig(os.path.join(report_dir, f"response_times_{timestamp}.png"))
            plt.savefig(os.path.join(report_dir, f"response_times_{timestamp}.pdf"))
            plt.close()
        
        # Test sonuç metriklerini logla
        test_metrics = {
            "total_predictions": num_predictions,
            "successful_predictions": successful_predictions,
            "success_rate": (successful_predictions/num_predictions)*100,
            "avg_response_time": np.mean(response_times),
            "min_response_time": min(response_times),
            "max_response_time": max(response_times)
        }
        
        if predictions:
            test_metrics.update({
                "avg_prediction": np.mean(predictions),
                "std_prediction": np.std(predictions),
                "min_prediction": min(predictions),
                "max_prediction": max(predictions)
            })
        
        log_performance_metrics(
            logger,
            test_metrics,
            "Prediction Test Summary",
            run_id=timestamp
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Test bitişini logla
        log_test_end(
            logger,
            "Prediction System Test",
            success=True,
            duration=duration,
            summary=f"Completed {successful_predictions}/{num_predictions} predictions successfully"
        )
        
        logger.info(f"\nTest tamamlandı!")
        logger.info(f"Toplam tahmin: {num_predictions}")
        logger.info(f"Başarılı tahmin: {successful_predictions}")
        logger.info(f"Başarı oranı: %{(successful_predictions/num_predictions)*100:.1f}")
        logger.info(f"\nTest raporu kaydedildi: {report_path}")
    
    except Exception as e:
        logger.critical(f"Test failed with error: {str(e)}")
        logger.critical(traceback.format_exc())
        
        # Test başarısızlığını logla
        log_test_end(
            logger, 
            "Prediction System Test", 
            success=False, 
            summary=f"Test failed with error: {str(e)}"
        )

if __name__ == "__main__":
    main() 