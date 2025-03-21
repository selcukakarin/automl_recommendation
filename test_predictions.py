import requests
import random
import time
from datetime import datetime
import os
import json
import matplotlib.pyplot as plt
import numpy as np

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
        print("\nAPI İsteği:")
        print(json.dumps(data, indent=2, ensure_ascii=False))
        
        response = requests.post(url, json=data)
        print("\nAPI Yanıtı:")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("Response JSON:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
            print(f"\n[{datetime.now()}] Tahmin: {result['predicted_rating']:.5f} "
                  f"(Güven Aralığı: [{result['confidence_interval'][0]:.5f}, {result['confidence_interval'][1]:.5f}])")
            return result
        else:
            print(f"Hata: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"İstek hatası: {str(e)}")
        return None

def check_model_health():
    """Model sağlığını kontrol eder ve sonuçları yazdırır"""
    url = "http://localhost:8000/model_health"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        print(f"Sağlık kontrolü hatası: {str(e)}")
        return None

def create_report_dir():
    """Test raporları için klasör oluşturur"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = "test_reports"
    # Ana test klasörü
    test_dir = os.path.join(base_dir, "prediction_tests")
    # Her çalıştırma için yeni zaman damgalı klasör
    run_dir = os.path.join(test_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir, timestamp

def main():
    """Ana fonksiyon"""
    print("Otomatik tahmin testi başlıyor...")
    
    # Test raporları için klasör oluştur
    report_dir, timestamp = create_report_dir()
    
    # Test raporu dosyası oluştur
    report_path = os.path.join(report_dir, f"prediction_test_report_{timestamp}.txt")
    
    # Test sonuçlarını saklamak için listeler
    predictions = []
    response_times = []
    
    # Kaç tahmin yapılacak
    num_predictions = 10
    
    successful_predictions = 0
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=== Tahmin Sistemi Test Raporu ===\n")
        f.write(f"Test Tarihi: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Planlanan Tahmin Sayısı: {num_predictions}\n\n")
        
        for i in range(num_predictions):
            print(f"\nTahmin {i+1}/{num_predictions}")
            
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
    if predictions:
        # Tahmin dağılımı grafiği
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
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(response_times) + 1), response_times, marker='o')
        plt.title('Yanıt Süreleri')
        plt.xlabel('Tahmin Sırası')
        plt.ylabel('Yanıt Süresi (saniye)')
        plt.grid(True)
        plt.savefig(os.path.join(report_dir, f"response_times_{timestamp}.png"))
        plt.savefig(os.path.join(report_dir, f"response_times_{timestamp}.pdf"))
        plt.close()
    
    print(f"\nTest tamamlandı!")
    print(f"Toplam tahmin: {num_predictions}")
    print(f"Başarılı tahmin: {successful_predictions}")
    print(f"Başarı oranı: %{(successful_predictions/num_predictions)*100:.1f}")
    print(f"\nTest raporu kaydedildi: {report_path}")

if __name__ == "__main__":
    main() 