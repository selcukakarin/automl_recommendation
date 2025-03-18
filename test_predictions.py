import requests
import random
import time
from datetime import datetime

def generate_random_request():
    """Rastgele parametrelerle tahmin isteği oluşturur"""
    return {
        "user_id": random.randint(1, 100),
        "item_id": random.randint(1, 50),
        "user_age": random.randint(18, 70),
        "user_gender": random.choice(["M", "F"]),
        "item_category": random.choice(["A", "B", "C"]),
        "item_price": round(random.uniform(10, 100), 2)
    }

def send_prediction_request(data):
    """Tahmin isteği gönderir ve sonucu yazdırır"""
    url = "http://localhost:8000/predict"
    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            result = response.json()
            print(f"[{datetime.now()}] Tahmin: {result['predicted_rating']:.2f} "
                  f"(Güven Aralığı: {result['confidence_interval']})")
            return True
        else:
            print(f"Hata: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"İstek hatası: {str(e)}")
        return False

def check_model_health():
    """Model sağlığını kontrol eder ve sonuçları yazdırır"""
    url = "http://localhost:8000/model_health"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            health = response.json()
            print("\nModel Sağlık Durumu:")
            print(f"Status: {health['status']}")
            print(f"Örnek Sayısı: {health['sample_size']}")
            print("Metrikler:")
            for metric, value in health['metrics'].items():
                print(f"- {metric}: {value:.4f}")
            print("-" * 50)
    except Exception as e:
        print(f"Sağlık kontrolü hatası: {str(e)}")

def main():
    """Ana fonksiyon"""
    print("Otomatik tahmin testi başlıyor...")
    
    # Kaç tahmin yapılacak
    num_predictions = 150
    
    # Tahminler arası bekleme süresi (saniye)
    delay = 1
    
    successful_predictions = 0
    
    for i in range(num_predictions):
        print(f"\nTahmin {i+1}/{num_predictions}")
        
        # Rastgele veri oluştur
        data = generate_random_request()
        print(f"Girdi: {data}")
        
        # Tahmin yap
        if send_prediction_request(data):
            successful_predictions += 1
        
        # Her 10 tahminde bir model sağlığını kontrol et
        if (i + 1) % 10 == 0:
            check_model_health()
        
        # Bekle
        time.sleep(delay)
    
    print(f"\nTest tamamlandı!")
    print(f"Toplam tahmin: {num_predictions}")
    print(f"Başarılı tahmin: {successful_predictions}")
    print(f"Başarı oranı: {(successful_predictions/num_predictions)*100:.1f}%")

if __name__ == "__main__":
    main() 