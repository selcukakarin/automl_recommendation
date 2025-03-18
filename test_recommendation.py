"""
E-ticaret Ürün Öneri Sistemi Test Scripti
-----------------------------------------

Bu script, MLflow tabanlı ürün öneri sisteminin çeşitli özelliklerini test etmek için kullanılır.
API'nin doğru çalıştığını ve önerilerin beklendiği gibi döndüğünü doğrulamak amacıyla 
farklı test senaryolarını çalıştırır.

Test Özellikleri:
1. Model Sağlık Durumu - Model performans metriklerini ve sağlık durumunu kontrol eder
2. Model Versiyonları - Mevcut tüm model versiyonlarını listeler
3. Ürün-tabanlı Öneriler - Belirli bir ürüne benzeyen diğer ürünleri önerir
4. Kullanıcı-tabanlı Öneriler - Kullanıcının geçmiş davranışlarına göre öneriler sunar

Nasıl Kullanılır:
----------------
1. Öneri API servisinin çalıştığından emin olun (serve.py dosyası)
2. test_recommendation.py scriptini çalıştırın
3. Test sonuçlarını konsol çıktısında inceleyin

API Endpoint'leri:
----------------
- /recommend - Ürün önerileri sunar
- /recommendation_model_health - Model sağlık durumunu kontrol eder
- /versions - Mevcut model versiyonlarını listeler

Çıktı:
-----
Script, her test senaryosu için ayrıntılı çıktılar gösterir. Önerilen ürünlerin 
ayrıntıları, model sağlık durumu ve versiyon bilgileri konsola yazdırılır.

Hata Durumu:
-----------
Herhangi bir API çağrısı başarısız olursa, script hata mesajlarını görüntüler ve 
diğer testlere devam eder. Bu, API'nin çalışmıyor olması durumunda bile tüm test 
senaryolarının çalıştırılmasını sağlar.
"""

import requests
import random
import pandas as pd
import time
from datetime import datetime
import json

# API endpoint'leri
BASE_URL = "http://localhost:8000"
RECOMMEND_URL = f"{BASE_URL}/recommend"
HEALTH_URL = f"{BASE_URL}/recommendation_model_health"
VERSIONS_URL = f"{BASE_URL}/versions"

def load_test_data():
    """
    Test için örnek kullanıcı ve ürün verilerini yükler.
    
    Bu fonksiyon, 'data/' dizinindeki CSV dosyalarından kullanıcı ve ürün verilerini okur.
    Bu veriler, test sırasında rastgele kullanıcı ve ürünler seçmek için kullanılır.
    
    Returns:
        tuple: İki pandas DataFrame içeren bir tuple:
            - users_df: Kullanıcı verileri
            - products_df: Ürün verileri
            
    Raises:
        Exception: Veri dosyaları bulunamazsa veya yüklenemezse hata yakalayıp yazdırır
    """
    try:
        # Veri dosyalarını oku
        users_df = pd.read_csv('data/users.csv')
        products_df = pd.read_csv('data/products.csv')
        return users_df, products_df
    except Exception as e:
        print(f"Veri yükleme hatası: {e}")
        print("Veri dosyalarının 'data/' dizininde olduğunu kontrol edin.")
        print("Eğer dosyalar yoksa, önce 'generate_data.py' scriptini çalıştırın.")
        return None, None

def pretty_print_json(json_data):
    """JSON verisini düzgün formatta yazdırır"""
    try:
        if isinstance(json_data, str):
            # Eğer string ise, JSON'a çevirmeye çalış
            data = json.loads(json_data)
        else:
            data = json_data
            
        print(json.dumps(data, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"JSON yazdırma hatası: {e}")
        print(json_data)  # Ham veriyi yazdır

def safe_api_call(method, url, json_data=None, print_response=False):
    """API'yi çağırır ve hataları güvenli bir şekilde işler"""
    try:
        if method.lower() == 'get':
            response = requests.get(url, timeout=10)
        else:
            response = requests.post(url, json=json_data, timeout=10)
        
        # HTTP durum kodunu kontrol et
        if response.status_code != 200:
            print(f"API hatası: {response.status_code}")
            try:
                error_data = response.json()
                print("Hata detayları:")
                pretty_print_json(error_data)
            except:
                print(f"Ham hata yanıtı: {response.text}")
            return None
        
        # Yanıtı JSON'a çevirmeye çalış
        try:
            result = response.json()
            if print_response:
                print("API yanıtı:")
                pretty_print_json(result)
            return result
        except Exception as e:
            print(f"JSON çözümleme hatası: {e}")
            print(f"Ham yanıt: {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        print(f"Bağlantı hatası: API servisi çalışıyor mu? ({url})")
        return None
    except requests.exceptions.Timeout:
        print(f"Zaman aşımı: API yanıt vermeyi durdurdu ({url})")
        return None
    except Exception as e:
        print(f"API çağrı hatası: {e}")
        return None

def test_item_based_recommendation():
    """
    Ürün bazlı öneri sistemini test eder.
    
    Bu test, rastgele seçilen bir ürüne benzer olan diğer ürünlerin önerilmesini ister.
    Bu, "bu ürünü beğendiyseniz, şunları da beğenebilirsiniz" tarzı öneriler için kullanılır.
    
    İşlem Adımları:
    1. Rastgele bir ürün seçilir
    2. Bu ürüne benzer ürünleri önermesi için API'ye istek yapılır
    3. Sonuçlar konsola yazdırılır
    
    Ürün benzerlikleri, kullanıcı-ürün etkileşim matrisindeki kosinüs benzerliğine dayalıdır.
    """
    print("\n=== Ürün Bazlı Öneri Testi ===")
    
    # Test verilerini yükle
    users_df, products_df = load_test_data()
    if users_df is None or products_df is None:
        return
    
    # Rastgele bir ürün seç
    test_item = random.choice(products_df['item_id'].tolist())
    test_user = random.choice(users_df['user_id'].tolist())
    
    print(f"\nTest Ürünü ID: {test_item}")
    print(f"Test Kullanıcı ID: {test_user}")
    
    # Öneri isteği gönder
    result = safe_api_call(
        'post',
        RECOMMEND_URL,
        {
            "user_id": test_user,
            "item_id": test_item,
            "num_recommendations": 5
        }
    )
    
    if result:
        print("\nÖnerilen Ürünler:")
        # API yanıt formatını kontrol et
        if 'recommendations' in result:
            for i, rec in enumerate(result['recommendations'], 1):
                if 'item_id' in rec:
                    try:
                        # Ürün bilgilerini bul
                        item_info = products_df[products_df['item_id'] == rec['item_id']]
                        if not item_info.empty:
                            item_info = item_info.iloc[0]
                            print(f"\n{i}. Ürün:")
                            print(f"   ID: {rec['item_id']}")
                            print(f"   İsim: {item_info['item_name']}")
                            print(f"   Kategori: {item_info['item_category']}")
                            print(f"   Fiyat: {item_info['item_price']:.2f} TL")
                            if 'similarity_score' in rec:
                                print(f"   Benzerlik Skoru: {rec['similarity_score']:.4f}")
                        else:
                            print(f"\n{i}. Ürün:")
                            print(f"   ID: {rec['item_id']}")
                            print(f"   Not: Ürün bilgileri bulunamadı")
                    except Exception as e:
                        print(f"\n{i}. Ürün:")
                        print(f"   ID: {rec['item_id']}")
                        print(f"   Hata: {e}")
                else:
                    print(f"\n{i}. Ürün:")
                    print(f"   Ürün bilgilerini görüntüleme hatası (item_id alanı bulunamadı)")
                    print(f"   Ham veri: {rec}")
        else:
            print("API yanıtı beklenen formatta değil. Ham yanıt:")
            pretty_print_json(result)

def test_user_based_recommendation():
    """Kullanıcı bazlı öneri sistemini test eder"""
    print("\n=== Kullanıcı Bazlı Öneri Testi ===")
    
    # Test verilerini yükle
    users_df, products_df = load_test_data()
    if users_df is None or products_df is None:
        return
    
    # Rastgele bir kullanıcı seç
    test_user = random.choice(users_df['user_id'].tolist())
    print(f"\nTest Kullanıcı ID: {test_user}")
    
    # Öneri isteği gönder
    result = safe_api_call(
        'post',
        RECOMMEND_URL,
        {
            "user_id": test_user,
            "num_recommendations": 5
        }
    )
    
    if result:
        print("\nKullanıcıya Özel Öneriler:")
        # API yanıt formatını kontrol et
        if 'recommendations' in result:
            for i, rec in enumerate(result['recommendations'], 1):
                if 'item_id' in rec:
                    try:
                        # Ürün bilgilerini bul
                        item_info = products_df[products_df['item_id'] == rec['item_id']]
                        if not item_info.empty:
                            item_info = item_info.iloc[0]
                            print(f"\n{i}. Ürün:")
                            print(f"   ID: {rec['item_id']}")
                            print(f"   İsim: {item_info['item_name']}")
                            print(f"   Kategori: {item_info['item_category']}")
                            print(f"   Fiyat: {item_info['item_price']:.2f} TL")
                            if 'predicted_rating' in rec:
                                print(f"   Tahmini Puan: {rec['predicted_rating']:.2f}")
                        else:
                            print(f"\n{i}. Ürün:")
                            print(f"   ID: {rec['item_id']}")
                            print(f"   Not: Ürün bilgileri bulunamadı")
                    except Exception as e:
                        print(f"\n{i}. Ürün:")
                        print(f"   ID: {rec['item_id']}")
                        print(f"   Hata: {e}")
                else:
                    print(f"\n{i}. Ürün:")
                    print(f"   Ürün bilgilerini görüntüleme hatası (item_id alanı bulunamadı)")
                    print(f"   Ham veri: {rec}")
        else:
            print("API yanıtı beklenen formatta değil. Ham yanıt:")
            pretty_print_json(result)

def test_model_health():
    """
    Model sağlık durumunu test eder.
    
    Bu test, öneri modelinin performans metriklerini ve sağlık durumunu kontrol eder.
    Sistem yöneticileri, model performansını izlemek için bu bilgileri kullanabilir.
    
    Döndürülen Bilgiler:
    - Durum: Modelin genel durumu (sağlıklı/uyarı/kritik)
    - Versiyon: Yüklü model versiyonu
    - Son Güncelleme: Modelin en son ne zaman güncellendiği
    - Metrikler: RMSE, MAE, başarılı tahmin oranı gibi performans metrikleri
    """
    print("\n=== Model Sağlık Kontrolü ===")
    
    # Sağlık durumu isteği gönder
    result = safe_api_call('get', HEALTH_URL)
    
    if result:
        print("\nModel Sağlık Bilgileri:")
        # API yanıt formatını kontrol et
        if 'status' in result:
            print(f"Durum: {result['status']}")
            
            if 'version' in result:
                print(f"Yüklü Model Versiyonu: {result['version']}")
            else:
                print("Uyarı: Model versiyonu bulunamadı")
                
            if 'last_updated' in result:
                print(f"Son Güncelleme: {result['last_updated']}")
            
            if 'metrics' in result and result['metrics']:
                print("\nModel Metrikleri:")
                metrics = result['metrics']
                if 'rmse' in metrics:
                    print(f"RMSE: {metrics['rmse']:.4f}")
                if 'mae' in metrics:
                    print(f"MAE: {metrics['mae']:.4f}")
                if 'prediction_ratio' in metrics:
                    # Yüzde formatında göster
                    ratio = metrics['prediction_ratio']
                    print(f"Tahmin Oranı: {ratio*100:.2f}%")
        else:
            print("API yanıtı beklenen formatta değil. Ham yanıt:")
            pretty_print_json(result)

def test_versions():
    """
    Mevcut model versiyonlarını listeler ve hangi endpoint'ler için kullanıldığını gösterir.
    
    Bu test, sistemde mevcut olan tüm model versiyonlarını ve detaylarını görüntüler.
    Her model için hangi API endpoint'i tarafından kullanılacağı bilgisini de içerir.
    
    Bu, farklı model versiyonları arasında geçiş yapmak ve her birinin
    performansını karşılaştırmak için kullanışlıdır.
    """
    print("\n=== Model Versiyonları ===")
    
    # Versiyon listesi isteği gönder
    result = safe_api_call('get', VERSIONS_URL)
    
    if result:
        # API yanıt formatını kontrol et
        if 'versions' in result and isinstance(result['versions'], list):
            versions = result['versions']
            if versions:
                print("\nKullanılabilir Model Versiyonları:")
                for version in versions:
                    print("\nVersiyon:", version.get('version_name', 'Bilinmiyor'))
                    if 'creation_date' in version:
                        print(f"Oluşturulma Tarihi: {version['creation_date']}")
                    
                    if 'model_type' in version:
                        print(f"Model Tipi: {version['model_type']}")
                    
                    if 'endpoint_info' in version:
                        print(f"API Endpoint: {version['endpoint_info']}")
                        
                    if 'metrics' in version and version['metrics']:
                        print("Performans Metrikleri:")
                        for metric_name, metric_value in version['metrics'].items():
                            print(f"  {metric_name}: {float(metric_value):.4f}")
            else:
                print("Henüz hiç model versiyonu bulunamadı.")
        else:
            print("API yanıtı beklenen formatta değil. Ham yanıt:")
            pretty_print_json(result)

def check_api_connection():
    """API bağlantısını kontrol eder"""
    print("API bağlantısı kontrol ediliyor...")
    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        if response.status_code == 200:
            print("✅ API servisine bağlantı başarılı")
            return True
        else:
            print(f"❌ API servisine bağlantı başarısız: HTTP {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ API servisine bağlantı hatası: {e}")
        print(f"ℹ️  API servisini başlatmak için: uvicorn serve:app --host 0.0.0.0 --port 8000")
        return False

def main():
    """Ana test fonksiyonu"""
    print("=== Öneri Sistemi Test Programı ===")
    print(f"Test Başlangıç Zamanı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Önce API bağlantısını kontrol et
    if not check_api_connection():
        print("\n⚠️  API servisi çalışmıyor veya erişilemiyor! Testler başarısız olabilir.")
        print("İşlemi iptal etmek için CTRL+C tuşlarına basabilir veya devam etmek için Enter'a basabilirsiniz.")
        try:
            input("Testlere devam etmek için Enter'a basın...")
        except KeyboardInterrupt:
            print("\nTestler iptal edildi.")
            return
    
    # Model sağlığını kontrol et
    test_model_health()
    
    # Mevcut versiyonları listele
    test_versions()
    
    # Ürün bazlı önerileri test et
    test_item_based_recommendation()
    
    # Kullanıcı bazlı önerileri test et
    test_user_based_recommendation()
    
    print(f"\nTest Bitiş Zamanı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main() 