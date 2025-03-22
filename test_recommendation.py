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
from datetime import datetime, timedelta
import json
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from tqdm import tqdm
import logging
import traceback
from logger import setup_logger, log_api_request, log_test_start, log_test_end, log_performance_metrics

# Logger ayarları
logger = setup_logger(
    name="recommendation_test",
    level="debug",
    log_file="test_recommendation_logs.log",  # Doğrudan bu isimli dosyaya yazılacak
    console_output=True,
    json_output=False
)

# Kök logger'ın seviyesini DEBUG olarak ayarla ve propagation'ı engelle
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)
logger.propagate = False  # Mesajların üst loggerlara iletilmesini engelle

# API endpoint'leri
BASE_URL = "http://localhost:8000"
RECOMMEND_URL = f"{BASE_URL}/recommend"
HEALTH_URL = f"{BASE_URL}/recommendation_model_health"
VERSIONS_URL = f"{BASE_URL}/versions"
METRICS_URL = f"{BASE_URL}/metrics"
LOAD_VERSION_URL = f"{BASE_URL}/load_recommendation_version"

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
        logger.info("Loading test data from CSV files...")
        # Veri dosyalarını oku
        users_df = pd.read_csv('data/users.csv')
        products_df = pd.read_csv('data/products.csv')
        
        logger.info(f"Loaded {len(users_df)} users and {len(products_df)} products")
        return users_df, products_df
    except Exception as e:
        error_msg = f"Data loading error: {e}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
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
        logger.warning(f"JSON formatting error: {e}")
        print(f"JSON yazdırma hatası: {e}")
        print(json_data)  # Ham veriyi yazdır

def safe_api_call(method, url, json_data=None, print_response=False):
    """API'yi çağırır ve hataları güvenli bir şekilde işler"""
    try:
        logger.debug(f"Making API call: {method.upper()} {url}")
        if json_data:
            logger.debug(f"Request data: {json_data}")
        
        start_time = time.time()
        
        if method.lower() == 'get':
            response = requests.get(url, timeout=10)
        else:
            response = requests.post(url, json=json_data, timeout=10)
        
        duration = time.time() - start_time
        logger.debug(f"Received response in {duration:.4f} seconds")
        
        # HTTP durum kodunu kontrol et
        if response.status_code != 200:
            error_msg = f"API error: {response.status_code}"
            logger.error(error_msg)
            
            try:
                error_data = response.json()
                logger.error(f"Error details: {error_data}")
            except:
                logger.error(f"Raw error response: {response.text}")
            
            # API hatasını logla
            log_api_request(
                logger,
                method=method.upper(),
                url=url,
                data=json_data,
                error=f"HTTP {response.status_code}: {response.text}",
                duration=duration
            )
            
            return None
        
        # Yanıtı JSON'a çevirmeye çalış
        try:
            result = response.json()
            
            # Başarılı API isteğini logla
            log_api_request(
                logger,
                method=method.upper(),
                url=url,
                data=json_data,
                response=result,
                duration=duration
            )
            
            if print_response:
                logger.debug(f"API response: {json.dumps(result, default=str)[:500]}...")
                print("API yanıtı:")
                pretty_print_json(result)
                
            return result
        except Exception as e:
            error_msg = f"JSON parsing error: {e}"
            logger.error(error_msg)
            logger.error(f"Raw response: {response.text}")
            
            # JSON dönüşüm hatasını logla
            log_api_request(
                logger,
                method=method.upper(),
                url=url,
                data=json_data,
                error=f"JSON parsing error: {e}",
                duration=duration
            )
            
            return None
            
    except requests.exceptions.ConnectionError:
        error_msg = f"Connection error: Is API service running? ({url})"
        logger.error(error_msg)
        
        # Bağlantı hatasını logla
        log_api_request(
            logger,
            method=method.upper(),
            url=url,
            data=json_data,
            error="Connection error"
        )
        
        return None
    except requests.exceptions.Timeout:
        error_msg = f"Timeout: API service stopped responding ({url})"
        logger.error(error_msg)
        
        # Zaman aşımı hatasını logla
        log_api_request(
            logger,
            method=method.upper(),
            url=url,
            data=json_data,
            error="Request timeout"
        )
        
        return None
    except Exception as e:
        error_msg = f"API call error: {e}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        
        # Genel hata durumunu logla
        log_api_request(
            logger,
            method=method.upper(),
            url=url,
            data=json_data,
            error=str(e)
        )
        
        return None

def check_api_connection():
    """API bağlantısını kontrol eder"""
    logger.info("Checking API connection...")
    try:
        start_time = time.time()
        response = requests.get(f"{BASE_URL}/", timeout=5)
        duration = time.time() - start_time
        
        if response.status_code == 200:
            logger.info(f"✅ API connection successful (Response time: {duration:.4f}s)")
            return True
        else:
            logger.error(f"❌ API connection failed: HTTP {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ API connection error: {e}")
        logger.error(f"ℹ️  To start API service: python serve.py")
        return False

def create_report_dir():
    """Test raporları için klasör oluşturur"""
    # Ana rapor dizini
    base_dir = "test_reports"
    
    # Öneri testleri için alt dizin
    test_dir = os.path.join(base_dir, "recommendation_tests")
    os.makedirs(test_dir, exist_ok=True)
    
    # Her çalıştırma için yeni zaman damgalı klasör
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(test_dir, f"run_{timestamp}")
    
    # Alt klasörleri oluştur
    os.makedirs(os.path.join(run_dir, "user_based"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "item_based"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "versions"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "health"), exist_ok=True)
    
    # Current run dosyasını güncelle
    current_run_file = os.path.join(test_dir, ".current_run")
    with open(current_run_file, "w") as f:
        f.write(run_dir)
    
    logger.info(f"Created report directory: {run_dir}")
    return run_dir, timestamp

def test_item_based_recommendation():
    """Ürün bazlı öneri testleri yapar"""
    # Ana rapor dizinini al
    test_dir = os.path.join("test_reports", "recommendation_tests")
    current_run_file = os.path.join(test_dir, ".current_run")
    
    # Mevcut çalışma dizinini oku
    with open(current_run_file, "r") as f:
        run_dir = f.read().strip()
    
    # Timestamp'i run_dir'den çıkar
    timestamp = os.path.basename(run_dir).replace("run_", "")
    
    # Item-based klasörüne rapor yaz
    report_path = os.path.join(run_dir, "item_based", f"item_based_test_report_{timestamp}.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=== Ürün Bazlı Öneri Testi Raporu ===\n")
        f.write(f"Test Tarihi: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Test verilerini yükle
        users_df, products_df = load_test_data()
        if users_df is None or products_df is None:
            error_msg = "Test verileri yüklenemedi!"
            f.write(f"{error_msg}\n")
            print(error_msg)
            return
        
        # Rastgele bir ürün ve kullanıcı seç
        test_item = random.choice(products_df['item_id'].tolist())
        test_user = random.choice(users_df['user_id'].tolist())
        
        test_info = f"Test Ürünü ID: {test_item}\nTest Kullanıcı ID: {test_user}"
        f.write(f"{test_info}\n\n")
        print(f"\n{test_info}")
        
        # Seçilen ürünün bilgilerini yaz
        item_info = products_df[products_df['item_id'] == test_item].iloc[0]
        item_details = (
            f"Test Ürünü Detayları:\n"
            f"İsim: {item_info['item_name']}\n"
            f"Kategori: {item_info['item_category']}\n"
            f"Fiyat: {item_info['item_price']:.2f} TL\n"
        )
        f.write(item_details)
        print(item_details)
        
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
            if 'recommendations' in result:
                recommendations = result['recommendations']
                header = f"\nÖnerilen Ürünler ({len(recommendations)}):"
                f.write(f"{header}\n")
                print(header)
                
                # Önerilen ürünlerin bilgilerini ve benzerlik skorlarını sakla
                recommended_items = []
                
                for i, rec in enumerate(recommendations, 1):
                    f.write("-" * 40 + "\n")
                    if 'item_id' in rec:
                        try:
                            item_info = products_df[products_df['item_id'] == rec['item_id']]
                            if not item_info.empty:
                                item_info = item_info.iloc[0]
                                item_str = (
                                    f"{i}. Ürün:\n"
                                    f"   ID: {rec['item_id']}\n"
                                    f"   İsim: {item_info['item_name']}\n"
                                    f"   Kategori: {item_info['item_category']}\n"
                                    f"   Fiyat: {item_info['item_price']:.2f} TL\n"
                                )
                                if 'similarity_score' in rec:
                                    item_str += f"   Benzerlik Skoru: {rec['similarity_score']:.4f}\n"
                                    recommended_items.append({
                                        'name': item_info['item_name'],
                                        'similarity': rec['similarity_score']
                                    })
                                
                                f.write(item_str)
                                print(item_str)
                            else:
                                error_str = (
                                    f"{i}. Ürün:\n"
                                    f"   ID: {rec['item_id']}\n"
                                    f"   Not: Ürün bilgileri bulunamadı\n"
                                )
                                f.write(error_str)
                                print(error_str)
                        except Exception as e:
                            error_str = (
                                f"{i}. Ürün:\n"
                                f"   ID: {rec['item_id']}\n"
                                f"   Hata: {e}\n"
                            )
                            f.write(error_str)
                            print(error_str)
                    else:
                        error_str = (
                            f"{i}. Ürün:\n"
                            f"   Ürün bilgilerini görüntüleme hatası (item_id alanı bulunamadı)\n"
                            f"   Ham veri: {rec}\n"
                        )
                        f.write(error_str)
                        print(error_str)
                
                # Benzerlik skorlarını görselleştir
                if recommended_items:
                    plt.figure(figsize=(10, 6))
                    names = [item['name'][:20] + '...' if len(item['name']) > 20 else item['name'] 
                            for item in recommended_items]
                    scores = [item['similarity'] for item in recommended_items]
                    
                    plt.bar(names, scores)
                    plt.title('Önerilen Ürünlerin Benzerlik Skorları')
                    plt.xlabel('Ürün İsmi')
                    plt.ylabel('Benzerlik Skoru')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    plt.savefig(os.path.join(run_dir, "item_based", f"item_similarity_{timestamp}.png"))
                    plt.savefig(os.path.join(run_dir, "item_based", f"item_similarity_{timestamp}.pdf"))
                    plt.close()
            else:
                error_msg = "API yanıtı beklenen formatta değil!"
                f.write(f"{error_msg}\n")
                print(error_msg)
                f.write("Ham yanıt:\n")
                json.dump(result, f, indent=2, ensure_ascii=False)
        else:
            error_msg = "Öneri alınamadı!"
            f.write(f"{error_msg}\n")
            print(error_msg)
    
    print(f"\nÜrün bazlı öneri raporu kaydedildi: {report_path}")

def test_user_based_recommendation():
    """Kullanıcı bazlı öneri testleri yapar"""
    # Ana rapor dizinini al
    test_dir = os.path.join("test_reports", "recommendation_tests")
    current_run_file = os.path.join(test_dir, ".current_run")
    
    # Mevcut çalışma dizinini oku
    with open(current_run_file, "r") as f:
        run_dir = f.read().strip()
    
    # Timestamp'i run_dir'den çıkar
    timestamp = os.path.basename(run_dir).replace("run_", "")
    
    # User-based klasörüne rapor yaz
    report_path = os.path.join(run_dir, "user_based", f"user_based_test_report_{timestamp}.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=== Kullanıcı Bazlı Öneri Testi Raporu ===\n")
        f.write(f"Test Tarihi: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Test verilerini yükle
        users_df, products_df = load_test_data()
        if users_df is None or products_df is None:
            error_msg = "Test verileri yüklenemedi!"
            f.write(f"{error_msg}\n")
            print(error_msg)
            return
        
        # Rastgele bir kullanıcı seç
        test_user = random.choice(users_df['user_id'].tolist())
        user_info = f"Test Kullanıcı ID: {test_user}"
        f.write(f"{user_info}\n\n")
        print(f"\n{user_info}")
        
        # Kullanıcı bilgilerini yaz
        user_data = users_df[users_df['user_id'] == test_user].iloc[0]
        user_details = (
            f"Kullanıcı Detayları:\n"
            f"Yaş: {user_data['user_age']}\n"
            f"Cinsiyet: {user_data['user_gender']}\n"
            f"Lokasyon: {user_data['user_location']}\n"
        )
        f.write(user_details)
        print(user_details)
        
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
            if 'recommendations' in result:
                recommendations = result['recommendations']
                header = f"\nKullanıcıya Özel Öneriler ({len(recommendations)}):"
                f.write(f"{header}\n")
                print(header)
                
                # Önerilen ürünlerin bilgilerini ve tahmin skorlarını sakla
                recommended_items = []
                
                for i, rec in enumerate(recommendations, 1):
                    f.write("-" * 40 + "\n")
                    if 'item_id' in rec:
                        try:
                            item_info = products_df[products_df['item_id'] == rec['item_id']]
                            if not item_info.empty:
                                item_info = item_info.iloc[0]
                                item_str = (
                                    f"{i}. Ürün:\n"
                                    f"   ID: {rec['item_id']}\n"
                                    f"   İsim: {item_info['item_name']}\n"
                                    f"   Kategori: {item_info['item_category']}\n"
                                    f"   Fiyat: {item_info['item_price']:.2f} TL\n"
                                )
                                if 'predicted_rating' in rec:
                                    item_str += f"   Tahmini Puan: {rec['predicted_rating']:.2f}\n"
                                    recommended_items.append({
                                        'name': item_info['item_name'],
                                        'rating': rec['predicted_rating']
                                    })
                                
                                f.write(item_str)
                                print(item_str)
                            else:
                                error_str = (
                                    f"{i}. Ürün:\n"
                                    f"   ID: {rec['item_id']}\n"
                                    f"   Not: Ürün bilgileri bulunamadı\n"
                                )
                                f.write(error_str)
                                print(error_str)
                        except Exception as e:
                            error_str = (
                                f"{i}. Ürün:\n"
                                f"   ID: {rec['item_id']}\n"
                                f"   Hata: {e}\n"
                            )
                            f.write(error_str)
                            print(error_str)
                    else:
                        error_str = (
                            f"{i}. Ürün:\n"
                            f"   Ürün bilgilerini görüntüleme hatası (item_id alanı bulunamadı)\n"
                            f"   Ham veri: {rec}\n"
                        )
                        f.write(error_str)
                        print(error_str)
                
                # Tahmin skorlarını görselleştir
                if recommended_items:
                    plt.figure(figsize=(10, 6))
                    names = [item['name'][:20] + '...' if len(item['name']) > 20 else item['name'] 
                            for item in recommended_items]
                    ratings = [item['rating'] for item in recommended_items]
                    
                    plt.bar(names, ratings)
                    plt.title('Önerilen Ürünlerin Tahmini Puanları')
                    plt.xlabel('Ürün İsmi')
                    plt.ylabel('Tahmini Puan')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    plt.savefig(os.path.join(run_dir, "user_based", f"user_predictions_{timestamp}.png"))
                    plt.savefig(os.path.join(run_dir, "user_based", f"user_predictions_{timestamp}.pdf"))
                    plt.close()
            else:
                error_msg = "API yanıtı beklenen formatta değil!"
                f.write(f"{error_msg}\n")
                print(error_msg)
                f.write("Ham yanıt:\n")
                json.dump(result, f, indent=2, ensure_ascii=False)
        else:
            error_msg = "Öneri alınamadı!"
            f.write(f"{error_msg}\n")
            print(error_msg)
    
    print(f"\nKullanıcı bazlı öneri raporu kaydedildi: {report_path}")

def test_model_health():
    """Model sağlık kontrolü yapar"""
    # Ana rapor dizinini al
    test_dir = os.path.join("test_reports", "recommendation_tests")
    current_run_file = os.path.join(test_dir, ".current_run")
    
    # Mevcut çalışma dizinini oku
    with open(current_run_file, "r") as f:
        run_dir = f.read().strip()
    
    # Timestamp'i run_dir'den çıkar
    timestamp = os.path.basename(run_dir).replace("run_", "")
    
    # Health klasörünü oluştur
    health_dir = os.path.join(run_dir, "health")
    os.makedirs(health_dir, exist_ok=True)
    
    # Health klasörüne rapor yaz
    report_path = os.path.join(health_dir, f"health_test_report_{timestamp}.txt")
    
    try:
        logger.info("Checking recommendation model health...")
        start_time = time.time()
        
        result = safe_api_call('get', HEALTH_URL)
        
        duration = time.time() - start_time
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=== Model Sağlık Kontrolü Raporu ===\n")
            f.write(f"Test Tarihi: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            if result:
                recommendation_model = result.get('recommendation_model', {})
                
                health_info = {
                    "status": recommendation_model.get("status", "Bilinmiyor"),
                    "model_version": recommendation_model.get("version", "Bilinmiyor"),
                    "last_trained": recommendation_model.get("last_updated", "Bilinmiyor"),
                    "metrics": recommendation_model.get("metrics", {
                        "average_rating": 0.0,
                        "rating_count": 0,
                        "unique_users": 0,
                        "unique_items": 0,
                        "sparsity": 0.0
                    }),
                    "model_info": recommendation_model.get("model_info", {})
                }
                
                # Model sağlık bilgilerini yaz
                f.write("Model Sağlık Bilgileri:\n\n")
                f.write(f"Model Durumu: {health_info['status']}\n")
                f.write(f"Model Versiyonu: {health_info['model_version']}\n")
                f.write(f"Son Güncelleme: {health_info['last_trained']}\n\n")
                
                # Model metriklerini yaz
                f.write("Model Metrikleri:\n")
                for metric_name, metric_value in health_info['metrics'].items():
                    f.write(f"- {metric_name}: {metric_value}\n")
                
                # Model bilgilerini yaz
                f.write("\nModel Bilgileri:\n")
                for info_name, info_value in health_info['model_info'].items():
                    f.write(f"- {info_name}: {info_value}\n")
                
                f.write("\nTest Tamamlandı.\n")
                
                logger.info(f"Model health status: {health_info['status']}")
                logger.info(f"Model version: {health_info['model_version']}")
                
                # Metrikleri logla
                log_performance_metrics(
                    logger,
                    health_info['metrics'],
                    "Recommendation Model Health Check"
                )
                
                return True
            else:
                error_msg = "Model sağlık bilgileri alınamadı!"
                f.write(f"\n{error_msg}\n")
                logger.error(error_msg)
                return False
                
    except Exception as e:
        error_msg = f"Health check error: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"Test Hatası: {error_msg}\n")
        return False

def test_versions():
    """Model versiyonlarını test eder"""
    # Ana rapor dizinini al
    test_dir = os.path.join("test_reports", "recommendation_tests")
    current_run_file = os.path.join(test_dir, ".current_run")
    
    # Mevcut çalışma dizinini oku
    with open(current_run_file, "r") as f:
        run_dir = f.read().strip()
    
    # Timestamp'i run_dir'den çıkar
    timestamp = os.path.basename(run_dir).replace("run_", "")
    
    # Versions klasörüne rapor yaz
    report_path = os.path.join(run_dir, "versions", f"versions_test_report_{timestamp}.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=== Model Versiyonları Raporu ===\n")
        f.write(f"Test Tarihi: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Versiyon listesi isteği gönder
        result = safe_api_call('get', VERSIONS_URL)
        
        if result:
            if 'versions' in result and isinstance(result['versions'], list):
                versions = result['versions']
                if versions:
                    header = "Kullanılabilir Model Versiyonları:"
                    f.write(f"{header}\n")
                    print(f"\n{header}")
                    
                    # Metrik değerlerini sakla
                    version_metrics = []
                    
                    for version in versions:
                        f.write("\n" + "="*50 + "\n")
                        version_name = version.get('version_name', 'Bilinmiyor')
                        f.write(f"Versiyon: {version_name}\n")
                        print(f"\nVersiyon: {version_name}")
                        
                        if 'creation_date' in version:
                            date_str = f"Oluşturulma Tarihi: {version['creation_date']}"
                            f.write(f"{date_str}\n")
                            print(date_str)
                        
                        if 'model_type' in version:
                            type_str = f"Model Tipi: {version['model_type']}"
                            f.write(f"{type_str}\n")
                            print(type_str)
                        
                        if 'endpoint' in version:
                            endpoint_str = f"API Endpoint: {version['endpoint']}"
                            f.write(f"{endpoint_str}\n")
                            print(endpoint_str)
                        
                        if 'metrics' in version and version['metrics']:
                            f.write("Performans Metrikleri:\n")
                            print("Performans Metrikleri:")
                            metrics = version['metrics']
                            for metric_name, metric_value in metrics.items():
                                metric_str = f"  {metric_name}: {float(metric_value):.4f}"
                                f.write(f"{metric_str}\n")
                                print(metric_str)
                            
                            # Metrik değerlerini sakla
                            version_metrics.append({
                                'version': version_name,
                                'metrics': {k: float(v) for k, v in metrics.items()}
                            })
                    
                    # Metrikleri görselleştir
                    if version_metrics:
                        # Her metrik için ayrı grafik
                        all_metrics = set()
                        for vm in version_metrics:
                            all_metrics.update(vm['metrics'].keys())
                        
                        for metric in all_metrics:
                            plt.figure(figsize=(10, 6))
                            versions = [vm['version'] for vm in version_metrics if metric in vm['metrics']]
                            values = [vm['metrics'][metric] for vm in version_metrics if metric in vm['metrics']]
                            
                            plt.bar(versions, values)
                            plt.title(f'{metric} Metriği - Versiyon Karşılaştırması')
                            plt.xticks(rotation=45)
                            plt.tight_layout()
                            plt.savefig(os.path.join(run_dir, "versions", f"version_comparison_{metric}_{timestamp}.png"))
                            plt.savefig(os.path.join(run_dir, "versions", f"version_comparison_{metric}_{timestamp}.pdf"))
                            plt.close()
                else:
                    msg = "Henüz hiç model versiyonu bulunamadı."
                    f.write(f"{msg}\n")
                    print(msg)
            else:
                msg = "API yanıtı beklenen formatta değil!"
                f.write(f"{msg}\n")
                print(msg)
                f.write("Ham yanıt:\n")
                json.dump(result, f, indent=2, ensure_ascii=False)
        else:
            error_msg = "Model versiyon bilgileri alınamadı!"
            f.write(f"{error_msg}\n")
            print(error_msg)
    
    print(f"\nVersiyon raporu kaydedildi: {report_path}")

def test_batch_inference_time():
    """Toplu tahmin performans testi yapar"""
    # Ana rapor dizinini al
    test_dir = os.path.join("test_reports", "recommendation_tests")
    current_run_file = os.path.join(test_dir, ".current_run")
    
    # Mevcut çalışma dizinini oku
    with open(current_run_file, "r") as f:
        run_dir = f.read().strip()
    
    # Timestamp'i run_dir'den çıkar
    timestamp = os.path.basename(run_dir).replace("run_", "")
    
    # Batch klasörüne rapor yaz
    report_path = os.path.join(run_dir, "batch", f"batch_test_report_{timestamp}.txt")
    os.makedirs(os.path.join(run_dir, "batch"), exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=== Toplu Çıkarsama Performans Testi Raporu ===\n")
        f.write(f"Test Tarihi: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Test verilerini yükle
        users_df, products_df = load_test_data()
        if users_df is None or products_df is None:
            return
        
        # Test için farklı boyutlarda gruplar
        batch_sizes = [1, 5, 10, 20, 50]
        results = {}
        
        for size in batch_sizes:
            print(f"\nBatch boyutu: {size} kullanıcı")
            users = random.sample(users_df['user_id'].tolist(), min(size, len(users_df)))
            
            # Öneri sürelerini ölç
            start_time = time.time()
            successful = 0
            
            for user_id in users:
                result = safe_api_call(
                    'post',
                    RECOMMEND_URL,
                    {
                        "user_id": user_id,
                        "num_recommendations": 5
                    },
                    print_response=False
                )
                if result:
                    successful += 1
            
            end_time = time.time()
            total_time = end_time - start_time
            avg_time = total_time / size if size > 0 else 0
            
            results[size] = {
                "total_time": total_time,
                "avg_time": avg_time,
                "success_rate": (successful / size) * 100 if size > 0 else 0
            }
            
            print(f"Toplam süre: {total_time:.2f} saniye")
            print(f"Ortalama süre: {avg_time:.4f} saniye/istek")
            print(f"Başarı oranı: %{results[size]['success_rate']:.1f}")
        
        # Sonuçları görselleştir ve kaydet
        try:
            # Grafik oluştur
            plt.figure(figsize=(10, 6))
            plt.plot(list(results.keys()), [r['avg_time'] for r in results.values()], marker='o')
            plt.title('Batch Boyutuna Göre Ortalama İstek Süresi')
            plt.xlabel('Batch Boyutu (Kullanıcı Sayısı)')
            plt.ylabel('Ortalama Süre (saniye)')
            plt.grid(True)
            
            # Grafiği kaydet
            plt.savefig(os.path.join(run_dir, "batch", f"batch_performance_{timestamp}.png"))
            plt.savefig(os.path.join(run_dir, "batch", f"batch_performance_{timestamp}.pdf"))
            plt.close()
            
            for size, metrics in results.items():
                f.write(f"\nBatch Boyutu: {size}\n")
                f.write(f"Toplam Süre: {metrics['total_time']:.2f} saniye\n")
                f.write(f"Ortalama Süre: {metrics['avg_time']:.4f} saniye/istek\n")
                f.write(f"Başarı Oranı: %{metrics['success_rate']:.1f}\n")
            
            print(f"\nTest raporu kaydedildi: {report_path}")
            
        except Exception as e:
            print(f"Rapor oluşturma hatası: {str(e)}")
            import traceback
            traceback.print_exc()

def test_model_drift():
    """Model kayma testi yapar"""
    # Ana rapor dizinini al
    test_dir = os.path.join("test_reports", "recommendation_tests")
    current_run_file = os.path.join(test_dir, ".current_run")
    
    # Mevcut çalışma dizinini oku
    with open(current_run_file, "r") as f:
        run_dir = f.read().strip()
    
    # Timestamp'i run_dir'den çıkar
    timestamp = os.path.basename(run_dir).replace("run_", "")
    
    # Drift klasörüne rapor yaz
    report_path = os.path.join(run_dir, "drift", f"drift_test_report_{timestamp}.txt")
    os.makedirs(os.path.join(run_dir, "drift"), exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=== Model Drift Analizi Raporu ===\n")
        f.write(f"Test Tarihi: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Mevcut versiyonları al
        versions_data = safe_api_call('get', VERSIONS_URL)
        if not versions_data or 'versions' not in versions_data:
            print("Model versiyonları alınamadı.")
            return
        
        versions = versions_data['versions']
        if not versions:
            print("Hiç model versiyonu bulunamadı.")
            return
        
        # Versiyonları tarihlerine göre sırala
        try:
            versions_with_date = []
            for v in versions:
                if 'creation_date' in v and v['creation_date']:
                    try:
                        date = datetime.strptime(v['creation_date'], "%Y-%m-%d %H:%M:%S")
                        versions_with_date.append((v, date))
                    except:
                        pass
            
            versions_with_date.sort(key=lambda x: x[1])
            
            if not versions_with_date:
                print("Tarih bilgisi olan versiyon bulunamadı.")
                return
            
            # Metrikleri karşılaştır ve raporla
            metrics_trends = {}
            for v, date in versions_with_date:
                version_name = v.get('version_name', 'bilinmiyor')
                f.write(f"\n{date.strftime('%Y-%m-%d %H:%M')} - {version_name}\n")
                
                if 'metrics' in v and v['metrics']:
                    for metric_name, metric_value in v['metrics'].items():
                        f.write(f"  {metric_name}: {float(metric_value):.4f}\n")
                        
                        if metric_name not in metrics_trends:
                            metrics_trends[metric_name] = []
                        metrics_trends[metric_name].append((date, float(metric_value)))
        
        except Exception as e:
            print(f"Model drift analizi hatası: {str(e)}")
            import traceback
            traceback.print_exc()

def test_ab_comparison():
    """A/B testi yapar"""
    test_type = "ab_test"
    report_dir, timestamp = create_report_dir()
    report_path = os.path.join(report_dir, "ab_test_report.txt")
    
    # Mevcut versiyonları al
    versions_data = safe_api_call('get', VERSIONS_URL)
    if not versions_data or 'versions' not in versions_data:
        print("Model versiyonları alınamadı.")
        return
    
    versions = [v['version_name'] for v in versions_data['versions'] 
               if 'version_name' in v and v.get('model_type') == 'collaborative_filtering']
    
    if len(versions) < 2:
        print("A/B testi için en az iki model versiyonu gereklidir.")
        return
    
    # Kullanıcıdan versiyon seçmesini iste
    print("\nMevcut model versiyonları:")
    for i, version in enumerate(versions, 1):
        print(f"{i}. {version}")
    
    try:
        choice1 = int(input("\nA grubu için versiyon seçin (1-{}): ".format(len(versions))))
        choice2 = int(input("B grubu için versiyon seçin (1-{}): ".format(len(versions))))
        
        if choice1 < 1 or choice1 > len(versions) or choice2 < 1 or choice2 > len(versions):
            print("Geçersiz seçim.")
            return
        
        version_a = versions[choice1-1]
        version_b = versions[choice2-1]
        
        # Test parametrelerini al
        num_users = int(input("Test edilecek kullanıcı sayısı: "))
        num_simulations = int(input("Simülasyon sayısı: "))
        
    except ValueError:
        print("Geçersiz giriş. Sayısal değer girin.")
        return
    except Exception as e:
        print(f"Giriş hatası: {e}")
        return
    
    # Test verilerini yükle
    users_df, products_df = load_test_data()
    if users_df is None or products_df is None:
        return
    
    # Test için kullanıcıları seç
    test_users = random.sample(users_df['user_id'].tolist(), min(num_users, len(users_df)))
    
    # A/B testi sonuçları
    results = {
        version_a: {"conversion_rate": 0, "avg_items_viewed": 0, "response_times": []},
        version_b: {"conversion_rate": 0, "avg_items_viewed": 0, "response_times": []}
    }
    
    print(f"\nA/B testi başlatılıyor: {version_a} vs {version_b}")
    print(f"Test kullanıcı sayısı: {len(test_users)}")
    print(f"Simülasyon sayısı: {num_simulations}")
    
    for version in [version_a, version_b]:
        # Modeli yükle
        result = safe_api_call('post', f"{LOAD_VERSION_URL}/{version}")
        if not result:
            print(f"Model {version} yüklenemedi. A/B testi yapılamıyor.")
            return
        
        print(f"\nModel {version} test ediliyor...")
        
        # Simülasyonu çalıştır
        conversions = 0
        total_items_viewed = 0
        
        for _ in range(num_simulations):
            for user_id in tqdm(test_users, desc=f"Simülasyon {_+1}/{num_simulations}"):
                start_time = time.time()
                result = safe_api_call(
                    'post',
                    RECOMMEND_URL,
                    {
                        "user_id": user_id,
                        "num_recommendations": 5
                    },
                    print_response=False
                )
                end_time = time.time()
                
                if result and 'recommendations' in result:
                    results[version]["response_times"].append(end_time - start_time)
                    
                    # Kullanıcı davranışını simüle et
                    items_viewed = random.randint(0, len(result['recommendations']))
                    total_items_viewed += items_viewed
                    
                    # Dönüşüm simülasyonu (kullanıcı en az bir ürüne tıkladı mı)
                    if items_viewed > 0:
                        conversions += 1
        
        # Sonuçları hesapla
        total_sessions = len(test_users) * num_simulations
        results[version]["conversion_rate"] = (conversions / total_sessions) * 100
        results[version]["avg_items_viewed"] = total_items_viewed / total_sessions
    
    # Sonuçları karşılaştır
    print("\n=== A/B Test Sonuçları ===")
    
    print(f"\nModel A ({version_a}):")
    print(f"- Dönüşüm Oranı: %{results[version_a]['conversion_rate']:.2f}")
    print(f"- Ortalama İncelenen Ürün: {results[version_a]['avg_items_viewed']:.2f}")
    print(f"- Ortalama Yanıt Süresi: {np.mean(results[version_a]['response_times']):.4f} saniye")
    
    print(f"\nModel B ({version_b}):")
    print(f"- Dönüşüm Oranı: %{results[version_b]['conversion_rate']:.2f}")
    print(f"- Ortalama İncelenen Ürün: {results[version_b]['avg_items_viewed']:.2f}")
    print(f"- Ortalama Yanıt Süresi: {np.mean(results[version_b]['response_times']):.4f} saniye")
    
    # İstatistiksel anlamlılık kontrolü (basit)
    conversion_diff = results[version_b]['conversion_rate'] - results[version_a]['conversion_rate']
    items_diff = results[version_b]['avg_items_viewed'] - results[version_a]['avg_items_viewed']
    time_diff = np.mean(results[version_b]['response_times']) - np.mean(results[version_a]['response_times'])
    
    print("\nKarşılaştırma (B - A):")
    print(f"- Dönüşüm Oranı Farkı: %{conversion_diff:.2f}")
    print(f"- Ortalama İncelenen Ürün Farkı: {items_diff:.2f}")
    print(f"- Yanıt Süresi Farkı: {time_diff:.4f} saniye")
    
    winner = version_b if conversion_diff > 0 else version_a
    print(f"\nKazanan Model: {winner} (dönüşüm oranı temel alındı)")

def test_robustness():
    """Dayanıklılık testi yapar"""
    test_type = "robustness"
    report_dir, timestamp = create_report_dir()
    report_path = os.path.join(report_dir, "robustness_report.txt")
    
    # Geçersiz kullanıcı ID'si
    print("\n1. Geçersiz Kullanıcı ID'si Testi")
    invalid_user_id = -999
    result = safe_api_call(
        'post',
        RECOMMEND_URL,
        {
            "user_id": invalid_user_id,
            "num_recommendations": 5
        }
    )
    if result:
        print("✅ Model, geçersiz kullanıcı ID'si ile çalışıyor")
        print("Önerilen ürünler muhtemelen popülerlik bazlı")
    else:
        print("❌ Model, geçersiz kullanıcı ID'si ile çalışmıyor")
    
    # Geçersiz ürün ID'si
    print("\n2. Geçersiz Ürün ID'si Testi")
    invalid_item_id = -999
    result = safe_api_call(
        'post',
        RECOMMEND_URL,
        {
            "user_id": 1,
            "item_id": invalid_item_id,
            "num_recommendations": 5
        }
    )
    if result:
        print("✅ Model, geçersiz ürün ID'si ile çalışıyor")
    else:
        print("❌ Model, geçersiz ürün ID'si ile çalışmıyor")
    
    # Çok büyük öneri sayısı
    print("\n3. Büyük Öneri Sayısı Testi")
    large_num = 1000
    result = safe_api_call(
        'post',
        RECOMMEND_URL,
        {
            "user_id": 1,
            "num_recommendations": large_num
        }
    )
    if result and 'recommendations' in result:
        print(f"✅ Model, yüksek öneri sayısı ile çalışıyor (İstenen: {large_num}, Dönen: {len(result['recommendations'])})")
    else:
        print("❌ Model, yüksek öneri sayısı ile çalışmıyor")
    
    # Eksik alanlar
    print("\n4. Eksik Alanlar Testi")
    result = safe_api_call(
        'post',
        RECOMMEND_URL,
        {}  # Hiçbir alan belirtilmedi
    )
    if result:
        print("✅ Model, tüm alanlar eksik olduğunda çalışıyor")
    else:
        print("❌ Model, tüm alanlar eksik olduğunda çalışmıyor")
    
    # Stres testi
    print("\n5. Hızlı İstek Stres Testi")
    num_requests = 20
    success_count = 0
    
    start_time = time.time()
    for _ in range(num_requests):
        result = safe_api_call(
            'post',
            RECOMMEND_URL,
            {
                "user_id": random.randint(1, 100),
                "num_recommendations": 5
            },
            print_response=False
        )
        if result:
            success_count += 1
    end_time = time.time()
    
    print(f"Başarı Oranı: {success_count}/{num_requests} (%{(success_count/num_requests)*100:.0f})")
    print(f"Toplam Süre: {end_time - start_time:.2f} saniye")
    print(f"Ortalama İstek Süresi: {(end_time - start_time)/num_requests:.4f} saniye")

def parse_arguments():
    """Komut satırı argümanlarını işler"""
    parser = argparse.ArgumentParser(description='Öneri Sistemi Test Programı')
    
    parser.add_argument('--all', action='store_true',
                        help='Tüm testleri çalıştır')
    
    parser.add_argument('--health', action='store_true',
                        help='Model sağlık durumunu kontrol et')
    
    parser.add_argument('--versions', action='store_true',
                        help='Model versiyonlarını listele')
    
    parser.add_argument('--item', action='store_true',
                        help='Ürün bazlı öneri testini çalıştır')
    
    parser.add_argument('--user', action='store_true',
                        help='Kullanıcı bazlı öneri testini çalıştır')
    
    parser.add_argument('--compare', nargs=2, metavar=('VERSION1', 'VERSION2'),
                        help='İki model versiyonunu karşılaştır')
    
    parser.add_argument('--batch', action='store_true',
                        help='Toplu çıkarsama performans testini çalıştır')
    
    parser.add_argument('--drift', action='store_true',
                        help='Model drift analizini çalıştır')
    
    parser.add_argument('--ab', action='store_true',
                        help='A/B test benzetimini çalıştır')
    
    parser.add_argument('--robust', action='store_true',
                        help='Model dayanıklılık testini çalıştır')
    
    return parser.parse_args()

def main():
    """Ana test fonksiyonu"""
    start_time = time.time()
    
    # Test başlangıcını logla
    log_test_start(
        logger, 
        "Recommendation System Test", 
        {"test_type": "full_system_test"}
    )
    
    args = parse_arguments()
    
    logger.info("=== Öneri Sistemi Test Programı ===")
    
    # Test dizinini oluştur ve timestamp al
    run_dir, timestamp = create_report_dir()
    
    # Ana rapor dosyasını oluştur
    main_report_path = os.path.join(run_dir, f"test_summary_{timestamp}.txt")
    
    try:
        with open(main_report_path, 'w', encoding='utf-8') as f:
            f.write(f"=== Öneri Sistemi Test Raporu ===\n")
            f.write(f"Test Tarihi: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # API bağlantı kontrolü
            api_status = check_api_connection()
            f.write(f"API Durumu: {'Çalışıyor' if api_status else 'Çalışmıyor'}\n\n")
            
            if not api_status:
                f.write("⚠️ API servisi çalışmıyor veya erişilemiyor!\n")
                logger.warning("\n⚠️ API servisi çalışmıyor veya erişilemiyor! Testler başarısız olabilir.")
                try:
                    input("Testlere devam etmek için Enter'a basın...")
                except KeyboardInterrupt:
                    logger.info("\nTestler iptal edildi.")
                    log_test_end(
                        logger, 
                        "Recommendation System Test", 
                        success=False, 
                        summary="Tests cancelled by user"
                    )
                    return
            
            f.write("Çalıştırılan Testler:\n")
            
            test_results = {
                "health": False,
                "versions": False,
                "item_based": False,
                "user_based": False,
                "batch": False,
                "drift": False
            }
            
            # Testleri çalıştır ve sonuçları raporla
            if args.all or (not any([args.health, args.versions, args.item, args.user, 
                                  args.compare, args.batch, args.drift, args.ab, args.robust])):
                f.write("- Tüm temel testler\n")
                
                # Her test için run_dir'i kullan
                test_results["health"] = test_model_health()
                test_results["versions"] = test_versions()
                test_results["item_based"] = test_item_based_recommendation()
                test_results["user_based"] = test_user_based_recommendation()
            else:
                if args.health:
                    f.write("- Model sağlık kontrolü\n")
                    test_results["health"] = test_model_health()
                
                if args.versions:
                    f.write("- Model versiyon listesi\n")
                    test_results["versions"] = test_versions()
                
                if args.item:
                    f.write("- Ürün bazlı öneri testi\n")
                    test_results["item_based"] = test_item_based_recommendation()
                
                if args.user:
                    f.write("- Kullanıcı bazlı öneri testi\n")
                    test_results["user_based"] = test_user_based_recommendation()
            
            # Gelişmiş testler
            if args.compare:
                f.write(f"- Model karşılaştırma: {args.compare[0]} vs {args.compare[1]}\n")
                test_model_comparison(args.compare[0], args.compare[1])
            
            if args.batch:
                f.write("- Toplu çıkarsama performans testi\n")
                test_results["batch"] = test_batch_inference_time()
            
            if args.drift:
                f.write("- Model drift analizi\n")
                test_results["drift"] = test_model_drift()
            
            if args.ab:
                f.write("- A/B test benzetimi\n")
                test_ab_comparison()
            
            if args.robust:
                f.write("- Model dayanıklılık testi\n")
                test_robustness()
            
            # Test raporlarını listele
            f.write("\nOluşturulan Test Raporları:\n")
            try:
                for root, dirs, files in os.walk(run_dir):
                    for dir_name in dirs:
                        dir_path = os.path.join(root, dir_name)
                        dir_size = sum(os.path.getsize(os.path.join(dir_path, f)) for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f)))
                        f.write(f"- {dir_name} ({dir_size} bytes)\n")
                    for file_name in files:
                        if file_name != os.path.basename(main_report_path):  # Ana rapor dosyasını listeleme
                            file_path = os.path.join(root, file_name)
                            file_size = os.path.getsize(file_path)
                            f.write(f"- {file_name} ({file_size} bytes)\n")
            except Exception as e:
                error_msg = f"Error listing report directory: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                f.write(f"Rapor dizini listelenirken hata: {str(e)}\n")
            
            end_time = time.time()
            duration = end_time - start_time
            
            f.write(f"\nTest Bitiş Zamanı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Toplam Test Süresi: {duration:.2f} saniye")
        
        logger.info(f"Test summary created: {main_report_path}")
        
        # Test başarıyla tamamlandı
        log_test_end(
            logger,
            "Recommendation System Test",
            success=True,
            duration=time.time() - start_time,
            summary=f"All specified tests completed successfully"
        )
        
    except Exception as e:
        error_msg = f"Test execution failed: {str(e)}"
        logger.critical(error_msg)
        logger.critical(traceback.format_exc())
        
        # Test başarısızlığını logla
        log_test_end(
            logger,
            "Recommendation System Test",
            success=False,
            duration=time.time() - start_time,
            summary=f"Test failed with error: {str(e)}"
        )

if __name__ == "__main__":
    main() 