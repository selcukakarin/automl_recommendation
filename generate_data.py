import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta

# Faker'ı başlat
fake = Faker('tr_TR')

def generate_users(n_users=10000):
    """Kullanıcı verilerini oluşturur"""
    users = []
    for user_id in range(1, n_users + 1):
        users.append({
            'user_id': user_id,
            'user_name': fake.name(),
            'user_age': random.randint(18, 70),
            'user_gender': random.choice(['M', 'F']),
            'user_location': fake.city(),
            'registration_date': fake.date_between(start_date='-2y', end_date='today')
        })
    return pd.DataFrame(users)

def generate_products(n_products=1000):
    """Ürün verilerini oluşturur"""
    categories = [
        'Elektronik', 'Giyim', 'Kitap', 'Ev & Yaşam', 'Spor', 'Kozmetik',
        'Oyuncak', 'Mücevher', 'Otomotiv', 'Bahçe', 'Mobilya', 'Gıda'
    ]
    
    products = []
    for item_id in range(1, n_products + 1):
        category = random.choice(categories)
        base_price = random.uniform(10, 1000)
        
        products.append({
            'item_id': item_id,
            'item_name': f"{fake.word().title()} {fake.word().title()}",
            'item_category': category,
            'item_subcategory': fake.word().title(),
            'item_price': round(base_price, 2),
            'item_rating_avg': round(random.uniform(3.5, 4.8), 2),
            'item_description': fake.text(max_nb_chars=200),
            'stock_quantity': random.randint(0, 1000),
            'launch_date': fake.date_between(start_date='-1y', end_date='today')
        })
    return pd.DataFrame(products)

def generate_interactions(users_df, products_df, n_interactions=100000):
    """Kullanıcı-ürün etkileşimlerini oluşturur"""
    interactions = []
    user_ids = users_df['user_id'].tolist()
    item_ids = products_df['item_id'].tolist()
    
    # Benzersiz kullanıcı-ürün çiftlerini takip etmek için set kullan
    used_pairs = set()
    
    # Her kullanıcı için en az bir etkileşim olsun
    for user_id in user_ids:
        n_user_interactions = random.randint(1, 20)  # Her kullanıcı 1-20 ürünle etkileşimde bulunsun
        
        # Bu kullanıcı için rastgele ürünler seç
        available_items = random.sample(item_ids, min(n_user_interactions, len(item_ids)))
        
        for item_id in available_items:
            pair = (user_id, item_id)
            if pair not in used_pairs:
                rating = max(1, min(5, random.gauss(4, 0.5)))  # Normal dağılımlı puanlar
                
                interactions.append({
                    'user_id': user_id,
                    'item_id': item_id,
                    'rating': round(rating, 1),
                    'timestamp': fake.date_time_between(start_date='-1y', end_date='now').strftime('%Y-%m-%d %H:%M:%S')
                })
                used_pairs.add(pair)
    
    # Kalan etkileşimleri rastgele oluştur
    attempts = 0
    max_attempts = n_interactions * 2  # Sonsuz döngüyü önlemek için maksimum deneme sayısı
    
    while len(interactions) < n_interactions and attempts < max_attempts:
        user_id = random.choice(user_ids)
        item_id = random.choice(item_ids)
        pair = (user_id, item_id)
        
        if pair not in used_pairs:
            rating = max(1, min(5, random.gauss(4, 0.5)))
            
            interactions.append({
                'user_id': user_id,
                'item_id': item_id,
                'rating': round(rating, 1),
                'timestamp': fake.date_time_between(start_date='-1y', end_date='now').strftime('%Y-%m-%d %H:%M:%S')
            })
            used_pairs.add(pair)
        
        attempts += 1
    
    return pd.DataFrame(interactions)

def main():
    print("Veri seti oluşturuluyor...")
    
    # Parametreler
    n_users = 10000
    n_products = 1000
    n_interactions = 100000
    
    # Verileri oluştur
    print("Kullanıcı verileri oluşturuluyor...")
    users_df = generate_users(n_users)
    
    print("Ürün verileri oluşturuluyor...")
    products_df = generate_products(n_products)
    
    print("Etkileşim verileri oluşturuluyor...")
    interactions_df = generate_interactions(users_df, products_df, n_interactions)
    
    # Verileri kaydet
    print("Veriler CSV dosyalarına kaydediliyor...")
    users_df.to_csv('data/users.csv', index=False)
    products_df.to_csv('data/products.csv', index=False)
    interactions_df.to_csv('data/interactions.csv', index=False)
    
    print("\nVeri seti istatistikleri:")
    print(f"Kullanıcı sayısı: {len(users_df)}")
    print(f"Ürün sayısı: {len(products_df)}")
    print(f"Etkileşim sayısı: {len(interactions_df)}")
    print(f"Ortalama kullanıcı başına etkileşim: {len(interactions_df) / len(users_df):.2f}")
    print(f"Ortalama ürün başına etkileşim: {len(interactions_df) / len(products_df):.2f}")

if __name__ == "__main__":
    # Data klasörünü oluştur
    import os
    os.makedirs('data', exist_ok=True)
    main() 