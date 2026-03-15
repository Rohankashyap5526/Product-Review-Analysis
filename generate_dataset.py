"""
generate_dataset.py
-------------------
Generates a synthetic e-commerce dataset for the project.
Run this script first to create data/ecommerce_data.csv
"""

import pandas as pd
import numpy as np
import random
import os
from datetime import datetime, timedelta

np.random.seed(42)
random.seed(42)

NUM_CUSTOMERS = 2000
NUM_RECORDS   = 15000

# ── Lookup tables ────────────────────────────────────────────────────────────
CATEGORIES   = ["Electronics","Clothing","Home & Kitchen","Books",
                 "Sports","Beauty","Toys","Automotive","Grocery","Jewelry"]
PRODUCTS     = {
    "Electronics":   ["Laptop","Smartphone","Headphones","Tablet","Camera"],
    "Clothing":      ["T-Shirt","Jeans","Jacket","Dress","Shoes"],
    "Home & Kitchen":["Blender","Coffee Maker","Vacuum","Cookware","Lamp"],
    "Books":         ["Fiction Novel","Science Book","Cookbook","Biography","Self-Help"],
    "Sports":        ["Running Shoes","Yoga Mat","Dumbbells","Bicycle","Tennis Racket"],
    "Beauty":        ["Face Cream","Shampoo","Lipstick","Perfume","Sunscreen"],
    "Toys":          ["LEGO Set","Board Game","Action Figure","Doll","Puzzle"],
    "Automotive":    ["Car Cover","Floor Mats","Air Freshener","Dash Cam","Tire Inflator"],
    "Grocery":       ["Organic Coffee","Protein Bar","Olive Oil","Nuts Mix","Green Tea"],
    "Jewelry":       ["Gold Ring","Silver Necklace","Diamond Earrings","Bracelet","Watch"],
}
GENDERS     = ["Male","Female","Other"]
LOCATIONS   = ["New York","Los Angeles","Chicago","Houston","Phoenix",
               "Philadelphia","San Antonio","San Diego","Dallas","San Jose"]
PAYMENT     = ["Credit Card","Debit Card","PayPal","Apple Pay","Google Pay"]

POSITIVE_REVIEWS = [
    "Absolutely love this product! Exceeded all my expectations.",
    "Great quality and fast shipping. Will definitely buy again.",
    "This is exactly what I needed. Highly recommend to everyone.",
    "Amazing product, works perfectly! Very satisfied with purchase.",
    "Excellent value for money. Works as described and ships quickly.",
    "Outstanding quality! The product is even better than the pictures.",
    "Very happy with this purchase. Great customer service too.",
    "Five stars all the way! Perfect product, fast delivery.",
    "Fantastic! This product changed my life. So glad I bought it.",
    "Superb quality. Will recommend to all my friends and family.",
]
NEUTRAL_REVIEWS = [
    "Product is okay, does the job but nothing special.",
    "Average quality. Meets basic requirements but could be better.",
    "It works as described. Nothing extraordinary about it.",
    "Decent product for the price. Some minor issues though.",
    "Not bad, not great. Pretty much what I expected for the price.",
    "It's fine I guess. Does what it says but looks cheap.",
    "Mediocre quality. Was expecting more based on the description.",
    "Acceptable product. Some assembly required and instructions unclear.",
    "It's alright. Nothing to write home about, just average.",
    "Works most of the time. Quality seems okay for everyday use.",
]
NEGATIVE_REVIEWS = [
    "Terrible product! Broke after just two days of use. Very disappointed.",
    "Complete waste of money. Does not work as advertised at all.",
    "Very poor quality. I would not recommend this product to anyone.",
    "Worst purchase I have made. Product arrived damaged and unusable.",
    "Extremely disappointed. Nothing like the pictures shown on website.",
    "Horrible experience! Customer service was useless and product failed.",
    "Total junk. Stopped working after one week. Avoid this product.",
    "Do not buy this! It is a scam. Product quality is absolutely terrible.",
    "Disgusting quality. Returning immediately. This is completely unacceptable.",
    "Waste of time and money. Product malfunctioned right out of the box.",
]

# ── Generate customers ────────────────────────────────────────────────────────
customer_ids  = [f"CUST{str(i).zfill(5)}" for i in range(1, NUM_CUSTOMERS + 1)]
customer_ages = np.random.randint(18, 70, NUM_CUSTOMERS)
customer_genders   = np.random.choice(GENDERS,   NUM_CUSTOMERS)
customer_locations = np.random.choice(LOCATIONS, NUM_CUSTOMERS)
customer_loyalty   = np.random.choice(["Bronze","Silver","Gold","Platinum"], NUM_CUSTOMERS,
                                       p=[0.4, 0.3, 0.2, 0.1])

customer_df = pd.DataFrame({
    "customer_id":      customer_ids,
    "age":              customer_ages,
    "gender":           customer_genders,
    "location":         customer_locations,
    "loyalty_tier":     customer_loyalty,
})

# ── Generate transactions ─────────────────────────────────────────────────────
records = []
start_date = datetime(2022, 1, 1)
end_date   = datetime(2024, 12, 31)

for _ in range(NUM_RECORDS):
    cust_idx  = np.random.randint(0, NUM_CUSTOMERS)
    cust_id   = customer_ids[cust_idx]
    category  = np.random.choice(CATEGORIES)
    product   = np.random.choice(PRODUCTS[category])

    base_price = {
        "Electronics": 300, "Clothing": 60, "Home & Kitchen": 80,
        "Books": 20,        "Sports": 100,  "Beauty": 35,
        "Toys": 45,         "Automotive": 55,"Grocery": 25, "Jewelry": 200,
    }[category]

    price    = round(base_price * np.random.uniform(0.5, 2.5), 2)
    quantity = np.random.randint(1, 6)
    rating   = np.random.choice([1,2,3,4,5], p=[0.05,0.10,0.20,0.35,0.30])

    # Sentiment skewed by rating
    if rating >= 4:
        review = random.choice(POSITIVE_REVIEWS)
        sentiment = "Positive"
    elif rating == 3:
        review = random.choice(NEUTRAL_REVIEWS)
        sentiment = "Neutral"
    else:
        review = random.choice(NEGATIVE_REVIEWS)
        sentiment = "Negative"

    # Inject noise: ~5 % bad/missing data
    if np.random.random() < 0.03:
        review = None
    if np.random.random() < 0.02:
        rating = None

    delta   = end_date - start_date
    rand_d  = random.randint(0, delta.days)
    ts      = start_date + timedelta(days=rand_d)

    # Purchase probability label (1 = purchased again within 30 days, naive proxy)
    will_purchase = 1 if (rating is not None and rating >= 4
                          and np.random.random() < 0.65) else int(np.random.random() < 0.25)

    records.append({
        "transaction_id":  f"TXN{str(len(records)+1).zfill(6)}",
        "customer_id":     cust_id,
        "age":             customer_df.loc[cust_idx,"age"],
        "gender":          customer_df.loc[cust_idx,"gender"],
        "location":        customer_df.loc[cust_idx,"location"],
        "loyalty_tier":    customer_df.loc[cust_idx,"loyalty_tier"],
        "category":        category,
        "product":         product,
        "price":           price,
        "quantity":        quantity,
        "total_amount":    round(price * quantity, 2),
        "rating":          rating,
        "review":          review,
        "sentiment":       sentiment,
        "payment_method":  np.random.choice(PAYMENT),
        "timestamp":       ts.strftime("%Y-%m-%d %H:%M:%S"),
        "will_purchase_again": will_purchase,
    })

df = pd.DataFrame(records)

# Introduce ~3 % duplicates deliberately so cleaning module can remove them
dup_rows = df.sample(frac=0.03, random_state=1)
df = pd.concat([df, dup_rows], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

os.makedirs("data", exist_ok=True)
df.to_csv("data/ecommerce_data.csv", index=False)
print(f"Dataset saved → data/ecommerce_data.csv  ({len(df):,} rows)")
