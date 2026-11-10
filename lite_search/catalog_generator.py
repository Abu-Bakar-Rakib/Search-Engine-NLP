import os
import sqlite3
import random
import json
import pandas as pd
from typing import List, Dict, Any, Tuple

def generate_ecommerce_catalog(n_products: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generates a deterministic 500-product retail e-commerce catalog with deliberate edge cases."""
    random.seed(seed)
    
    categories = [
        # Apparel
        ("t-shirt", "apparel/tops", "clothing", ["cotton", "polyester", "linen"], ["men", "women", "kids", "unisex"], ["S,M,L,XL", "M,L", "S,M", "XXL"], (350, 2500)),
        ("shirt", "apparel/formal", "clothing", ["cotton", "oxford cotton", "silk"], ["men", "women"], ["38,40,42,44", "S,M,L,XL"], (1200, 4500)),
        ("panjabi", "apparel/traditional", "ethnic", ["cotton", "silk", "semi-pure silk"], ["men", "boys"], ["38,40,42,44", "36,38,40"], (1500, 8500)),
        ("hoodie", "apparel/winter", "winterwear", ["fleece", "cotton blend"], ["men", "women", "unisex"], ["M,L,XL", "S,M,L"], (1400, 3800)),
        ("jacket", "apparel/outerwear", "winterwear", ["leather", "polyester", "nylon"], ["men", "women", "unisex"], ["M,L,XL", "L,XL"], (2500, 12000)),
        ("jeans", "apparel/bottoms", "denim", ["denim", "stretch denim"], ["men", "women", "kids"], ["30,32,34,36", "28,30,32"], (1200, 4800)),
        ("saree", "apparel/traditional", "ethnic", ["cotton", "jamdani", "georgette", "silk"], ["women"], [""], (1800, 15000)),
        ("kurti", "apparel/women", "ethnic", ["cotton", "viscose", "lawn"], ["women", "girls"], ["S,M,L,XL"], (850, 3200)),
        
        # Footwear
        ("running shoes", "footwear/sports", "shoes", ["mesh", "synthetic leather"], ["men", "women", "unisex"], ["7,8,9,10,11", "5,6,7,8", "6,7,8,9"], (2200, 9500)),
        ("sneakers", "footwear/casual", "shoes", ["canvas", "leather", "suede"], ["men", "women", "kids", "unisex"], ["7,8,9,10", "6,7,8", "4,5,6"], (1800, 7500)),
        ("leather shoes", "footwear/formal", "shoes", ["genuine leather", "pu leather"], ["men"], ["39,40,41,42,43,44"], (2800, 11000)),
        ("sandals", "footwear/casual", "footwear", ["leather", "rubber", "eva"], ["men", "women", "kids"], ["38,39,40,41,42"], (650, 2400)),
        
        # Electronics
        ("smartphone", "electronics/mobile", "devices", ["glass and aluminum", "plastic"], ["unisex"], [""], (8500, 65000)),
        ("laptop", "electronics/computing", "devices", ["aluminum", "carbon fiber"], ["unisex"], [""], (35000, 140000)),
        ("wireless headphones", "electronics/audio", "audio", ["matte plastic", "aluminum"], ["unisex"], [""], (1200, 18000)),
        ("smartwatch", "electronics/wearable", "wearable", ["silicone", "leather", "stainless steel"], ["men", "women", "unisex"], [""], (1800, 22000)),
        ("bluetooth speaker", "electronics/audio", "audio", ["fabric mesh", "plastic"], ["unisex"], [""], (1100, 12500)),
        ("tablet", "electronics/computing", "devices", ["metal", "glass"], ["unisex"], [""], (14000, 55000)),
        ("power bank", "electronics/accessories", "charging", ["abs plastic", "aluminum"], ["unisex"], [""], (800, 3500)),
        ("webcam", "electronics/computing", "accessories", ["plastic"], ["unisex"], [""], (1500, 6500)),

        # Accessories
        ("leather wallet", "accessories/wallets", "small leather goods", ["genuine cowhide leather", "synthetic leather"], ["men"], [""], (600, 3200)),
        ("leather belt", "accessories/belts", "leather goods", ["leather", "metal buckle"], ["men"], ["32,34,36,38,40"], (500, 2200)),
        ("sunglasses", "accessories/eyewear", "eyewear", ["polycarbonate", "metal frame"], ["men", "women", "unisex"], [""], (750, 6500)),
        ("backpack", "accessories/bags", "bags", ["waterproof nylon", "polyester"], ["unisex", "youth"], [""], (1100, 5200)),
        ("travel bag", "accessories/luggage", "luggage", ["oxford cloth", "hard polycarbonate"], ["unisex"], [""], (2500, 14000)),
        ("wristwatch", "accessories/watches", "watches", ["stainless steel", "leather strap"], ["men", "women", "unisex"], [""], (1200, 24000)),

        # Home & Kitchen
        ("stainless steel water bottle", "home/kitchen", "drinkware", ["food-grade stainless steel"], ["unisex"], ["500ml,750ml,1000ml"], (450, 1800)),
        ("ceramic coffee mug", "home/kitchen", "drinkware", ["ceramic", "porcelain"], ["unisex"], ["350ml"], (250, 850)),
        ("desk lamp", "home/lighting", "lighting", ["metal", "abs plastic"], ["unisex"], [""], (800, 2800)),
        ("scented candle", "home/decor", "decor", ["soy wax", "essential oils"], ["unisex"], [""], (350, 1600)),
        ("air purifier", "home/appliances", "appliances", ["hepa filter", "plastic"], ["unisex"], [""], (7500, 32000)),

        # Toys & Kids
        ("building blocks", "toys/construction", "educational", ["non-toxic abs plastic"], ["kids", "toddler"], [""], (650, 4500)),
        ("educational toy", "toys/learning", "educational", ["wood", "bpa-free plastic"], ["kids", "toddler"], [""], (450, 3200)),
        ("remote control car", "toys/vehicles", "action", ["plastic", "electronic parts"], ["kids", "boys"], [""], (1200, 6800)),
        ("soft teddy bear", "toys/plush", "plush", ["soft cotton", "plush fabric"], ["kids", "girls", "toddler"], ["30cm,50cm,80cm"], (400, 2400)),
        ("puzzle board", "toys/puzzles", "educational", ["cardboard", "wood"], ["kids"], [""], (300, 1500)),

        # Stationery
        ("notebook", "stationery/paper", "study", ["recycled paper", "leatherette cover"], ["unisex", "youth"], ["A5,B5"], (150, 850)),
        ("pen set", "stationery/writing", "writing", ["metal", "stainless steel"], ["unisex"], [""], (350, 3500)),
        ("scientific calculator", "stationery/electronics", "study", ["matte plastic"], ["unisex", "youth"], [""], (950, 3800)),
        ("study desk organizer", "stationery/organization", "office", ["metal mesh", "wood"], ["unisex"], [""], (450, 1800)),

        # Personal Care
        ("electric toothbrush", "personal_care/oral", "grooming", ["waterproof plastic"], ["unisex", "adult"], [""], (1200, 6500)),
        ("beard trimmer", "personal_care/grooming", "grooming", ["stainless steel blade", "plastic"], ["men"], [""], (1400, 5200)),
        ("perfume", "personal_care/fragrance", "fragrance", ["glass bottle", "fragrance oils"], ["men", "women", "unisex"], ["50ml,100ml"], (950, 9500)),
        ("hair dryer", "personal_care/hair", "styling", ["heat-resistant plastic"], ["women", "unisex"], [""], (1200, 4800))
    ]

    brands_by_cat = {
        "apparel": ["Aarong", "Yellow", "Cats Eye", "Ecstasy", "Richman", "Zara", "H&M", "Nike", "Adidas", "Lotto"],
        "footwear": ["Apex", "Bata", "Lotto", "Nike", "Adidas", "Puma", "Bay Emporium"],
        "electronics": ["Samsung", "Apple", "Xiaomi", "Sony", "Dell", "HP", "Lenovo", "Anker", "Logitech"],
        "accessories": ["Casio", "Titan", "Fossil", "Aarong", "Fastrack", "Wildcraft"],
        "home": ["Philips", "Prestige", "IKEA", "Milton", "Havells"],
        "toys": ["Lego", "Hot Wheels", "Fisher-Price", "Barbie", "Funskool"],
        "stationery": ["Matador", "Casio", "Parker", "Moleskine", "Deli"],
        "personal_care": ["Philips", "Braun", "Nivea", "L'Oreal", "The Body Shop"]
    }

    colors = ["black", "white", "blue", "navy blue", "red", "grey", "brown", "green", "pink", "beige", "gold", "silver"]
    
    products = []
    
    for i in range(1, n_products + 1):
        cat_spec = random.choice(categories)
        cat_leaf, cat_path, broad_group, materials, genders, sizes_opts, price_range = cat_spec
        
        # Determine broad brand group
        b_key = "apparel"
        if "footwear" in cat_path: b_key = "footwear"
        elif "electronics" in cat_path: b_key = "electronics"
        elif "accessories" in cat_path: b_key = "accessories"
        elif "home" in cat_path: b_key = "home"
        elif "toys" in cat_path: b_key = "toys"
        elif "stationery" in cat_path: b_key = "stationery"
        elif "personal_care" in cat_path: b_key = "personal_care"
        
        brand = random.choice(brands_by_cat.get(b_key, ["Generic"]))
        color = random.choice(colors)
        material = random.choice(materials)
        gender = random.choice(genders)
        size = random.choice(sizes_opts)
        
        # Audience logic
        if gender in ("kids", "boys", "girls"):
            audience = "kids"
        elif "toddler" in cat_leaf or "plush" in cat_path or "toddler" in gender:
            audience = "toddler"
        elif "study" in cat_path or cat_leaf in ("notebook", "scientific calculator", "backpack"):
            audience = random.choice(["youth", "adult"])
        else:
            audience = "adult"

        # Price generation
        base_price = round(random.uniform(price_range[0], price_range[1]), -1)
        
        # Edge cases:
        # 1. Price limit edge cases (round numbers under 1000, 3000, 5000)
        if random.random() < 0.2:
            target_cutoffs = [950, 999, 1000, 2800, 2999, 3000, 4800, 4999, 5000, 9500, 10000]
            base_price = float(random.choice(target_cutoffs))

        # 2. Out of stock & Inactive products
        stock_qty = 0 if (random.random() < 0.08) else random.randint(5, 250)
        is_active = 0 if (random.random() < 0.04) else 1

        # 3. Missing metadata (color, material, sizes missing occasionally)
        if random.random() < 0.10:
            color = ""
        if random.random() < 0.12:
            material = ""

        # Title formatting
        title_patterns = [
            f"{brand} {color.capitalize()} {cat_leaf.title()}",
            f"{color.capitalize()} {material.capitalize()} {cat_leaf.title()} by {brand}",
            f"{brand} {cat_leaf.title()} for {gender.capitalize()}",
            f"{brand} Premium {cat_leaf.title()}"
        ]
        title = random.choice(title_patterns).replace("  ", " ").strip()
        
        # Description formatting with intent hooks (gift, office, travel, study, coffee, workout)
        intent_hooks = []
        if cat_leaf in ("leather wallet", "wristwatch", "perfume", "ceramic coffee mug", "panjabi", "saree", "building blocks"):
            intent_hooks.append("Makes a thoughtful gift for birthdays, anniversaries, Eid, or special occasions.")
        if cat_leaf in ("wireless headphones", "tablet", "laptop", "webcam", "desk lamp"):
            intent_hooks.append("Ideal for online classes, Zoom meetings, and quiet home study.")
        if cat_leaf in ("travel bag", "backpack", "power bank", "stainless steel water bottle", "sunglasses"):
            intent_hooks.append("Perfect travel companion for weekend trips, tours, and outdoor journeys.")
        if cat_leaf in ("stainless steel water bottle", "ceramic coffee mug"):
            intent_hooks.append("Vacuum insulation designed to keep coffee hot or iced drinks cold for up to 12 hours on long drives.")
        if cat_leaf in ("running shoes", "smartwatch", "t-shirt"):
            intent_hooks.append("Engineered for daily workout, marathon running, gym fitness, and active lifestyle.")
        if cat_leaf in ("building blocks", "educational toy", "puzzle board"):
            intent_hooks.append("Fun educational gift for toddlers and kids that loves building, problem solving, and creative play.")

        hook_text = " ".join(intent_hooks)
        desc = (
            f"High quality {color} {material} {cat_leaf} crafted with excellence. "
            f"Designed for {gender} and suitable for {audience} use. "
            f"{hook_text} Authentic {brand} warranty included."
        ).strip()

        # Tags
        tags = [cat_leaf, brand.lower(), color.lower(), gender, audience]
        if "gift" in hook_text.lower(): tags.append("gift")
        if "travel" in hook_text.lower(): tags.append("travel")
        if "online class" in hook_text.lower(): tags.append("online_class")
        if "workout" in hook_text.lower(): tags.append("fitness")

        # Attributes
        attributes = {
            "origin": random.choice(["Bangladesh", "Vietnam", "China", "India", "Germany"]),
            "warranty_months": random.choice([0, 6, 12, 24]),
            "weight_grams": random.randint(100, 2500)
        }

        products.append({
            "id": f"P{i:04d}",
            "sku": f"SKU-{brand[:3].upper()}-{cat_leaf[:3].upper()}-{i:04d}",
            "title": title,
            "description": desc,
            "category_path": cat_path,
            "category": cat_leaf,
            "brand": brand,
            "color": color,
            "material": material,
            "gender": gender,
            "audience": audience,
            "sizes": size,
            "price": float(base_price),
            "currency": "BDT",
            "stock_qty": stock_qty,
            "rating": round(random.uniform(3.5, 4.9), 1),
            "review_count": random.randint(0, 480),
            "attributes": json.dumps(attributes),
            "tags": ",".join(set(tags)),
            "is_active": is_active,
            "updated_at": f"2026-03-{random.randint(1, 28):02d} 10:00:00"
        })

    return pd.DataFrame(products)


def generate_industrial_catalog(n_products: int = 150, seed: int = 99) -> pd.DataFrame:
    """Generates a second independent synthetic catalog (Industrial & Lab Gear) to test domain portability."""
    random.seed(seed)
    
    categories = [
        ("rotary hammer drill", "power_tools/drilling", "Bosch Professional", "blue", "reinforced steel", "trade_professional", (85.0, 350.0)),
        ("angle grinder", "power_tools/cutting", "Makita", "teal", "aluminum housing", "trade_professional", (65.0, 220.0)),
        ("circular saw", "power_tools/woodworking", "DeWalt", "yellow", "carbide steel", "trade_professional", (95.0, 280.0)),
        ("digital multimeter", "measurement/electrical", "Fluke", "yellow", "high-impact plastic", "trade_professional", (120.0, 580.0)),
        ("laser distance meter", "measurement/optics", "Bosch Professional", "blue", "rubberized polymer", "trade_professional", (70.0, 240.0)),
        ("infrared thermometer", "measurement/thermal", "Fluke", "yellow", "durable abs", "trade_professional", (90.0, 310.0)),
        ("digital precision balance", "laboratory/measuring", "Mettler Toledo", "silver", "cast aluminum", "scientist", (250.0, 1800.0)),
        ("magnetic stirrer", "laboratory/mixing", "Corning", "white", "ceramic plate", "scientist", (150.0, 650.0)),
        ("microscope trinocular", "laboratory/optics", "Olympus", "white", "optical glass", "scientist", (450.0, 3200.0)),
        ("centrifuge machine", "laboratory/separation", "Eppendorf", "grey", "stainless rotor", "scientist", (600.0, 4200.0)),
        ("respiratory face mask", "safety/ppe", "3M Safety", "white", "electrostatic meltblown", "industrial", (25.0, 85.0)),
        ("welding helmet auto-darkening", "safety/welding", "Lincoln Electric", "black", "polycarbonate", "industrial", (75.0, 320.0)),
        ("cut-resistant gloves", "safety/hand_protection", "Ansell", "grey", "kevlar nitrile", "industrial", (15.0, 55.0)),
        ("safety harness fall protection", "safety/fall_arrest", "Miller Fall Protection", "yellow", "high-strength polyester", "industrial", (60.0, 210.0)),
        ("torque wrench", "hand_tools/fastening", "Stanley FatMax", "chrome", "chrome vanadium", "trade_professional", (45.0, 190.0)),
        ("hydraulic crimping tool", "hand_tools/crimping", "Klauke", "black", "alloy steel", "trade_professional", (80.0, 350.0))
    ]

    items = []
    for i in range(1, n_products + 1):
        cat_leaf, path, brand, color, mat, sector, price_range = random.choice(categories)
        cost = round(random.uniform(price_range[0], price_range[1]), 2)
        
        desc = (
            f"Heavy duty industrial grade {cat_leaf} designed for high precision and rugged site operations. "
            f"Built with {mat} for {sector} safety compliance. Certified by international standards."
        )
        
        items.append({
            "item_code": f"IND-{i:04d}",
            "model_no": f"MOD-{brand[:3].upper()}-{i:04d}",
            "item_name": f"{brand} {cat_leaf.title()} ({color.capitalize()})",
            "specification_summary": desc,
            "division_path": path,
            "sub_category": cat_leaf,
            "manufacturer": brand,
            "finish_color": color,
            "build_material": mat,
            "operator_type": "unisex",
            "target_sector": sector,
            "gauge_size": "Standard",
            "unit_cost": cost,
            "currency": "USD",
            "inventory_count": random.randint(5, 50),
            "audit_score": round(random.uniform(4.0, 5.0), 1),
            "cert_count": random.randint(1, 10),
            "tech_specs": json.dumps({"power_rating": "750W", "voltage": "220V"}),
            "keywords": f"{cat_leaf},{brand.lower()},heavy_duty,precision",
            "is_available": 1,
            "last_inspected": "2026-03-15"
        })
        
    return pd.DataFrame(items)


def build_and_save_demo_catalogs(data_dir: str = "data"):
    """Creates SQLite databases and CSV exports for both catalogs deterministically."""
    os.makedirs(data_dir, exist_ok=True)
    
    # Catalog 1: E-Commerce
    df1 = generate_ecommerce_catalog(500, seed=42)
    db1_path = os.path.join(data_dir, "synthetic_catalog_1.sqlite")
    csv1_path = os.path.join(data_dir, "synthetic_catalog_1.csv")
    
    # Save to SQLite
    conn1 = sqlite3.connect(db1_path)
    df1.to_sql("products", conn1, if_exists="replace", index=False)
    conn1.close()
    
    # Save to CSV
    df1.to_csv(csv1_path, index=False)
    print(f"Generated E-Commerce Catalog 1: {len(df1)} products -> {db1_path} & {csv1_path}")
    
    # Catalog 2: Industrial
    df2 = generate_industrial_catalog(150, seed=99)
    db2_path = os.path.join(data_dir, "synthetic_catalog_2.sqlite")
    csv2_path = os.path.join(data_dir, "synthetic_catalog_2.csv")
    
    conn2 = sqlite3.connect(db2_path)
    df2.to_sql("equipment", conn2, if_exists="replace", index=False)
    conn2.close()
    
    df2.to_csv(csv2_path, index=False)
    print(f"Generated Industrial Catalog 2: {len(df2)} products -> {db2_path} & {csv2_path}")


if __name__ == "__main__":
    build_and_save_demo_catalogs()
