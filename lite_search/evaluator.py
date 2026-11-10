import time
import math
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Callable, Optional, Tuple
from dataclasses import dataclass

@dataclass
class LabeledQuery:
    query: str
    query_type: str  # english, bangla, banglish, typo, recipient, gift, usecase, budget, vague, brand, audience, no_match
    predicate: Callable[[Any], bool]
    expected_category: Optional[str] = None
    expected_brand: Optional[str] = None
    expected_audience: Optional[str] = None
    expected_gender: Optional[str] = None
    max_price: Optional[float] = None


def generate_120_labeled_queries() -> List[LabeledQuery]:
    """Generates 120+ structured labeled evaluation queries with ground truth predicates."""
    tests = [
        # --- Group 1: English Explicit Queries (15 queries) ---
        LabeledQuery("nike running shoes under 5000", "english",
                     lambda p: "running shoes" in p.category_leaf and p.brand.lower() == "nike" and p.price <= 5000,
                     expected_category="running shoes", expected_brand="Nike", max_price=5000),
        LabeledQuery("black cotton t-shirt for men", "english",
                     lambda p: "t-shirt" in p.category_leaf and p.gender in ("men", "unisex") and "black" in p.color.lower(),
                     expected_category="t-shirt", expected_gender="men"),
        LabeledQuery("blue shirt size L for men below 3000", "english",
                     lambda p: "shirt" in p.category_leaf and "blue" in p.color.lower() and p.gender in ("men", "unisex") and p.price <= 3000,
                     expected_category="shirt", expected_gender="men", max_price=3000),
        LabeledQuery("leather wallet for men under 2000", "english",
                     lambda p: "wallet" in p.category_leaf and p.gender in ("men", "unisex") and p.price <= 2000,
                     expected_category="leather wallet", max_price=2000),
        LabeledQuery("stainless steel water bottle under 1000", "english",
                     lambda p: "water bottle" in p.category_leaf and p.price <= 1000,
                     expected_category="stainless steel water bottle", max_price=1000),
        LabeledQuery("ceramic coffee mug under 500", "english",
                     lambda p: "mug" in p.category_leaf and p.price <= 500,
                     expected_category="ceramic coffee mug", max_price=500),
        LabeledQuery("casio wristwatch for men", "english",
                     lambda p: "watch" in p.category_leaf and p.brand.lower() == "casio",
                     expected_category="wristwatch", expected_brand="Casio"),
        LabeledQuery("wireless headphones under 4000", "english",
                     lambda p: "headphones" in p.category_leaf and p.price <= 4000,
                     expected_category="wireless headphones", max_price=4000),
        LabeledQuery("samsung smartphone", "english",
                     lambda p: "smartphone" in p.category_leaf and p.brand.lower() == "samsung",
                     expected_category="smartphone", expected_brand="Samsung"),
        LabeledQuery("dell laptop", "english",
                     lambda p: "laptop" in p.category_leaf and p.brand.lower() == "dell",
                     expected_category="laptop", expected_brand="Dell"),
        LabeledQuery("electric toothbrush under 3000", "english",
                     lambda p: "toothbrush" in p.category_leaf and p.price <= 3000,
                     expected_category="electric toothbrush", max_price=3000),
        LabeledQuery("philips beard trimmer for men", "english",
                     lambda p: "trimmer" in p.category_leaf and p.brand.lower() == "philips",
                     expected_category="beard trimmer", expected_brand="Philips"),
        LabeledQuery("educational toy for kids under 2000", "english",
                     lambda p: p.audience in ("kids", "toddler") and p.price <= 2000,
                     expected_audience="kids", max_price=2000),
        LabeledQuery("building blocks for toddler", "english",
                     lambda p: "building blocks" in p.category_leaf or p.audience in ("kids", "toddler"),
                     expected_category="building blocks", expected_audience="toddler"),
        LabeledQuery("scientific calculator for students under 2000", "english",
                     lambda p: "calculator" in p.category_leaf and p.price <= 2000,
                     expected_category="scientific calculator", max_price=2000),

        # --- Group 2: Bangla Queries (15 queries) ---
        LabeledQuery("বাবার জন্য গিফট ৩০০০ টাকার মধ্যে", "bangla",
                     lambda p: p.gender in ("men", "unisex") and p.audience == "adult" and p.price <= 3000 and p.category_leaf in ("leather wallet", "leather belt", "panjabi", "wristwatch", "ceramic coffee mug"),
                     expected_gender="men", max_price=3000),
        LabeledQuery("বাচ্চাদের খেলনা ১৫০০ টাকার নিচে", "bangla",
                     lambda p: p.audience in ("kids", "toddler") and p.price <= 1500,
                     expected_audience="kids", max_price=1500),
        LabeledQuery("ছেলেদের জুতা ২৫০০ টাকার মধ্যে", "bangla",
                     lambda p: p.category_leaf in ("running shoes", "sneakers", "leather shoes", "sandals") and p.price <= 2500,
                     expected_gender="men", max_price=2500),
        LabeledQuery("মায়ের জন্য শাড়ি ৫০০০ টাকার নিচে", "bangla",
                     lambda p: p.category_leaf == "saree" and p.price <= 5000,
                     expected_category="saree", max_price=5000),
        LabeledQuery("ছোটদের পুতুল ১০০০ টাকার মধ্যে", "bangla",
                     lambda p: p.audience in ("kids", "toddler") and p.price <= 1000,
                     expected_audience="kids", max_price=1000),
        LabeledQuery("পুরুষদের পাঞ্জাবি ২০০০ টাকার মধ্যে", "bangla",
                     lambda p: "panjabi" in p.category_leaf and p.price <= 2000,
                     expected_category="panjabi", max_price=2000),
        LabeledQuery("পানির বোতল ৫০০ টাকার মধ্যে", "bangla",
                     lambda p: "bottle" in p.category_leaf and p.price <= 500,
                     expected_category="stainless steel water bottle", max_price=500),
        LabeledQuery("কফি খাওয়ার মগ ৫০০ টাকার নিচে", "bangla",
                     lambda p: "mug" in p.category_leaf and p.price <= 500,
                     expected_category="ceramic coffee mug", max_price=500),
        LabeledQuery("স্মার্ট ঘড়ি ৩০০০ টাকার মধ্যে", "bangla",
                     lambda p: "smartwatch" in p.category_leaf or "watch" in p.category_leaf and p.price <= 3000,
                     max_price=3000),
        LabeledQuery("পড়ার টেবিলের ল্যাম্প", "bangla",
                     lambda p: "desk lamp" in p.category_leaf or "lamp" in p.title.lower(),
                     expected_category="desk lamp"),
        LabeledQuery("ছেলেদের টি শার্ট ৫০০ টাকার মধ্যে", "bangla",
                     lambda p: "t-shirt" in p.category_leaf and p.price <= 500,
                     expected_category="t-shirt", max_price=500),
        LabeledQuery("মেয়েদের পারফিউম ২০০০ টাকার নিচে", "bangla",
                     lambda p: "perfume" in p.category_leaf and p.gender in ("women", "unisex") and p.price <= 2000,
                     expected_category="perfume", max_price=2000),
        LabeledQuery("ব্লুটুথ স্পিকার ২০০০ টাকার মধ্যে", "bangla",
                     lambda p: "speaker" in p.category_leaf and p.price <= 2000,
                     expected_category="bluetooth speaker", max_price=2000),
        LabeledQuery("ছাত্রদের ব্যাকপ্যাক ১৫০০ টাকার নিচে", "bangla",
                     lambda p: "backpack" in p.category_leaf and p.price <= 1500,
                     expected_category="backpack", max_price=1500),
        LabeledQuery("দাড়ি কাটার ট্রিমার ২০০০ টাকার মধ্যে", "bangla",
                     lambda p: "trimmer" in p.category_leaf and p.price <= 2000,
                     expected_category="beard trimmer", max_price=2000),

        # --- Group 3: Banglish Queries (15 queries) ---
        LabeledQuery("babar jonno gift under 3000", "banglish",
                     lambda p: p.gender in ("men", "unisex") and p.audience == "adult" and p.price <= 3000 and p.category_leaf in ("leather wallet", "leather belt", "panjabi", "wristwatch", "ceramic coffee mug"),
                     expected_gender="men", max_price=3000),
        LabeledQuery("baccha der khelna 1500 er niche", "banglish",
                     lambda p: p.audience in ("kids", "toddler") and p.price <= 1500,
                     expected_audience="kids", max_price=1500),
        LabeledQuery("panjabi 2000 er niche", "banglish",
                     lambda p: "panjabi" in p.category_leaf and p.price <= 2000,
                     expected_category="panjabi", max_price=2000),
        LabeledQuery("cheleder juta 3000 er moddhe", "banglish",
                     lambda p: p.category_leaf in ("running shoes", "sneakers", "leather shoes", "sandals") and p.price <= 3000,
                     expected_gender="men", max_price=3000),
        LabeledQuery("kom dam e valo phone", "banglish",
                     lambda p: "smartphone" in p.category_leaf or "mobile" in p.description.lower(),
                     expected_category="smartphone"),
        LabeledQuery("mayer jonno saree 4000 er vitor", "banglish",
                     lambda p: "saree" in p.category_leaf and p.price <= 4000,
                     expected_category="saree", max_price=4000),
        LabeledQuery("choto bacchar soft teddy bear under 1000", "banglish",
                     lambda p: "teddy bear" in p.category_leaf and p.price <= 1000,
                     expected_category="soft teddy bear", max_price=1000),
        LabeledQuery("ghori for men under 2500", "banglish",
                     lambda p: "watch" in p.category_leaf and p.price <= 2500,
                     expected_category="wristwatch", max_price=2500),
        LabeledQuery("choshma under 1500", "banglish",
                     lambda p: "sunglasses" in p.category_leaf and p.price <= 1500,
                     expected_category="sunglasses", max_price=1500),
        LabeledQuery("khata notebook 500 er niche", "banglish",
                     lambda p: "notebook" in p.category_leaf and p.price <= 500,
                     expected_category="notebook", max_price=500),
        LabeledQuery("soundbox bluetooth under 3000", "banglish",
                     lambda p: "speaker" in p.category_leaf and p.price <= 3000,
                     expected_category="bluetooth speaker", max_price=3000),
        LabeledQuery("shart blue color men below 2000", "banglish",
                     lambda p: "shirt" in p.category_leaf and p.price <= 2000,
                     expected_category="shirt", max_price=2000),
        LabeledQuery("porashonar jonno desk lamp", "banglish",
                     lambda p: "lamp" in p.category_leaf or "desk lamp" in p.category_leaf,
                     expected_category="desk lamp"),
        LabeledQuery("pani khabar botol under 800", "banglish",
                     lambda p: "bottle" in p.category_leaf and p.price <= 800,
                     expected_category="stainless steel water bottle", max_price=800),
        LabeledQuery("manibag leather under 1500", "banglish",
                     lambda p: "wallet" in p.category_leaf and p.price <= 1500,
                     expected_category="leather wallet", max_price=1500),

        # --- Group 4: Typos and Misspellings (10 queries) ---
        LabeledQuery("adydas running shoos", "typo",
                     lambda p: "running shoes" in p.category_leaf and p.brand.lower() == "adidas",
                     expected_category="running shoes", expected_brand="Adidas"),
        LabeledQuery("samsang mobail phon", "typo",
                     lambda p: "smartphone" in p.category_leaf and p.brand.lower() == "samsung",
                     expected_category="smartphone", expected_brand="Samsung"),
        LabeledQuery("nik snikers black", "typo",
                     lambda p: "sneakers" in p.category_leaf and p.brand.lower() == "nike",
                     expected_category="sneakers", expected_brand="Nike"),
        LabeledQuery("kashio hand wotch", "typo",
                     lambda p: "watch" in p.category_leaf and p.brand.lower() == "casio",
                     expected_category="wristwatch", expected_brand="Casio"),
        LabeledQuery("lethar wallat for man", "typo",
                     lambda p: "wallet" in p.category_leaf and p.gender in ("men", "unisex"),
                     expected_category="leather wallet"),
        LabeledQuery("blutooth hedphone wierless", "typo",
                     lambda p: "headphones" in p.category_leaf,
                     expected_category="wireless headphones"),
        LabeledQuery("toddlar bilding bloks", "typo",
                     lambda p: "building blocks" in p.category_leaf or p.audience in ("kids", "toddler"),
                     expected_category="building blocks"),
        LabeledQuery("airpode wirless under 3000", "typo",
                     lambda p: "headphones" in p.category_leaf and p.price <= 3000,
                     expected_category="wireless headphones", max_price=3000),
        LabeledQuery("cotton punjaby under 3000", "typo",
                     lambda p: "panjabi" in p.category_leaf and p.price <= 3000,
                     expected_category="panjabi", max_price=3000),
        LabeledQuery("sunglas uv protekt", "typo",
                     lambda p: "sunglasses" in p.category_leaf,
                     expected_category="sunglasses"),

        # --- Group 5: Recipient and Persona Queries (10 queries) ---
        LabeledQuery("gift for my father under 3000", "recipient",
                     lambda p: p.gender in ("men", "unisex") and p.audience == "adult" and p.price <= 3000,
                     expected_gender="men", max_price=3000),
        LabeledQuery("gift for mom under 4000", "recipient",
                     lambda p: p.gender in ("women", "unisex") and p.audience == "adult" and p.price <= 4000,
                     expected_gender="women", max_price=4000),
        LabeledQuery("present for my husband under 5000", "recipient",
                     lambda p: p.gender in ("men", "unisex") and p.price <= 5000,
                     expected_gender="men", max_price=5000),
        LabeledQuery("gift for my wife under 6000", "recipient",
                     lambda p: p.gender in ("women", "unisex") and p.price <= 6000,
                     expected_gender="women", max_price=6000),
        LabeledQuery("gift for a 5 year old boy under 2000", "recipient",
                     lambda p: p.audience in ("kids", "toddler") and p.price <= 2000,
                     expected_audience="kids", max_price=2000),
        LabeledQuery("gift for university student", "recipient",
                     lambda p: p.category_leaf in ("backpack", "laptop", "tablet", "notebook", "desk lamp", "scientific calculator"),
                     expected_audience="youth"),
        LabeledQuery("gift for a toddler who loves building things", "recipient",
                     lambda p: ("building blocks" in p.category_leaf or "toy" in p.category_leaf) and p.audience in ("kids", "toddler"),
                     expected_audience="toddler"),
        LabeledQuery("birthday gift for brother under 2500", "recipient",
                     lambda p: p.gender in ("men", "unisex") and p.price <= 2500,
                     expected_gender="men", max_price=2500),
        LabeledQuery("gift for sister under 3000", "recipient",
                     lambda p: p.gender in ("women", "unisex") and p.price <= 3000,
                     expected_gender="women", max_price=3000),
        LabeledQuery("presents for kids under 1500", "recipient",
                     lambda p: p.audience in ("kids", "toddler") and p.price <= 1500,
                     expected_audience="kids", max_price=1500),

        # --- Group 6: Occasion and Use-Case Queries (10 queries) ---
        LabeledQuery("online class er jonno", "usecase",
                     lambda p: p.category_leaf in ("webcam", "wireless headphones", "tablet", "laptop", "desk lamp"),
                     expected_category="electronics"),
        LabeledQuery("keep my coffee hot on a long drive", "usecase",
                     lambda p: p.category_leaf in ("stainless steel water bottle", "ceramic coffee mug"),
                     expected_category="drinkware"),
        LabeledQuery("something comfortable for summer travel", "usecase",
                     lambda p: p.category_leaf in ("travel bag", "backpack", "running shoes", "sunglasses", "stainless steel water bottle"),
                     expected_category="travel"),
        LabeledQuery("noise cancelling headphones for flights", "usecase",
                     lambda p: "headphones" in p.category_leaf,
                     expected_category="wireless headphones"),
        LabeledQuery("running shoes for daily morning workout", "usecase",
                     lambda p: "running shoes" in p.category_leaf,
                     expected_category="running shoes"),
        LabeledQuery("formal leather shoes for office meeting", "usecase",
                     lambda p: "leather shoes" in p.category_leaf and p.gender in ("men", "unisex"),
                     expected_category="leather shoes"),
        LabeledQuery("eid special panjabi for men under 4000", "usecase",
                     lambda p: "panjabi" in p.category_leaf and p.price <= 4000,
                     expected_category="panjabi", max_price=4000),
        LabeledQuery("gym water bottle 1000ml", "usecase",
                     lambda p: "water bottle" in p.category_leaf,
                     expected_category="stainless steel water bottle"),
        LabeledQuery("laptop bag for university classes", "usecase",
                     lambda p: "backpack" in p.category_leaf,
                     expected_category="backpack"),
        LabeledQuery("power bank for long travel", "usecase",
                     lambda p: "power bank" in p.category_leaf,
                     expected_category="power bank"),

        # --- Group 7: Budget and Price Limit Queries (15 queries) ---
        LabeledQuery("t-shirt under 500", "budget",
                     lambda p: "t-shirt" in p.category_leaf and p.price <= 500,
                     expected_category="t-shirt", max_price=500),
        LabeledQuery("shirt under 1500", "budget",
                     lambda p: "shirt" in p.category_leaf and p.price <= 1500,
                     expected_category="shirt", max_price=1500),
        LabeledQuery("running shoes below 3000", "budget",
                     lambda p: "running shoes" in p.category_leaf and p.price <= 3000,
                     expected_category="running shoes", max_price=3000),
        LabeledQuery("smartwatch under 3000", "budget",
                     lambda p: "smartwatch" in p.category_leaf and p.price <= 3000,
                     expected_category="smartwatch", max_price=3000),
        LabeledQuery("wireless headphones below 2000", "budget",
                     lambda p: "headphones" in p.category_leaf and p.price <= 2000,
                     expected_category="wireless headphones", max_price=2000),
        LabeledQuery("backpack under 1500", "budget",
                     lambda p: "backpack" in p.category_leaf and p.price <= 1500,
                     expected_category="backpack", max_price=1500),
        LabeledQuery("leather wallet under 1000", "budget",
                     lambda p: "wallet" in p.category_leaf and p.price <= 1000,
                     expected_category="leather wallet", max_price=1000),
        LabeledQuery("sunglasses under 1200", "budget",
                     lambda p: "sunglasses" in p.category_leaf and p.price <= 1200,
                     expected_category="sunglasses", max_price=1200),
        LabeledQuery("toy under 800", "budget",
                     lambda p: p.audience in ("kids", "toddler") and p.price <= 800,
                     expected_audience="kids", max_price=800),
        LabeledQuery("perfume under 1500", "budget",
                     lambda p: "perfume" in p.category_leaf and p.price <= 1500,
                     expected_category="perfume", max_price=1500),
        LabeledQuery("desk lamp under 1500", "budget",
                     lambda p: "desk lamp" in p.category_leaf and p.price <= 1500,
                     expected_category="desk lamp", max_price=1500),
        LabeledQuery("scientific calculator under 1800", "budget",
                     lambda p: "calculator" in p.category_leaf and p.price <= 1800,
                     expected_category="scientific calculator", max_price=1800),
        LabeledQuery("smartphone under 15000", "budget",
                     lambda p: "smartphone" in p.category_leaf and p.price <= 15000,
                     expected_category="smartphone", max_price=15000),
        LabeledQuery("beard trimmer under 2000", "budget",
                     lambda p: "trimmer" in p.category_leaf and p.price <= 2000,
                     expected_category="beard trimmer", max_price=2000),
        LabeledQuery("panjabi under 2500", "budget",
                     lambda p: "panjabi" in p.category_leaf and p.price <= 2500,
                     expected_category="panjabi", max_price=2500),

        # --- Group 8: Exact Brand and Product Queries (15 queries) ---
        LabeledQuery("Nike running shoes", "brand",
                     lambda p: "running shoes" in p.category_leaf and p.brand.lower() == "nike",
                     expected_category="running shoes", expected_brand="Nike"),
        LabeledQuery("Adidas sneakers", "brand",
                     lambda p: "sneakers" in p.category_leaf and p.brand.lower() == "adidas",
                     expected_category="sneakers", expected_brand="Adidas"),
        LabeledQuery("Samsung smartphone", "brand",
                     lambda p: "smartphone" in p.category_leaf and p.brand.lower() == "samsung",
                     expected_category="smartphone", expected_brand="Samsung"),
        LabeledQuery("Apple laptop", "brand",
                     lambda p: "laptop" in p.category_leaf and p.brand.lower() == "apple",
                     expected_category="laptop", expected_brand="Apple"),
        LabeledQuery("Sony wireless headphones", "brand",
                     lambda p: "headphones" in p.category_leaf and p.brand.lower() == "sony",
                     expected_category="wireless headphones", expected_brand="Sony"),
        LabeledQuery("Casio wristwatch", "brand",
                     lambda p: "watch" in p.category_leaf and p.brand.lower() == "casio",
                     expected_category="wristwatch", expected_brand="Casio"),
        LabeledQuery("Aarong panjabi", "brand",
                     lambda p: "panjabi" in p.category_leaf and p.brand.lower() == "aarong",
                     expected_category="panjabi", expected_brand="Aarong"),
        LabeledQuery("Apex leather shoes", "brand",
                     lambda p: "leather shoes" in p.category_leaf and p.brand.lower() == "apex",
                     expected_category="leather shoes", expected_brand="Apex"),
        LabeledQuery("Bata sandals", "brand",
                     lambda p: "sandals" in p.category_leaf and p.brand.lower() == "bata",
                     expected_category="sandals", expected_brand="Bata"),
        LabeledQuery("Philips electric toothbrush", "brand",
                     lambda p: "toothbrush" in p.category_leaf and p.brand.lower() == "philips",
                     expected_category="electric toothbrush", expected_brand="Philips"),
        LabeledQuery("Dell laptop for office", "brand",
                     lambda p: "laptop" in p.category_leaf and p.brand.lower() == "dell",
                     expected_category="laptop", expected_brand="Dell"),
        LabeledQuery("Lego building blocks", "brand",
                     lambda p: "building blocks" in p.category_leaf and p.brand.lower() == "lego",
                     expected_category="building blocks", expected_brand="Lego"),
        LabeledQuery("Xiaomi smartphone", "brand",
                     lambda p: "smartphone" in p.category_leaf and p.brand.lower() == "xiaomi",
                     expected_category="smartphone", expected_brand="Xiaomi"),
        LabeledQuery("Hot Wheels toy car", "brand",
                     lambda p: "car" in p.title.lower() or p.brand.lower() == "hot wheels",
                     expected_brand="Hot Wheels"),
        LabeledQuery("HP laptop", "brand",
                     lambda p: "laptop" in p.category_leaf and p.brand.lower() == "hp",
                     expected_category="laptop", expected_brand="HP"),

        # --- Group 9: Audience-Sensitive & Demographic Queries (10 queries) ---
        LabeledQuery("shoes for 5 year old boy", "audience",
                     lambda p: p.audience in ("kids", "toddler") or p.gender in ("kids", "boys"),
                     expected_audience="kids"),
        LabeledQuery("toys for 2 year old toddler", "audience",
                     lambda p: p.audience in ("kids", "toddler"),
                     expected_audience="toddler"),
        LabeledQuery("formal shirt for adult men", "audience",
                     lambda p: "shirt" in p.category_leaf and p.gender in ("men", "unisex") and p.audience == "adult",
                     expected_gender="men", expected_audience="adult"),
        LabeledQuery("saree for adult women", "audience",
                     lambda p: "saree" in p.category_leaf and p.gender in ("women", "unisex"),
                     expected_category="saree", expected_gender="women"),
        LabeledQuery("running shoes for women", "audience",
                     lambda p: "running shoes" in p.category_leaf and p.gender in ("women", "unisex"),
                     expected_category="running shoes", expected_gender="women"),
        LabeledQuery("kids t-shirt under 600", "audience",
                     lambda p: "t-shirt" in p.category_leaf and p.audience == "kids" and p.price <= 600,
                     expected_audience="kids", max_price=600),
        LabeledQuery("baby girl soft plush toy", "audience",
                     lambda p: p.audience in ("kids", "toddler"),
                     expected_audience="toddler"),
        LabeledQuery("leather wallet for men", "audience",
                     lambda p: "wallet" in p.category_leaf and p.gender in ("men", "unisex"),
                     expected_gender="men"),
        LabeledQuery("women kurti under 2000", "audience",
                     lambda p: "kurti" in p.category_leaf and p.gender in ("women", "unisex") and p.price <= 2000,
                     expected_gender="women", max_price=2000),
        LabeledQuery("boys remote control car", "audience",
                     lambda p: p.audience in ("kids", "toddler") or "car" in p.category_leaf,
                     expected_audience="kids"),

        # --- Group 10: Vague and Underspecified Queries (10 queries) ---
        LabeledQuery("something comfortable for home", "vague",
                     lambda p: p.category_leaf in ("t-shirt", "sandals", "slippers", "ceramic coffee mug", "scented candle")),
        LabeledQuery("nice gadget", "vague",
                     lambda p: p.category_leaf in ("smartwatch", "wireless headphones", "smartphone", "tablet", "bluetooth speaker")),
        LabeledQuery("something to write on", "vague",
                     lambda p: "notebook" in p.category_leaf or "pen" in p.category_leaf),
        LabeledQuery("item to keep drinks", "vague",
                     lambda p: p.category_leaf in ("stainless steel water bottle", "ceramic coffee mug")),
        LabeledQuery("portable audio device", "vague",
                     lambda p: p.category_leaf in ("wireless headphones", "bluetooth speaker")),
        LabeledQuery("winter protection wear", "vague",
                     lambda p: p.category_leaf in ("hoodie", "jacket")),
        LabeledQuery("useful gift item", "vague",
                     lambda p: p.category_leaf in ("leather wallet", "wristwatch", "ceramic coffee mug", "perfume", "stainless steel water bottle")),
        LabeledQuery("study essentials", "vague",
                     lambda p: p.category_leaf in ("notebook", "pen set", "scientific calculator", "desk lamp", "backpack")),
        LabeledQuery("daily grooming tool", "vague",
                     lambda p: p.category_leaf in ("electric toothbrush", "beard trimmer", "hair dryer")),
        LabeledQuery("fun game for weekend", "vague",
                     lambda p: p.category_leaf in ("puzzle board", "building blocks", "remote control car")),

        # --- Group 11: No-Exact-Match / Out-of-Domain Queries (10 queries) ---
        LabeledQuery("spacecraft rocket engine NASA", "no_match",
                     lambda p: False),
        LabeledQuery("real dinosaur fossil bone", "no_match",
                     lambda p: False),
        LabeledQuery("submarine nuclear reactor", "no_match",
                     lambda p: False),
        LabeledQuery("xyzabc 123456789 qwerasdf", "no_match",
                     lambda p: False),
        LabeledQuery("flying magic broomstick Harry Potter", "no_match",
                     lambda p: False),
        LabeledQuery("fresh live giraffe animal", "no_match",
                     lambda p: False),
        LabeledQuery("industrial heavy blast furnace", "no_match",
                     lambda p: False),
        LabeledQuery("quantum teleportation chamber", "no_match",
                     lambda p: False),
        LabeledQuery("ancient egyptian gold sarcophagus", "no_match",
                     lambda p: False),
        LabeledQuery("agricultural tractor harvester 500hp", "no_match",
                     lambda p: False)
    ]
    return tests


class BenchmarkEvaluator:
    """Evaluates search engine over labeled query set, computes IR metrics and runs ablations."""
    
    def __init__(self, engine, labeled_queries: List[LabeledQuery]):
        self.engine = engine
        self.labeled_queries = labeled_queries

    def evaluate_pipeline(self, ablation_mode: str = "full_pipeline", k: int = 5) -> Dict[str, float]:
        hit_1 = 0
        precisions = []
        recalls = []
        ndcgs = []
        reciprocal_ranks = []
        wrong_audience_count = 0
        constraint_violations = 0
        latencies = []
        cache_hits = 0

        # Evaluate only on valid queries (exclude no_match queries for IR relevance calculation)
        eval_queries = [q for q in self.labeled_queries if q.query_type != "no_match"]

        for tq in eval_queries:
            t0 = time.perf_counter()
            res = self.engine.search(tq.query, k=k, ablation_mode=ablation_mode, use_cache=False)
            lat = (time.perf_counter() - t0) * 1000
            latencies.append(lat)
            if res.is_cache_hit:
                cache_hits += 1

            retrieved = res.products[:k]
            if not retrieved:
                precisions.append(0.0)
                recalls.append(0.0)
                ndcgs.append(0.0)
                reciprocal_ranks.append(0.0)
                continue

            # Ground truth matches
            matches = [1 if tq.predicate(p) else 0 for p in retrieved]
            
            # Hit@1
            if matches[0] == 1:
                hit_1 += 1

            # Precision@K
            p_at_k = sum(matches) / float(k)
            precisions.append(p_at_k)

            # Recall@K (assuming target has at least 3 matching items in catalog)
            r_at_k = min(sum(matches) / 3.0, 1.0)
            recalls.append(r_at_k)

            # NDCG@K
            dcg = sum((2**m - 1) / math.log2(rank + 2) for rank, m in enumerate(matches))
            # Ideal DCG
            ideal_matches = sorted(matches, reverse=True)
            idcg = sum((2**m - 1) / math.log2(rank + 2) for rank, m in enumerate(ideal_matches))
            ndcg_val = (dcg / idcg) if idcg > 0 else 0.0
            ndcgs.append(ndcg_val)

            # MRR
            rr = 0.0
            for rank, m in enumerate(matches):
                if m == 1:
                    rr = 1.0 / (rank + 1)
                    break
            reciprocal_ranks.append(rr)

            # Constraint checks: Wrong audience
            if tq.expected_audience in ("kids", "toddler"):
                for p in retrieved:
                    if p.audience not in ("kids", "toddler") and p.gender not in ("kids", "boys", "girls") and "toy" not in p.category_leaf:
                        wrong_audience_count += 1
                        break

            # Constraint checks: Budget violation
            if tq.max_price is not None:
                for p in retrieved:
                    if p.price > tq.max_price:
                        constraint_violations += 1
                        break

        n_eval = len(eval_queries)
        return {
            "ablation": ablation_mode,
            "hit_at_1": round(hit_1 / n_eval, 3),
            "precision_at_5": round(float(np.mean(precisions)), 3),
            "recall_at_5": round(float(np.mean(recalls)), 3),
            "ndcg_at_5": round(float(np.mean(ndcgs)), 3),
            "mrr": round(float(np.mean(reciprocal_ranks)), 3),
            "wrong_audience_rate": round(wrong_audience_count / max(n_eval, 1), 3),
            "hard_constraint_violation_rate": round(constraint_violations / max(n_eval, 1), 3),
            "p50_latency_ms": round(float(np.median(latencies)), 1),
            "p95_latency_ms": round(float(np.percentile(latencies, 95)), 1)
        }

    def run_all_ablations(self) -> pd.DataFrame:
        """Runs evaluation across all 7 defined ablation configurations."""
        modes = [
            "bm25_only",
            "dense_only",
            "hybrid_only",
            "hybrid_intent",
            "hybrid_intent_rerank",
            "full_pipeline"
        ]
        results = []
        for m in modes:
            print(f"Evaluating ablation: {m}...")
            res = self.evaluate_pipeline(ablation_mode=m, k=5)
            results.append(res)
        return pd.DataFrame(results)
