"""Language data: Bangla/Banglish/Hindi aliases, phrase fixes, gender cues and product groups.
Everything here is plain data - extend it (or override with JSON files, see Settings.lexicon_file) without touching code.
"""
ALIASES = {
    "juta": "shoes", "jutha": "shoes", "joota": "shoes", "jutta": "shoes", "jooto": "shoes", "juto": "shoes",
    "bike": "bicycle", "bikes": "bicycle", "cycle": "bicycle", "cycles": "bicycle",
    "chappal": "sandals", "chapal": "sandals", "chappals": "sandals", "slipper": "sandals", "slippers": "sandals",
    "sandal": "sandals", "keds": "sneakers", "moja": "socks",
    "sari": "saree", "shari": "saree", "sarees": "saree", "punjabi": "panjabi",
    "pant": "trousers", "pants": "trousers", "pents": "trousers", "pent": "trousers", "tshirt": "t-shirt",
    "churi": "kitchen knife", "chhuri": "kitchen knife", "chaku": "kitchen knife", "knife": "kitchen knife", "knives": "kitchen knife",
    "ketli": "electric kettle", "ketly": "electric kettle", "kettle": "electric kettle",
    "tupi": "cap", "topi": "cap", "ghori": "watch", "ghadi": "watch",
    "chosma": "sunglasses", "chashma": "sunglasses", "chasma": "sunglasses", "shades": "sunglasses",
    "baag": "bag", "beg": "bag", "byag": "bag", "thola": "bag",
    "chata": "umbrella", "chhata": "umbrella", "chaata": "umbrella", "chhaata": "umbrella",
    "mobile": "mobile", "mobail": "smartphone", "cellphone": "smartphone", "cell": "smartphone",
    "earphones": "earbuds", "earphone": "earbuds",
    "ac": "air conditioner", "pankha": "fan",
    "kolom": "pen", "kalam": "pen", "khata": "notebook", "khatha": "notebook", "boi": "book",
    "powerbank": "power bank", "smartwatch": "smartwatch", "handwatch": "watch",
    "facewash": "face wash", "sunblock": "sunscreen", "lipstik": "lipstick", "shampo": "shampoo",
    "lal": "red", "nil": "blue", "sada": "white", "shada": "white", "kalo": "black", "kala": "black",
    "shobuj": "green", "sabuj": "green", "sobuj": "green", "badami": "brown",
    "khoyeri": "brown", "holud": "yellow", "golapi": "pink",
    "jama": "clothes", "jaama": "clothes", "poshak": "clothing", "posak": "clothing",
    "kapor": "clothes", "kapod": "clothes",
    "chele": "men", "cheler": "men", "cheleder": "men", "purush": "men", "purusher": "men",
    "meye": "women", "meyer": "women", "meyeder": "women", "mohila": "women", "mohilader": "women",
    "baccha": "kids", "bachcha": "kids", "bacchader": "kids",
    "shishuder": "kids", "teen": "teenagers", "teens": "teenagers", "teenager": "teenagers",
    "unisex": "unisex", "sobai": "everyone",
    "ma": "mother", "maa": "mother", "amma": "mother", "ammu": "mother",
    "baba": "father", "abbu": "father", "abba": "father",
    "bou": "wife", "bhai": "brother", "vai": "brother", "apu": "sister", "didi": "sister",
    "dam": "", "daam": "",
    "give": "", "show": "", "find": "", "get": "", "buy": "", "search": "",
    "please": "", "pls": "", "plz": "", "me": "", "want": "", "need": ""
}

GENDER_CUES = {
    "men": ["men", "mens", "men's", "man", "male", "gents", "gentlemen", "father", "dad", "daddy", "papa", "abba", "abbu",
            "baba", "husband", "brother", "bhai", "boro bhai", "choto bhai", "grandfather", "grandpa", "dada", "dadu",
            "nana", "boyfriend", "uncle", "kaku", "chacha", "him", "his", "purush", "purusher", "chele", "cheler", "cheleder"],
    "women": ["women", "womens", "women's", "woman", "female", "ladies", "lady", "mother", "mom", "mum", "mommy", "mummy",
              "amma", "wife", "sister", "bon", "boro bon", "choto bon", "grandmother", "grandma", "dadi", "dida", "nani",
              "girlfriend", "aunt", "aunty", "khala", "khalamoni", "fupu", "pishi", "her", "hers", "mohila", "mohilader",
              "meye", "meyer", "meyeder"],
    "boys": ["boy", "boys", "son", "nephew", "little boy", "baby boy", "schoolboy", "choto chele", "baccha chele"],
    "girls": ["girl", "girls", "daughter", "niece", "little girl", "baby girl", "schoolgirl", "choto meye", "baccha meye"],
    "unisex": ["unisex", "gender neutral", "gender-neutral", "for everyone", "all genders", "sobai"],
    "baby": ["baby", "babies", "newborn", "infant", "toddler", "shishu", "baccha", "bachcha"],
    "kids": ["kid", "kids", "child", "children", "toddler", "kids wear", "for kids", "bacchader", "bachader", "shishuder"],
    "teenagers": ["teen", "teens", "teenager", "teenagers", "adolescent", "adolescents", "kishor", "kishori"],
}

PHRASES = [
    ("notebook computer", "laptop"), ("notebook pc", "laptop"),
    ("air conditioner", "air conditioner"),
    ("tee shirt", "t-shirt"), ("tee-shirt", "t-shirt"), ("tshirt", "t-shirt"),
    ("jeans pant", "jeans"), ("denim pant", "jeans"),
    ("sports shoe", "sports shoes"), ("running shoe", "running shoes"),
    ("formal shoe", "formal shoes"), ("school shoe", "school shoes"),
    ("laptop bag", "laptop bag"), ("ladies handbag", "women handbag"),
    ("mobile phone", "mobile phone"), ("smart phone", "smartphone"),
    ("smart watch", "smartwatch"), ("power bank", "power bank"),
    ("phone cover", "phone case"), ("mobile cover", "phone case"), ("back cover", "phone case"),
    ("phone charger", "phone charger"), ("mobile charger", "phone charger"),
    ("bed sheet", "bedsheet"), ("electric kettle", "electric kettle"),
    ("face wash", "face wash"), ("sun screen", "sunscreen"),
    ("note book", "notebook"), ("toy car", "toy car"),
]

PRICE_RULES = [
    (r"(\d[\d,]*)\s*(?:tk|taka|bdt)?\s*(?:er\s*)?(?:niche|nichay|kom|kome)\b", r"under \1"),
    (r"(\d[\d,]*)\s*(?:tk|taka|bdt)?\s*(?:er\s*)?(?:upore|opore|beshi)\b", r"over \1"),
]

# Few-shot examples used by the LLM translator (Banglish -> English)
TRANS_SHOTS = [
    ("juta", "shoes"), ("running er juta", "running shoes"), ("meyeder chappal", "women sandals"),
    ("panjabi dam 2000 er niche", "panjabi under 2000"), ("kalo panjabi", "black panjabi"),
    ("lal sharee", "red saree"), ("laptop er bag", "laptop bag"), ("bichana chador", "bedsheet"),
    ("shit er jama", "winter clothes"), ("1000 takar moddhe", "under 1000"),
]

KID_WORDS = {"kid", "kids", "child", "children", "infant", "infants", "toddler", "toddlers",
             "boy", "boys", "girl", "girls", "toy", "toys"}          # no "baby": "baby pink" is a colour
KID_TITLE = {"kid", "kids", "child", "children", "infant", "infants", "toddler", "toddlers"}

GROUPS = {
    "footwear":  ({"shoe", "shoes", "footwear", "sneaker", "sneakers", "sandal", "sandals", "boot", "boots",
                   "clog", "clogs", "loafer", "loafers", "heel", "heels", "flats"},
                  ["running shoes", "sneakers", "formal shoes", "sandals", "walking shoes", "boots", "clogs",
                   "training shoes", "school shoes", "loafers", "heels", "flats"]),
    "bags":      ({"bag", "bags", "backpack", "backpacks", "rucksack", "handbag", "suitcase", "luggage", "trolley",
                   "duffel","laptop bag", "tote"},
                  ["backpack", "duffel bag", "travel bag", "handbag", "gym bag", "sling bag", "suitcase"]),
    "electronics": (
        {"phone", "phones", "smartphone", "smartphones", "mobile", "mobiles",
         "tablet", "tablets", "ipad", "laptop", "laptops", "computer", "computers",
         "desktop", "monitor", "monitors", "keyboard", "keyboards", "mouse",
         "webcam", "printer", "router", "projector"},
        ["smartphone", "mobile phone", "phone", "button phone", "5g smartphone", "laptop", "tablet", "monitor",
         "keyboard", "computer mouse", "printer", "router", "webcam"]
    ),
    "phone_accessories": (
        {"charger", "chargers", "adapter", "adapters", "cable", "cables", "powerbank", "powerbanks",
         "power bank", "power banks", "phone case", "phone cases", "mobile cover", "phone cover",
         "back cover", "screen protector", "phone holder", "mobile holder",
         "phone stand", "mobile stand"},
        ["phone charger", "charger", "adapter", "USB cable", "power bank", "phone case",
         "screen protector", "phone holder", "wireless charger"]
    ),
    "home_appliances": (
        {"refrigerator", "fridge", "freezer", "washing machine",
         "microwave", "oven", "blender", "mixer", "juicer",
         "rice cooker", "air fryer", "vacuum", "vacuum cleaner",
         "iron", "fan", "air conditioner", "air cooler", "ac", "heater"},
        ["refrigerator", "washing machine", "microwave oven",
         "rice cooker", "blender", "air fryer", "vacuum cleaner",
         "electric iron", "ceiling fan", "fan", "air conditioner", "air cooler"]
    ),
    "kitchenware": (
        {"cookware", "pan", "pans", "frying pan", "pot", "pots",
         "pressure cooker", "kettle", "knife", "knives", "cutlery",
         "churi", "chhuri", "chaku", "choora", "chhora",
         "plate", "plates", "bowl", "bowls", "mug", "mugs",
         "glass", "glasses", "bottle", "bottles", "lunchbox",
         "lunchboxes", "thermos", "cup", "cups"},
        ["cookware set", "cookware", "frying pan", "pressure cooker",
         "electric kettle", "kettle", "kitchen knife", "dinner set",
         "water bottle", "stainless steel water bottle", "lunch box",
         "food storage container", "ceramic coffee mug", "mug", "travel mug", "thermos"]
    ),
    "beauty": (
        {"makeup", "cosmetics", "lipstick", "mascara", "foundation",
         "concealer", "eyeliner", "blush", "highlighter", "perfume",
         "fragrance", "nail polish", "nailpaint", "makeup brush",
         "makeup brushes"},
        ["lipstick", "makeup", "foundation", "mascara", "eyeliner",
         "makeup kit", "makeup brushes", "perfume", "nail polish"]
    ),
    "haircare": (
        {"shampoo", "conditioner", "hair oil", "hair serum", "hair mask"},
        ["shampoo", "hair", "conditioner", "hair care"]
    ),
    "skincare": (
        {"skincare", "moisturizer", "moisturiser", "sunscreen", "sunblock",
         "serum", "cleanser", "facewash", "face wash", "toner",
         "exfoliator", "face cream", "lip balm", "face mask"},
        ["face wash", "moisturizer", "sunscreen", "face serum", "serum",
         "toner", "face cream", "lip balm", "face mask", "skin"]
    ),
    "personal_care": (
        {"toothbrush", "toothpaste", "mouthwash", "razor", "razors",
         "trimmer", "beard trimmer", "shaver", "epilator", "deodorant",
         "body wash", "body lotion", "soap", "sanitizer",
         "sanitiser", "electric toothbrush"},
        ["toothbrush", "toothpaste", "electric trimmer", "beard trimmer", "trimmer", "mens trimmers", "razor",
         "deodorant", "body wash", "body lotion", "hand sanitizer"]
    ),
    "toys": (
        {"toy", "toys", "lego", "doll", "dolls", "teddy bear",
         "puzzle", "puzzles", "remote control car", "rc car",
         "building blocks", "board game", "board games",
         "action figure", "action figures"},
        ["building blocks", "dolls", "puzzles", "board games",
         "remote control cars", "teddy bears", "educational toys", "toy"]
    ),
    "traditional_bangladeshi": (
        {"panjabi", "punjabi", "lungi", "gamcha", "nakshi kantha",
         "jamdani", "jamdani saree", "muslin", "shalwar kameez",
         "prayer mat", "tasbih", "topor", "fotua", "three piece", "three-piece"},
        ["panjabi", "lungi", "gamcha", "nakshi kantha",
         "jamdani saree", "muslin clothing", "prayer mat", "tasbih", "saree", "kurti", "salwar kameez", "traditional_bangladeshi"]
    ),
    "watches":   ({"watch", "watches", "smartwatch", "smart watch", "wristwatch", "handwatch"}, ["watch", "smartwatch", "smart watch", "wearable technologies", "fitness band"]),
    "eyewear":   ({"sunglasses", "glasses", "goggles"}, ["sunglasses", "goggles"]),
    "ethnic":    ({"panjabi", "saree", "kurti", "salwar", "kameez", "fotua", "three piece", "three-piece"},
                  ["panjabi", "saree", "kurti", "salwar kameez", "traditional_bangladeshi"]),
    "tops":      ({"shirt", "shirts", "tee", "tees", "polo", "hoodie", "hoodies", "sweater", "cardigan"},
                  ["t-shirt", "shirt", "hoodie", "sweater"]),
    "bottoms":   ({"jeans", "trousers", "shorts", "leggings", "chinos", "pant", "pants", "pents", "pent", "joggers", "trackpants", "pajama", "pajamas"},
                  ["jeans", "trousers", "shorts", "leggings", "track pants", "pajamas"]),
    "outerwear": ({"jacket", "jackets", "raincoat", "poncho", "coat"}, ["jacket", "raincoat"]),
    "audio":     ({"headphone", "headphones", "earbuds", "earphone", "earphones", "speaker", "speakers", "ইয়ারবাডস", "ইয়ারফোন", "হেডফোন"},
                  ["earbuds", "wireless headphones", "headphones", "speaker"]),
    "kettle":    ({"kettle", "kettles", "electric kettle", "ketli", "ketly", "কেটলি", "কেটলী", "ইলেকট্রিক কেটলি"},
                  ["electric kettle", "kettle"]),
    "cycling":   ({"bike", "bikes", "bicycle", "bicycles", "cycle", "cycles", "cycling", "সাইকেল", "বাইসাইকেল"},
                  ["bicycle", "helmet", "sports shoes", "water bottle"]),
    "clothing":  ({"dress", "dresses", "clothing", "cloth", "clothes", "apparel", "wear", "outfit", "attire",
                   "jama", "jaama", "kapor", "kapod", "poshak", "posak",
                   "পোশাক", "পোষাক", "ড্রেস", "জামা", "কাপড়", "কাপড়", "বস্ত্র", "ক্লথ"},
                  ["shirt", "t-shirt", "panjabi", "jeans", "trousers", "jacket", "hoodie", "sweater", "saree",
                   "kurti", "salwar kameez", "salwar", "dress", "fotua", "lungi", "pajamas", "shorts", "coat",
                   "raincoat", "track pants", "clothing", "traditional_bangladeshi"]),
}
