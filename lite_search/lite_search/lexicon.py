"""Language data: Bangla/Banglish/Hindi aliases, phrase fixes, gender cues and product groups.
Everything here is plain data - extend it (or override with JSON files, see Settings.lexicon_file) without touching code.
"""
ALIASES = {
    "juta": "shoes", "jutha": "shoes", "joota": "shoes", "jutta": "shoes", "jooto": "shoes", "juto": "shoes",
    "chappal": "sandals", "chapal": "sandals", "chappals": "sandals", "slipper": "sandals", "slippers": "sandals",
    "sandal": "sandals", "keds": "sneakers", "moja": "socks",
    "sari": "saree", "shari": "saree", "sarees": "saree", "punjabi": "panjabi",
    "pant": "trousers", "pants": "trousers", "tshirt": "t-shirt",
    "tupi": "cap", "topi": "cap", "ghori": "watch", "ghadi": "watch",
    "chosma": "sunglasses", "chashma": "sunglasses", "chasma": "sunglasses", "shades": "sunglasses",
    "baag": "bag", "beg": "bag", "byag": "bag", "thola": "bag",
    "chata": "umbrella", "chhata": "umbrella", "chaata": "umbrella", "chhaata": "umbrella",
    "mobile": "smartphone", "mobail": "smartphone", "pankha": "fan",
    "kolom": "pen", "kalam": "pen", "khata": "notebook", "khatha": "notebook", "boi": "book",
    "lal": "red", "nil": "blue", "sada": "white", "shada": "white", "kalo": "black", "kala": "black",
    "shobuj": "green", "sabuj": "green", "sobuj": "green", "badami": "brown",
    "khoyeri": "brown", "holud": "yellow", "golapi": "pink",
    "chele": "boys", "cheler": "boys", "meye": "girls", "meyer": "girls", "meyeder": "girls",
    "cheleder": "boys", "purush": "men", "purusher": "men",
    "mohila": "women", "mohilader": "women", "baccha": "kids", "bachcha": "kids", "bacchader": "kids",
    "shishuder": "kids", "teen": "teenagers", "teens": "teenagers", "teenager": "teenagers",
    "unisex": "unisex", "sobai": "everyone",
    "ma": "mother", "maa": "mother", "amma": "mother", "ammu": "mother",
    "baba": "father", "abbu": "father", "abba": "father",
    "bou": "wife", "bhai": "brother", "vai": "brother", "apu": "sister", "didi": "sister",
    "dam": "", "daam": "",
    "give": "", "show": "", "find": "", "get": "", "buy": "", "search": "",
    "please": "", "pls": "", "plz": "", "me": "", "want": "", "need": ""
}

PHRASES = [
    ("tee shirt", "t-shirt"), ("tee-shirt", "t-shirt"), ("tshirt", "t-shirt"),
    ("jeans pant", "jeans"), ("denim pant", "jeans"),
    ("sports shoe", "sports shoes"), ("running shoe", "running shoes"),
    ("formal shoe", "formal shoes"), ("school shoe", "school shoes"),
    ("laptop bag", "laptop bag"), ("ladies handbag", "women handbag"),
    ("mobile phone", "mobile phone"), ("smart phone", "smartphone"),
    ("bed sheet", "bedsheet"), ("electric kettle", "electric kettle"),
    ("face wash", "facial cleanser"), ("sun screen", "sunscreen"),
    ("note book", "notebook"), ("toy car", "toy car")
]

GENDER_CUES = {
    "men": ["men", "mens", "men's", "man", "male", "gents", "gentlemen", "father", "dad", "daddy", "papa", "abba", "abbu",
            "baba", "husband", "brother", "bhai", "boro bhai", "choto bhai", "grandfather", "grandpa", "dada", "dadu",
            "nana", "boyfriend", "uncle", "kaku", "chacha", "him", "his", "purush", "purusher"],
    "women": ["women", "womens", "women's", "woman", "female", "ladies", "lady", "mother", "mom", "mum", "mommy", "mummy",
              "amma", "wife", "sister", "bon", "boro bon", "choto bon", "grandmother", "grandma", "dadi", "dida", "nani",
              "girlfriend", "aunt", "aunty", "khala", "khalamoni", "fupu", "pishi", "her", "hers", "mohila", "mohilader"],
    "boys": ["boy", "boys", "son", "nephew", "little boy", "baby boy", "schoolboy", "chele", "cheler", "cheleder"],
    "girls": ["girl", "girls", "daughter", "niece", "little girl", "baby girl", "schoolgirl", "meye", "meyer", "meyeder"],
    "unisex": ["unisex", "gender neutral", "gender-neutral", "for everyone", "all genders", "sobai"],
    "baby": ["baby", "babies", "newborn", "infant", "toddler", "shishu", "baccha", "bachcha"],
    "kids": ["kid", "kids", "child", "children", "toddler", "kids wear", "for kids", "bacchader", "bachader", "shishuder"],
    "teenagers": ["teen", "teens", "teenager", "teenagers", "adolescent", "adolescents", "kishor", "kishori"],
}

PHRASES = [
    ("tee shirt", "t-shirt"), ("tee-shirt", "t-shirt"), ("tshirt", "t-shirt"),
    ("jeans pant", "jeans"), ("denim pant", "jeans"),
    ("sports shoe", "sports shoes"), ("running shoe", "running shoes"),
    ("formal shoe", "formal shoes"), ("school shoe", "school shoes"),
    ("laptop bag", "laptop bag"), ("ladies handbag", "women handbag"),
    ("mobile phone", "mobile phone"), ("smart phone", "smartphone"),
    ("bed sheet", "bedsheet"), ("electric kettle", "electric kettle"),
    ("face wash", "facial cleanser"), ("sun screen", "sunscreen"),
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
        ["smartphone", "mobile phone", "laptop", "tablet", "monitor",
         "keyboard", "computer mouse", "printer", "router", "webcam"]
    ),
    "phone_accessories": (
        {"charger", "chargers", "cable", "cables", "powerbank", "powerbanks",
         "power bank", "power banks", "phone case", "phone cases", "cover",
         "covers", "screen protector", "phone holder", "mobile holder",
         "phone stand", "mobile stand"},
        ["phone charger", "USB cable", "power bank", "phone case",
         "screen protector", "phone holder", "wireless charger"]
    ),
    "home_appliances": (
        {"refrigerator", "fridge", "freezer", "washing machine",
         "microwave", "oven", "blender", "mixer", "juicer",
         "rice cooker", "air fryer", "vacuum", "vacuum cleaner",
         "iron", "fan", "air conditioner", "ac", "heater"},
        ["refrigerator", "washing machine", "microwave oven",
         "rice cooker", "blender", "air fryer", "vacuum cleaner",
         "electric iron", "ceiling fan", "air conditioner"]
    ),
    "kitchenware": (
        {"cookware", "pan", "pans", "frying pan", "pot", "pots",
         "pressure cooker", "kettle", "knife", "knives", "cutlery",
         "plate", "plates", "bowl", "bowls", "mug", "mugs",
         "glass", "glasses", "bottle", "bottles", "lunchbox",
         "lunchboxes", "thermos"},
        ["cookware set", "frying pan", "pressure cooker",
         "electric kettle", "kitchen knife", "dinner set",
         "water bottle", "lunch box", "food storage container"]
    ),
    "beauty": (
        {"makeup", "cosmetics", "lipstick", "mascara", "foundation",
         "concealer", "eyeliner", "blush", "highlighter", "perfume",
         "fragrance", "nail polish", "nailpaint", "makeup brush",
         "makeup brushes"},
        ["lipstick", "foundation", "mascara", "eyeliner",
         "makeup kit", "makeup brushes", "perfume", "nail polish"]
    ),
    "skincare": (
        {"skincare", "moisturizer", "moisturiser", "sunscreen",
         "serum", "cleanser", "facewash", "face wash", "toner",
         "exfoliator", "face cream", "lip balm", "face mask"},
        ["face wash", "moisturizer", "sunscreen", "face serum",
         "toner", "face cream", "lip balm", "face mask"]
    ),
    "personal_care": (
        {"toothbrush", "toothpaste", "mouthwash", "razor", "razors",
         "trimmer", "shaver", "epilator", "deodorant",
         "body wash", "body lotion", "soap", "sanitizer",
         "sanitiser", "electric toothbrush"},
        ["toothbrush", "toothpaste", "electric trimmer", "razor",
         "deodorant", "body wash", "body lotion", "hand sanitizer"]
    ),
    "toys": (
        {"toy", "toys", "lego", "doll", "dolls", "teddy bear",
         "puzzle", "puzzles", "remote control car", "rc car",
         "building blocks", "board game", "board games",
         "action figure", "action figures"},
        ["building blocks", "dolls", "puzzles", "board games",
         "remote control cars", "teddy bears", "educational toys"]
    ),
    "traditional_bangladeshi": (
        {"panjabi", "punjabi", "lungi", "gamcha", "nakshi kantha",
         "jamdani", "jamdani saree", "muslin", "shalwar kameez",
         "prayer mat", "tasbih", "topor"},
        ["panjabi", "lungi", "gamcha", "nakshi kantha",
         "jamdani saree", "muslin clothing", "prayer mat", "tasbih"]
    ),
    "watches":   ({"watch", "watches", "smartwatch", "wristwatch"}, ["watch", "smartwatch", "fitness band"]),
    "eyewear":   ({"sunglasses", "glasses", "goggles"}, ["sunglasses", "goggles"]),
    "ethnic":    ({"panjabi", "saree", "kurti", "salwar", "kameez"}, ["panjabi", "saree", "kurti", "salwar kameez"]),
    "tops":      ({"shirt", "shirts", "tee", "tees", "polo", "hoodie", "hoodies", "sweater", "cardigan"},
                  ["t-shirt", "shirt", "hoodie", "sweater"]),
    "bottoms":   ({"jeans", "trousers", "shorts", "leggings", "chinos"},
                  ["jeans", "trousers", "shorts", "leggings", "track pants"]),
    "outerwear": ({"jacket", "jackets", "raincoat", "poncho", "coat"}, ["jacket", "raincoat"]),
    "audio":     ({"headphone", "headphones", "earbuds", "earphone", "earphones", "speaker", "speakers"},
                  ["headphones", "earbuds", "speaker"]),
}
