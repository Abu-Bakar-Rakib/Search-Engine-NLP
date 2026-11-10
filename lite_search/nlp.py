"""
Advanced NLP query parser for Bangla, Banglish, and English.
Supports: price extraction, size, persona, intent, brand, color, material.
Merged from Desktop SEARCH ENGINE project.
"""
from __future__ import annotations

import copy
import json
import math
import os
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Domain Pack — loads config/lexicon/taxonomy from JSON
# ---------------------------------------------------------------------------

class DomainPack:
    """Encapsulates company/domain taxonomy, lexicon, personas, intents, and locales."""

    def __init__(self, data: Dict[str, Any]):
        self.data = data
        self.schema_map: Dict[str, str] = data.get("schema_map", {})
        self.taxonomy_map: Dict[str, List[str]] = data.get("taxonomy_map", {})
        self.lexicon: Dict[str, Any] = data.get("lexicon", {})
        self.persona_config: Dict[str, Any] = data.get("persona_config", {})
        self.intent_config: Dict[str, Any] = data.get("intent_config", {})
        self.locale_config: Dict[str, Any] = data.get("locale_config", {})
        self.retrieval_config: Dict[str, Any] = data.get("retrieval_config", {})
        self.threshold_config: Dict[str, Any] = data.get("threshold_config", {})

        self.aliases: Dict[str, str] = {k.lower(): v.lower() for k, v in self.lexicon.get("aliases", {}).items()}
        raw_b_aliases = {k.lower(): v for k, v in self.lexicon.get("brand_aliases", {}).items()}
        # Include canonical values themselves and essential catalog brands (Gree, Dyson, etc.)
        for v in list(raw_b_aliases.values()):
            if v.lower() not in raw_b_aliases:
                raw_b_aliases[v.lower()] = v
        for common_b in ["gree", "dyson", "philips", "walton", "panasonic", "samsung", "apple", "xiaomi", "sony", "hp", "dell", "lenovo", "asus", "acer", "casio"]:
            if common_b not in raw_b_aliases:
                raw_b_aliases[common_b] = common_b.capitalize()
        self.brand_aliases: Dict[str, str] = raw_b_aliases
        self.bangla_dict: Dict[str, str] = {
            unicodedata.normalize("NFC", k): v
            for k, v in self.lexicon.get("bangla_dictionary", {}).items()
        }

        self.leaf_to_group: Dict[str, str] = {}
        for group, leaves in self.taxonomy_map.items():
            for leaf in leaves:
                self.leaf_to_group[leaf.lower()] = group

        b_digits = self.locale_config.get("bangla_digits", {})
        self.digit_trans_table = str.maketrans(b_digits) if b_digits else str.maketrans({})

        # Precompiled patterns for fast resolution
        self._brand_patterns: List[Tuple[re.Pattern, str]] = [
            (re.compile(rf"(?<![\w\u0980-\u09FF]){re.escape(variant.lower())}(?![\w\u0980-\u09FF])"), canonical)
            for variant, canonical in self.brand_aliases.items()
        ]
        self._persona_patterns: List[Tuple[re.Pattern, Dict[str, Any]]] = []
        for persona_name, cfg in self.persona_config.items():
            for alias in cfg.get("aliases", []):
                pat = re.compile(rf"(?<![\w\u0980-\u09FF]){re.escape(alias.lower())}(?![\w\u0980-\u09FF])")
                self._persona_patterns.append((pat, {
                    "persona": persona_name,
                    "matched_term": alias,
                    "target_audience": cfg.get("target_audience", "adult"),
                    "target_gender": cfg.get("target_gender", "unisex"),
                    "affinities": cfg.get("affinities", []),
                }))

        self._intent_patterns: List[Tuple[re.Pattern, str, str, List[str]]] = []
        for intent_name, cfg in self.intent_config.items():
            for trigger in cfg.get("triggers", []):
                pat = re.compile(rf"(?<![\w\u0980-\u09FF]){re.escape(trigger.lower())}(?![\w\u0980-\u09FF])")
                self._intent_patterns.append((pat, intent_name, trigger, cfg.get("boost_affinities", [])))

        self.persona_alias_prefixes: Tuple[str, ...] = tuple(
            alias.lower()
            for cfg in self.persona_config.values()
            for alias in cfg.get("aliases", [])
        )
        self.bangla_dict_items = sorted(self.bangla_dict.items(), key=lambda x: -len(x[0]))

    @classmethod
    def from_file(cls, filepath: str) -> "DomainPack":
        with open(filepath, "r", encoding="utf-8") as f:
            return cls(json.load(f))

    @classmethod
    def default(cls) -> "DomainPack":
        """Load embedded domain pack from lite_search package directory."""
        pkg_dir = os.path.dirname(__file__)
        path = os.path.join(pkg_dir, "domain_pack.json")
        if os.path.exists(path):
            return cls.from_file(path)
        return cls({})  # Empty fallback

    def convert_bangla_digits(self, text: str) -> str:
        return text.translate(self.digit_trans_table)

    def resolve_alias(self, token: str) -> str:
        t_low = token.lower()
        if t_low in self.aliases:
            return self.aliases[t_low]
        if token in self.bangla_dict:
            return self.bangla_dict[token]
        return token

    def resolve_brand(self, text: str) -> Optional[str]:
        t_low = text.lower()
        for pat, canonical in self._brand_patterns:
            if pat.search(t_low):
                return canonical
        return None

    def match_persona(self, query: str) -> Optional[Dict[str, Any]]:
        q_clean = query.lower()
        for pat, res in self._persona_patterns:
            if pat.search(q_clean):
                return res
        return None

    def match_intent(self, query: str) -> List[Dict[str, Any]]:
        q_clean = query.lower()
        matched = []
        seen = set()
        for pat, intent_name, trigger, affs in self._intent_patterns:
            if intent_name in seen:
                continue
            if pat.search(q_clean):
                seen.add(intent_name)
                matched.append({
                    "intent": intent_name,
                    "trigger": trigger,
                    "affinities": affs,
                })
        return matched


# ---------------------------------------------------------------------------
# Structured Intent — output of query parsing
# ---------------------------------------------------------------------------

@dataclass
class StructuredIntent:
    raw_query: str
    normalized_query: str
    retrieval_query: str
    language: str = "en"

    product_type: Optional[str] = None
    brand: Optional[str] = None
    color: Optional[str] = None
    material: Optional[str] = None
    size: Optional[str] = None

    recipient: Optional[str] = None
    target_audience: Optional[str] = None
    target_gender: Optional[str] = None

    intents: List[str] = field(default_factory=list)
    affinities: List[str] = field(default_factory=list)

    min_price: Optional[float] = None
    max_price: Optional[float] = None
    currency: str = "BDT"
    corrections: List[str] = field(default_factory=list)

    hard_constraints: Dict[str, Any] = field(default_factory=dict)
    soft_preferences: Dict[str, Any] = field(default_factory=dict)
    excluded_brands: List[str] = field(default_factory=list)
    excluded_colors: List[str] = field(default_factory=list)
    excluded_categories: List[str] = field(default_factory=list)
    confidence: float = 0.70
    provenance: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_query": self.raw_query,
            "normalized_query": self.normalized_query,
            "retrieval_query": self.retrieval_query,
            "language": self.language,
            "corrections": self.corrections,
            "product_type": self.product_type,
            "brand": self.brand,
            "color": self.color,
            "material": self.material,
            "size": self.size,
            "recipient": self.recipient,
            "target_audience": self.target_audience,
            "target_gender": self.target_gender,
            "intents": self.intents,
            "affinities": self.affinities,
            "min_price": self.min_price,
            "max_price": self.max_price,
            "currency": self.currency,
            "hard_constraints": self.hard_constraints,
            "soft_preferences": self.soft_preferences,
            "confidence": self.confidence,
        }


# ---------------------------------------------------------------------------
# QueryParser — the main NLP engine
# ---------------------------------------------------------------------------

_BANGLA_RANGE = "\u0980-\u09FF"

_COLOR_CANONICAL = {
    # English & phonetic Banglish
    "black": "black", "kalo": "black",
    "red": "red", "lal": "red",
    "blue": "blue", "nil": "blue", "neeli": "blue",
    "green": "green", "shobuj": "green", "sobuj": "green",
    "white": "white", "shada": "white",
    "yellow": "yellow", "holud": "yellow",
    "pink": "pink", "golapi": "pink",
    "orange": "orange", "komla": "orange",
    "brown": "brown", "badami": "brown",
    "grey": "grey", "gray": "grey", "dhusor": "grey",
    "purple": "purple", "beguni": "purple",
    "gold": "gold", "sonali": "gold",
    "silver": "silver", "rupali": "silver",
    "navy": "navy", "khaki": "khaki", "maroon": "maroon", "beige": "beige",
    # Native Bengali script
    "কালো": "black", "লাল": "red", "নীল": "blue", "সবুজ": "green",
    "সাদা": "white", "হলুদ": "yellow", "গোলাপি": "pink", "কমলা": "orange",
    "বাদামী": "brown", "বাদামি": "brown", "ধূসর": "grey", "বেগুনি": "purple",
    "সোনালী": "gold", "রূপালী": "silver", "নেভি": "navy",
}

_MATERIAL_CANONICAL = {
    # English & phonetic Banglish
    "cotton": "cotton", "suti": "cotton",
    "leather": "leather", "chamra": "leather",
    "wool": "wool", "poshom": "wool",
    "silk": "silk",
    "polyester": "polyester",
    "denim": "denim",
    "canvas": "canvas",
    "mesh": "mesh",
    "nylon": "nylon",
    "linen": "linen",
    "velvet": "velvet",
    "suede": "suede",
    "plastic": "plastic",
    "metal": "metal",
    "steel": "steel",
    "aluminium": "aluminium",
    "rubber": "rubber",
    "ceramic": "ceramic",
    "glass": "glass",
    "wood": "wood",
    "paper": "paper",
    # Native Bengali script
    "সুতি": "cotton", "চামড়া": "leather", "চামড়ার": "leather",
    "পশমি": "wool", "সিল্ক": "silk", "ডেনিম": "denim",
    "প্লাস্টিক": "plastic", "মেটাল": "metal", "স্টিল": "steel",
    "সিরামিক": "ceramic", "কাচ": "glass", "কাঠ": "wood", "কাগজ": "paper",
}

_COLORS = list(_COLOR_CANONICAL.keys())
_MATERIALS = list(_MATERIAL_CANONICAL.keys())
_BANGLA_ALPHABET_RE = re.compile(f"[{_BANGLA_RANGE}]")

_BANGLA_SUFFIXES = [
    # Multi-char suffixes
    "গুলোর", "গুলির", "গুলো", "গুলি", "দেরকে", "দের", "খানা", "খানি", "টুকু", "সমূহ",
    # Genitive, Locative, Dative, Definite
    "য়ের", "ের", "তে", "য়ে", "কে", "টা", "টি", "র", "ে", "য়"
]


def stem_bangla(word: str) -> List[str]:
    """Generates candidate root stems for inflected Bangla words."""
    candidates = [word]
    for suf in _BANGLA_SUFFIXES:
        if word.endswith(suf) and len(word) > len(suf) + 1:
            candidates.append(word[:-len(suf)])
    return candidates


def levenshtein_dist(s1: str, s2: str) -> int:
    """Computes Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_dist(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            ins = prev[j + 1] + 1
            dele = curr[j] + 1
            sub = prev[j] + (c1 != c2)
            curr.append(min(ins, dele, sub))
        prev = curr
    return prev[-1]


_TYPO_MAP = {
    # Brands
    "adiddas": "adidas", "adydas": "adidas", "addidas": "adidas", "adida": "adidas", "adids": "adidas",
    "samsang": "samsung", "sam sung": "samsung", "samsumg": "samsung", "samsong": "samsung",
    "nik": "nike", "naik": "nike", "nkie": "nike",
    "kashio": "casio", "cacio": "casio", "casioo": "casio",
    "aple": "apple", "apel": "apple", "appil": "apple",
    "xiomi": "xiaomi", "redmi": "xiaomi", "realme": "realme",
    "arong": "aarong", "aron": "aarong", "arongg": "aarong",
    "bata": "bata", "batabd": "bata",
    "logitec": "logitech", "logitek": "logitech",

    # Tech & Electronics
    "mobail": "mobile", "mobil": "mobile", "mobile": "mobile", "phon": "phone", "fone": "phone", "fon": "phone",
    "smartphon": "smartphone", "samrtphone": "smartphone",
    "headphon": "wireless headphones", "hedphone": "wireless headphones", "hedfon": "wireless headphones",
    "hedfone": "wireless headphones", "headphone": "wireless headphones", "headphones": "wireless headphones",
    "blutooth": "bluetooth", "bluetoth": "bluetooth", "blututh": "bluetooth", "bluetoothh": "bluetooth",
    "wierless": "wireless", "wirless": "wireless", "wireles": "wireless",
    "airpode": "earbuds", "airpod": "earbuds", "airpods": "earbuds", "erpod": "earbuds",
    "erphone": "earbuds", "earphon": "earbuds", "earphone": "earbuds", "earphones": "earbuds",
    "soundbox": "bluetooth speaker", "spikar": "bluetooth speaker", "speeker": "bluetooth speaker",
    "leptop": "laptop", "laptap": "laptop", "labtop": "laptop",
    "tab": "tablet", "teb": "tablet",
    "smartwotch": "smartwatch", "smartwach": "smartwatch",
    "powerbank": "power bank", "handwatch": "watch",

    # Beauty & Skincare
    "lipstik": "lipstick", "shampo": "shampoo", "sunblock": "sunscreen", "facewash": "face wash",

    # Footwear & Apparel
    "pant": "trousers", "pants": "trousers", "pents": "trousers", "pent": "trousers",
    "shoos": "shoes", "shoo": "shoes", "soes": "shoes",
    "snikers": "sneakers", "sneaker": "sneakers", "sniker": "sneakers", "snekers": "sneakers",
    "trainer": "running shoes", "trainers": "running shoes", "jogging": "running shoes", "joging": "running shoes",
    "runing": "running", "runing shoes": "running shoes",
    "shart": "shirt", "shrt": "shirt",
    "tshit": "t-shirt", "tshir": "t-shirt", "teeshirt": "t-shirt", "t-shart": "t-shirt", "tshart": "t-shirt", "tshirt": "t-shirt",
    "punjaby": "panjabi", "panjabe": "panjabi", "panzabi": "panjabi", "punjabi": "panjabi",
    "shari": "saree", "sari": "saree", "sharee": "saree", "shadi": "saree",
    "kurty": "kurti", "kurte": "kurti",
    "kamej": "salwar kameez", "kamij": "salwar kameez", "kamiz": "salwar kameez", "kameez": "salwar kameez",
    "lungy": "lungi",
    "gamchaa": "gamcha",
    "juta": "shoes", "juto": "shoes", "jutta": "shoes", "jutha": "shoes", "joota": "shoes", "jooto": "shoes",
    "chappal": "sandals", "sendel": "sandals", "chappals": "sandals", "slipper": "sandals", "slippers": "sandals",
    "sunglas": "sunglasses", "sunglass": "sunglasses", "shades": "sunglasses",
    "chasma": "sunglasses", "chosma": "sunglasses", "chashma": "sunglasses",
    "wotch": "wristwatch", "hand wotch": "wristwatch", "ghori": "wristwatch", "ghadi": "wristwatch",
    "jaket": "jacket", "jakit": "jacket",
    "hudi": "hoodie", "hoody": "hoodie",
    "jinse": "jeans", "jin": "jeans",

    # Accessories, Kitchen & Kids
    "churi": "kitchen knife", "chhuri": "kitchen knife", "chaku": "kitchen knife", "knife": "kitchen knife", "knives": "kitchen knife", "knif": "kitchen knife",
    "lethar": "leather", "lether": "leather", "chomra": "leather", "chamra": "leather",
    "wallat": "leather wallet", "walet": "leather wallet",
    "botol": "stainless steel water bottle", "bottol": "stainless steel water bottle", "botal": "stainless steel water bottle",
    "toddlar": "toddler", "todler": "toddler",
    "bilding": "building", "bild": "build",
    "bloks": "blocks", "blok": "block",
    "protekt": "protect", "protec": "protect",
    "beg": "backpack", "byag": "backpack", "bagpack": "backpack",
    "khelna": "educational toy",
    # Bicycle / bike typos
    "byke": "bicycle", "baike": "bicycle", "bycycle": "bicycle", "bicycal": "bicycle",
    "saikel": "bicycle", "saikil": "bicycle", "motar": "motor",
}

_BANGLA_SPELLING_VARIANTS = {
    "প্যান্ট": "trousers", "পেন্ট": "trousers", "পায়জামা": "trousers", "পাজামা": "trousers",
    "ছুরি": "kitchen knife", "ছুড়ি": "kitchen knife", "চাকু": "kitchen knife", "ছোরা": "kitchen knife",
    "পাঞ্জাবী": "পাঞ্জাবি", "পাঞ্জাবীর": "পাঞ্জাবি",
    "শাড়ী": "শাড়ি", "শাড়ীর": "শাড়ি", "শাড়ি": "শাড়ি",
    "ঘড়ী": "ঘড়ি", "ঘড়ির": "ঘড়ি",
    "জুতা": "জুতো", "জুতার": "জুতো",
    "স্নিকার্স": "স্নিকার",
    "টি শার্ট": "টি-শার্ট", "টিশার্ট": "টি-শার্ট", "টি-শার্টের": "টি-শার্ট",
    "মোবাইল": "mobile", "ফোন": "phone", "স্মার্টফোন": "smartphone",
    "ল্যাপটপ": "laptop", "ট্যাব": "tablet",
    "হেডফোন": "headphones", "ব্লুটুথ": "bluetooth",
    "ঘড়ি": "ঘড়ি", "চশমা": "সানগ্লাস",
    "ব্যাগ": "ব্যাগ", "মানিব্যাগ": "leather wallet",
    "সুতি": "cotton", "চামড়া": "leather", "চামড়ার": "leather",
    "বোটল": "water bottle", "পানির বোতল": "water bottle",
    "সাইকেল": "bicycle", "বাইসাইকেল": "bicycle"
}

_CANONICAL_TARGETS = [
    "samsung", "apple", "xiaomi", "nike", "adidas", "puma", "casio", "sony", "hp", "dell",
    "lenovo", "aarong", "taaga", "bata", "apex", "lotto", "logitech", "philips", "bosch",
    "dewalt", "makita", "fluke", "fossil", "fastrack", "gree", "dyson", "panasonic", "walton",
    "oraimo", "nivea", "anker", "baseus", "hoco", "awei", "colmi",
    "smartphone", "phone", "laptop", "tablet", "webcam", "headphones", "earbuds", "bluetooth", "wireless",
    "speaker", "smartwatch", "running shoes", "sneakers", "shoes", "sandals", "slippers",
    "boots", "panjabi", "saree", "kurti", "shirt", "t-shirt", "jacket", "hoodie", "jeans",
    "trousers", "wallet", "wristwatch", "sunglasses", "backpack", "water bottle", "coffee mug",
    "kitchen knife", "knife", "bicycle", "cycle", "helmet", "camera", "perfume", "router",
    "umbrella", "tent", "tea", "rice", "cookware", "blender", "iron", "fan", "refrigerator",
    "lipstick", "shampoo", "sunscreen", "face wash", "serum", "moisturizer", "charger", "adapter", "trimmer", "cover",
    "football", "dumbbells", "yoga mat", "badminton racket", "cricket bat",
    "building blocks", "educational toy", "electric toothbrush", "hair dryer",
    "leather", "cotton", "denim", "canvas", "black", "white", "blue", "red", "green", "yellow",
    "toddler", "building", "blocks", "protect"
]

# Comprehensive common English words vocabulary to protect valid real words from false typo correction
_ENGLISH_COMMON_WORDS = {
    "bike", "bikes", "bicycle", "bicycles", "cycling", "cycle", "motorcycle", "biker", "cyclist",
    "bell", "pure", "pool", "cool", "tool", "wool", "fool", "milk", "silk", "rice", "mice", "nice",
    "game", "ball", "wall", "hall", "mall", "tall", "fall", "call", "card", "cord", "hard", "yard",
    "book", "cook", "look", "hook", "like", "hike", "mike", "lake", "bake", "cake", "make", "take",
    "wake", "wire", "fire", "tire", "hire", "suit", "coat", "boat", "goat", "desk", "chair", "table",
    "door", "floor", "gold", "silver", "wood", "steel", "iron", "lamp", "light", "plate", "bowl", "cup",
    "dish", "soap", "hair", "face", "body", "hand", "foot", "leg", "arm", "head", "eye", "back", "bed",
    "sofa", "room", "home", "house", "shop", "store", "price", "sale", "deal", "pack", "set", "kit",
    "size", "fit", "look", "wear", "walk", "run", "play", "work", "live", "stay", "free", "fast", "slow",
    "hard", "soft", "warm", "cold", "hot", "cool", "dry", "wet", "clean", "fresh", "dark", "deep", "high",
    "low", "long", "short", "wide", "thin", "thick", "fine", "good", "best", "great", "well", "real",
    "true", "easy", "safe", "rich", "poor", "full", "half", "part", "side", "top", "end", "line", "point",
    "spot", "band", "ring", "chain", "belt", "strap", "case", "box", "bag", "wrap", "hold", "stand",
    "rest", "stop", "move", "turn", "open", "shut", "lock", "key", "cut", "drop", "rise", "lift", "carry",
    "keep", "give", "bring", "send", "show", "find", "lose", "gain", "win", "beat", "hit", "kick", "ride",
    "drive", "fly", "sail", "swim", "jump", "climb", "hang", "pull", "push", "press", "roll", "spin",
    "blow", "burn", "boil", "fry", "mix", "pour", "wash", "wipe", "brush", "comb", "draw", "paint",
    "write", "read", "hear", "listen", "speak", "talk", "tell", "say", "ask", "cry", "laugh", "smile",
    "care", "help", "save", "serve", "feed", "meet", "join", "leave", "wait", "start", "done", "used",
    "made", "born", "grow", "fill", "empty", "close", "hide", "seek", "hunt", "catch", "trap", "kill",
    "feel", "think", "know", "mean", "wish", "hope", "fear", "love", "hate", "want", "need", "have",
    "own", "spend", "earn", "lay", "sit", "lie", "sleep", "eat", "drink", "dress", "neat", "tidy",
    "dirty", "clear", "bright", "dull", "dim", "faint", "loud", "quiet", "still", "calm", "rough",
    "wild", "gentle", "kind", "sweet", "sour", "bitter", "salty", "stale", "ripe", "raw", "damp",
    "heavy", "firm", "tight", "loose", "stiff", "limp", "tough", "weak", "strong", "sharp", "flat",
    "round", "square", "straight", "bent", "curved", "solid", "broad", "narrow", "shallow", "far",
    "near", "early", "late", "quick", "swift", "prompt", "ready", "able", "apt", "due", "bound", "sure",
    "false", "fake", "right", "wrong", "fair", "foul", "proud", "humble", "bold", "brave", "shy", "candid",
    "smart", "clever", "wise", "glad", "merry", "jolly", "sad", "grim", "stern", "fond", "dear", "rare",
    "common", "plain", "bare", "sheer", "mere", "sole", "lone", "alone", "whole", "each", "every", "both",
    "either", "neither", "some", "any", "none", "one", "two", "three", "four", "five", "six", "seven",
    "eight", "nine", "ten", "while", "where", "which", "whose", "with", "thing", "things", "phone",
    "time", "item", "someone", "something", "weather", "summer", "winter", "autumn", "spring", "rain",
    "rainy", "casual", "formal", "office", "party", "travel", "college", "school", "waterproof", "water",
    "daily", "everyday", "better", "cheap", "affordable", "gift", "present", "comfortable", "walking",
    "running", "jogging", "training", "exercise", "workout", "fitness", "budget", "performance", "men",
    "women", "kids", "boy", "boys", "girl", "girls", "baby", "adult", "teen", "student", "teacher",
    "father", "mother", "sister", "brother", "friend", "wife", "husband", "son", "daughter"
}


_B = r"(?<![\w\u0980-\u09FF])"
_E = r"(?![\w\u0980-\u09FF])"

_EN_PHRASE_MAPS = [
    (r"\bnotebook\s+(?:computer|pc)\b", "laptop"),
    (r"\btee\s*shirt\b", "t-shirt"),
    (r"\b(?:mountain\s+)?bicycles?\b", "bicycle"),
    (r"\b(?:mountain\s+)?bikes?\b", "bicycle"),
    (rf"{_B}(?:বাইসাইকেল|সাইকেল){_E}", "bicycle"),
    (r"\b(?:apple\s+)?iphones?\b(?!\s+(?:case|cover|protector|charger|cable|holder|stand))", "Apple smartphone iPhone"),
    # Functional / semantic descriptions -> clear product concepts
    (r"\b(?:something\s+to\s+)?protect\s+(?:my\s+)?skin\s+from\s+(?:the\s+)?sun\b", "sunscreen"),
    (r"\b(?:gadget\s+to\s+)?trim\s+(?:my\s+)?beard\b", "beard trimmer"),
    (r"\b(?:to\s+)?keep\s+(?:the\s+)?room\s+cool(?:\s+in\s+summer)?\b", "air conditioner air cooler fan"),
    (r"\bcover\s+to\s+protect\s+(?:my\s+)?(?:phone|mobile|smartphone)\b", "phone case"),
    (r"\bcharger\s+to\s+charge\s+(?:my\s+)?(?:phone|mobile|smartphone)(?:\s+fast)?\b", "fast phone charger"),
    # Accessory compounds BEFORE phone/mobile rules
    (r"\b(?:phone|mobile|cellphone)\s+(?:er\s+)?(?:case|cover|back\s*cover)\b", "phone case"),
    (r"\b(?:phone|mobile)\s+(?:er\s+)?(?:charger|adapter)\b", "phone charger"),
    (r"\b(?:case|cover|back\s*cover)\b", "phone case"),
    (r"\bsmart\s+watch\b", "smartwatch"),
    (r"\b(macbook(?:\s+air|\s+pro)?)\b", r"Apple laptop \1"),
    (r"\b(galaxy\s+s\d+(?:\s+\w+)?)\b", r"Samsung smartphone \1"),
    (r"\binverter\s+ac\b", "inverter air conditioner"),
    (r"\bac\b", "air conditioner"),
    (r"\b(?:charge(?:\s+\w+)*\s+on\s+the\s+go|portable\s+charger)\b", "power bank"),
    (r"\b(?:shoes?\s+(?:for\s+)?(?:long\s+distance\s+)?(?:running|jogging)|(?:running|jogging)\s+shoes?|trainers\s+for\s+jogging)\b", "running shoes"),
    (r"\bcomfortable\s+shoes\s+for\s+(?:long\s+distance\s+)?running\b", "comfortable running shoes"),
    (r"\btrainers\b", "running shoes"),
    (r"\bjogging\b", "running shoes"),
    (r"ভালো\s+পারফরম্?য?য?া?ন্সের\s+ল্যাপটপ", "high performance laptop"),
    (r"\bplaystation\b", "uncatalogued_item_playstation"),
    (r"\btesla\b", "uncatalogued_item_tesla"),
    # Bengali Brand Names (boundary-anchored)
    (rf"{_B}(?:ওয়ালটন|ওয়ালটন){_E}", "Walton"),
    (rf"{_B}(?:সিম্ফনি|সিম্ফোনি){_E}", "Symphony"),
    (rf"{_B}(?:আড়ং|আড়ং){_E}", "Aarong"),
    (rf"{_B}এপেক্স{_E}", "Apex"),
    (rf"{_B}বাটা{_E}", "Bata"),
    (rf"{_B}লোটো{_E}", "Lotto"),
    (rf"{_B}সেইলর{_E}", "Sailor"),
    (rf"{_B}(?:ইয়েলো|ইয়েলো){_E}", "Yellow"),
    (rf"{_B}রিচম্যান{_E}", "Richman"),
    (rf"{_B}ক্যাটস\s*আই{_E}", "Cats Eye"),
    (rf"{_B}কে\s*ক্রাফট{_E}", "Kay Kraft"),
    (rf"{_B}অঞ্জনস{_E}", "Anjans"),
    (rf"{_B}পাকিজা{_E}", "Pakiza"),
    (rf"{_B}আমানত\s*শাহ{_E}", "Amanat Shah"),
    (rf"{_B}প্রাণ{_E}", "Pran"),
    (rf"{_B}(?:রাঁধুনী|রাধুনী){_E}", "Radhuni"),
    (rf"{_B}(?:বাঘাবাড়ী|বাঘাবাড়ী){_E}", "Baghabari"),
    (rf"{_B}ইস্পাহানি{_E}", "Ispahani"),
    (rf"{_B}টাইটান{_E}", "Titan"),
    (rf"{_B}ক্যাসিও{_E}", "Casio"),
    (rf"{_B}ফাস্টট্র্যাক{_E}", "Fastrack"),
    (rf"{_B}স্যামসাং{_E}", "Samsung"),
    (rf"{_B}অ্যাপল{_E}", "Apple"),
    # Bengali Category / Product Phrases (boundary-anchored; accessories before phone)
    (rf"{_B}(?:ফোনের|মোবাইলের|ফোন|মোবাইল)\s*কভার{_E}", "phone case"),
    (rf"{_B}(?:ফোনের|মোবাইলের|ফোন|মোবাইল)\s*চার্জার{_E}", "phone charger"),
    (rf"{_B}(?:ফেস\s*ওয়াশ|ফেস\s*ওয়াশ|ফেসওয়াশ|ফেসওয়াশ){_E}", "face wash"),
    (rf"{_B}(?:সানস্ক্রিন|সান\s*স্ক্রিন|সানব্লক){_E}", "sunscreen"),
    (rf"{_B}লিপস্টিক{_E}", "lipstick"),
    (rf"{_B}শ্যাম্পু{_E}", "shampoo"),
    (rf"{_B}সিরাম{_E}", "serum"),
    (rf"{_B}ময়েশ্চারাইজার{_E}", "moisturizer"),
    (rf"{_B}(?:ইয়ারবাডস|ইয়ারবাড|ইয়ারবাডস|ইয়ারবাড|ইয়ারফোন|ইয়ারফোন){_E}", "earbuds"),
    (rf"{_B}হেডফোন{_E}", "headphones"),
    (rf"{_B}(?:ব্লুটুথ\s*স্পিকার|সাউন্ডবক্স){_E}", "bluetooth speaker"),
    (rf"{_B}(?:স্মার্টওয়াচ|স্মার্টওয়াচ|স্মার্ট\s*ওয়াচ|স্মার্ট\s*ওয়াচ|স্মার্ট\s*ঘড়ি|স্মার্ট\s*ঘড়ি){_E}", "smartwatch"),
    (rf"{_B}(?:হাতঘড়ি|হাতঘড়ি|ঘড়ি|ঘড়ি){_E}", "wristwatch"),
    (rf"{_B}(?:রোদচশমা|সানগ্লাস){_E}", "sunglasses"),
    (rf"{_B}(?:পাওয়ার\s*ব্যাংক|পাওয়ার\s*ব্যাংক){_E}", "power bank"),
    (rf"{_B}স্মার্টফোন{_E}", "smartphone"),
    (rf"{_B}(?:মোবাইল\s*ফোন|মোবাইল|ফোন|ফোনের){_E}", "phone"),
    (rf"{_B}রাইস\s*কুকার{_E}", "rice cooker"),
    (rf"{_B}(?:ইন্ডাকশন\s*কুকার|চুলা){_E}", "induction cooker"),
    (rf"{_B}প্রেসার\s*কুকার{_E}", "pressure cooker"),
    (rf"{_B}(?:ইলেকট্রিক\s*কেটলি|কেটলি|কেটলী){_E}", "electric kettle"),
    (rf"{_B}(?:ফুড\s*প্রসেসর|চপার|ব্লেন্ডার){_E}", "blender"),
    (rf"{_B}(?:ড্রাই\s*আয়রন|ইস্ত্রি){_E}", "iron"),
    (rf"{_B}(?:সিলিং\s*ফ্যান|টেবিল\s*ফ্যান|ফ্যান|পাখা){_E}", "fan"),
    (rf"{_B}(?:রেফ্রিজারেটর|ফ্রিজ){_E}", "refrigerator"),
    (rf"{_B}(?:পানির\s*বোতল|বোতল){_E}", "stainless steel water bottle"),
    (rf"{_B}(?:মানিব্যাগ|ওয়ালেট){_E}", "leather wallet"),
    (rf"{_B}(?:ব্যাকপ্যাক|ল্যাপটপ\s*ব্যাগ|ব্যাগ){_E}", "backpack"),
    (rf"{_B}(?:ট্রাভেল\s*ব্যাগ|ট্রলি|স্যুটকেস){_E}", "travel bag"),
    (rf"{_B}(?:নকশী\s*কাঁথা|নকশি\s*কাঁথা|কাঁথা){_E}", "nakshi kantha"),
    (rf"{_B}(?:জায়নামাজ|জায়নামায|জায়নামাজ){_E}", "prayer mat"),
    (rf"{_B}(?:তসবিহ|তসবি){_E}", "tasbih"),
    (rf"{_B}(?:লুঙ্গি|লুঙ্গী){_E}", "lungi"),
    (rf"{_B}গামছা{_E}", "gamcha"),
    (rf"{_B}(?:পাঞ্জাবি|পাঞ্জাবী){_E}", "panjabi"),
    (rf"{_B}(?:শাড়ি|শাড়ী|শাড়ি|শাড়ী){_E}", "saree"),
    (rf"{_B}(?:থ্রি\s*পিস|সালোয়ার\s*কামিজ|কামিজ){_E}", "salwar kameez"),
    (rf"{_B}কুর্তি{_E}", "kurti"),
    (rf"{_B}(?:ফতুয়া|ফতুয়া){_E}", "fotua"),
    (rf"{_B}(?:পায়জামা|পাজামা){_E}", "pajamas"),
    (rf"{_B}(?:স্যান্ডেল|স্যান্ডাল){_E}", "sandals"),
    (rf"{_B}(?:স্নিকার্স|স্নিকার|কেডস){_E}", "sneakers"),
    (rf"{_B}(?:রানিং\s*জুতা|জুতো|জুতা){_E}", "running shoes"),
    (rf"{_B}ফরমাল\s*জুতা{_E}", "formal shoes"),
    (rf"{_B}(?:টি-শার্ট|টিশার্ট){_E}", "t-shirt"),
    (rf"{_B}শার্ট{_E}", "shirt"),
    (rf"{_B}(?:জিন্স\s*প্যান্ট|জিন্স){_E}", "jeans"),
    (rf"{_B}(?:ট্রাউজার|প্যান্ট){_E}", "trousers"),
    (rf"{_B}জ্যাকেট{_E}", "jacket"),
    (rf"{_B}হুডি{_E}", "hoodie"),
    (rf"{_B}মধু{_E}", "honey"),
    (rf"{_B}(?:সরিষার\s*তেল|তেল){_E}", "mustard oil"),
    (rf"{_B}ঘি{_E}", "ghee"),
    (rf"{_B}(?:চা|টি){_E}", "tea"),
    (rf"{_B}(?:পোলাও\s*চাল|চাল){_E}", "rice"),
    (rf"{_B}(?:মসলা|আচার){_E}", "spices"),
    (rf"{_B}(?:টিকিট|টিকেট){_E}", "ticket"),
    (rf"{_B}(?:বিমানের|বিমান){_E}", "airplane"),
]
_COMPILED_EN_PHRASE_MAPS = [(re.compile(pat, flags=re.IGNORECASE), rep) for pat, rep in _EN_PHRASE_MAPS]


class QueryParser:
    """Full Bangla/Banglish/English query parser with structured intent extraction."""

    def __init__(self, domain_pack: Optional[DomainPack] = None):
        self.dp = domain_pack or DomainPack.default()
        self.colors = _COLORS
        self.materials = _MATERIALS

        # Precompiled patterns for attributes & categories (compiled once)
        self._color_patterns: List[Tuple[re.Pattern, str, str]] = [
            (re.compile(rf"(?<![\w\u0980-\u09FF]){re.escape(c_tok)}(?![\w\u0980-\u09FF])"), c_tok, c_canon)
            for c_tok, c_canon in _COLOR_CANONICAL.items()
        ]
        self._material_patterns: List[Tuple[re.Pattern, str, str]] = [
            (re.compile(rf"(?<![\w\u0980-\u09FF]){re.escape(m_tok)}(?![\w\u0980-\u09FF])"), m_tok, m_canon)
            for m_tok, m_canon in _MATERIAL_CANONICAL.items()
        ]
        self._leaf_patterns: List[Tuple[re.Pattern, str]] = [
            (re.compile(rf"(?<![\w\u0980-\u09FF]){re.escape(leaf)}(?![\w\u0980-\u09FF])"), leaf)
            for leaf in sorted(self.dp.leaf_to_group.keys(), key=lambda x: -len(x))
        ]
        self._men_pats = [re.compile(rf"(?<![\w\u0980-\u09FF]){re.escape(w)}(?![\w\u0980-\u09FF])")
                          for w in ("men", "man", "male", "purush", "chele", "cheler", "cheleder", "ছেলেদের", "পুরুষ")]
        self._women_pats = [re.compile(rf"(?<![\w\u0980-\u09FF]){re.escape(w)}(?![\w\u0980-\u09FF])")
                            for w in ("women", "woman", "female", "mohila", "meye", "meyer", "meyeder", "মেয়েদের", "মহিলা")]
        self._kids_pats = [re.compile(rf"(?<![\w\u0980-\u09FF]){re.escape(w)}(?![\w\u0980-\u09FF])")
                           for w in ("kids", "boy", "girl", "child", "children", "baccha", "choto", "বাচ্চাদের", "ছোটদের")]

        # Query parsing memoization cache
        self._parse_cache: Dict[str, StructuredIntent] = {}

        # Dynamically populated from the active Catalog via set_catalog()
        self.catalog_words: set = set()
        self.catalog_brands: Dict[str, str] = {}
        self.dynamic_targets: List[str] = list(_CANONICAL_TARGETS)

    def set_catalog(self, catalog) -> None:
        """Bind active Catalog vocabulary so typo correction & negative exclusions stay catalog-agnostic."""
        self.catalog_words = set()
        self.catalog_brands = {}
        targets = set(_CANONICAL_TARGETS)
        if catalog is not None:
            for w, cnt in catalog.doc_freq.items():
                if cnt >= 3:
                    self.catalog_words.add(w.lower())
                if cnt >= 5 and w.isalpha() and len(w) >= 5:
                    targets.add(w.lower())
            from .parser import _NOISY_BRANDS
            for b in catalog.vocab_values("brand"):
                bl = b.strip().lower()
                if len(bl) >= 2 and bl not in _NOISY_BRANDS:
                    self.catalog_brands[bl] = b.strip()
                    self.catalog_words.add(bl)
                    if bl.isalpha() and len(bl) >= 5:
                        targets.add(bl)
            for c_val in catalog.cat_uni:
                for tok in re.findall(r"[a-z]+", str(c_val).lower()):
                    if len(tok) >= 5:
                        targets.add(tok)
        self.dynamic_targets = sorted(targets)

    # ------------------------------------------------------------------
    # Typo correction & Text normalization
    # ------------------------------------------------------------------

    def correct_typos(self, text: str) -> Tuple[str, List[str]]:
        """Correct misspelled or phonetically distorted words in English, Banglish, and Bangla."""
        corrections = []
        tokens = text.split()
        out_tokens = []
        for t in tokens:
            t_low = t.lower()
            # 1. Exact Bangla spelling variant
            if t in _BANGLA_SPELLING_VARIANTS:
                canon = _BANGLA_SPELLING_VARIANTS[t]
                if canon != t:
                    corrections.append(f"'{t}' -> '{canon}'")
                out_tokens.append(canon)
                continue
            # 2. Known typo dictionary
            if t_low in _TYPO_MAP:
                canon = _TYPO_MAP[t_low]
                corrections.append(f"'{t}' -> '{canon}'")
                out_tokens.append(canon)
                continue
            # 3. Known brand aliases
            if t_low == "iphone":
                out_tokens.append("iPhone")
                continue
            if t_low in self.dp.brand_aliases:
                canon = self.dp.brand_aliases[t_low]
                if canon.lower() != t_low:
                    corrections.append(f"'{t}' -> '{canon}'")
                out_tokens.append(canon)
                continue
            # 4. Known aliases
            if t_low in self.dp.aliases:
                canon = self.dp.aliases[t_low]
                if canon.lower() != t_low:
                    corrections.append(f"'{t}' -> '{canon}'")
                out_tokens.append(canon)
                continue

            # 5. Preserve persona keywords and known real English, catalog, or Bangla words
            is_persona = t_low.startswith(self.dp.persona_alias_prefixes)
            is_known = (
                is_persona
                or t in self.dp.bangla_dict
                or t_low in ("baba", "babar", "ma", "ammu", "abbu", "vai", "bhai", "apu", "bon", "chele", "meye", "baccha", "bachcha")
                or t_low in self.catalog_words
                or t_low in _ENGLISH_COMMON_WORDS
                or t_low in self.dp.aliases
                or t_low in self.dp.brand_aliases
                or bool(_BANGLA_ALPHABET_RE.search(t))
            )
            if is_known:
                out_tokens.append(t)
                continue

            # 6. Guarded fuzzy matching against canonical + active catalog terms
            # ONLY for words >= 5 letters that are NOT recognized in English or catalog vocabulary
            if len(t_low) >= 5 and t_low.isalpha():
                best_match = None
                min_d = 99
                for target in self.dynamic_targets:
                    # Target must share the initial character to preserve phonetic integrity
                    if not target or target[0] != t_low[0]:
                        continue
                    if abs(len(t_low) - len(target)) > 1:
                        continue
                    d = levenshtein_dist(t_low, target)
                    if d < min_d:
                        min_d = d
                        best_match = target
                max_allowed = 1 if len(t_low) <= 7 else 2
                if best_match and min_d <= max_allowed and best_match != t_low:
                    corrections.append(f"'{t}' -> '{best_match}'")
                    out_tokens.append(best_match)
                    continue

            out_tokens.append(t)
        return " ".join(out_tokens), corrections

    def normalize_text(self, text: str) -> Tuple[str, str, List[str]]:
        """Convert Bangla digits, correct typos, apply multi-word and word-level Bangla translations."""
        text = unicodedata.normalize("NFC", text)
        text = self.dp.convert_bangla_digits(text)
        has_bangla = bool(_BANGLA_ALPHABET_RE.search(text))

        # 1. Multi-word English / model phrase mappings (before token-level typo correction)
        for pat, rep in _COMPILED_EN_PHRASE_MAPS:
            text = pat.sub(rep, text)

        # 2. Typo correction & normalization
        text, corrections = self.correct_typos(text)

        # 3. Multi-word phrases in bangla_dict first (longest phrases first, boundary-safe)
        for bn_word, en_trans in self.dp.bangla_dict_items:
            if " " in bn_word and bn_word in text:
                text = re.sub(rf"{_B}{re.escape(bn_word)}{_E}", en_trans, text)
                has_bangla = True

        # 4. Token-level dictionary lookup with stemming
        tokens = text.split()
        out = []
        for tok in tokens:
            mapped = self.dp.bangla_dict.get(tok)
            if not mapped and _BANGLA_ALPHABET_RE.search(tok):
                for stem in stem_bangla(tok):
                    if stem in self.dp.bangla_dict:
                        mapped = self.dp.bangla_dict[stem]
                        break
            if mapped:
                out.append(mapped)
                has_bangla = True
            else:
                out.append(tok)
        normalized = " ".join(out).lower().strip()
        normalized = re.sub(r"\s+", " ", normalized)
        lang = "bn" if has_bangla else "en"
        return normalized, lang, corrections

    # ------------------------------------------------------------------
    # Price extraction
    # ------------------------------------------------------------------

    def extract_price(self, text: str) -> Tuple[Optional[float], Optional[float], str]:
        stripped = text
        min_p: Optional[float] = None
        max_p: Optional[float] = None

        CURR = r"(?:tk|taka|takar|bdt|টাকা|টাকার|টাকায়)"

        # Range: "between X and Y" / "X to Y" / "X theke Y" / "X theke Y taka" / "X থেকে Y টাকা"
        range_match = re.search(
            rf"(?:between|from|theke|থেকে)?\s*(\d+)\s*(?:and|to|-|theke|থেকে)\s*(\d+)\s*{CURR}?",
            stripped, re.IGNORECASE,
        )
        if range_match:
            p1, p2 = float(range_match.group(1)), float(range_match.group(2))
            min_p, max_p = min(p1, p2), max(p1, p2)
            stripped = stripped[: range_match.start()] + " " + stripped[range_match.end() :]

        # Max price / under
        if max_p is None:
            under_patterns = [
                rf"(?:under|below|less than|max|maximum|budget|upto|within)\s*[:=]?\s*(\d+)\s*{CURR}?",
                rf"(\d+)\s*{CURR}?\s*(?:er niche|niche|er kom|kom|moddhe|er moddhe|er vitor|ভিতর|নিচে|কম|মধ্যে|ভেতর|পর্যন্ত)",
                rf"(\d+)\s*{CURR}\s*(?:budget)?",
                rf"\b(?:ac|air\s*conditioner|laptop|phone|smartphone|tv|fridge|refrigerator|panjabi|saree|shoes?)\s+(\d{{4,7}})\b",
            ]
            for pat in under_patterns:
                m = re.search(pat, stripped, re.IGNORECASE)
                if m:
                    max_p = float(m.group(1))
                    stripped = stripped[: m.start()] + " " + stripped[m.end() :]
                    break

        # Min price / above
        if min_p is None:
            above_patterns = [
                rf"(?:above|over|more than|min|minimum)\s*[:=]?\s*(\d+)\s*{CURR}?",
                rf"(\d+)\s*{CURR}?\s*(?:er beshi|beshi|er upore|upore|বেশি|উপরে)",
            ]
            for pat in above_patterns:
                m = re.search(pat, stripped, re.IGNORECASE)
                if m:
                    min_p = float(m.group(1))
                    stripped = stripped[: m.start()] + " " + stripped[m.end() :]
                    break

        stripped = re.sub(r"\s+", " ", stripped).strip()
        return min_p, max_p, stripped

    # ------------------------------------------------------------------
    # Size extraction
    # ------------------------------------------------------------------

    def extract_size(self, text: str) -> Tuple[Optional[str], str]:
        stripped = text
        m = re.search(r"(?<![\w\u0980-\u09FF])(?:size|সাইজ|মাপ)\s*[:=]?\s*([a-zA-Z0-9]+)(?![\w\u0980-\u09FF])", stripped, re.IGNORECASE)
        if m:
            size_val = m.group(1).upper()
            stripped = stripped[: m.start()] + " " + stripped[m.end() :]
        else:
            m2 = re.search(r"\b(XXL|XL|XS|[SML])\b", stripped)
            if m2 and any(
                k in stripped.lower()
                for k in ("shirt", "t-shirt", "panjabi", "hoodie", "jacket", "for men", "for women")
            ):
                size_val = m2.group(1).upper()
                stripped = stripped[: m2.start()] + " " + stripped[m2.end() :]
            else:
                size_val = None
        stripped = re.sub(r"\s+", " ", stripped).strip()
        return size_val, stripped

    # ------------------------------------------------------------------
    # Full parse pipeline
    # ------------------------------------------------------------------

    def parse(self, raw_query: str) -> StructuredIntent:
        """Parse raw query → StructuredIntent with all attributes extracted."""
        cache_key = raw_query.strip().lower()
        if cache_key in self._parse_cache:
            return copy.deepcopy(self._parse_cache[cache_key])

        norm_query, lang, corrections = self.normalize_text(raw_query)

        # 1. Price
        min_p, max_p, q_no_price = self.extract_price(norm_query)

        # 2. Size
        size_val, q_no_size = self.extract_size(q_no_price)

        # 3. Persona / recipient
        persona_info = self.dp.match_persona(norm_query)
        recipient = persona_info["persona"] if persona_info else None
        target_audience = persona_info["target_audience"] if persona_info else None
        target_gender = persona_info["target_gender"] if persona_info else None

        # Explicit gender
        q_low = norm_query.lower()
        if any(p.search(q_low) for p in self._men_pats):
            target_gender = "men"
        elif any(p.search(q_low) for p in self._women_pats):
            target_gender = "women"
        elif any(p.search(q_low) for p in self._kids_pats):
            target_audience = "kids"

        if target_gender in ("boys", "girls") and size_val in ("S", "M", "L", "XL", "XXL", "38", "40", "42", "44"):
            target_gender = "men" if target_gender == "boys" else "women"

        # 4. Intents
        intents_info = self.dp.match_intent(norm_query)
        intent_names = [i["intent"] for i in intents_info]

        # 5. Affinities (persona + intent combined)
        affinities: List[str] = []
        if persona_info:
            affinities.extend(persona_info.get("affinities", []))
        for item in intents_info:
            item_affs = item.get("affinities", [])
            # Adult/parent gifts should not be diluted by children's toys
            if recipient in ("mother", "father", "wife", "husband") or target_audience in ("adult", "women", "men") or target_gender in ("women", "men"):
                item_affs = [a for a in item_affs if a not in ("building blocks", "educational toy", "toy car", "toy")]
            affinities.extend(item_affs)
        
        # Dedicated charging / power bank intent
        if re.search(r"\b(?:charge(?:\s+\w+)*\s+(?:on\s+the\s+go|travell?ing|trip|road)|portable\s+charger|power\s*bank)\b", norm_query.lower()):
            affinities.insert(0, "power bank")
            if "travel" not in intent_names:
                intent_names.append("travel")
        
        # Casual intent should only apply affinities when query does not explicitly specify a non-apparel product
        has_non_apparel = any(re.search(rf"\b{re.escape(term)}\b", norm_query.lower()) for term in (
            "phone", "smartphone", "laptop", "computer", "pc", "ac", "air conditioner", "fan",
            "refrigerator", "fridge", "tv", "television", "power bank", "headphone", "headphones",
            "earbud", "earbuds", "earphone", "earphones", "microwave", "blender", "kettle"
        ))
        if has_non_apparel and "casual" in intent_names:
            intent_names.remove("casual")
            clothing_affs = {"t-shirt", "jeans", "sneakers", "shirt", "hoodie", "shorts", "sandals"}
            affinities = [a for a in affinities if a not in clothing_affs]

        # Room/air cooling queries should not trigger summer apparel (t-shirt/sandals)
        if "room_cooling" in intent_names or any(w in norm_query.lower() for w in ("room", "air cool", "cooling")):
            if "summer" in intent_names:
                intent_names.remove("summer")
            apparel_affs = {"t-shirt", "jeans", "sneakers", "shirt", "hoodie", "shorts", "sandals", "cotton", "sunglasses"}
            affinities = [a for a in affinities if a not in apparel_affs]

        affinities = list(dict.fromkeys(affinities))

        # Negative exclusions extraction (e.g., 'not Nike', 'except Apple', 'not black', 'ছাড়া', 'ব্যতীত')
        excluded_brands: List[str] = []
        excluded_colors: List[str] = []
        excluded_categories: List[str] = []

        neg_patterns = [
            r"\b(?:not|except|without|no)\s+([a-zA-Z0-9\u0980-\u09FF\s\-]+?)(?=\s+(?:under|below|above|between|for|size|with|and|,|\.|$)|$)",
            r"([a-zA-Z0-9\u0980-\u09FF]+)\s+(?:ছাড়|ছাড়া|ব্যতীত|বিনা)",
        ]
        neg_matches = []
        for pat in neg_patterns:
            for m in re.finditer(pat, q_low):
                neg_phrase = m.group(1).strip()
                neg_matches.append(neg_phrase)

        for neg_phrase in neg_matches:
            res_b = self.dp.resolve_brand(neg_phrase)
            if res_b:
                excluded_brands.append(res_b)
            for b_alias, b_canon in self.dp.brand_aliases.items():
                if re.search(r"\b" + re.escape(b_alias) + r"\b", neg_phrase):
                    excluded_brands.append(b_canon)
            for b_low, b_canon in self.catalog_brands.items():
                if re.search(rf"(?<![a-z0-9]){re.escape(b_low)}(?![a-z0-9])", neg_phrase):
                    excluded_brands.append(b_canon)
            # Also catch any literal brand token in negative phrase even if not in alias map
            for neg_tok in re.findall(r"[a-z0-9]{2,}", neg_phrase.lower()):
                if neg_tok not in ("the", "and", "or", "for", "from", "with", "any"):
                    if neg_tok in self.catalog_brands:
                        excluded_brands.append(self.catalog_brands[neg_tok])
            for c_token, c_canonical in _COLOR_CANONICAL.items():
                if re.search(r"\b" + re.escape(c_token) + r"\b", neg_phrase):
                    excluded_colors.append(c_canonical)
            for leaf in self.dp.leaf_to_group.keys():
                if re.search(r"\b" + re.escape(leaf) + r"\b", neg_phrase):
                    excluded_categories.append(leaf)

        excluded_brands = list(dict.fromkeys(excluded_brands))
        excluded_colors = list(dict.fromkeys(excluded_colors))
        excluded_categories = list(dict.fromkeys(excluded_categories))

        # 6. Brand (only from positive portion of query, and never an excluded brand)
        q_pos_only = q_low
        for neg_phrase in neg_matches:
            q_pos_only = re.sub(rf"\b(?:not|except|without|no)\s+{re.escape(neg_phrase)}\b", " ", q_pos_only, flags=re.IGNORECASE)
            q_pos_only = re.sub(rf"{_B}{re.escape(neg_phrase)}\s+(?:ছাড়|ছাড়া|ব্যতীত|বিনা){_E}", " ", q_pos_only)
        raw_brand = self.dp.resolve_brand(q_pos_only)
        ex_b_low = {b.lower() for b in excluded_brands}
        brand = raw_brand if (raw_brand and raw_brand.lower() not in ex_b_low) else None

        # 7. Color (with canonical mapping to catalog colors, only if not excluded)
        color = None
        for pat, c_token, c_canonical in self._color_patterns:
            if c_canonical in excluded_colors:
                continue
            is_negated = any(re.search(r"\b" + re.escape(c_token) + r"\b", n) for n in neg_matches)
            if not is_negated and pat.search(q_low):
                color = c_canonical
                break

        # Disambiguate: If a token matches both brand and color (e.g. 'yellow', 'orange', 'pink'),
        # it is a color unless explicitly designated as brand
        if brand and color and brand.lower().strip() == color.lower().strip():
            brand = None

        # 8. Material (with canonical mapping)
        material = None
        for pat, m_token, m_canonical in self._material_patterns:
            if pat.search(q_low):
                material = m_canonical
                break

        # 9. Category resolution + alias substitution
        product_type: Optional[str] = None

        # Check taxonomy leaf phrases in q_no_size and norm_query (longest first)
        target_check_texts = [q_no_size.lower(), norm_query.lower(), raw_query.lower()]
        for pat, leaf in self._leaf_patterns:
            if leaf in excluded_categories:
                continue
            if any(pat.search(t) for t in target_check_texts):
                product_type = leaf
                break

        # Running shoes semantic cues
        if not product_type:
            if any(re.search(r"\b(?:running\s+(?:.*?\s+)?shoes?|shoes?\s+(?:.*?\s+)?(?:running|jogging)|trainers?\s+(?:for\s+)?jogging)\b", t) for t in target_check_texts):
                product_type = "running shoes"

        # Explicit iPhone check (only when not asking for accessories)
        if any(re.search(r"\biphones?\b", t) for t in target_check_texts):
            if not any(w in t for t in target_check_texts for w in ("case", "cover", "protector", "charger", "cable", "holder", "stand")):
                product_type = "smartphone"
                if not brand and "apple" not in ex_b_low:
                    brand = "Apple"

        tokens = q_no_size.split()
        resolved_tokens = []
        for tok in tokens:
            canon = self.dp.resolve_alias(tok)
            if canon == tok and _BANGLA_ALPHABET_RE.search(tok):
                for stem in stem_bangla(tok):
                    if stem in self.dp.bangla_dict:
                        canon = self.dp.bangla_dict[stem]
                        break
            resolved_tokens.append(canon)
            if not product_type and canon.lower() in self.dp.leaf_to_group and canon.lower() not in excluded_categories:
                product_type = canon.lower()

        # Build clean retrieval text (strip out negative terms so they don't pollute BM25/Dense)
        clean_q = " ".join(resolved_tokens)
        for neg_phrase in neg_matches:
            clean_q = re.sub(rf"\b(?:not|except|without|no)\s+{re.escape(neg_phrase)}\b", " ", clean_q, flags=re.IGNORECASE)
            clean_q = re.sub(rf"{_B}{re.escape(neg_phrase)}\s+(?:ছাড়|ছাড়া|ব্যতীত|বিনা){_E}", " ", clean_q)
        for ex_b in excluded_brands:
            clean_q = re.sub(rf"\b{re.escape(ex_b)}\b", " ", clean_q, flags=re.IGNORECASE)
        for ex_c in excluded_colors:
            clean_q = re.sub(rf"\b{re.escape(ex_c)}\b", " ", clean_q, flags=re.IGNORECASE)
        clean_q = re.sub(
            r"(?<!-)\b(for|my|in|a|an|the|er|jonno|under|below|and|or|budget|debo|debe|din|chai|dorkar|ke|r)\b(?!-)",
            " ", clean_q, flags=re.IGNORECASE,
        )
        clean_q = re.sub(
            r"(?<![\w\u0980-\u09FF])(জন্য|কে|র|এর|চাই|দরকার|দেবো|দেবে|দিন|টাকা|টাকার|টাকায়|মধ্যে|নিচে|কম|উপরে|বেশি|ভেতর|ভিতর|পর্যন্ত|বাজেট|সাইজ|মাপ|পরার|পড়ার|ব্যবহারের|ব্যবহার|করার|কেনার|কিনতে|নিতে|লাগবে|খুঁজছি|দেখান|দেখাও|দাও|একটি|একটা|কয়েকটি|সহ|ছাড়া|দিয়ে|নিয়ে|থেকে|হতে|ভালো|সেরা|সুন্দর|আরামদায়ক|নতুন|সস্তা|কিছু|কোন|কোনো)(?![\w\u0980-\u09FF])",
            " ", clean_q,
        )
        clean_q = re.sub(r"\b(\w+)(?:\s+\1\b)+", r"\1", clean_q, flags=re.IGNORECASE)
        clean_q = re.sub(r"\s+", " ", clean_q).strip() or norm_query

        # Match longest leaf categories from the cleaned query text
        clean_low = clean_q.lower()
        if "power bank" in affinities and ("charge" in norm_query.lower() or "power bank" in norm_query.lower() or "portable" in norm_query.lower()):
            product_type = "power bank"
        else:
            for leaf in sorted(self.dp.leaf_to_group.keys(), key=lambda x: -len(x)):
                if leaf in excluded_categories:
                    continue
                if re.search(r"\b" + re.escape(leaf) + r"\b", clean_low) or (len(leaf) > 5 and leaf in clean_low):
                    product_type = leaf
                    break

        # Standalone phone / mobile query without accessory cues -> smartphone category
        if not product_type and re.search(r"\b(?:phone|phones|mobile|mobiles)\b", clean_low):
            if not any(w in clean_low for w in ("case", "cover", "charger", "adapter", "cable", "protector", "holder", "stand", "gimbal")):
                product_type = "smartphone"

        # Hard constraints
        hard: Dict[str, Any] = {}
        if max_p is not None:
            hard["max_price"] = max_p
        if min_p is not None:
            hard["min_price"] = min_p
        if size_val:
            hard["size"] = size_val
        if target_gender in ("men", "women", "kids"):
            hard["gender"] = target_gender
        if target_audience in ("kids", "toddler"):
            hard["audience"] = target_audience

        # Soft preferences
        soft: Dict[str, Any] = {}
        if brand:
            soft["brand"] = brand
        if color:
            soft["color"] = color
        if material:
            soft["material"] = material
        if affinities:
            soft["affinities"] = affinities

        intent_res = StructuredIntent(
            raw_query=raw_query,
            normalized_query=norm_query,
            retrieval_query=clean_q,
            language=lang,
            corrections=corrections,
            product_type=product_type,
            brand=brand,
            color=color,
            material=material,
            size=size_val,
            recipient=recipient,
            target_audience=target_audience,
            target_gender=target_gender,
            intents=intent_names,
            affinities=affinities,
            min_price=min_p,
            max_price=max_p,
            currency="BDT",
            hard_constraints=hard,
            soft_preferences=soft,
            excluded_brands=excluded_brands,
            excluded_colors=excluded_colors,
            excluded_categories=excluded_categories,
            confidence=0.95 if (product_type or brand or recipient or max_p) else 0.70,
            provenance={
                "normalization": "rule",
                "price": "regex_rule" if (min_p or max_p) else "none",
                "recipient": "lexicon" if recipient else "none",
                "intents": "rule" if intent_names else "none",
            },
        )
        if len(self._parse_cache) < 10000:
            self._parse_cache[cache_key] = intent_res
        return intent_res

