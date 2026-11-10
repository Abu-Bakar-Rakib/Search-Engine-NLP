import re
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Set, Optional, Tuple
import pandas as pd
import numpy as np

CANONICAL_FIELDS = [
    "product_id",
    "sku",
    "title",
    "description",
    "category_path",
    "category_leaf",
    "brand",
    "color",
    "material",
    "gender",
    "audience",
    "sizes",
    "price",
    "currency",
    "stock_qty",
    "rating",
    "review_count",
    "attributes",
    "tags",
    "is_active",
    "updated_at"
]

MANDATORY_FIELDS = ["product_id", "title", "price"]

@dataclass
class CanonicalProduct:
    product_id: str
    title: str
    price: float
    sku: str = ""
    description: str = ""
    category_path: str = ""
    category_leaf: str = "general"
    brand: str = ""
    color: str = ""
    material: str = ""
    gender: str = "unisex"
    audience: str = "adult"
    sizes: str = ""
    currency: str = "BDT"
    stock_qty: int = 100
    rating: float = 4.0
    review_count: int = 0
    attributes: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    is_active: bool = True
    updated_at: str = ""
    provenance: Dict[str, str] = field(default_factory=dict)  # "catalog" vs "derived"
    
    # Pre-computed helper representations
    size_set: Set[str] = field(default_factory=set)
    search_text: str = ""

    def __post_init__(self):
        # Convert sizes string to normalized set
        if isinstance(self.sizes, str) and self.sizes.strip():
            self.size_set = {s.strip().upper() for s in self.sizes.split(",") if s.strip()}
        elif isinstance(self.sizes, (list, set)):
            self.size_set = {str(s).strip().upper() for s in self.sizes if str(s).strip()}
            self.sizes = ",".join(sorted(self.size_set))
        else:
            self.size_set = set()
            self.sizes = ""

        # Normalize gender
        g = str(self.gender).lower().strip()
        if g in ("men", "man", "male", "boys", "boy"):
            self.gender = "men"
        elif g in ("women", "woman", "female", "girls", "girl"):
            self.gender = "women"
        elif g in ("kids", "child", "children", "baby", "toddler"):
            self.gender = "kids"
        else:
            self.gender = "unisex"

        # Build comprehensive search text representation for indexing
        tags_str = " ".join(self.tags) if isinstance(self.tags, list) else str(self.tags)
        attrs_str = " ".join(f"{k} {v}" for k, v in self.attributes.items()) if isinstance(self.attributes, dict) else ""
        self.search_text = (
            f"{self.title}. Category: {self.category_leaf} ({self.category_path}). "
            f"Brand: {self.brand}. Color: {self.color}. Material: {self.material}. "
            f"Audience: {self.audience} {self.gender}. Description: {self.description}. "
            f"{tags_str} {attrs_str}"
        ).strip()


class SchemaProfiler:
    """Profiles raw datasets, detects nulls, duplicate IDs, price outliers, and suggests schema maps."""
    
    @staticmethod
    def profile_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
        total_rows = len(df)
        columns = list(df.columns)
        
        # Missing values
        null_counts = df.isnull().sum().to_dict()
        null_pct = {col: round(cnt / total_rows * 100, 2) for col, cnt in null_counts.items()}
        
        # ID uniqueness check
        id_candidates = [c for c in columns if any(k in c.lower() for k in ("id", "code", "sku", "key"))]
        id_stats = {}
        for c in id_candidates:
            unique_count = df[c].nunique()
            id_stats[c] = {
                "unique": unique_count,
                "is_unique": unique_count == total_rows,
                "duplicates": total_rows - unique_count
            }
            
        # Price analysis
        price_cols = [c for c in columns if any(k in c.lower() for k in ("price", "cost", "amount", "rate", "mrp"))]
        price_stats = {}
        for c in price_cols:
            numeric_prices = pd.to_numeric(df[c], errors="coerce").dropna()
            if len(numeric_prices) > 0:
                q25 = numeric_prices.quantile(0.25)
                q75 = numeric_prices.quantile(0.75)
                iqr = q75 - q25
                outliers = numeric_prices[(numeric_prices < (q25 - 1.5 * iqr)) | (numeric_prices > (q75 + 1.5 * iqr))]
                price_stats[c] = {
                    "min": float(numeric_prices.min()),
                    "max": float(numeric_prices.max()),
                    "mean": round(float(numeric_prices.mean()), 2),
                    "median": float(numeric_prices.median()),
                    "outlier_count": len(outliers)
                }
                
        # Suggest schema map
        suggested_map = SchemaProfiler.suggest_schema_map(columns)
        
        return {
            "total_rows": total_rows,
            "columns": columns,
            "null_percentages": null_pct,
            "id_profiling": id_stats,
            "price_profiling": price_stats,
            "suggested_schema_map": suggested_map
        }

    @staticmethod
    def suggest_schema_map(columns: List[str]) -> Dict[str, str]:
        """Fuzzy matches raw column names to canonical schema fields."""
        heuristics = {
            "product_id": ["product_id", "item_code", "id", "item_id", "productid", "asin", "sku_id"],
            "sku": ["sku", "model_no", "model", "item_sku", "product_code"],
            "title": ["title", "item_name", "product_name", "name", "product_title"],
            "description": ["description", "specification_summary", "details", "desc", "summary", "body"],
            "category_path": ["category_path", "division_path", "cat_hierarchy", "breadcrumb"],
            "category_leaf": ["category_leaf", "sub_category", "category", "cat", "product_type", "type"],
            "brand": ["brand", "manufacturer", "brand_name", "maker", "vendor"],
            "color": ["color", "finish_color", "colour", "shade"],
            "material": ["material", "build_material", "fabric", "composition"],
            "gender": ["gender", "operator_type", "target_gender", "sex"],
            "audience": ["audience", "target_sector", "age_group", "persona"],
            "sizes": ["sizes", "gauge_size", "size", "available_sizes"],
            "price": ["price", "unit_cost", "unit_price", "amount", "mrp", "cost", "sale_price"],
            "currency": ["currency", "curr", "price_currency"],
            "stock_qty": ["stock_qty", "inventory_count", "quantity", "stock", "qty", "inventory"],
            "rating": ["rating", "audit_score", "avg_rating", "stars", "score"],
            "review_count": ["review_count", "cert_count", "reviews", "num_reviews"],
            "attributes": ["attributes", "tech_specs", "specs", "metadata"],
            "tags": ["tags", "keywords", "labels"],
            "is_active": ["is_active", "is_available", "active", "status", "in_stock"],
            "updated_at": ["updated_at", "last_inspected", "modified_at", "created_at"]
        }
        
        mapping = {}
        cols_lower = {c.lower(): c for c in columns}
        
        for canonical, candidates in heuristics.items():
            matched = None
            for cand in candidates:
                if cand in cols_lower:
                    matched = cols_lower[cand]
                    break
            if matched:
                mapping[canonical] = matched
        return mapping


class SchemaValidator:
    """Validates and transforms raw dictionaries or dataframes into CanonicalProduct records."""
    
    def __init__(self, schema_map: Dict[str, str]):
        self.schema_map = schema_map
        # Invert schema map to look up raw column -> canonical field
        self.inv_map = {v: k for k, v in schema_map.items()}

    def transform_row(self, row: Dict[str, Any]) -> CanonicalProduct:
        mapped_data: Dict[str, Any] = {}
        provenance: Dict[str, str] = {}
        
        for canonical_field, source_col in self.schema_map.items():
            if source_col in row and row[source_col] is not None and not (isinstance(row[source_col], float) and np.isnan(row[source_col])):
                mapped_data[canonical_field] = row[source_col]
                provenance[canonical_field] = "catalog"
            else:
                mapped_data[canonical_field] = None
                provenance[canonical_field] = "missing"

        # Check mandatory fields
        p_id = mapped_data.get("product_id")
        if not p_id or str(p_id).strip() == "":
            p_id = f"AUTO_ID_{np.random.randint(100000, 999999)}"
            provenance["product_id"] = "derived"

        title = mapped_data.get("title")
        if not title or str(title).strip() == "":
            title = "Untitled Product"
            provenance["title"] = "derived"

        price_raw = mapped_data.get("price")
        try:
            price = float(price_raw) if price_raw is not None else 0.0
            if np.isnan(price) or price < 0:
                price = 0.0
        except (ValueError, TypeError):
            price = 0.0
            provenance["price"] = "derived"

        # Extract attributes & tags safely
        raw_attrs = mapped_data.get("attributes")
        attributes = {}
        if isinstance(raw_attrs, dict):
            attributes = raw_attrs
        elif isinstance(raw_attrs, str) and raw_attrs.strip().startswith("{"):
            try:
                attributes = json.loads(raw_attrs)
            except Exception:
                attributes = {}

        raw_tags = mapped_data.get("tags")
        tags = []
        if isinstance(raw_tags, list):
            tags = [str(t).strip() for t in raw_tags]
        elif isinstance(raw_tags, str) and raw_tags.strip():
            tags = [t.strip() for t in raw_tags.split(",") if t.strip()]

        # Stock qty
        try:
            stock_qty = int(mapped_data.get("stock_qty", 100))
        except (ValueError, TypeError):
            stock_qty = 100

        # Rating
        try:
            rating = float(mapped_data.get("rating", 4.0))
        except (ValueError, TypeError):
            rating = 4.0

        # Active flag
        is_active = mapped_data.get("is_active", True)
        if isinstance(is_active, str):
            is_active = is_active.lower() in ("true", "1", "yes", "active")
        else:
            is_active = bool(is_active)

        return CanonicalProduct(
            product_id=str(p_id),
            title=str(title),
            price=price,
            sku=str(mapped_data.get("sku") or ""),
            description=str(mapped_data.get("description") or ""),
            category_path=str(mapped_data.get("category_path") or ""),
            category_leaf=str(mapped_data.get("category_leaf") or "general"),
            brand=str(mapped_data.get("brand") or ""),
            color=str(mapped_data.get("color") or ""),
            material=str(mapped_data.get("material") or ""),
            gender=str(mapped_data.get("gender") or "unisex"),
            audience=str(mapped_data.get("audience") or "adult"),
            sizes=str(mapped_data.get("sizes") or ""),
            currency=str(mapped_data.get("currency") or "BDT"),
            stock_qty=stock_qty,
            rating=rating,
            review_count=int(mapped_data.get("review_count") or 0),
            attributes=attributes,
            tags=tags,
            is_active=is_active,
            updated_at=str(mapped_data.get("updated_at") or ""),
            provenance=provenance
        )

    def transform_dataframe(self, df: pd.DataFrame) -> List[CanonicalProduct]:
        products = []
        for _, row in df.iterrows():
            products.append(self.transform_row(row.to_dict()))
        return products
