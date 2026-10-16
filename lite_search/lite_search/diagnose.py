"""`python -m lite_search --check`: connect to the data source and report whether the column mapping is sane,
BEFORE building embeddings. Run this first on a new company database."""
from .text import is_missing

EXPECTED = ["id", "title", "category", "brand", "color", "material", "gender", "sizes", "price", "description"]


def diagnose(df):
    raw = df.attrs.get("raw_columns", [])
    L = [f"products loaded : {len(df)}", f"source columns  : {', '.join(map(str, raw)) or '(unknown)'}", "",
         "column        filled   note"]
    warnings = []
    for c in EXPECTED:
        s = df[c]
        filled = (s.map(lambda v: not is_missing(v)).mean() * 100) if c != "price" else (df[c] > 0).mean() * 100
        note = ""
        if c == "gender" and (s == "unisex").all():
            filled, note = 0.0, "no gender data - every product treated as unisex"
        L.append(f"{c:<13} {filled:5.0f}%   {note}")
        if c == "title" and filled < 95:
            warnings.append("title is mostly empty - map your product-name column: COLUMN_MAP=<your_col>:title")
        if c == "category" and filled < 50:
            warnings.append("category mostly empty - rules and product groups work best with it (COLUMN_MAP=<col>:category)")
        if c == "price" and filled < 50:
            warnings.append("price mostly 0/empty - 'under 5000' style queries will not work (COLUMN_MAP=<col>:price)")
    if len(df):
        L += ["", f"price range     : {df['price'].min():.0f} .. {df['price'].max():.0f}",
              f"categories      : {df['category'].nunique()}  top: {', '.join(df['category'].value_counts().head(8).index)}",
              f"brands          : {df['brand'].nunique()}  top: {', '.join(df['brand'].value_counts().head(8).index)}",
              "", "sample rows:"]
        for r in df.head(3).itertuples():
            L.append(f"  {r.id} | {r.title[:50]} | {r.category} | {r.brand} | {r.color} | {r.price:.0f}")
        if df["id"].duplicated().any():
            warnings.append(f"{int(df['id'].duplicated().sum())} duplicate ids - ids must be unique")
    L += [""] + (["WARNINGS:"] + [" - " + w for w in warnings] if warnings else ["Looks good."])
    return "\n".join(L)
