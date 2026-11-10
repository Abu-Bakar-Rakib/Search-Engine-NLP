import sys
import requests

# Ensure UTF-8 output for Windows console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_URL = "http://localhost:8000"

def test_search_api():
    print("=" * 70)
    print("  TESTING SEARCH ENGINE REST API CLIENT")
    print("=" * 70)

    # Check if server is running
    try:
        health = requests.get(f"{BASE_URL}/", timeout=3)
        print("[OK] Connected to API Server:", health.json())
    except requests.exceptions.ConnectionError:
        print("\n[!] ERROR: API server is not running!")
        print("Please open another terminal and start the server first with:")
        print("   uvicorn api:app --reload --port 8000\n")
        return

    # 1. Test Search Query
    query_text = "gift for my father under 3000"
    print(f"\n--- 1. Testing GET /search for: '{query_text}' ---")
    response = requests.get(
        f"{BASE_URL}/search",
        params={"q": query_text, "limit": 3}
    )
    data = response.json()

    print(f"Query       : {data['query']}")
    print(f"Latency     : {data['latency_ms']} ms | Cache Hit: {data['is_cache_hit']}")
    print(f"Intent      : Recipient={data['parsed_intent']['recipient']} | Budget={data['parsed_intent']['max_price']}")
    print("\nResults:")
    for item in data["results"]:
        print(f"  -> [{item['product_id']}] {item['title']} - {item['price']} {item['currency']} ({item['category']})")
        print(f"     Decision: {item['decision_explanation']}")

    # 2. Test Intent Extraction Endpoint
    bangla_query = "বাচ্চাদের খেলনা ১৫০০ টাকার নিচে"
    print(f"\n--- 2. Testing POST /intent/parse for: '{bangla_query}' ---")
    intent_res = requests.post(
        f"{BASE_URL}/intent/parse",
        json={"query": bangla_query}
    )
    intent_data = intent_res.json()
    print("Language    :", intent_data["language"])
    print("Demographics:", intent_data["demographics"])
    print("Budget      :", intent_data["budget_constraints"])
    print("Affinities  :", intent_data["inferred_preferences"]["affinities"][:4])

    # 3. Test Catalog Statistics
    print("\n--- 3. Testing GET /catalog/stats ---")
    stats_res = requests.get(f"{BASE_URL}/catalog/stats")
    stats = stats_res.json()
    print(f"Catalog Name : {stats['domain_name']}")
    print(f"Total Items  : {stats['total_products']} products across {stats['unique_categories']} categories")
    print(f"Price Range  : {stats['price_summary']['min_price']} to {stats['price_summary']['max_price']} {stats['price_summary']['currency']}")
    print("=" * 70)

if __name__ == "__main__":
    test_search_api()
