"""CLI:  python -m lite_search --check | "red nike shoes under 5000" | --repl | --serve"""
import argparse
import json
import logging

from .config import Settings
from .engine import build_engine


def main():
    ap = argparse.ArgumentParser(prog="lite_search")
    ap.add_argument("query", nargs="?")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--repl", action="store_true", help="interactive prompt")
    ap.add_argument("--check", action="store_true", help="test the DB/CSV connection and column mapping, then exit")
    ap.add_argument("--serve", action="store_true", help="start the HTTP API")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    if a.serve:
        import uvicorn
        uvicorn.run("lite_search.server:app", host=a.host, port=a.port)
        return
    if a.check:
        from .catalog import load_dataframe
        from .diagnose import diagnose
        print(diagnose(load_dataframe(Settings.from_env())))
        return
    eng = build_engine(Settings.from_env())

    def show(q):
        r = eng.search(q, k=a.k)
        print(f"\n[{r['source']}] intent={json.dumps(r['intent'], ensure_ascii=False)} {r['took_ms']} ms"
              f"{' (cached)' if r['cached'] else ''}")
        for n in r["notes"]:
            print("  note:", n)
        for x in r["results"]:
            print(f"  {x['rank']:>2}. {x['title']} | {x['brand']} | {x['color']} | {x['price']:.0f} | {x['matched']}")

    if a.repl or not a.query:
        while True:
            try:
                q = input("query> ").strip()
            except EOFError:
                break
            if q in ("exit", "quit"):
                break
            if q:
                show(q)
    else:
        show(a.query)


main()
