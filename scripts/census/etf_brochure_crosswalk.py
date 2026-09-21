"""etf_brochure_crosswalk.py -- fund -> investment-adviser CRD -> Form ADV Part 2A brochure text.

The strategy-classification input (research/etf_brochure_type_prompt.txt) needs, per admitted fund,
the local path(s) of its adviser's brochure text. The adviser CRD is read straight from the fund's
Form N-CEN filing (ADVISER.tsv, column CRD_NUM, in ncen_data/*_ncen*.zip) -- no name matching. The
CRD keys the brochure folder adv_data/brochures/<crd>/text.txt (per-CRD fetch or FOIA bulk, see
adv_fetch.py). Writes research/etf_universe/etf_brochure_crosswalk.csv (one row per admitted fund:
its CRDs, the brochure text paths on disk, and a coverage status) and prints coverage by strategy.

  python research/etf_brochure_crosswalk.py            # admitted funds (universe_sample.in_sample)
  python research/etf_brochure_crosswalk.py --scored   # every scored fund
"""
import os, re, glob, sys, zipfile
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
U = os.path.join(ROOT, "research", "etf_universe")
BROCH = os.path.join(ROOT, "adv_data", "brochures")
OUT = os.path.join(U, "etf_brochure_crosswalk.csv")
SER = re.compile(r"(S\d{9,})")


def series_to_crds():
    """series id -> [(crd, adviser_name)], newest N-CEN filing per (series, crd)."""
    rows = []
    for zp in sorted(glob.glob(os.path.join(ROOT, "ncen_data", "*_ncen*.zip"))):
        q = os.path.basename(zp).split("_")[0]
        with zipfile.ZipFile(zp) as z:
            if "ADVISER.tsv" not in z.namelist():
                continue
            with z.open("ADVISER.tsv") as f:
                a = pd.read_csv(f, sep="\t", dtype=str, keep_default_na=False,
                                usecols=["FUND_ID", "ADVISER_NAME", "CRD_NUM"])
        a["series"] = a.FUND_ID.str.extract(SER)[0]
        a["q"] = q
        rows.append(a.dropna(subset=["series"]))
    adv = pd.concat(rows, ignore_index=True)
    adv = adv[adv.CRD_NUM.str.strip() != ""]
    adv["crd"] = adv.CRD_NUM.str.lstrip("0")                 # brochure folders carry no leading zeros
    adv = adv.sort_values("q").drop_duplicates(["series", "crd"], keep="last")
    return adv.groupby("series").apply(
        lambda g: list(zip(g.crd, g.ADVISER_NAME)), include_groups=False).to_dict()


def text_paths(crds):
    out = []
    for c in str(crds).split(";"):
        p = os.path.join(BROCH, c, "text.txt")
        if c and os.path.exists(p) and os.path.getsize(p) > 200:
            out.append(f"adv_data/brochures/{c}/text.txt")
    return ";".join(out)


def main():
    scored = "--scored" in sys.argv
    uni = pd.read_csv(os.path.join(U, "universe_sample.csv"), dtype=str)
    funds = uni if scored else uni[uni.in_sample.astype(str).str.lower().isin(("true", "1", "yes", "y"))]
    keep = set(funds.ticker.str.upper())

    ser2crd = series_to_crds()
    cen = pd.read_csv(os.path.join(U, "ncen_active_etfs.csv"), dtype=str)
    t2s = {t.strip().upper(): str(r.SERIES_ID)
           for _, r in cen.iterrows() for t in str(r.tickers).replace(";", ",").split(",") if t.strip()}
    typ = pd.read_csv(os.path.join(U, "ncen_active_types.csv"), dtype=str)
    t2type = {t.strip().upper(): r.get("type", "")
              for _, r in typ.iterrows() for t in str(r.get("tickers", "")).replace(";", ",").split(",") if t.strip()}

    rows = []
    for _, f in funds.iterrows():
        tk = str(f.ticker).upper()
        crds = ser2crd.get(t2s.get(tk, ""), [])
        tp = text_paths(";".join(c for c, _ in crds))
        rows.append(dict(ticker=tk, name_type=t2type.get(tk, ""),
                         phi_63=f.get("phi_63", ""), phibar_63=f.get("phibar_63", ""),
                         status_book=f.get("status", ""), n_advisers=len(crds),
                         crds=";".join(c for c, _ in crds),
                         adviser_names=" | ".join(n for _, n in crds),
                         text_paths=tp,
                         coverage=("on_disk" if tp else ("bulk_needed" if crds else "no_crd"))))
    X = pd.DataFrame(rows)
    X.to_csv(OUT, index=False)
    print(f"wrote {os.path.relpath(OUT, ROOT)}: {len(X)} funds")
    print(X.coverage.value_counts().to_string())
    print("\ncoverage by strategy (covered / total):")
    X["covered"] = X.text_paths != ""
    for nt, g in X.groupby("name_type"):
        print(f"  {nt or '(blank)':20s} {int(g.covered.sum()):3d} / {len(g)}")


if __name__ == "__main__":
    main()
