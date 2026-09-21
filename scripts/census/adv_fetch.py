"""
adv_fetch.py -- fetch utilities for SEC Form ADV bulk data, EDGAR 13F filer lists and Form ADV Part 2A
brochures. SEC asks for a User-Agent with contact info and no more than ~10 requests/second; this script
sends the header and sleeps between requests.

Subcommands, all writing under adv_data/:

  python research/adv_fetch.py adv
      Download the latest SEC Form ADV bulk data zip (Part 1A base + Schedule D sections) from the
      SEC's Form ADV data page and extract it. Locates the base file and the Schedule D 7.B.(1)
      private-fund file by name pattern and prints what it found. If the page layout has changed,
      set ADV_INDEX / ADV_ZIP_RE below; the script prints every zip link it sees.

  python research/adv_fetch.py 13f-cover 2025q2 2025q1
      From the EDGAR full index for the given quarters, list every 13F-HR filer (name + CIK) into
      adv_data/<q>_13f_filers.tsv, for an ADV -> 13F crosswalk.

  python research/adv_fetch.py 13f-value adv_data/review.csv
      For each candidate marked pass with a CIK, the latest 13F-HR cover-page total value against its
      ADV regulatory AUM, with the count of XML-era 13F-HR quarters. Writes adv_data/coverage13f.csv.

  python research/adv_fetch.py brochures-bulk auto --candidates adv_data/candidates_etf.csv
      Bulk Part 2A path. Download the FOIA monthly Part 2A brochure archives (annual-update months
      first), keep only the members whose CRD is in the candidates file, and extract each to
      adv_data/brochures/<crd>/ and write text.txt (lowercased Item 4/8 narrative) from it. Stops once
      every candidate is covered. 'auto' = the latest twelve months; or name <year>-<month> specs;
      'list' prints the archive names. --candidates defaults to adv_data/candidates.csv.

  python research/adv_fetch.py brochures adv_data/candidates_etf.csv
      Targeted Part 2A path (no large archives): for each CRD, download the Part 2A brochure by its
      version id (BRCHR_VRSN_ID, from the ADV_Brochure_Mapping CSVs on disk) via the IAPD brochure
      endpoint, and write text.txt. CRDs whose brochure month is not mapped on disk are written to
      candidates_need_bulk.csv with the brochures-bulk command to finish them. The
      reports.adviserinfo .../PDF/<crd>.pdf report URL is Form ADV Part 1 (a checkbox filing with no
      Item 4/8 narrative) and is not used.
"""
import io
import json
import os
import re
import sys
import time
import zipfile
import urllib.request

UA = os.environ.get("SEC_USER_AGENT", "research-reproduction contact@example.com")   # SEC requires a contact User-Agent: set SEC_USER_AGENT to your name and email
SLEEP = 0.25                                     # <= 4 req/s, well under the SEC ceiling
OUT = "adv_data"

# --- Form ADV bulk data ---------------------------------------------------------------------
# The SEC posts monthly Form ADV data as zip files; the index page lists them. If the index moves,
# set ADV_INDEX to the current "Form ADV data" page under sec.gov and re-run.
ADV_INDEX_CANDIDATES = [   # tried in order; or pass an index / .zip URL on the command line to skip the search
    "https://www.sec.gov/foia-services/frequently-requested-documents/form-adv-data",
    "https://www.sec.gov/data-research/sec-markets-data/information-about-registered-investment-advisers-exempt-reporting-advisers",
    "https://www.sec.gov/about/divisions-offices/division-investment-management/investment-adviser-data",
    "https://www.sec.gov/foia/docs/form-adv-data",
]
ADV_ZIP_RE = re.compile(r'href="([^"]+\.zip)"', re.I)
ADV_BASE_PAT = re.compile(r"^IA_ADV_Base_A_")             # registered advisers, Part 1A base
ADV_7B1_PAT = re.compile(r"^IA_Schedule_D_7B1_\d")        # private-fund master table (not ERA_, not 7B1A##)

# --- 13F filers from the EDGAR full index (stable URL, lists every filing with form type, name, CIK)
F13_IDX_URL = "https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{q}/form.idx"
F13_LINE = re.compile(r"^(13F-HR(?:/A)?)\s{2,}(.+?)\s{2,}(\d+)\s{2,}(\d{4}-\d{2}-\d{2})\s{2,}(\S+)\s*$")

# --- ADV Part 2A brochures (IAPD) ------------------------------------------------------------
# The per-CRD download needs the Part 2A brochure, not the Part 1 report. The Part 1 report PDF
# (reports.adviserinfo.../reports/ADV/<crd>/PDF/<crd>.pdf) is the checkbox filing and has no Item 4/8
# narrative. The Part 2A brochure is the per-brochure IAPD endpoint below, keyed by the brochure
# version id (BRCHR_VRSN_ID), which is the BrochureID column of the FOIA ADV_Brochure_Mapping CSVs
# (_brochure_ids).
BROCHURE_URL = "https://reports.adviserinfo.sec.gov/reports/ADV/{crd}/PDF/{crd}.pdf"           # Part 1 report (not the Part 2A brochure)
BROCHURE_PART2A = "https://files.adviserinfo.sec.gov/IAPD/Content/Common/crd_iapd_Brochure.aspx?BRCHR_VRSN_ID={id}"


def get_to_file(url, path):
    """stream a large download to disk (the filing-data set is multi-hundred MB)."""
    import shutil
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Encoding": "identity"})
    with urllib.request.urlopen(req, timeout=900) as r, open(path, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    time.sleep(SLEEP)
    return os.path.getsize(path)


def get(url, binary=False):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Encoding": "identity"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()
    time.sleep(SLEEP)
    return data if binary else data.decode("utf-8", errors="replace")


def _latest(links):
    """prefer the link carrying the largest date-like number (YYYYMM / YYYYMMDD)."""
    def key(l):
        nums = re.findall(r"(20\d{4,6})", l)
        return int(nums[-1]) if nums else -1
    return sorted(links, key=key, reverse=True)[0]


def fetch_adv(url_arg=None):
    os.makedirs(OUT, exist_ok=True)
    if url_arg and url_arg.lower().endswith(".zip"):
        urls = [url_arg]
    else:
        candidates = [url_arg] if url_arg else ADV_INDEX_CANDIDATES
        links, used = [], None
        for idx in candidates:
            try:
                html = get(idx)
            except Exception as e:
                print(f"  index {idx}: {e}"); continue
            found = ADV_ZIP_RE.findall(html)
            print(f"  index {idx}: {len(found)} zip links")
            if found:
                links, used = found, idx; break
        if not links:
            sys.exit("no ADV zip links on any candidate index. Open sec.gov, search 'Form ADV data', copy the "
                     "zip link, and run: python research/adv_fetch.py adv <zip-url>")
        for l in links[:10]:
            print("   ", l)
        # FOIA page products: adv-filing-data-<from>-<to>-partN.zip = Form ADV Part 1 filing data, split
        # into chunks ("-partN" is a chunk number, not a form part); adv-brochures-<year>-<month>.zip =
        # Part 2A brochures; advpart-3-*.zip = Form CRS. The filing data is needed, every chunk.
        fd = [l for l in links if "adv-filing-data" in l.lower()]
        if not fd:
            sys.exit("no adv-filing-data links on the index; paste the zip list so the pattern can be set")
        stems = {}
        for l in fd:
            stems.setdefault(re.sub(r"-part\d+\.zip$", ".zip", l, flags=re.I), []).append(l)
        best = max(stems, key=lambda s: int((re.findall(r"(20\d{6})", s) or ["0"])[-1]))
        chunks = sorted(stems[best])
        print("filing-data set:", best, "| chunks:", chunks)
        urls = [c if c.startswith("http") else "https://www.sec.gov" + c for c in chunks]
        print("index used:", used)
    pdir = os.path.join(OUT, "adv_part1")
    os.makedirs(pdir, exist_ok=True)
    for k, url in enumerate(urls):
        zpath = os.path.join(pdir, f"chunk{k}.zip")
        if os.path.exists(zpath) and os.path.getsize(zpath) > 1e6:
            print("have", zpath); continue
        print("downloading", url)
        mb = get_to_file(url, zpath) / 1e6
        print(f"  {mb:.0f} MB -> {zpath}")
    for k in range(len(urls)):
        zipfile.ZipFile(os.path.join(pdir, f"chunk{k}.zip")).extractall(pdir)
    # archives nest zips inside zips: extract every nested zip once, then scan the tree for the CSVs
    for _ in range(3):
        nested = [os.path.join(r, f) for r, _, fs in os.walk(pdir) for f in fs
                  if f.lower().endswith(".zip") and not f.startswith("chunk") and not os.path.exists(os.path.join(r, f + ".extracted"))]
        if not nested:
            break
        for nz in nested:
            try:
                zipfile.ZipFile(nz).extractall(os.path.dirname(nz)); open(nz + ".extracted", "w").close()
            except Exception as e:
                print("  nested zip failed:", nz, e)
    csvs = [os.path.relpath(os.path.join(r, f), OUT) for r, _, fs in os.walk(pdir) for f in fs if f.lower().endswith(".csv")]
    base = [c for c in csvs if ADV_BASE_PAT.search(os.path.basename(c))]
    d7 = [c for c in csvs if ADV_7B1_PAT.search(os.path.basename(c))]
    print("csv files found:", len(csvs))
    for c in csvs[:40]:
        print("   ", c)
    print("  base candidates:", base)
    print("  7.B.(1) candidates:", d7)
    json.dump({"zip": urls, "base": base, "d7b1": d7, "members": csvs}, open(os.path.join(OUT, "adv_manifest.json"), "w"), indent=1)
    print("manifest written. If base/7B1 lists are empty, paste the csv list above so the patterns can be set.")


def fetch_13f_cover(quarters):
    """13F-HR filer names and CIKs from the EDGAR full index, one file per quarter (any recent quarter
    serves the crosswalk; 13F-HR filings for a quarter-end land in the following quarter's index)."""
    os.makedirs(OUT, exist_ok=True)
    for q in quarters:
        m = re.match(r"(\d{4})q([1-4])$", q.lower())
        if not m:
            sys.exit(f"quarter must look like 2025q2, got {q}")
        url = F13_IDX_URL.format(year=m.group(1), q=m.group(2))
        print("downloading", url)
        txt = get(url)
        rows, seen = [], set()
        for line in txt.splitlines():
            mm = F13_LINE.match(line)
            if mm and mm.group(3) not in seen:
                seen.add(mm.group(3)); rows.append((mm.group(2).strip(), mm.group(3)))
        dest = os.path.join(OUT, f"{q}_13f_filers.tsv")
        with open(dest, "w", encoding="utf-8") as f:
            f.write("filer\tcik\n"); f.writelines(f"{n}\t{c}\n" for n, c in rows)
        print(f"  {len(rows)} 13F-HR filers -> {dest}")


def _candidate_crds(path=None):
    """CRDs to fetch. 13F candidates.csv gates on a matched CIK; a plain CRD list (e.g.
    candidates_etf.csv, one 'crd' column) has no CIK, so accept any row with a crd there."""
    import csv
    p = path or os.path.join(OUT, "candidates.csv")
    if not os.path.exists(p):
        return set()
    rows = list(csv.DictReader(open(p, newline="", encoding="utf-8")))
    if rows and "cik" in rows[0]:
        return {r["crd"] for r in rows if r.get("cik") and r.get("crd")}
    return {r["crd"].lstrip("0") for r in rows if r.get("crd") and r["crd"] not in ("", "nan", "N/A")}


def _extract_text(pdf_bytes, crd):
    """Write adv_data/brochures/<crd>/text.txt (lowercased) from a Part 2A PDF, so the classifier
    reads text.txt directly. Returns True on a usable extraction (>=200 chars of narrative)."""
    import io
    try:
        from pypdf import PdfReader
    except ImportError:
        from PyPDF2 import PdfReader
    try:
        pages = []
        for pg in PdfReader(io.BytesIO(pdf_bytes)).pages:
            try:
                pages.append(pg.extract_text() or "")
            except Exception:
                pass
        text = "\n".join(pages).lower()
    except Exception as e:
        print(f"    text extract failed crd={crd}: {e}"); return False
    if len(text.strip()) < 200:
        return False
    d = os.path.join(OUT, "brochures", str(crd)); os.makedirs(d, exist_ok=True)
    tp = os.path.join(d, "text.txt")
    if os.path.exists(tp) and os.path.getsize(tp) >= len(text.encode("utf-8")):
        return True                                  # a fund can file a full Part 2A and a short Item-18 appendix; keep the longer
    with open(tp, "w", encoding="utf-8") as f:
        f.write(text)
    return True


def _brochure_ids(crds):
    """CRD -> list of Part 2A brochure version ids, newest first, from every ADV_Brochure_Mapping*.csv
    on disk (columns CRDNumber, BrochureID, BrochureName, DateFiled). Prefers full 'PART 2A' brochures
    over 'ITEM 18'-only appendices. The BrochureID is the BRCHR_VRSN_ID the IAPD brochure endpoint wants."""
    import csv, glob
    ids = {}
    for mp in glob.glob(os.path.join(OUT, "**", "ADV_Brochure_Mapping*.csv"), recursive=True):
        for r in csv.DictReader(open(mp, newline="", encoding="utf-8")):
            crd = str(r.get("CRDNumber", "")).lstrip("0")
            if crd in crds and r.get("BrochureID"):
                appendix = "item 18" in str(r.get("BrochureName", "")).lower() and "part 2a" not in str(r.get("BrochureName", "")).lower()
                ids.setdefault(crd, []).append((r.get("DateFiled", ""), appendix, r["BrochureID"], r.get("BrochureName", "")))
    return {c: [(vid, nm) for _, _, vid, nm in sorted(v, key=lambda x: (x[1], _dkey(x[0])), reverse=False)][::-1]
            for c, v in ids.items()}   # non-appendix first, newest first


def _dkey(d):
    m = re.search(r"(\d{1,2})/(\d{1,2})/(\d{4})", str(d))
    return (int(m.group(3)), int(m.group(1)), int(m.group(2))) if m else (0, 0, 0)


MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]


def _archive_months(link):
    """(year, [month indices]) parsed from an archive name; names on the index vary:
    adv-brochures-2024-december, adv_brochures_2022_sep, adv_brochures_2021_mar_1 (chunk),
    part2adv_brochures_2022_oct-dec (quarter)."""
    b = os.path.basename(link).lower()
    y = re.search(r"(20\d\d)", b)
    if not y:
        return None
    toks = re.split(r"[-_.]", b)
    ms = [i for i, m in enumerate(MONTHS) if any(t.startswith(m) for t in toks)]
    return (int(y.group(1)), ms) if ms else None


def fetch_brochures_bulk(months, cand_path=None):
    """Monthly Part 2A brochure archives from the FOIA index (link names vary: adv-brochures-2024-december,
    adv_brochures_2024_october, ...). Members are keyed <CRD>_<brochureID>_<version>_<YYYYMMDD>.pdf.
    Each archive holds only the brochures filed that month, and most advisers file the annual update in
    March-April, so a full twelve-month cycle is needed for coverage. Downloads each requested
    <year>-<month> and extracts only the members whose CRD is a candidate into
    adv_data/brochures/<crd>/ (the layout the classification step reads), printing candidate
    coverage after every month and stopping once every candidate has a brochure. 'list' prints the archives."""
    os.makedirs(OUT, exist_ok=True)
    html = None
    for idx in ADV_INDEX_CANDIDATES:
        try:
            html = get(idx); break
        except Exception as e:
            print(f"  index {idx}: {e}")
    if html is None:
        sys.exit("no ADV index reachable")
    bro = [l for l in ADV_ZIP_RE.findall(html) if "brochure" in l.lower()]
    print(f"{len(bro)} brochure archives on the index")
    if months == ["list"]:
        for l in bro:
            print("  ", l)
        return
    want = _candidate_crds(cand_path)
    if not want:
        sys.exit(f"{cand_path or 'adv_data/candidates.csv'} missing or has no CRDs")
    bdir = os.path.join(OUT, "brochures"); os.makedirs(bdir, exist_ok=True)
    have = {d for d in os.listdir(bdir) if os.listdir(os.path.join(bdir, d))} & want
    print(f"{len(want)} candidate CRDs; {len(have)} already have a brochure")
    # group archives by (year, month); a month may be split into chunks, a quarter archive covers three
    by_month = {}
    for l in bro:
        p = _archive_months(l)
        if p:
            for m in p[1]:
                by_month.setdefault((p[0], m), []).append(l)
    if months == ["auto"]:                          # latest twelve months on the index, annual-update months first
        keys = sorted(by_month, reverse=True)[:12]
        keys = sorted(keys, key=lambda k: (k[1] not in (2, 3), -k[0], -k[1]))
    else:
        keys = []
        for spec in months:
            y, mo = spec.lower().split("-", 1)
            k = (int(y), next((i for i, m in enumerate(MONTHS) if mo.startswith(m)), -1))
            if k not in by_month:
                print(f"  no archive for {spec} (run 'brochures-bulk list' to see the names)"); continue
            keys.append(k)
    for y, m in keys:
        spec = f"{y}-{MONTHS[m]}"
        zdir = os.path.join(OUT, "brochures_bulk", spec); os.makedirs(zdir, exist_ok=True)
        took = total = 0
        for j, l in enumerate(sorted(set(by_month[(y, m)]))):
            url = l if l.startswith("http") else "https://www.sec.gov" + l
            zpath = os.path.join(zdir, f"archive_{j}.zip")
            if not (os.path.exists(zpath) and os.path.getsize(zpath) > 1e6):
                print("downloading", url); mb = get_to_file(url, zpath) / 1e6; print(f"  {mb:.0f} MB")
            try:
                z = zipfile.ZipFile(zpath)
            except zipfile.BadZipFile as e:
                print(f"  bad zip {zpath}: {e}"); continue
            names = z.namelist(); total += len(names)
            for n in names:
                crd = os.path.basename(n).split("_", 1)[0].lstrip("0")
                if crd in want and n.lower().endswith(".pdf"):
                    blob = z.read(n)
                    dest = os.path.join(bdir, crd); os.makedirs(dest, exist_ok=True)
                    open(os.path.join(dest, os.path.basename(n)), "wb").write(blob)
                    if _extract_text(blob, crd):     # write Part 2A text.txt (keeps the longer of full brochure vs Item-18 appendix)
                        have.add(crd); took += 1
        print(f"  {spec}: {total} members, {took} for candidates; coverage {len(have)}/{len(want)}")
        if have >= want:
            print("  all candidates covered; stopping"); break
    missing = sorted(want - have, key=int)
    if missing:
        print(f"{len(missing)} candidates still without a brochure (first 20): {missing[:20]}")


def fetch_13f_value(review_csv):
    """13F coverage check: each passing candidate's latest 13F-HR cover page (primary_doc.xml,
    tableValueTotal, dollars from 2023 on) against its ADV regulatory AUM. One submissions JSON plus one
    cover page per CIK, cached in adv_data/cover13f/<cik>.json. Writes adv_data/coverage13f.csv."""
    import csv
    rows = [r for r in csv.DictReader(open(review_csv, newline="", encoding="utf-8"))
            if r.get("pass", "").lower() == "true" and r.get("cik")]
    cdir = os.path.join(OUT, "cover13f"); os.makedirs(cdir, exist_ok=True)
    out = []
    for i, r in enumerate(rows):
        cik = int(r["cik"]); cf = os.path.join(cdir, f"{cik}.json")
        rec = json.load(open(cf)) if os.path.exists(cf) else None
        if rec is None or "n_13f" not in rec:            # an older cache entry lacks the filing-history count
            rec = {"cik": cik, "report": "", "filed": "", "value": None, "entries": None, "error": "",
                   "n_13f": 0, "first_report": ""}
            try:
                sub = json.loads(get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json"))
                rc = sub["filings"]["recent"]
                hr = [(fd, rd, acc) for f, fd, rd, acc in zip(rc["form"], rc["filingDate"], rc["reportDate"], rc["accessionNumber"]) if f == "13F-HR"]
                for extra in sub["filings"].get("files", []):     # long filers: older filings are paginated
                    e = json.loads(get("https://data.sec.gov/submissions/" + extra["name"]))
                    hr += [(fd, rd, acc) for f, fd, rd, acc in zip(e["form"], e["filingDate"], e["reportDate"], e["accessionNumber"]) if f == "13F-HR"]
                era = sorted({rd for _, rd, _ in hr if rd >= "2013-06-30"})   # distinct quarters, XML era
                rec["n_13f"] = len(era); rec["first_report"] = era[0] if era else ""
                if not hr:
                    rec["error"] = "no 13F-HR in filings"
                else:
                    fd, rd, acc = sorted(hr)[-1]
                    x = get(f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc.replace('-', '')}/primary_doc.xml")
                    def tag(t):
                        m = re.search(rf"<(?:\w+:)?{t}>(.*?)</(?:\w+:)?{t}>", x, flags=re.S)
                        return m.group(1).strip() if m else ""
                    rec.update(report=rd, filed=fd, value=float(tag("tableValueTotal") or "nan"),
                               entries=int(tag("tableEntryTotal") or 0))
            except Exception as e:                    # recorded per CIK; the loop continues
                rec["error"] = f"{type(e).__name__}: {e}"
            json.dump(rec, open(cf, "w"))
        raum = float(r.get("raum") or "nan")
        val = rec["value"] if rec["value"] is not None else float("nan")
        if rec["report"] and rec["report"] < "2023-01-01":
            val *= 1e3                                 # pre-2023 covers report in thousands
        out.append({"crd": r["crd"], "name": r["name"], "cik": cik, "report": rec["report"], "value_13f": val,
                    "entries": rec["entries"], "n_13f": rec.get("n_13f", 0), "first_report": rec.get("first_report", ""),
                    "raum": raum, "coverage": val / raum if raum else float("nan"), "error": rec["error"]})
        if i % 50 == 0:
            print(f"  {i}/{len(rows)}", flush=True)
    import pandas as pd
    df = pd.DataFrame(out); df.to_csv(os.path.join(OUT, "coverage13f.csv"), index=False)
    ok = df[df.coverage >= 0.70]
    print(f"{len(df)} passing candidates; {int(df.error.astype(bool).sum())} fetch errors; "
          f"coverage >= 70%: {len(ok)}; quartiles {df.coverage.quantile([.25, .5, .75]).round(2).tolist()}; "
          f"of which >= 12 XML-era filings: {int((ok.n_13f >= 12).sum())}")
    print("-> adv_data/coverage13f.csv")


def fetch_brochures(candidates_csv):
    """Download each CRD's Form ADV Part 2A brochure (the Item 4/8 narrative) and extract text.txt.
    Version ids come from the on-disk ADV_Brochure_Mapping CSVs; CRDs not covered by a mapping on disk
    are listed at the end for the bulk path (brochures-bulk auto --candidates <file>), which pulls the
    monthly FOIA archives that contain them. Skips a CRD whose text.txt already exists and is non-trivial."""
    crds = _candidate_crds(candidates_csv)
    idmap = _brochure_ids(crds)
    print(f"{len(crds)} CRDs; {len(idmap)} resolvable from brochure mappings on disk; Part 2A endpoint")
    ok = fail = skip = 0; need_bulk = []
    for crd in sorted(crds, key=lambda c: int(c) if c.isdigit() else 0):
        tp = os.path.join(OUT, "brochures", str(crd), "text.txt")
        if os.path.exists(tp) and os.path.getsize(tp) > 200:
            skip += 1; continue
        vids = idmap.get(crd)
        if not vids:
            need_bulk.append(crd); continue
        got = False
        for vid, name in vids:                       # try newest full Part 2A first
            url = BROCHURE_PART2A.format(id=vid)
            try:
                blob = get(url, binary=True)
                if not blob.startswith(b"%PDF"):
                    raise ValueError(f"not a PDF ({len(blob)} bytes); {blob[:80]!r}")
                d = os.path.join(OUT, "brochures", str(crd)); os.makedirs(d, exist_ok=True)
                open(os.path.join(d, f"{crd}_{vid}.pdf"), "wb").write(blob)
                if _extract_text(blob, crd):
                    ok += 1; got = True; break
            except Exception as e:
                print(f"  brochure crd={crd} vid={vid}: {e}  url={url}")
        if not got:
            fail += 1
    print(f"brochures Part 2A: {ok} fetched+extracted, {skip} already had text, {fail} failed, {len(need_bulk)} not in on-disk mappings")
    if need_bulk:
        import csv as _csv
        nb = os.path.join(OUT, "candidates_need_bulk.csv")
        with open(nb, "w", newline="", encoding="utf-8") as f:
            w = _csv.writer(f); w.writerow(["crd"]); w.writerows([[c] for c in sorted(need_bulk, key=int)])
        print(f"  {len(need_bulk)} CRDs need the bulk archives (their brochure month is not mapped on disk):\n"
              f"  python research/adv_fetch.py brochures-bulk auto --candidates {nb}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "adv":
        fetch_adv(sys.argv[2] if len(sys.argv) > 2 else None)
    elif cmd == "13f-cover":
        fetch_13f_cover(sys.argv[2:] or ["2025q2", "2025q1"])
    elif cmd == "brochures-bulk":
        argv = sys.argv[2:]
        cand = None
        if "--candidates" in argv:
            i = argv.index("--candidates")
            if i + 1 >= len(argv):
                sys.exit("--candidates needs a file path, e.g. --candidates adv_data/candidates_etf.csv")
            cand = argv[i + 1]; argv = argv[:i] + argv[i + 2:]
        fetch_brochures_bulk(argv or ["auto"], cand)     # default: the latest 12 months (annual-update months first)
    elif cmd == "13f-value":
        fetch_13f_value(sys.argv[2] if len(sys.argv) > 2 else os.path.join(OUT, "review.csv"))
    elif cmd == "brochures":
        fetch_brochures(sys.argv[2] if len(sys.argv) > 2 else os.path.join(OUT, "candidates.csv"))
    else:
        print(__doc__)
