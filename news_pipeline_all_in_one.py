# -*- coding: utf-8 -*-
"""
뉴스 파이프라인 올인원 (clean → dedup → summarize → csv → hourly/topic summaries)
- 1차 클린은 파일 출력하지 않고 메모리로만 처리합니다.
- 산출물 디렉터리에 단계 번호 접두(2_, 3_, 4_)를 붙입니다.

■ 실행 예시
python news_pipeline_all_in_one.py \
  --base "//Users/yujimin/KB AI CHALLENGE/project_pipeline/news" \
  --date 20250805 \
  --stocks 삼성전자 NAVER 카카오 현대차

■ 요구 패키지
pip install pandas numpy scikit-learn rapidfuzz transformers torch tqdm
"""

import argparse, json, re, sys, torch
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from transformers import pipeline
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering

# ─────────────────────────────────────────────────────────
# 공통 인자
# ─────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, help="베이스 데이터 디렉터리 (…/news)")
    p.add_argument("--date", required=True, help="타깃 날짜: YYYYMMDD")
    p.add_argument("--stocks", nargs="*", default=[], help="요약/후속 단계 처리할 종목명 리스트(미지정 시 dedup 파일명에서 추론)")
    p.add_argument("--cpu", action="store_true", help="요약을 CPU 강제")
    return p.parse_args()

# ─────────────────────────────────────────────────────────
# [clean] 보수적 클린 (메모리 처리)
# ─────────────────────────────────────────────────────────
def normalize_invisible_spaces(s: str) -> str:
    return (s.replace("\u200b", " ")
             .replace("\u200c", " ")
             .replace("\u200d", " ")
             .replace("\ufeff", " ")
             .replace("\xa0", " "))

RE_HTML         = re.compile(r"<[^>]+>")
RE_JS_DECL      = re.compile(r"(?:var|let|const)\s+[A-Za-z_$][A-Za-z0-9_$]*\s*=\s*[^;]+;")
RE_TOP_REDIRECT = re.compile(r"top\.location\.href\s*=\s*['\"][^'\"]+['\"];?")
RE_CREDIT_SRC   = re.compile(r"[\(\[]\s*(?:사진|영상|자료|그래픽|출처)\s*=\s*[^)\]]{1,200}[\)\]]")
RE_CREDIT_PROV  = re.compile(r"[\(\[]\s*[^)\]]{0,80}제공\s*[\)\]]")
RE_CREDIT_GRAB  = re.compile(r"[\(\[]\s*[^)\]]{0,80}갈무리\s*[\)\]]")
RE_CREDIT_DB    = re.compile(r"[\(\[]\s*[^)\]]{0,40}\bDB\b\s*[\)\]]")
RE_DECOS        = re.compile(r"[ⓒ©※◆■□▲△▼▽▶▷◀◁●○◐◑★☆◆◇■□•▪▫…─—–―︱│｜]+")
RE_SEPS         = re.compile(r"(?:={3,}|-{3,}|_{3,}|~{3,})")
RE_URL          = re.compile(r"https?://[^\s]+")
RE_MAIL         = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
RE_SPACE        = re.compile(r"\s+")

def clean_content(text: str) -> str:
    if not isinstance(text, str): return ""
    t = normalize_invisible_spaces(text)
    t = RE_HTML.sub(" ", t)
    t = RE_JS_DECL.sub(" ", t)
    t = RE_TOP_REDIRECT.sub(" ", t)
    t = RE_CREDIT_SRC.sub(" ", t)
    t = RE_CREDIT_PROV.sub(" ", t)
    t = RE_CREDIT_GRAB.sub(" ", t)
    t = RE_CREDIT_DB.sub(" ", t)
    t = RE_DECOS.sub(" ", t)
    t = RE_SEPS.sub(" ", t)
    t = RE_URL.sub(" ", t)
    t = RE_MAIL.sub(" ", t)
    t = RE_SPACE.sub(" ", t).strip()
    return t

def step_clean_in_memory(raw_dir: Path) -> Dict[str, List[dict]]:
    """
    raw_dir/{ticker}.jsonl → {ticker: [cleaned rows]}
    파일로 저장하지 않고 메모리로만 반환.
    """
    files = sorted(raw_dir.glob("*.jsonl"))
    if not files:
        print(f"[CLEAN][WARN] No jsonl under {raw_dir}")
        return {}
    print(f"[CLEAN] Found {len(files)} file(s) in {raw_dir}")

    rows_by_file: Dict[str, List[dict]] = {}
    for fp in files:
        total = kept = 0
        bucket: List[dict] = []
        with open(fp, "r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line: continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                total += 1
                obj["content"] = clean_content(obj.get("content", ""))
                bucket.append(obj)
                kept += 1
        rows_by_file[fp.stem] = bucket
        print(f"[CLEAN][OK] {fp.name}: {kept}/{total} (in-memory)")
    return rows_by_file

# ─────────────────────────────────────────────────────────
# [dedup] 보수적 중복제거 (파일 출력: 3_{YYYYMMDD}_dedup)
# ─────────────────────────────────────────────────────────
SP_MULTI = re.compile(r"\s+")
def normalize_text(text: str) -> str:
    if not isinstance(text, str): return ""
    t = (text.replace("\u200b"," ").replace("\u200c"," ").replace("\u200d"," ")
              .replace("\ufeff"," ").replace("\xa0"," "))
    return SP_MULTI.sub(" ", t).strip()

def dt_min(dt: str) -> str:  # "YYYY.MM.DD HH:MM"
    return (dt or "")[:16]

def dt_date(dt: str) -> str:  # "YYYY.MM.DD"
    return (dt or "")[:10]

def step_dedup_from_memory(clean_rows: Dict[str, List[dict]], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    SIM_ALMOST_SAME = 99
    LEN_DELTA_PCT   = 0.03
    SHORT_LEN_TH    = 180
    SIM_SHORT_GROUP = 97
    SIM_FINAL_GATE  = 98

    def almost_same(a: dict, b: dict) -> bool:
        if dt_min(a.get("datetime","")) != dt_min(b.get("datetime","")):
            return False
        s = fuzz.token_set_ratio(a["_norm"], b["_norm"])
        if s < SIM_ALMOST_SAME:
            return False
        la, lb = len(a["_norm"]), len(b["_norm"])
        if la == 0 or lb == 0: return True
        return abs(la - lb) / max(la, lb) <= LEN_DELTA_PCT

    def save_jsonl(rows: List[dict], path: Path):
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total_before = total_after = 0
    for stem, rows in sorted(clean_rows.items()):
        # A) 정규화
        for r in rows:
            r["_norm"] = normalize_text(r.get("content", ""))

        # A) 완전중복
        stats = dict(total=len(rows), exact=0, strong=0, shortgrp=0, final_sim=0)
        seen_hash, exact_kept = {}, []
        for r in rows:
            h = hash(r["_norm"])
            if h not in seen_hash:
                seen_hash[h] = r
                exact_kept.append(r)
            else:
                stats["exact"] += 1

        # B) (title, source, minute)
        strong_map: Dict[Tuple[str,str,str], List[dict]] = {}
        for r in exact_kept:
            key = (r.get("title"), r.get("source"), dt_min(r.get("datetime","")))
            strong_map.setdefault(key, []).append(r)

        strong_kept = []
        for key, group in strong_map.items():
            if len(group) == 1:
                strong_kept.append(group[0]); continue
            kept = []
            for g in sorted(group, key=lambda x: len(x["_norm"]), reverse=True):
                if any(almost_same(g, k) for k in kept):
                    stats["strong"] += 1
                    continue
                kept.append(g)
            strong_kept.extend(kept)

        # C) 단문 군집
        group_ts_date: Dict[Tuple[str,str,str], List[dict]] = {}
        for r in strong_kept:
            key = (r.get("title"), r.get("source"), dt_date(r.get("datetime","")))
            group_ts_date.setdefault(key, []).append(r)

        after_short = []
        for key, group in group_ts_date.items():
            shorts = [g for g in group if len(g["_norm"]) <= SHORT_LEN_TH]
            longs  = [g for g in group if len(g["_norm"]) > SHORT_LEN_TH]
            removed_here = 0
            if len(shorts) > 1:
                shorts_sorted = sorted(shorts, key=lambda x: len(x["_norm"]), reverse=True)
                picked = []
                for cand in shorts_sorted:
                    if all(fuzz.token_set_ratio(cand["_norm"], p["_norm"]) < SIM_SHORT_GROUP for p in picked):
                        picked.append(cand)
                    else:
                        removed_here += 1
                after_short.extend(picked)
            else:
                after_short.extend(shorts)
            after_short.extend(longs)
            stats["shortgrp"] += removed_here

        # D) 마지막 게이트 (title, source)
        final_map: Dict[Tuple[str,str], List[dict]] = {}
        for r in after_short:
            final_map.setdefault((r.get("title"), r.get("source")), []).append(r)

        final_rows = []
        for key, group in final_map.items():
            if len(group) == 1:
                final_rows.append(group[0]); continue
            kept: List[dict] = []
            for cand in sorted(group, key=lambda x: (len(x["_norm"]), x.get("datetime","")), reverse=True):
                if any(fuzz.token_set_ratio(cand["_norm"], k["_norm"]) >= SIM_FINAL_GATE for k in kept):
                    stats["final_sim"] += 1
                    continue
                kept.append(cand)
            final_rows.extend(kept)

        for r in final_rows:
            r.pop("_norm", None)

        out_fp = out_dir / f"{stem}.jsonl"
        save_jsonl(final_rows, out_fp)

        total_before += stats["total"]; total_after += len(final_rows)
        print(f"[DEDUP] {stem}.jsonl: {stats['total']} → {len(final_rows)} "
              f"(removed {stats['total']-len(final_rows)}) "
              f"[exact:{stats['exact']}, strong:{stats['strong']}, shortgrp:{stats['shortgrp']}, final:{stats['final_sim']}]")

    print(f"[DEDUP][ALL] {total_before} → {total_after} (removed {total_before-total_after})")
    print(f"[DEDUP][OUT] {out_dir}")
    return out_dir

# ─────────────────────────────────────────────────────────
# [summarize] (파일 출력: 4_news_summarized)
# ─────────────────────────────────────────────────────────
def chunk_text(text: str, max_chars: int = 1800):
    text = (text or "").strip()
    if len(text) <= max_chars: return [text] if text else []
    sents = re.split(r'(?<=[\.!?])\s+', text)
    chunks, buf = [], ""
    for s in sents:
        if len(buf) + len(s) + 1 > max_chars:
            if buf: chunks.append(buf.strip())
            buf = s
        else:
            buf = (buf + " " + s).strip() if buf else s
    if buf: chunks.append(buf.strip())
    return chunks

def build_summarizer(use_cpu: bool):
    device = -1 if use_cpu or (not torch.cuda.is_available()) else 0
    print("[SUMM] Using device:", "GPU" if device == 0 else "CPU")
    summarizer = pipeline(
        "summarization",
        model="gogamza/kobart-summarization",
        tokenizer="gogamza/kobart-summarization",
        device=device,
        model_kwargs={"torch_dtype": torch.float32},
    )
    summarizer.tokenizer.model_max_length = 1024
    return summarizer

def summarize_long(summarizer, text: str,
                   chunk_chars=1800, max_new_tokens=128, min_new_tokens=30, second_pass=True):
    if not text or len(text) < 50: return text or ""
    chunks = chunk_text(text, max_chars=chunk_chars)
    parts = []
    for ch in chunks:
        try:
            parts.append(summarizer(
                ch, truncation=True,
                max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens,
                do_sample=False
            )[0]["summary_text"])
        except Exception:
            subchunks = chunk_text(ch, max_chars=max(800, chunk_chars // 2))
            for sch in subchunks:
                try:
                    parts.append(summarizer(
                        sch, truncation=True,
                        max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens,
                        do_sample=False
                    )[0]["summary_text"])
                except:
                    pass
    if not parts: return ""
    if len(parts) == 1 or not second_pass: return parts[0]
    joined = " ".join(parts)
    try:
        return summarizer(
            joined, truncation=True,
            max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens,
            do_sample=False
        )[0]["summary_text"]
    except Exception:
        return joined[:max_new_tokens*10]

def step_group_and_summarize(dedup_dir: Path, stocks: List[str], out_dir: Path, use_cpu: bool) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    summarizer = build_summarizer(use_cpu)

    if not stocks:
        stocks = sorted({fp.stem.split("_")[0] for fp in dedup_dir.glob("*.jsonl")})

    for stock in stocks:
        input_file = dedup_dir / f"{stock}.jsonl"
        if not input_file.exists():
            print(f"[SUMM][MISS] {stock} 파일 없음: {input_file}")
            continue

        rows = []
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                try: rows.append(json.loads(line))
                except: pass
        if not rows:
            print(f"[SUMM][WARN] {stock}: 입력 비어있음")
            continue

        df = pd.DataFrame(rows)
        for col in ["datetime","title","content","link"]:
            if col not in df: df[col] = ""
        df['minute'] = pd.to_datetime(df['datetime'], errors="coerce").dt.strftime("%Y-%m-%d %H:%M").fillna("unknown")

        grouped = df.groupby(['minute', 'title']).agg({
            'content': lambda x: " ".join(map(str, x)),
            'link': list
        }).reset_index()

        summaries = []
        for _, row in tqdm(grouped.iterrows(), total=len(grouped), desc=f"{stock} summarizing"):
            summary = summarize_long(summarizer, row['content'])
            summaries.append({
                "minute": row['minute'],
                "title": row['title'],
                "summary": summary,
                "links": row['link']
            })

        output_file = out_dir / f"{stock}_summarized.jsonl"
        with open(output_file, "w", encoding="utf-8") as f:
            for s in summaries:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"[SUMM][OK] {stock} → {output_file}")
    return out_dir

# ─────────────────────────────────────────────────────────
# [csv] JSONL → CSV
# ─────────────────────────────────────────────────────────
def step_jsonl_to_csv(summ_dir: Path) -> List[Path]:
    csv_paths = []
    for jsonl_file in sorted(summ_dir.glob("*_summarized.jsonl")):
        rows = []
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                try: rows.append(json.loads(line))
                except: pass
        if not rows:
            print(f"[CSV][SKIP] empty: {jsonl_file}")
            continue
        df = pd.DataFrame(rows)
        if "links" in df.columns:
            df = df.drop(columns=["links"])
        csv_path = jsonl_file.with_suffix(".csv")
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        csv_paths.append(csv_path)
        print(f"[CSV][OK] {csv_path}")
    return csv_paths

# ─────────────────────────────────────────────────────────
# [summary] 시간대/토픽 요약 CSV
# ─────────────────────────────────────────────────────────
def normalize_space(s: str) -> str:
    if not isinstance(s, str): return ""
    return " ".join(s.replace("\xa0", " ").split()).strip()

def sentence_join(texts, max_len=400):
    out = []; total = 0
    for t in texts:
        t = normalize_space(t)
        if not t: continue
        if total + len(t) + 1 > max_len: break
        out.append(t); total += len(t) + 1
    return " ".join(out)

def build_hourly_summary(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["minute_dt"] = pd.to_datetime(df["minute"], errors="coerce")
    df = df.dropna(subset=["minute_dt"])
    df["hour"] = df["minute_dt"].dt.strftime("%H:00")

    by_hour = df.groupby("hour").agg(event_count=("title", "count")).reset_index().sort_values("hour")

    def rep_titles(sub, k=3):
        sub = sub.sort_values("minute_dt")
        seen = set(); reps = []
        for t in sub["title"]:
            key = re.sub(r"\s+", " ", str(t).lower())
            if key not in seen:
                seen.add(key); reps.append(t)
            if len(reps) >= k: break
        return " | ".join(reps)

    reps = df.groupby("hour").apply(rep_titles).reset_index(name="sample_titles")

    def top_keywords(sub, k=6):
        titles = [normalize_space(t) for t in sub["title"].tolist()]
        titles = [t for t in titles if t]
        if not titles: return ""
        vec = TfidfVectorizer(max_features=20000, ngram_range=(1,2), token_pattern=r"[가-힣A-Za-z]{2,}")
        X = vec.fit_transform(titles)
        scores = np.asarray(X.sum(axis=0)).ravel()
        idxs = np.argsort(-scores)[:k]
        feats = np.array(vec.get_feature_names_out())[idxs]
        return ", ".join(feats)

    kws = df.groupby("hour").apply(top_keywords).reset_index(name="top_keywords")

    def hour_summary(sub):
        texts = sub.sort_values("minute_dt")["summary"].fillna("").tolist()
        return sentence_join(texts, max_len=400)

    hrsum = df.groupby("hour").apply(hour_summary).reset_index(name="hour_summary")

    out = by_hour.merge(reps, on="hour").merge(kws, on="hour").merge(hrsum, on="hour")
    return out

def build_topic_summary(df: pd.DataFrame, cosine_cutoff: float = 0.55) -> pd.DataFrame:
    df = df.copy()
    df["minute_dt"] = pd.to_datetime(df["minute"], errors="coerce")
    df = df.dropna(subset=["minute_dt"]).reset_index(drop=True)

    docs = [(normalize_space(t) + " || " + normalize_space(s))
            for t, s in zip(df["title"], df["summary"])]

    if not docs:
        return pd.DataFrame(columns=[
            "event_id","n_articles","min_time","max_time","title_canonical","summary_concat"
        ])

    vec = TfidfVectorizer(max_features=30000, ngram_range=(1,2), token_pattern=r"[가-힣A-Za-z]{2,}")
    X = vec.fit_transform(docs).toarray().astype(np.float32)

    dist_th = 1.0 - cosine_cutoff
    agg = AgglomerativeClustering(metric="cosine", linkage="average",
                                  distance_threshold=dist_th, n_clusters=None)
    labels = agg.fit_predict(X)

    out_rows = []
    for lab in sorted(np.unique(labels)):
        idxs = np.where(labels == lab)[0].tolist()
        sub = df.iloc[idxs].copy()
        times = sub["minute_dt"].sort_values()
        min_time = times.iloc[0]; max_time = times.iloc[-1]
        title_canon = sub.iloc[0]["title"]
        summary_concat = sentence_join(sub["summary"].fillna("").tolist(), max_len=600)
        out_rows.append({
            "event_id": f"EVT_{lab:03d}",
            "n_articles": len(idxs),
            "min_time": min_time.strftime("%Y-%m-%d %H:%M"),
            "max_time": max_time.strftime("%Y-%m-%d %H:%M"),
            "title_canonical": title_canon,
            "summary_concat": summary_concat
        })
    return pd.DataFrame(out_rows)

def step_hourly_topic(csv_paths: List[Path]):
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        for c in ["minute","title","summary"]:
            if c not in df.columns:
                print(f"[SUMMARY][SKIP] '{csv_path.name}' 누락 컬럼: {c}")
                break
        else:
            hourly = build_hourly_summary(df)
            topic  = build_topic_summary(df, cosine_cutoff=0.55)
            stem   = csv_path.stem.replace("_summarized", "")
            hourly_path = csv_path.parent / f"{stem}_hourly_summary.csv"
            topic_path  = csv_path.parent / f"{stem}_topic_summary.csv"
            hourly.to_csv(hourly_path, index=False, encoding="utf-8-sig")
            topic.to_csv(topic_path, index=False, encoding="utf-8-sig")
            print(f"[SUMMARY][OK] {hourly_path.name}, {topic_path.name}")

# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────
def main():
    args = parse_args()
    base = Path(args.base).expanduser()
    yyyymmdd = args.date

    RAW_DIR   = base / "news_raw" / yyyymmdd
    DEDUP_DIR = base / "news_cleaned" / f"3_{yyyymmdd}_dedup"
    SUMM_DIR  = base / "news_cleaned" / "4_news_summarized"

    # 1) 클린(메모리)
    clean_rows = step_clean_in_memory(RAW_DIR)

    # 2) 중복 제거(파일 출력)
    dedup_dir = step_dedup_from_memory(clean_rows, DEDUP_DIR)

    # 3) 요약(파일 출력)
    summ_dir = step_group_and_summarize(dedup_dir, args.stocks, SUMM_DIR, use_cpu=args.cpu)

    # 4) JSONL → CSV
    csv_paths = step_jsonl_to_csv(summ_dir)

    # 5) 시간대/토픽 요약 CSV
    step_hourly_topic(csv_paths)

    print("\n[PIPELINE][DONE]")
    print(f"- Raw dir     : {RAW_DIR}")
    print(f"- Dedup dir   : {dedup_dir}")
    print(f"- Summ. dir   : {summ_dir}")
    print(f"- CSV files   : {[p.name for p in csv_paths]}")

if __name__ == "__main__":
    main()
