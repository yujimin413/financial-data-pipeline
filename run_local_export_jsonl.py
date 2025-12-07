# run_local_export_jsonl.py
# -*- coding: utf-8 -*-
import json
from pathlib import Path
from datetime import datetime, timedelta

from crawler_core import crawl_many

# ====== 크롤링 대상 종목 ======
STOCKS = [
    {"name": "삼성전자", "code": "005930"},
    {"name": "NAVER",   "code": "035420"},
    {"name": "카카오",   "code": "035720"},
    {"name": "현대차",   "code": "005380"},
]

# ====== 출력 루트 ======
OUT_ROOT = Path("./news_raw")

# ====== 수집할 날짜 구간(포함 범위) ======
# 요구: 7.7~7.11, 7.14~7.18, 7.21~7.25, 7.28~7.31, 8.4
# 연도는 예시 스크립트와 동일하게 2025년으로 가정
DATE_BLOCKS = [
    ("2025.08.06", "2025.08.8"),
    ("2025.08.11", "2025.08.14"),
]
SINGLE_DATES = ["2025.08.04"]  # 단일 날짜

# ====== 옵션 ======
MAX_PAGES_PER_STOCK = 200  # 네이버 종목뉴스 페이지 탐색 최대 페이지
PROGRESS_EVERY = 10        # 진행상황 표시 주기(건)

def safe_filename(name: str) -> str:
    """파일/폴더명 안전화: 금지문자 제거, 공백 압축"""
    import re
    s = str(name).strip()
    s = re.sub(r"[\\/:*?\"<>|]+", "_", s)  # 금지문자 → _
    s = re.sub(r"\s+", "_", s)             # 다중 공백 → _
    return s

def date_token(date_str: str) -> str:
    """'YYYY.MM.DD' → 'YYYYMMDD'"""
    return date_str.replace(".", "")

def _daterange_inclusive(start_str: str, end_str: str):
    """'YYYY.MM.DD' ~ 'YYYY.MM.DD' 포함 범위로 하루씩 생성"""
    start = datetime.strptime(start_str, "%Y.%m.%d")
    end   = datetime.strptime(end_str,   "%Y.%m.%d")
    cur = start
    while cur <= end:
        yield cur.strftime("%Y.%m.%d")
        cur += timedelta(days=1)

def _all_target_dates():
    dates = []
    for s, e in DATE_BLOCKS:
        dates.extend(list(_daterange_inclusive(s, e)))
    dates.extend(SINGLE_DATES)
    # 중복 제거 및 정렬
    dates = sorted(set(dates))
    return dates

def crawl_one_day(day_str: str):
    """하루치(YYYY.MM.DD) 수집 → news_raw/YYYYMMDD/{종목}.jsonl"""
    dt_tok = date_token(day_str)
    day_dir = OUT_ROOT / dt_tok
    day_dir.mkdir(parents=True, exist_ok=True)

    for stock in STOCKS:
        stock_name = stock["name"]
        out_path = day_dir / f"{safe_filename(stock_name)}.jsonl"

        print(f"\n📌 [{stock_name}] {day_str} 뉴스 크롤링 시작...")
        count = 0
        with open(out_path, "w", encoding="utf-8") as f:
            for r in crawl_many([stock], day_str, max_pages=MAX_PAGES_PER_STOCK):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                count += 1
                if count % PROGRESS_EVERY == 0:
                    print(f"  ├─ 진행: {count}건 수집 완료")
        print(f"✅ [{stock_name}] 완료: {count}건 저장 → {out_path}")

def main():
    targets = _all_target_dates()
    print("=== 수집 대상 날짜 ===")
    for d in targets:
        print(" -", d)
    print("=====================\n")

    for day in targets:
        crawl_one_day(day)

if __name__ == "__main__":
    main()
