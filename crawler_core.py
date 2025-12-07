# crawler_core.py
# -*- coding: utf-8 -*-
import requests, re, time
from bs4 import BeautifulSoup
from datetime import datetime

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/114.0.0.0 Safari/537.36"
    ),
    "Referer": "https://finance.naver.com/",
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7"
}

def _clean_text(html_fragment):
    return re.sub(r'<[^>]+>', '', str(html_fragment)).strip() if html_fragment else ""

def _fetch_article_content(link, session, depth=0):
    if depth > 2:
        return ""
    try:
        resp = session.get(link, timeout=10)
        resp.encoding = resp.apparent_encoding
        soup = BeautifulSoup(resp.text, "html.parser")

        # JS redirect 대응
        redirect = soup.find("script", text=re.compile(r"top\.location\.href"))
        if redirect and redirect.string:
            m = re.search(r"top\.location\.href\s*=\s*'([^']+)';", redirect.string)
            if m:
                return _fetch_article_content(m.group(1), session, depth+1)

        for sel in ("#articleBodyContents",
                    "div#contents .newsct_body",
                    "article#dic_area",
                    "div.newsct_body"):
            body = soup.select_one(sel)
            if body:
                return _clean_text(body)
        return _clean_text(soup)
    except Exception:
        return ""

def crawl_one_stock(stock_name, stock_code, target_date_str, max_pages=200, sleep_sec=0.2):
    """
    yield dict(record) 형태:
    {
      'stock_name', 'datetime'(YYYY.MM.DD HH:MM),
      'title','source','content','link','stock_code'
    }
    """
    base_url = "https://finance.naver.com"
    url_tpl  = f"{base_url}/item/news_news.naver?code={stock_code}&page={{page}}&clusterId="
    target_dt = datetime.strptime(target_date_str, "%Y.%m.%d")

    session = requests.Session()
    session.headers.update(HEADERS)
    visited = set()

    for page in range(1, max_pages + 1):
        resp = session.get(url_tpl.format(page=page), timeout=10)
        resp.encoding = resp.apparent_encoding
        soup = BeautifulSoup(resp.text, "html.parser")
        rows = soup.select('table[summary="종목뉴스의 제목, 정보제공, 날짜"] tbody > tr')
        if not rows:
            break

        for tr in rows:
            td_date = tr.find("td", class_="date")
            if not td_date:
                continue

            date_time = " ".join(td_date.stripped_strings)  # 'YYYY.MM.DD HH:MM'
            date_part = date_time.split()[0]
            row_dt = datetime.strptime(date_part, "%Y.%m.%d")

            # 날짜 필터: target_date '하루'만 수집
            if row_dt < target_dt:
                return
            if row_dt > target_dt:
                continue

            a_tag = tr.find("a", class_="tit")
            info  = tr.find("td", class_="info")
            if not a_tag or not info:
                continue

            href  = a_tag.get("href", "")
            m_id  = re.search(r"article_id=(\d+).+office_id=(\d+)", href)
            if m_id:
                key = f"{m_id.group(2)}_{m_id.group(1)}"
                if key in visited:
                    continue
                visited.add(key)

            link = href if href.startswith("http") else base_url + href
            content = _fetch_article_content(link, session)
            time.sleep(sleep_sec)

            yield {
                "stock_name": stock_name,
                "stock_code": stock_code,
                "datetime": date_time,
                "title": a_tag.get_text(strip=True),
                "source": info.get_text(strip=True),
                "content": content,
                "link": link,
            }

def crawl_many(stocks, target_date_str, max_pages=200):
    for s in stocks:
        yield from crawl_one_stock(s["name"], s["code"], target_date_str, max_pages=max_pages)

# -----------------------------
# (옵션) 가벼운 전처리 유틸
# -----------------------------
def basic_preprocess(records):
    """
    - content 공란 제거
    - link 기준 중복 제거
    - 제목/본문 짧은 글 필터링(옵션)
    """
    seen = set()
    out = []
    for r in records:
        if not r.get("content", "").strip():
            continue
        link = r.get("link")
        if link in seen:
            continue
        seen.add(link)
        if len(r.get("title","")) < 3:
            continue
        if len(r.get("content","")) < 20:
            continue
        out.append(r)
    return out
