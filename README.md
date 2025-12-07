# financial-data-pipeline
Hybrid Financial Data Pipeline combining Statistical Logic (TF-IDF, Clustering) and LLMs.

# 📈 M-able 룰틴 (M-able Rule-tine): Financial Data Pipeline
> **Cost-Efficient Hybrid AI Pipeline for Personalized Investment Coaching**

[![Python](https://img.shields.io/badge/Python-3.9-blue?logo=python)](https://www.python.org/) [![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/) [![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Transformers-yellow)](https://huggingface.co/) [![PostgreSQL](https://img.shields.io/badge/PostgreSQL-14-336791?logo=postgresql)](https://www.postgresql.org/)

---

## 📝 Project Overview
**M-able 룰틴** 프로젝트에서 **금융 뉴스 데이터 수집, 전처리, 분석 파이프라인**을 전담하여 개발했습니다.
본 리포지토리는 무분별한 LLM 사용을 지양하고, **통계적 기법(TF-IDF, Clustering)과 경량 AI를 혼합한 하이브리드 설계**를 통해 데이터 처리 비용을 최적화한 **Data Engineering 로직**을 담고 있습니다.

### 🎯 My Data Engineering Contributions
* **Hybrid News Pipeline:** `RapidFuzz`와 `TF-IDF`로 전처리 후 `KoBART`로 요약하는 4단계 파이프라인 구축
* **Advanced Deduplication:** 단순 텍스트 매칭이 아닌, 시간/출처/토큰 유사도를 복합적으로 고려한 **Multi-Stage 중복 제거 로직** 구현
* **Topic Clustering:** `AgglomerativeClustering`을 활용해 파편화된 뉴스를 '이벤트(Event)' 단위로 묶어 시장의 맥락(Context) 추출
* **Data Schema Design:** 분석에 용이하도록 정규화된 뉴스/매매 로그 데이터베이스 스키마 설계

---

## 🏗️ Data Processing Architecture
고비용 리소스(LLM) 투입 전, CPU 기반의 통계적 필터링을 선행하여 비용 효율을 극대화한 구조입니다.

```mermaid
graph TD
    A["Raw Crawling Data"] -->|"1. Cleansing (Regex)"| B("Noise Reduction")
    B -->|"2. Multi-Stage Dedup (RapidFuzz)"| C{"Unique Articles"}
    C -->|"3. Summary (KoBART)"| D["Short Context"]
    D -->|"4. Topic Clustering (TF-IDF/Cosine)"| E["Market Events"]
    E -->|"5. Insight Analysis (GPT-4)"| F["Structured Data"]
````

### 🔨 Key Implementation Details

#### 1\. Robust Crawler & Cleansing [`crawler_core.py`]

  * **Anti-Bot & Redirect Handling:** 네이버 금융의 JS Redirect(`top.location.href`) 패턴을 추적하여 실제 기사 본문 원본 확보.
  * **Regex Cleansing:** 기자 이메일, 광고성 문구, 불필요한 HTML 태그를 정규식으로 제거하여 분석 가능한 텍스트로 정제.

#### 2\. Multi-Stage Deduplication [`news_pipeline_all_in_one.py`]

단순히 똑같은 글만 지우는 것이 아니라, 4단계 거름망을 통해 **정보 가치는 같고 표현만 다른 기사**를 제거했습니다.

1.  **Exact Match:** 해시값 기반 완전 중복 제거.
2.  **Strong Match:** 같은 시간, 같은 제목, 같은 출처의 기사 그룹핑.
3.  **Short Text Grouping:** 단신 기사들의 유사도 비교.
4.  **Final Gate:** `RapidFuzz` Token Set Ratio \> 98%인 경우 통합.

#### 3\. Topic Clustering & Summarization [`news_pipeline_all_in_one.py`]

  * **Hierarchical Clustering:** `TfidfVectorizer`로 벡터화된 뉴스 제목/요약본을 `AgglomerativeClustering` (Cosine Distance)으로 묶어, 개별 기사가 아닌 **'하나의 사건(Event)'** 단위로 구조화했습니다.
  * **Two-Pass Summarization:** `KoBART` 모델을 사용하여 1800자가 넘는 장문 기사를 청킹(Chunking) 후 재요약하는 방식으로 정보 손실을 최소화했습니다.

-----

## 💡 Problem Solving (Deep Dive)

### 🚀 Issue 1: 데이터 노이즈와 비용 효율 (Cost Efficiency)

> *"하루 수만 건의 뉴스 데이터를 전부 GPT에 넣으면 API 비용이 감당 불가능한 수준"*

  * **Solution:** **Hybrid Pipeline Design**
      * Python 라이브러리(`Scikit-learn`, `RapidFuzz`)를 활용한 전처리 단계에서 중복/노이즈를 **약 60% 사전 제거**.
      * 정제된 '고밀도 정보'만 LLM에 입력하여 토큰 비용 최소화 및 할루시네이션 방지.

### 🧠 Issue 2: 과거 데이터 공백 (Data Gap)

> *"API가 과거 분봉(Minute-candle) 데이터를 제공하지 않아 매매 당시 차트 복기 불가능"*

  * **Solution:** **OHLCV 추세 역산출 알고리즘**
      * 확보 가능한 **일봉(OHLCV)** 데이터를 활용해 당시의 변동성(Volatility)과 추세 위치를 추정하는 통계적 보간 로직 구현.
      * 매매 당시 사용자가 '추격 매수'를 했는지, '저점 매수'를 했는지 패턴 분석 성공.

-----

## 🛠️ Tech Stack

| Category | Technology | Usage in Project |
| :--- | :--- | :--- |
| **Language** | Python 3.9 | Data Pipeline Implementation |
| **NLP / ML** | **Scikit-learn** | `TfidfVectorizer`, `AgglomerativeClustering` for Topic Modeling |
| **NLP / ML** | **RapidFuzz** | High-performance String Matching for Deduplication |
| **AI Model** | **KoBART** | Korean Text Summarization (`gogamza/kobart-summarization`) |
| **Crawling** | BeautifulSoup4 | Static/Dynamic Content Parsing |
| **Data Storage** | PostgreSQL | Data Schema Design & Logging |

-----

## 📂 Directory Structure

```bash
financial-data-pipeline/
├── crawler_core.py             # 네이버 금융 뉴스 크롤링 엔진 (JS Redirect 처리)
├── run_local_export_jsonl.py   # 종목별/일자별 크롤링 실행 스크립트
├── news_pipeline_all_in_one.py # [Core] 전처리-중복제거-요약-클러스터링 파이프라인
├── requirements.txt            # 의존성 패키지 (transformers, rapidfuzz 등)
└── README.md                   # 프로젝트 설명서
```
