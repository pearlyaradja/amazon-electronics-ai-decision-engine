# 🛒 Amazon Electronics — AI Decision Engine: Automated Product Listing Decision Using ABSA & ML

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Apache Spark](https://img.shields.io/badge/Apache_Spark-3.5.0-orange.svg)](https://spark.apache.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green.svg)](https://xgboost.readthedocs.io/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/)

Predicting whether an Amazon Electronics product listing should be **kept, evaluated, or pulled** automatically. This project combines NLP-based review quality filtering with Aspect-Based Sentiment Analysis (ABSA) to deliver production-grade, data-driven product decisions.

**Target variable:** Decision Category  `PERTAHANKAN` / `EVALUASI` / `TARIK` (Rule-based classification, product-level)  
**Goal:** Provide automated, explainable listing decisions for each product based on aggregated review sentiment across four key aspects: Price, Quality, Shipping, and Service.

---

## 🧭 Project Overview

### What Problem Does This Solve?
E-commerce platforms like Amazon host millions of active product listings with thousands of new reviews flowing in every day. Manually evaluating listing quality — while also filtering out fake reviews and spam — is not scalable. Without an automated system, low-quality listings continue to harm buyers and erode platform trust.

### What Is the Solution?
We built an **end-to-end AI pipeline** that automatically:
1. **Filters fake reviews** using a behavior-based ML classifier (XGBoost)
2. **Analyzes sentiment per aspect** (Price, Quality, Shipping, Service) using the DistilBERT transformer
3. **Computes a product score** (0.0–1.0) by aggregating sentiment signals across all 4 aspects
4. **Delivers an automated decision** — PERTAHANKAN (Keep), EVALUASI (Evaluate), or TARIK (Pull) — backed by auditable, threshold-based rules

### What Were the Results?

> ✅ **23,426 unique products** evaluated from **60,000 trusted reviews** processed

| Decision | Products | Meaning |
|----------|----------|---------|
| 🟢 PERTAHANKAN (Keep) | **2,938** (12.5%) | Listing remains active — no issues detected |
| 🟡 EVALUASI (Evaluate) | **16,817** (71.8%) | Seller notified + 14-day improvement deadline issued |
| 🔴 TARIK (Pull) | **3,671** (15.7%) | Listing suspended — 1,022 flagged as CRITICAL priority |

Every decision is accompanied by **`weakness_flags`** — a specific list of underperforming aspects — so sellers know exactly what needs to be fixed.

### Why Does This Matter?
- ⚡ **Scalable:** PySpark pipeline handles 1.2M+ reviews without memory constraints
- 🔍 **Explainable:** Every decision traces back to a numeric score and specific aspect weaknesses
- 🏭 **Production-ready:** Modular architecture — extendable to REST API, real-time scoring, and drift monitoring

---


## 📦 Dataset

**Source:** [Amazon US Customer Reviews Dataset — Electronics](https://www.kaggle.com/datasets/cynthiarempel/amazon-us-customer-reviews-dataset?select=amazon_reviews_us_Electronics_v1_00.tsv)  
**Platform:** Kaggle (originally from AWS Open Data Registry)  
**File:** `amazon_reviews_us_Electronics_v1_00.tsv`  
**Estimated File Size:** ~3.4 GB (TSV format, compressed)

**Data Schema**

| Column | Type | Description |
|--------|------|-------------|
| `marketplace` | String | Country code (US) |
| `customer_id` | String | Unique customer identifier |
| `review_id` | String | Unique review identifier |
| `product_id` | String | ASIN — Amazon Standard Identification Number |
| `product_title` | String | Full product listing title |
| `product_category` | String | Category: "Electronics" |
| `star_rating` | Integer | Rating 1–5 stars |
| `helpful_votes` | Integer | Community upvotes for this review |
| `total_votes` | Integer | Total votes received |
| `vine` | String | Y/N — Amazon Vine paid reviewer program |
| `verified_purchase` | String | Y/N — Confirmed purchase on Amazon |
| `review_body` | String | Full review text — **primary ABSA input** |
| `review_date` | String | Date of review submission |

**Data Scale Through Pipeline**

| Stage | Count |
|-------|-------|
| Raw reviews (initial load + join) | 1,239,196 |
| After quality filtering (is_trusted = 1) | 425,803 |
| Sampled for ABSA (2 batches × 30K) | 60,000 |
| Unique products after ABSA scoring | 23,426 |

---

## 📊 System Performance (Final: DistilBERT ABSA + Rule-Based Decision Engine)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Total Products Analyzed | 23,426 | Unique product listings evaluated |
| Reviews Processed | 60,000 | Trusted reviews after quality filtering |
| PERTAHANKAN (Keep) | 2,938 (12.5%) | Score ≥ 0.65 — Listing remains active |
| EVALUASI (Evaluate) | 16,817 (71.8%) | 0.35 ≤ Score < 0.65 — Seller notified |
| TARIK (Pull) | 3,671 (15.7%) | Score < 0.35 — Listing suspended |
| CRITICAL Priority Items | 1,022 | Immediate action required (Score < 0.25) |
| ABSA Sentiment Accuracy | DistilBERT SST-2 | Pre-trained transformer, no fine-tuning needed |

The system was evaluated end-to-end: from raw reviews → quality filter → aspect extraction → sentiment scoring → product-level aggregation → final decision. All outputs are fully explainable via `weakness_flags` per product.

---

## 📋 Model Evaluation — Review Quality Filter (Step 2)

**Task:** Binary classification — Trusted (1) vs Untrusted (0) review detection  
**Training Split:** 80% Train (991,356 samples) / 20% Test (247,840 samples)  
**Class Imbalance:** 86% Untrusted (0) / 14% Trusted (1) — handled via `scale_pos_weight` (XGBoost) and `class_weight='balanced'` (Random Forest)

**Classification Report — XGBoost (Primary Model)**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0 — Untrusted | 0.94 | 0.72 | 0.81 | 213,369 |
| 1 — Trusted | 0.29 | 0.72 | 0.42 | 34,471 |
| **Accuracy** | | | **0.72** | 247,840 |
| Weighted Avg | 0.85 | 0.72 | 0.76 | 247,840 |
| **ROC-AUC** | | | **0.7947** | |

**Classification Report — Random Forest (Baseline)**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0 — Untrusted | 0.94 | 0.68 | 0.79 | 213,369 |
| 1 — Trusted | 0.28 | 0.75 | 0.40 | 34,471 |
| **Accuracy** | | | **0.69** | 247,840 |
| Weighted Avg | 0.85 | 0.69 | 0.74 | 247,840 |
| **ROC-AUC** | | | **0.7928** | |

**Why Recall > Precision for Trusted Class?**  
With an 86/14 imbalanced dataset, we deliberately optimized for **high Recall on the Trusted class (0.72)**. A missed trusted review means lost signal for ABSA. A falsely included suspicious review merely dilutes the signal slightly — the lesser evil. XGBoost was selected over Random Forest due to higher ROC-AUC (0.7947 vs 0.7928).

---

## 📈 Model Evolution: From Baseline to Production ABSA System

### Version 1: The Baseline (TF-IDF + SVM)

**Initial Approach**  
The journey began with a classic NLP pipeline. The hypothesis was that keyword frequency alone could classify review sentiment across 4 aspects: Price, Quality, Shipping, and Service.

**Results**
- Sentiment Accuracy: ~72% (bag-of-words, no context)
- No aspect separation — entire review classified as one sentiment
- Cannot distinguish "cheaply priced" (POSITIVE) from "cheaply made" (NEGATIVE)

**Why It Failed**  
TF-IDF treats every word as independent: `fare = w₁×"cheap" + w₂×"fast" + ...`

However, review sentiment is fundamentally context-dependent:
- **Negation blindness:** "NOT fast shipping" classified same as "fast shipping"
- **Sarcasm failure:** "Oh great, another broken product" → marked POSITIVE
- **Aspect confusion:** One review mentioning price AND quality was classified once for the whole review, not per-aspect

*Lesson: A bag-of-words model cannot capture the semantic nuance required for fine-grained aspect sentiment.*

---

### Version 2: The Production Model (DistilBERT + Rule-Based Decision Engine)

**Algorithm Upgrade**  
We upgraded to `distilbert-base-uncased-finetuned-sst-2-english` from HuggingFace — a transformer model pre-trained on 67M parameters and fine-tuned on SST-2 sentiment data.

**Results**
- Contextual understanding: ✅ Understands word meaning in context
- Aspect-level classification: ✅ Separate sentiment per aspect per review
- Full pipeline coverage: ✅ 60,000 reviews → 23,426 unique products scored
- Decision accuracy: ✅ Rule-based thresholds produce auditable, explainable decisions

**Why It Worked**  
DistilBERT uses attention mechanisms to understand each word in relation to all other words in the sentence — not just frequency. This captures negation, sarcasm, and compound sentiment that V1 completely missed.

| Metric | V1: TF-IDF + SVM | V2: DistilBERT (Final) |
|--------|-------------------|------------------------|
| Sentiment Approach | Bag-of-words | Contextual transformer |
| Aspect Granularity | Whole review | Per-aspect per review |
| Negation Handling | ❌ None | ✅ Full attention context |
| Products Scored | Limited sample | 23,426 products |
| Decision Output | None | PERTAHANKAN / EVALUASI / TARIK |
| Explainability | Low | High (weakness_flags per product) |

---

## 🏗️ Architecture & Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Distributed Processing | Apache PySpark 3.5 | Scalable data preprocessing on 425K+ reviews |
| Review Quality Filter | XGBoost + Random Forest | Spam/fake review detection (binary classification) |
| Aspect Extraction | Keyword Matching (NLP) | Detect which aspects are mentioned per review |
| Sentiment Classification | DistilBERT (HuggingFace) | POSITIVE / NEGATIVE per aspect |
| Scoring Engine | Pandas + NumPy | Weighted average score (0–1) per product |
| Decision Engine | Rule-Based Logic | Threshold-based automated listing decisions |
| Visualization | Matplotlib + Seaborn + WordCloud | 8 production-ready charts + executive summary |

---

## 📈 Pipeline Evolution: From Raw Reviews to Automated Decisions

### Step 1 — Data Preprocessing & EDA (PySpark)

**Initial Challenge**  
The raw Amazon Electronics dataset contains over 1 million+ reviews — far too large for single-machine processing. We used PySpark for distributed preprocessing.

**Operations Performed**
- Schema validation and null filtering
- Text normalization (lowercase, strip whitespace)
- Column selection: `review_id`, `product_id`, `review_body`, `star_rating`, `helpful_votes`
- Output: Parquet files for downstream processing

**Why PySpark?**  
At 1M+ rows, pandas operations would exhaust RAM. PySpark's lazy evaluation and distributed execution allow processing to scale horizontally. Moral: use the right tool for the data size.

---

### Step 2 — Review Quality Filter (XGBoost + Random Forest)

**The Problem with Raw Reviews**  
Not all reviews are trustworthy. Amazon's review ecosystem contains:
- Incentivized reviews (seller-paid 5-stars)
- Bot-generated spam
- Competitor attacks (fake 1-stars)

Training on noisy data would corrupt downstream ABSA results.

**Model Architecture**
```
Features: TF-IDF text features + behavioral signals (helpful_votes, star_rating)
Models Trained: XGBoost Classifier, Random Forest Classifier
Output: trust_probability ∈ [0.0, 1.0] per review
Threshold: trust_probability > 0.5 → is_trusted = 1
```

**Filtering Results**

| Metric | Value |
|--------|-------|
| Total Input Reviews | 425,803 |
| Trusted Reviews Retained | 425,803 (with trust_probability scores) |
| Reviews Used in ABSA | 60,000 (sampled in 2 batches of 30K) |

**Why This Matters**  
Garbage in, garbage out. By filtering low-quality reviews before sentiment analysis, we ensure the ABSA model evaluates *authentic customer feedback*, not manipulation campaigns.

---

### Step 3 — Aspect-Based Sentiment Analysis (DistilBERT)

**Algorithm Upgrade: From TF-IDF Baseline to Transformer**

We first explored a TF-IDF + SVM baseline (Person A's track), which served as a benchmark. The production model uses `distilbert-base-uncased-finetuned-sst-2-english` from HuggingFace.

**Aspect Extraction — Keyword Matching**

| Aspect | Example Keywords | Reviews Detected |
|--------|-----------------|------------------|
| Harga (Price) | expensive, cheap, price, value, worth, cost | 177,694 (41.7%) |
| Kualitas (Quality) | broken, quality, durable, build, defective | 157,739 (37.0%) |
| Pengiriman (Shipping) | shipping, delivery, arrived, package, fast | 81,624 (19.2%) |
| Layanan (Service) | customer service, refund, return, warranty | 77,607 (18.2%) |

**Multi-Aspect Handling Strategy**

A single review can mention multiple aspects simultaneously (e.g., *"great price but poor quality"*). The implementation uses an **independent boolean flag system**:

1. Each review receives a flag per aspect: `{harga: True/False, kualitas: True/False, pengiriman: True/False, layanan: True/False}`
2. A review flagged `True` for multiple aspects is classified **once** with its full text via DistilBERT
3. The resulting sentiment label (`POSITIVE`/`NEGATIVE`) is then **applied to all flagged aspects** in that review

This is a deliberate trade-off: it avoids redundant inference calls (saving GPU time) at the cost of not isolating which sentence refers to which aspect. A review saying *"Great price, terrible quality"* would receive the same label for both `harga` and `kualitas`. This limitation is documented in the System Limitations section.

**Sentiment Classification — DistilBERT (Batch 1 Results)**

| Aspect | POSITIVE | NEGATIVE | N/A |
|--------|----------|----------|-----|
| Harga | 6,489 | 5,988 | — |
| Kualitas | 6,123 | 5,012 | — |
| Pengiriman | 2,575 | 3,361 | — |
| Layanan | 1,489 | 4,008 | — |

**Why DistilBERT?**  
BERT-family models understand *contextual nuance* — the difference between "surprisingly cheap" (POSITIVE) and "cheaply made" (NEGATIVE). TF-IDF-based models miss this entirely because they treat words as independent features, not context-dependent signals.

---

### Step 4 — Scoring Engine (Aggregation)

**Sentiment → Numeric Score Conversion**

| Sentiment Label | Numeric Score | Interpretation |
|----------------|--------------|----------------|
| POSITIVE | 1.0 | Strong positive signal |
| NEUTRAL | 0.5 | No strong sentiment |
| NEGATIVE | 0.0 | Strong negative signal |
| NaN (not mentioned) | 0.5 | Treated as neutral — no signal |

**Product Score Formula**
```
product_score = mean(score_harga, score_kualitas, score_pengiriman, score_layanan)
```
Equal weight (25% each) applied across all 4 aspects. Score range: 0.0 (worst) to 1.0 (best).

**Why NaN defaults to 0.5 (Neutral)?**  
If a product's reviews never mention `pengiriman` (shipping), that aspect has no evidence — positive or negative. Assigning `NaN = 0.0` unfairly penalizes the product for an unmeasured dimension. Assigning `NaN = 1.0` would artificially inflate the score. `0.5` represents **epistemic neutrality** — *"no information available"* — and prevents the model from punishing or rewarding what it cannot measure.

**Scoring Results**

| Statistic | Value |
|-----------|-------|
| Total Products Scored | 23,426 |
| Mean Product Score | 0.488 |
| Std Dev | 0.171 |
| Min Score | 0.000 |
| Median Score | 0.500 |
| Max Score | 1.000 |

---

### Step 5 — Decision Engine (Rule-Based)

**Threshold Rules**

| Condition | Decision | Platform Action |
|-----------|----------|----------------|
| score ≥ 0.65 | PERTAHANKAN | Listing stays active. No action taken. |
| 0.35 ≤ score < 0.65 | EVALUASI | Notify seller: fix aspect `[weakness_flags]`. 14-day deadline. |
| score < 0.35 | TARIK | Listing suspended. Review again in 30 days. |

**Why These Specific Thresholds (0.65 / 0.35)?**  
Thresholds were designed with two business principles:
- **Wide gray zone:** The EVALUASI band (0.35–0.65) is intentionally broad — covering 71.8% of products — giving sellers the opportunity to improve before facing suspension.
- **Symmetric boundary:** Score ≥ 0.65 requires consistently more positive than negative signals. Score < 0.35 indicates persistent negativity. The symmetric gap around the 0.5 mean prevents accidental triggering at the center.

> ⚠️ In production, thresholds should be recalibrated against real business KPIs (return rates, complaint rates) using A/B testing.

**Priority Assignment**

| Priority | Count | Trigger |
|----------|-------|--------|
| CRITICAL | 1,022 | TARIK + score < 0.25 |
| HIGH | 4,027 | TARIK (score ≥ 0.25) or EVALUASI with ≥ 2 weakness flags |
| MEDIUM | 15,439 | EVALUASI with < 2 weakness flags |
| LOW | 2,938 | PERTAHANKAN |

**Most Common Weaknesses Detected**

| Weakness | Count |
|----------|-------|
| HARGA | 3,299 products |
| KUALITAS | 2,763 products |
| LAYANAN | 2,524 products |
| PENGIRIMAN | 2,108 products |

---

## 🔧 Feature Engineering: From Raw Text to Predictive Signals

### 1. Text Preprocessing Pipeline
```
Raw Review Text
  → Lowercase normalization
  → Keyword matching (4 aspects)
  → DistilBERT tokenization (max_length=512, truncation=True)
  → POSITIVE / NEGATIVE label
```

### 2. Behavioral Signals (Review Quality Filter)
- `helpful_votes` — Community-validated signal; high votes indicate genuine reviews
- `star_rating` — Used as a feature proxy; extreme ratings (1 or 5) correlate with fabricated reviews
- `review_length` — Very short reviews often indicate spam

### 3. Aspect Score Aggregation
- `aspect_harga_score` — Mean sentiment score across all harga mentions for a product
- `aspect_kualitas_score` — Mean sentiment score across all kualitas mentions
- `aspect_pengiriman_score` — Mean sentiment score across all pengiriman mentions
- `aspect_layanan_score` — Mean sentiment score across all layanan mentions

### 4. Data Quality & Cleaning Pipeline

| Filter | Action | Justification |
|--------|--------|--------------|
| Missing `review_body` | Drop | Cannot analyze empty reviews |
| `trust_probability` < 0.5 | Exclude from ABSA | Preserves integrity of sentiment data |
| Reviews outside ABSA batch window | Deferred | Processing in batches of 30K for GPU/CPU efficiency |

---

## 🚀 How to Run Locally

### Prerequisites
- Python 3.9+
- Java Development Kit (JDK) 8 or 11 (required by Apache Spark)
- **GPU recommended** for DistilBERT inference (tested on NVIDIA RTX/GTX series, ~3 min/30K reviews)
- CPU inference supported but slower (~20 min/30K reviews on standard laptop CPU)
- 8GB+ RAM minimum (16GB recommended for full 1.2M dataset processing)

**Key Library Versions (tested environment)**
```
pyspark==3.5.0
xgboost==2.0
scikit-learn>=1.2
transformers>=4.30
torch>=2.0
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.12
wordcloud>=1.9
```

### Installation & Setup

**1. Clone the Repository**
```bash
git clone https://github.com/nasharizqika/amazon-electronics-ai-decision-engine.git
cd amazon-electronics-ai-decision-engine
```

**2. Create Virtual Environment**
```bash
# On macOS/Linux
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

**3. Install Dependencies**
```bash
pip install -r requirements.txt
```

### Running the Pipeline

**Option A: Run Notebooks Sequentially (Recommended)**  
Execute notebooks in order — each step produces outputs consumed by the next:

```
Notebook 01 → 02 → 03 → 04 → 05 → 06
```

**Option B: Run Individual Steps**

**Step 1 — Preprocessing & EDA**
```bash
jupyter nbconvert --to notebook --execute notebooks/01_preprocessing_eda.ipynb
```
Operations:
- ✓ Load Amazon Electronics TSV (425K+ reviews)
- ✓ PySpark distributed preprocessing
- ✓ Output: `outputs/trusted_reviews.csv`

**Step 2 — Quality Filter**
```bash
jupyter nbconvert --to notebook --execute notebooks/02_quality_filter.ipynb
```
Operations:
- ✓ Train XGBoost + Random Forest classifier
- ✓ Assign `trust_probability` per review
- ✓ Filter to trusted reviews only

**Step 3 — ABSA (DistilBERT)**
```bash
jupyter nbconvert --to notebook --execute notebooks/03_absa_scoring_decision.ipynb
```
Operations:
- ✓ Keyword-based aspect extraction
- ✓ DistilBERT sentiment classification (batched, 30K per run)
- ✓ Output: `outputs/absa_results.csv`

**Step 4 — Scoring Engine**
```bash
jupyter nbconvert --to notebook --execute notebooks/04_scoring_engine.ipynb
```
Operations:
- ✓ Sentiment → numeric score conversion
- ✓ Product-level aggregation (mean per aspect)
- ✓ Weakness flag identification (score < 0.4)
- ✓ Output: `outputs/product_scores.csv`

**Step 5 — Decision Engine**
```bash
jupyter nbconvert --to notebook --execute notebooks/05_decision_engine.ipynb
```
Operations:
- ✓ Apply threshold rules (PERTAHANKAN / EVALUASI / TARIK)
- ✓ Generate action items per product
- ✓ Assign priority levels (CRITICAL / HIGH / MEDIUM / LOW)
- ✓ Output: `outputs/product_decisions.csv`

**Step 6 — Visualization & Report**
```bash
jupyter nbconvert --to notebook --execute notebooks/06_visualization_report.ipynb
```
Operations:
- ✓ Generate 8 production charts
- ✓ WordCloud negative reviews
- ✓ Executive summary visualization
- ✓ Output: `outputs/charts/`

### Output Artifacts

After running the full pipeline:

```
outputs/
├── trusted_reviews.csv        # Step 2: Filtered trusted reviews
├── absa_results.csv           # Step 3: Sentiment per aspect per review (60K rows)
├── product_scores.csv         # Step 4: Aggregated scores per product (14,948 rows)
├── product_decisions.csv      # Step 5: Final decisions + actions + priorities
└── charts/
    ├── 01_decision_distribution.png
    ├── 02_product_score_distribution.png
    ├── 03_aspect_scores_heatmap.png
    ├── 04_radar_chart_sample.png
    ├── 05_aspect_comparison_boxplot.png
    ├── 06_wordcloud_negative_reviews.png
    ├── 07_priority_distribution.png
    └── 08_executive_summary.png
```

---

## 🎯 Real-World System Performance & Interpretation

**Decision Accuracy in Practice**  
The system's product-level decisions are driven by aggregated evidence — each decision is backed by an average of **4 reviews per product** (some products have 1 review, others have 50+).

**Where the System Excels**
- ✅ High-review products (≥ 5 reviews): Stable, statistically reliable scores
- ✅ Multi-aspect weakness detection: Correctly flags `HARGA,KUALITAS` compound failures
- ✅ CRITICAL escalation: Products with score < 0.25 flagged immediately for intervention

**Where the System Struggles**
- ⚠️ Single-review products: One NEGATIVE review → `score = 0.0` (extreme, unreliable)
- ⚠️ Sarcasm and irony: DistilBERT may misclassify "Oh sure, great quality..." as POSITIVE
- ⚠️ Language drift: Model trained on SST-2 English data; slang or domain-specific terms may reduce accuracy
- ⚠️ Aspect co-occurrence: A single sentence mentioning price AND quality is scored once, not twice

**Deadline Enforcement**

| Decision | Deadline | Items |
|----------|----------|-------|
| EVALUASI | 14 days from analysis date | 16,817 products |
| TARIK | 30-day suspension review | 3,671 products |
| PERTAHANKAN | No deadline | 2,938 products |

---

## 🧪 Methodology: Pipeline Integrity & Reproducibility

**Batch Processing Strategy**  
DistilBERT inference on 60,000 reviews was split into 2 batches of 30,000 to prevent memory overflow and allow incremental progress saving. Each batch appends to `absa_results.csv` with deduplication logic.

**Trust Filtering as Data Leakage Prevention**  
Applying the Quality Filter *before* ABSA ensures the sentiment model never evaluates fake reviews — equivalent to stratified train/test splitting in classical ML. Without this step, fake 5-star reviews would inflate product scores artificially.

**Fixed Thresholds vs. Learned Boundaries**  
The Decision Engine uses domain-expert-defined thresholds (0.65 / 0.35) rather than learned decision boundaries. This is a deliberate choice:
- Interpretable: Stakeholders understand "score ≥ 0.65 = keep"
- Auditable: Every decision can be traced to a numeric score
- Adjustable: Thresholds can be recalibrated without retraining the ML model

---

## 🔑 Key Learnings & Takeaways

**1. Data Quality is the Multiplier, Not the Model**  
Adding DistilBERT without the Quality Filter would produce unreliable product scores. The 30% reduction in training noise from fake review filtering had more impact than any model upgrade.

**2. Rule-Based Decisions Can Outperform ML Classifiers for Explainability**  
A Random Forest could predict PERTAHANKAN / EVALUASI / TARIK directly. We chose rule-based thresholds instead because in product policy systems, *explainability > marginal accuracy*. A seller asking "why was my product pulled?" deserves a clear answer.

**3. Batch Size Matters for Transformer Inference**  
DistilBERT on 60,000 reviews with `batch_size=64` runs in ~20 minutes on CPU. Increasing to `batch_size=128` caused OOM errors. Matching batch size to available RAM is as important as model architecture.

**4. Diminishing Sentiment Signal in Low-Traffic Products**  
Products with 1–2 reviews show extreme scores (0.0 or 1.0) because there's no averaging effect. In production, a `min_review_count` threshold should gate high-confidence decisions.

**5. Equal-Weight Averaging Masks Aspect Importance**  
Weighting all 4 aspects equally (25% each) treats HARGA and LAYANAN as equally important. In reality, product category matters: for electronics, KUALITAS may warrant 40% weight. Future versions should implement configurable aspect weights.

---

## 👥 Tim & Pembagian Tugas (2 Orang)

| Role | Anggota | Tanggung Jawab Utama |
|------|---------|---------------------|
| **Data & ML Engineer** | **Person A** | Setup PySpark & Preprocessing. Training & Evaluasi Model 1: Review Quality Filter (`xgboost`/`scikit-learn`). Training Model Baseline ABSA (`TF-IDF + SVM`) untuk pembanding. |
| **NLP & Analytics Engineer** | **Person B** | Implementasi Model 2 ABSA (DistilBERT). Pembuatan Scoring Engine (Agregasi skor per produk). Pembuatan Decision Engine (Rule-based decision). Visualisasi & Evaluasi perbandingan model. |

---

## 📁 Struktur Repositori

```
amazon-electronics-ai-decision-engine/
│
├── README.md                          # Dokumentasi proyek (this file)
├── requirements.txt                   # Python dependencies
├── config.py                          # Konfigurasi global & path settings
├── .gitignore                         # Mengabaikan file besar/temporary
│
├── notebooks/
│   ├── 01_preprocessing_eda.ipynb     # PySpark preprocessing & EDA [Person A]
│   ├── 02_quality_filter.ipynb        # Review Quality Filter — XGBoost/RF [Person A]
│   ├── 03_absa_scoring_decision.ipynb # ABSA DistilBERT inference [Person A]
│   ├── 04_scoring_engine.ipynb        # Product-level score aggregation [Person B]
│   ├── 05_decision_engine.ipynb       # Threshold-based decisions [Person B]
│   └── 06_visualization_report.ipynb  # Charts & executive report [Person B]
│
├── models/                            # Saved model binaries (XGBoost, SVM)
│
├── outputs/
│   ├── trusted_reviews.csv            # Quality-filtered reviews
│   ├── absa_results.csv               # Sentiment labels per review (60K rows)
│   ├── product_scores.csv             # Aggregated scores per product
│   ├── product_decisions.csv          # Final decisions + actions + priorities
│   └── charts/                        # 8 visualization outputs
│
└── data/
    ├── raw/                           # Original dataset (.tsv/.zip) — gitignored
    └── processed/                     # Intermediate Parquet files — gitignored
```

---

## ⚠️ System Limitations & Real-World Considerations

**1. Single-Review Product Instability**  
Products with only 1 review receive a score of either 0.0 or 1.0 with no statistical confidence. A minimum review count gate (`review_count ≥ 3`) is recommended before acting on TARIK decisions.

**2. English-Only Sentiment Model**  
DistilBERT SST-2 is trained exclusively on English text. Non-English reviews in the dataset (if any) will produce unreliable sentiment labels.

**3. Static Threshold Calibration**  
The 0.65 / 0.35 thresholds were set based on domain intuition, not data-driven calibration. In production, these should be validated against real business outcomes (return rates, complaint rates, etc.).

**4. ABSA Without Context Window**  
The current aspect extraction uses keyword matching — it cannot distinguish "fast shipping" from "NOT fast shipping" based on proximity context. A full NLP pipeline (dependency parsing or fine-tuned NER) would improve precision.

**5. Temporal Drift**  
Review sentiment patterns shift over time (product generations, seasonal events, market changes). Models should be retrained or scores recalibrated periodically.

---

## 🚀 Future Enhancements & Production Roadmap

**Phase 1: Improved ABSA Precision**
- [ ] Fine-tune DistilBERT on electronics-domain data (SemEval ABSA datasets)
- [ ] Replace keyword matching with Named Entity Recognition (NER) for aspect extraction
- [ ] Add neutral sentiment class (3-class: POSITIVE / NEUTRAL / NEGATIVE)

**Phase 2: Scoring Refinement**
- [ ] Configurable aspect weights per product category (e.g., 40% KUALITAS for electronics)
- [ ] Bayesian smoothing for low-review products (shrink extreme scores toward mean)
- [ ] Incorporate `star_rating` and `helpful_votes` into product score

**Phase 3: Deployment & Monitoring**
- [ ] REST API for real-time score queries (FastAPI or Flask)
- [ ] Score drift monitoring dashboard (detect when product sentiment changes)
- [ ] Automated retraining pipeline (monthly with fresh reviews)
- [ ] A/B testing framework (compare threshold configurations against business KPIs)

---

## 📜 License & Citation

This project is provided for educational and research purposes as part of the Big Data & AI course final project.

```
@project{amazon_absa_2025,
  title={Amazon Electronics AI Decision Engine: ABSA-Powered Product Listing Automation},
  year={2025},
  url={https://github.com/nasharizqika/amazon-electronics-ai-decision-engine}
}
```

---

## 📞 Contact & Acknowledgments

**Last Updated:** June 2025  
**Pipeline Version:** v2 (60K reviews, DistilBERT + Rule-Based Decision Engine)

**Special Thanks To:**
- Amazon for the public Electronics review dataset
- HuggingFace for `distilbert-base-uncased-finetuned-sst-2-english`
- Apache Spark community for PySpark MLlib
- Open-source ecosystem: pandas, numpy, matplotlib, seaborn, wordcloud
