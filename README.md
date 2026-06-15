# Task-6-coretech-knowledge-base-semantic-search
# 🔍 CoreTech Knowledge Base — Semantic Search System
### Internship Task 6 | TF-IDF + Cosine Similarity

---

##  Project Overview

This project builds a **semantic search system** over a custom knowledge base for a fictional IT company, **CoreTech Solutions**. Given a natural language query, the system retrieves the **top 3 most relevant records** from the knowledge base and displays their **cosine similarity scores**.

---

##  Project Files

| File | Description |
|------|-------------|
| `coretech_knowledge_base.csv` | Knowledge base dataset (34 records) |
| `coretech_semantic_search.ipynb` | Google Colab notebook with full implementation |
| `README.md` | This file |

---

##  Dataset — `coretech_knowledge_base.csv`

The dataset contains **34 records** across 3 categories:

| Category | Count | Description |
|----------|-------|-------------|
| Service Description | 11 | Descriptions of CoreTech's IT services |
| FAQ | 14 | Common questions and answers |
| Company Info | 9 | About the company, leadership, offices, CSR |

### Columns

| Column | Type | Description |
|--------|------|-------------|
| `id` | int | Unique record identifier |
| `category` | str | One of: Service Description, FAQ, Company Info |
| `title` | str | Short title of the record |
| `content` | str | Full text content (100–250 words) |

---

##  How the Search System Works

### 1. Text Preprocessing
- The `title` and `content` fields are concatenated into a single `search_text` field per record.

### 2. TF-IDF Vectorisation
- **TF (Term Frequency):** How often a word appears in a document.
- **IDF (Inverse Document Frequency):** How rare/important the word is across all documents.
- **TF-IDF score** = TF × IDF — words that are frequent in a document but rare globally get high scores.
- Configuration used:
  - `stop_words='english'` — removes common words like "the", "is", "and"
  - `ngram_range=(1, 2)` — captures both single words and two-word phrases
  - `sublinear_tf=True` — applies log normalization to reduce impact of very frequent terms

### 3. Cosine Similarity
- The user query is transformed into a TF-IDF vector using the same fitted vectorizer.
- **Cosine similarity** is computed between the query vector and every document vector:

```
similarity = (A · B) / (||A|| × ||B||)
```

- Result is a value between **0** (no match) and **1** (perfect match).
- The **top 3** documents with the highest similarity scores are returned.

---

##  How to Run

### Option A — Google Colab (Recommended)

1. Go to [https://colab.research.google.com](https://colab.research.google.com)
2. Click **File → Upload notebook** and upload `coretech_semantic_search.ipynb`
3. Run all cells in order (`Runtime → Run all`)
4. The dataset is created automatically inside the notebook — no upload needed
5. In **Step 6**, type your own query in the input box and press Enter

### Option B — Run Locally

```bash
# 1. Clone / download the files
# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Jupyter
jupyter notebook coretech_semantic_search.ipynb
```

---

##  Requirements

```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.1.0
matplotlib>=3.6.0
```

>  All of the above are **pre-installed in Google Colab** — no manual installation required.

To install locally:
```bash
pip install pandas numpy scikit-learn matplotlib
```

---

##  Example Queries & Results

| Query | Top Match | Score |
|-------|-----------|-------|
| `How does CoreTech protect against cyber attacks?` | Cybersecurity Solutions | ~0.55 |
| `cloud hosting pricing and uptime` | Cloud Hosting Services | ~0.62 |
| `What is the response time for IT support tickets?` | SLA for Managed IT Support | ~0.58 |
| `When was CoreTech founded?` | About CoreTech | ~0.61 |
| `backup and disaster recovery RTO RPO` | Backup & Disaster Recovery | ~0.71 |

---

## 📐 Architecture Diagram

```
User Query
    │
    ▼
TfidfVectorizer.transform(query)
    │
    ▼
Query TF-IDF Vector  ──►  cosine_similarity()  ◄──  TF-IDF Matrix (34 docs)
                                  │
                                  ▼
                         Score Array [0.0 → 1.0]
                                  │
                                  ▼
                      argsort() → Top 3 Indices
                                  │
                                  ▼
                     Display: Rank | Score | Title | Content
```

---
