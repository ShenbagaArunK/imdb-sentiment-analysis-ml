# IMDB Movie Review Sentiment Analysis

> 50,000 movie reviews. Two vectorization strategies. One question: can a machine read a room?

An end-to-end NLP pipeline that classifies IMDB movie reviews as positive or negative — from raw text cleaning through feature engineering to a head-to-head comparison of **Bag-of-Words vs TF-IDF**, with both models saved for real-time prediction.

---

## 📌 What This Project Does

Raw movie reviews are messy — HTML tags, punctuation, casing, and stopwords that mostly add noise. This pipeline cleans them, converts text into numerical features two different ways, trains a Naive Bayes classifier on each, and compares which representation wins.

---

## 🔬 Pipeline

**1. Text Preprocessing**
- Lowercasing, HTML tag removal, punctuation stripping (regex)
- Tokenization + **Lancaster stemming**
- Custom stopword handling — **negation words (`not`, `won't`, `didn't`...) are deliberately kept**, since they carry sentiment and dropping them would flip meaning

**2. Exploratory Visualization**
- Word frequency bar charts
- Word clouds for the full corpus, and separately for positive vs negative reviews

**3. Feature Engineering — two approaches compared**
- **Bag-of-Words** (`CountVectorizer`, 1000 features)
- **TF-IDF** (`TfidfVectorizer`, 1000 features, bigrams via `ngram_range=(1,2)`)

**4. Modeling**
- `GaussianNB` trained on both feature sets
- MinMax scaling, 80/20 train-test split
- Evaluated with accuracy, classification report, and confusion matrix

**5. Real-Time Prediction**
- Helper functions wrap preprocessing + vectorization + inference so any new review string returns a sentiment instantly

**6. Model Persistence**
- Both trained models serialized with `joblib` (`.pkl`) for reuse

---

## 🛠 Stack

`Python` `scikit-learn` `NLTK` `Pandas` `NumPy` `Matplotlib` `Seaborn` `WordCloud` `joblib`

**Techniques:** Text Preprocessing · Stemming · CountVectorizer · TF-IDF · Naive Bayes · Model Comparison

---

## 🗂 Repository Structure

```
sentiment-analysis.ipynb              → Full pipeline notebook
IMDB_Dataset.csv                      → 50K labeled movie reviews
Movie_review_sentiment_BOW.pkl        → Saved Bag-of-Words model
Movie_review_sentiment_TF-IDF.pkl     → Saved TF-IDF model
```

---

## 🚀 Run It

```bash
pip install scikit-learn nltk pandas numpy matplotlib seaborn wordcloud mlxtend joblib
jupyter notebook sentiment-analysis.ipynb
```

```python
import nltk
nltk.download('stopwords')
```

---

## 📬 Connect

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/shenbaga-arun/)
[![Portfolio](https://img.shields.io/badge/Portfolio-FF5722?style=flat&logo=todoist&logoColor=white)](https://shenbagaarunk.github.io/)
