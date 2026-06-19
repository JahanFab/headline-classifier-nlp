# Headline Classifier with NLP

**Multi-Class Text Classification: A Comparative Analysis**

A comparative study of **news headline topic classification** using both traditional machine learning and deep learning. The project measures how different **preprocessing strategies**, **word representations**, and **model architectures** affect classification performance — and finds that deep sequence models with Skip-gram embeddings clearly outperform classical approaches, with a **Bidirectional GRU reaching 0.9203 accuracy**.

> Academic project — Department of Computer Science & Engineering, Brac University, Dhaka, Bangladesh.


## Problem

Text classification is a core NLP task underpinning applications like sentiment analysis and spam detection. With the volume of digital news growing, automatically sorting headlines into topics is a practical need. This project frames that as a **multi-class classification** problem and systematically compares the components of an NLP pipeline rather than chasing a single model.

## Dataset

| | |
|---|---|
| Training samples | 95,440 |
| Test samples | 12,000 |
| Columns | `News Headline`, `News Topic` |
| Classes | Business, Sports, World News, Science & Technology |

Class distribution (training): Business 30,563 · Sports 26,773 · World News 24,415 · Science & Technology 13,689. The data had no significant null values; duplicate headlines were intentionally kept to preserve class balance.

**Exploratory Data Analysis** covered class distribution plots, word- and character-length analysis (overall and per class), vocabulary richness, top words / bigrams / trigrams, stopword ratios, and word clouds before and after cleaning.

## Approach

### Preprocessing variants
Three levels of cleaning were compared head-to-head:

- **Raw** — no preprocessing
- **Extreme** — aggressive Lancaster stemming (+ lowercasing, stopword removal, tokenization)
- **Optimum** — lemmatization with Porter / Snowball stemming; **Snowball performed best**

### Word representations
- **TF-IDF** — captures term importance across documents, including n-grams
- **Skip-gram Word2Vec** — semantic embeddings trained on the corpus with gensim

### Models
- **Classical ML:** Logistic Regression (used for preprocessing comparison) and **Random Forest** (with hyperparameter tuning)
- **Deep learning:** Deep Neural Network (TF-IDF), and the recurrent family — **SimpleRNN, GRU, LSTM**, plus their **Bidirectional** variants — on Skip-gram embeddings

Key training techniques: hyperparameter tuning, early stopping, dropout regularization, and batch normalization.

## Results

Models were evaluated with **accuracy**, **macro-F1**, and **confusion matrices**. Best and worst model per preprocessing variant:

| Preprocessing | Best model | Accuracy | F1 (macro) | Worst model | Accuracy |
|---|---|---|---|---|---|
| Raw | GRU (Skip-gram) | 0.9197 | 0.9194 | Random Forest (TF-IDF) | 0.8523 |
| Extreme | DNN (TF-IDF) | 0.9115 | 0.9112 | Random Forest (TF-IDF) | 0.8700 |
| **Optimum** | **Bidirectional GRU (Skip-gram)** | **0.9203** | **0.9202** | Random Forest (TF-IDF) | 0.8728 |

**Takeaways:**

- Deep learning models substantially outperformed the classical ML baseline.
- **Bidirectional architectures beat their unidirectional counterparts** — context from both directions helps.
- **GRU and LSTM** outperformed SimpleRNN; the **Bidirectional GRU** was the overall best.
- Skip-gram embeddings gave a slight edge over TF-IDF for the sequence models.
- Preprocessing meaningfully shifted results, with **Optimum (Snowball)** giving the best top-end performance.
- Random Forest was a respectable baseline (0.85+) but was consistently outperformed by neural networks; the DNN beat Random Forest but lost to the sequence models.

## Tech stack

- **Data / EDA:** pandas, numpy, matplotlib, seaborn, wordcloud
- **Classical NLP & ML:** NLTK (stemmers, stopwords, lemmatizer), scikit-learn (TF-IDF, Random Forest, Logistic Regression, metrics)
- **Embeddings:** gensim (Word2Vec, Skip-gram)
- **Deep learning:** PyTorch (DNN, RNN / GRU / LSTM, with GPU support when available)

## Requirements

```bash
pip install pandas numpy matplotlib seaborn nltk wordcloud gensim scikit-learn torch
```

The notebook downloads the required NLTK corpora (`stopwords`, `wordnet`, `omw-1.4`) on first run. A CUDA GPU is used automatically for the neural models if available, otherwise it falls back to CPU.

## Usage
```bash
jupyter notebook headline_classifier.ipynb
```

Run the cells top to bottom. The EDA and preprocessing cells are quick; the RNN experiments are the most compute-intensive and benefit from a GPU.

> **Note:** the dataset paths are currently hardcoded to a local Windows path (`D:\fabia\...`). Update the `pd.read_csv(...)` lines near the top of the notebook to point to your own data (e.g. a `data/` folder).



