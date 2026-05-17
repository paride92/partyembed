# partyembed

> **This package is a repackaging of [lrheault/partyembed](https://github.com/lrheault/partyembed) by Ludovic Rheault and Christopher Cochrane. All credit for the methodology, algorithms, and research belongs to the original authors. This project only adds a flexible API layer so the analysis can be applied to any text corpus. If you use this in research, cite the original paper (see below).**

A flexible Python package for analysing ideological positioning in political text using Doc2Vec embeddings enriched with group metadata (party, session, legislature, etc.).

The original [partyembed](https://github.com/lrheault/partyembed) was designed for pre-built US/Canada/UK legislative datasets. This version generalises the same approach so it can be applied to **any** text corpus by specifying column names, metadata structure, preprocessing options, and language.

---

## Installation

```bash
pip install git+https://github.com/paride92/partyembed.git
```

### Requirements

- Python 3.9+
- gensim >= 4.0
- numpy, pandas, scikit-learn, scipy, matplotlib, nltk

---

## Overview

The package follows a three-step pipeline:

```
Raw text file  →  Preprocessor  →  Trainer  →  Explore
```

1. **Configure** how to read your data (`CorpusConfig`) and how to clean it (`PreprocessConfig`).
2. **Train** a Doc2Vec model that tags each document with group metadata (`Trainer`).
3. **Explore** ideological placement, topic ownership, and polarization (`Explore`).

---

## Step 1 — Configure

### `PreprocessConfig`

Controls text cleaning.

```python
from partyembed import PreprocessConfig

prep_cfg = PreprocessConfig(
    custom_stopwords=["presidente", "onorevole", "ministro"],  # domain-specific words to remove
    nltk_languages=["italian"],        # NLTK stopword list(s) to include
    use_sklearn_stopwords=False,       # sklearn list is English-only; skip for other languages
    expand_contractions=False,         # set True for English only
    apostrophe_handling="space",       # 'space' | 'remove' | 'keep'
    remove_accents=False,              # keep è, à, ù etc. for non-English text
    lowercase=True,
    min_token_length=3,
    remove_digits=True,
)
```

| Parameter | Default | Description |
|---|---|---|
| `custom_stopwords` | `[]` | Any extra words to remove |
| `nltk_languages` | `[]` | NLTK language names, e.g. `['italian', 'english']`. Supports ~25 languages |
| `use_sklearn_stopwords` | `True` | sklearn English stopword list (~318 words) |
| `expand_contractions` | `True` | Expand English contractions ("won't" → "will not") |
| `apostrophe_handling` | `"space"` | `"space"` → `l'acqua` becomes `l acqua`; `"remove"` → `lacqua`; `"keep"` → untouched |
| `remove_accents` | `True` | Strip diacritics via ASCII normalisation — set `False` for non-English text |
| `min_token_length` | `3` | Discard tokens shorter than this |
| `remove_digits` | `True` | Discard purely numeric tokens |

**NLTK supported languages include:** arabic, danish, dutch, english, finnish, french, german, hungarian, italian, norwegian, portuguese, romanian, russian, spanish, swedish, turkish. The stopword corpus is downloaded automatically on first use.

---

### `CorpusConfig`

Describes your data file layout.

```python
from partyembed import CorpusConfig

corpus_cfg = CorpusConfig(
    text_column="speech_text",        # column name (str) or index (int) containing text
    tag_columns=["party", "congress"],# columns used to build document tags
    tag_separator="_",                # joins tag columns: "PD_18" (party + congress)
    file_format="csv",                # 'csv' | 'tsv' | 'json' | 'parquet'
    encoding="utf-8",
    header=True,
    filter_column="chamber",          # optional: keep only rows where...
    filter_value="Camera",            # ...this column equals this value
    preprocessed=False,               # True if text is already cleaned
)
```

The composite document tag is built by joining `tag_columns` with `tag_separator`. For example, with `tag_columns=["party", "congress"]` and `tag_separator="_"`, a row with `party="PD"` and `congress=18` produces tag `"PD_18"`.

To use a custom format string instead:

```python
corpus_cfg = CorpusConfig(
    tag_columns=["party", "legislature"],
    tag_format="{party}_{legislature}",   # explicit format, overrides tag_separator
)
```

---

### `ModelConfig`

Doc2Vec hyperparameters.

```python
from partyembed import ModelConfig

model_cfg = ModelConfig(
    vector_size=200,       # embedding dimensionality
    window=20,             # context window size
    min_count=50,          # ignore words appearing fewer than this many times
    workers=8,             # parallel threads
    epochs=5,              # training passes over the corpus
    dm=0,                  # 0 = DBOW (faster, usually better for short docs); 1 = DM
    use_bigrams=True,      # detect multi-word expressions
    use_trigrams=True,
    bigram_min_count=5,    # minimum count for a phrase to be kept
    bigram_threshold=10.0, # higher = fewer phrases detected
)
```

---

## Step 2 — Train

```python
from partyembed import Trainer
import logging
logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)

trainer = Trainer(corpus_cfg, model_cfg, prep_cfg)  # accepts PreprocessConfig or Preprocessor
model = trainer.train("data/speeches.csv", save_path="models/italian_parliament")
```

`trainer.train()` runs in three phases:
1. **Phrase detection** — detects bigrams (and trigrams if enabled) across the corpus.
2. **Vocabulary building** — counts words and builds the model vocabulary.
3. **Training** — runs Doc2Vec for the specified number of epochs.

Progress is printed via Python's `logging` module. To enable it, add the `basicConfig` line shown above before calling `train()`.

### Saving and loading phrases

If you want to reuse the phrase detectors without retraining:

```python
trainer.fit_phrases("data/speeches.csv", save_path="models/phrases")
# later:
trainer.load_phrases("models/phrases_bigram", "models/phrases_trigram")
model = trainer.train("data/speeches.csv")
```

### Loading a saved model

```python
from gensim.models.doc2vec import Doc2Vec
model = Doc2Vec.load("models/italian_parliament")
```

---

## Step 3 — Explore

### Loading

```python
from partyembed import Explore

m = Explore(
    model="models/italian_parliament",   # path string or Doc2Vec object
    group_labels={"PD": "Partito Democratico", "FI": "Forza Italia", "M5S": "Movimento 5 Stelle"},
    group_colors={"PD": "#CC0000", "FI": "#0070C0", "M5S": "#FFD700"},
)
```

By default, `Explore` includes all document tags in the model and assumes tags are formatted as `{group}_{time}` (everything before the last `_` is the group; everything after is the time period). You can customise this:

```python
m = Explore(
    model=model,
    tags=["PD_17", "PD_18", "FI_17", "FI_18"],   # restrict to specific tags
    group_fn=lambda t: t.split("_")[0],            # tag → group key
    time_fn=lambda t: t.split("_")[1],             # tag → time period label
)
```

### Scatter plot — party positions

```python
m.plot()
m.plot(axis_labels=("Component 1", "Component 2"), save_path="figures/positions.png")
```

### Time-series plot — ideological change over time

```python
m.plot_timeseries(dimension=1)   # first component over time
m.plot_timeseries(dimension=2)   # second component over time
```

### PCA vs. guided projection

By default `method='pca'` performs principal component analysis on the group vectors. Alternatively, `method='guided'` projects onto axes defined by a political lexicon:

```python
# With the default English lexicon (axis 1 = Economic, axis 2 = Social)
m = Explore(model, method='guided')

# With a custom Italian lexicon (4 word lists required for 2-D)
my_lexicon = [
    # [0] Left on economic axis
    ["povertà", "disuguaglianza", "salario_minimo", "sindacati", "redistribuzione"],
    # [1] Right on economic axis
    ["privatizzazione", "libero_mercato", "imprese", "competitività", "contribuenti"],
    # [2] Progressive on social axis
    ["diritti_civili", "ambiente", "clima", "parità"],
    # [3] Conservative on social axis
    ["sicurezza", "tradizione", "famiglia", "confini", "chiesa"],
]
m = Explore(model, method='guided', custom_lexicon=my_lexicon)
m.plot(axis_labels=("Sinistra–Destra Economica", "Progressista–Conservatore"))
```

Words not found in the model vocabulary are silently skipped. Check coverage first:

```python
vocab = set(model.wv.key_to_index)
for i, word_list in enumerate(my_lexicon):
    found = [w for w in word_list if w in vocab]
    print(f"List {i}: {len(found)}/{len(word_list)} found — {found}")
```

### Axis orientation

By default axes are not flipped. To orient them so that a specific group appears on a specific side:

```python
m = Explore(
    model,
    orientation={
        "dim1": ("PD_18", "FI_18"),   # PD should be left of FI on dim 1
    }
)
```

---

### Issue ownership — topic salience by group

Measures how closely each group's embedding aligns with a topic over time.

```python
df = m.issue(
    "sanita",      # seed word (must be in model vocabulary)
    n=30,          # words used to build the topic vector (seed + n-1 neighbours)
    bootstrap=True,# compute 95% confidence intervals
    n_boot=1000,
    smooth=3,      # rolling average over 3 time periods (0 = no smoothing)
)
```

The returned DataFrame has columns `tag`, `group`, `time`, `similarity`, and (if `bootstrap=True`) `lb` and `ub`.

```python
# Wide format — one column per group
df.pivot(index="time", columns="group", values="similarity")

# Which group scores highest on average?
df.groupby("group")["similarity"].mean().sort_values(ascending=False)

# Plot over time
import matplotlib.pyplot as plt
for grp, sub in df.groupby("group"):
    plt.plot(sub["time"], sub["similarity"], label=grp)
    if "lb" in df.columns:
        plt.fill_between(sub["time"], sub["lb"], sub["ub"], alpha=0.2)
plt.legend(); plt.show()
```

**Interpreting similarity scores:** higher values mean the group's speech occupies a region of the embedding space closer to the topic. Scores are relative — compare groups against each other within the same model, not across models. If the `lb` of one group exceeds the `ub` of another, the difference is statistically meaningful.

---

### Polarization

Returns the Euclidean distance between two groups' document vectors at each time period.

```python
df = m.polarization(group1_prefix="PD_", group2_prefix="FI_")
# columns: time, tag1, tag2, euclidean_distance
```

A larger distance means the two groups occupy more distant regions of the embedding space in that period — a proxy for ideological divergence.

---

### Interpret — words driving each dimension

Finds the vocabulary words closest to the extremes of each PCA dimension. Only available with `method='pca'`.

```python
m.interpret(top_words=20, min_count=100)
```

---

### Get raw placement scores

```python
df = m.get_placement()
# columns: tag, group, time, label, dim1, dim2
```

---

## Preprocessing files in bulk

If your corpus is large, preprocess it once and save, then train on the preprocessed file:

```python
from partyembed import Preprocessor, PreprocessConfig

prep = Preprocessor(PreprocessConfig(nltk_languages=["italian"], remove_accents=False))
prep.preprocess_file(
    input_path="data/raw_speeches.tsv",
    output_path="data/clean_speeches.tsv",
    text_col=2,          # zero-based column index of raw text
    sep="\t",
    append_column=True,  # adds cleaned text as an extra column; False = replace the whole line
)
```

Then set `preprocessed=True` in `CorpusConfig` and point `text_column` at the new column.

---

## Full example

```python
import logging
from partyembed import PreprocessConfig, CorpusConfig, ModelConfig, Trainer, Explore

logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)

# 1. Configure
prep_cfg = PreprocessConfig(
    nltk_languages=["italian"],
    use_sklearn_stopwords=False,
    remove_accents=False,
    expand_contractions=False,
    apostrophe_handling="space",
    custom_stopwords=["presidente", "onorevole", "ministro", "camera", "senato"],
)
corpus_cfg = CorpusConfig(
    text_column="speech_text",
    tag_columns=["party", "legislature"],
    file_format="csv",
    filter_column="chamber",
    filter_value="Camera",
)
model_cfg = ModelConfig(vector_size=200, window=20, min_count=50, epochs=5)

# 2. Train
trainer = Trainer(corpus_cfg, model_cfg, prep_cfg)
model = trainer.train("data/speeches.csv", save_path="models/italian_parliament")

# 3. Explore
m = Explore(
    model,
    group_labels={"PD": "Partito Democratico", "FI": "Forza Italia", "M5S": "Movimento 5 Stelle"},
    group_colors={"PD": "#CC0000", "FI": "#0070C0", "M5S": "#FFD700"},
    orientation={"dim1": ("PD_18", "FI_18")},
)

m.plot()
m.plot_timeseries(dimension=1)
m.interpret(top_words=20)

df_pol = m.polarization(group1_prefix="PD_", group2_prefix="FI_")
df_health = m.issue("sanita", n=30, smooth=3)
df_health.pivot(index="time", columns="group", values="similarity")
```

---

## Reference

Original research: Rheault, L. & Cochrane, C. (2020). *Word Embeddings for the Analysis of Ideological Placement in Parliamentary Corpora.* Political Analysis, 28(1), 112–133. https://doi.org/10.1017/pan.2019.26
