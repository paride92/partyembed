from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PreprocessConfig:
    """Configuration for text preprocessing.

    Parameters
    ----------
    custom_stopwords:
        Any extra words to remove on top of the built-in lists.
    use_sklearn_stopwords:
        Include sklearn's English stopword list (~318 words).
    nltk_languages:
        One or more NLTK language names whose stopword lists should be loaded,
        e.g. ``['english', 'french', 'italian']``.  NLTK supports ~25 languages
        including arabic, danish, dutch, english, finnish, french, german,
        hungarian, italian, norwegian, portuguese, romanian, russian, spanish,
        swedish, and turkish.  Requires ``nltk.download('stopwords')`` once.
    expand_contractions:
        Expand English contractions (e.g. "won't" → "will not").
    remove_accents:
        Strip diacritical marks via ASCII normalisation.  Set ``False`` for
        non-English text where accents are meaningful.
    min_token_length:
        Discard tokens shorter than this many characters.
    remove_digits:
        Discard purely numeric tokens.
    lowercase:
        Lowercase the text before tokenisation.
    """
    custom_stopwords: List[str] = field(default_factory=list)
    use_sklearn_stopwords: bool = True
    nltk_languages: List[str] = field(default_factory=list)
    expand_contractions: bool = True
    apostrophe_handling: str = "space"  # 'space' | 'remove' | 'keep'
    remove_accents: bool = True  # set False for non-English text
    min_token_length: int = 3
    remove_digits: bool = True
    lowercase: bool = True


@dataclass
class CorpusConfig:
    """Configuration for reading and tagging documents from a file.

    Parameters
    ----------
    text_column:
        Column name (str) or index (int) containing the raw/preprocessed text.
    tag_columns:
        One or more columns used to build document tags (e.g. ['party', 'congress']).
    tag_separator:
        String used to join multiple tag column values into a single composite tag.
    tag_format:
        Optional Python format string using column names as keys, e.g.
        ``'{party}_{congress}'``.  When set, overrides ``tag_separator``.
    file_format:
        One of ``'csv'``, ``'tsv'``, ``'json'``, ``'parquet'``.
    encoding:
        File encoding.
    header:
        Whether the file has a header row.
    filter_column:
        Optional column to filter on before iterating.
    filter_value:
        Keep only rows where ``filter_column == filter_value``.
    preprocessed:
        Set to ``True`` if the text column already contains preprocessed tokens
        (space-separated) so the Preprocessor is skipped.
    """
    text_column: str | int = "text"
    tag_columns: List[str | int] = field(default_factory=lambda: ["group", "period"])
    tag_separator: str = "_"
    tag_format: Optional[str] = None
    file_format: str = "tsv"
    encoding: str = "utf-8"
    header: bool = True
    filter_column: Optional[str | int] = None
    filter_value: Optional[str] = None
    preprocessed: bool = False


@dataclass
class ModelConfig:
    """Configuration for Doc2Vec model training."""
    vector_size: int = 200
    window: int = 20
    min_count: int = 50
    workers: int = 8
    epochs: int = 5
    dm: int = 0
    use_bigrams: bool = True
    use_trigrams: bool = True
    bigram_min_count: int = 5
    bigram_threshold: float = 10.0
