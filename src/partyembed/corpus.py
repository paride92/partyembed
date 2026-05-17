from __future__ import annotations

import logging
from collections import namedtuple
from typing import Callable, Iterator, List, Optional

import pandas as pd

from partyembed.config import CorpusConfig
from partyembed.preprocess import Preprocessor

logger = logging.getLogger(__name__)

_Document = namedtuple("Document", ["words", "tags"])


def _load_df(config: CorpusConfig, path: str) -> pd.DataFrame:
    fmt = config.file_format.lower()
    sep = "\t" if fmt in ("tsv", "tab") else ","
    if fmt in ("tsv", "csv", "tab"):
        df = pd.read_csv(
            path,
            sep=sep,
            encoding=config.encoding,
            header=0 if config.header else None,
            low_memory=False,
        )
    elif fmt == "json":
        df = pd.read_json(path, encoding=config.encoding)
    elif fmt == "parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file format: {fmt!r}")
    if config.filter_column is not None and config.filter_value is not None:
        df = df[df[config.filter_column].astype(str) == str(config.filter_value)]
    return df


def _build_tags(row, config: CorpusConfig) -> List[str]:
    if config.tag_format:
        values = {str(col): str(row[col]) for col in config.tag_columns}
        return [config.tag_format.format(**values)]
    parts = [str(row[col]) for col in config.tag_columns]
    composite = config.tag_separator.join(parts)
    # Return composite + individual parts as separate tags so the model
    # learns a vector for each party-period combination AND each component.
    return [composite] + parts


class CorpusIterator:
    """Iterate over a dataset yielding tagged documents for Doc2Vec.

    Parameters
    ----------
    input_path:
        Path to the data file.
    corpus_config:
        How to read and tag documents.
    preprocessor:
        If provided, and ``corpus_config.preprocessed`` is ``False``, each
        document will be cleaned before tokenisation.
    bigram / trigram:
        Fitted :class:`gensim.models.phrases.Phraser` objects.  Applied to
        token lists when provided.

    Examples
    --------
    >>> config = CorpusConfig(text_column='speech', tag_columns=['party', 'year'])
    >>> corpus = CorpusIterator('data.tsv', config, preprocessor=prep)
    >>> for doc in corpus:
    ...     print(doc.tags)
    """

    def __init__(
        self,
        input_path: str,
        corpus_config: CorpusConfig,
        preprocessor: Optional[Preprocessor] = None,
        bigram=None,
        trigram=None,
    ):
        self.input_path = input_path
        self.config = corpus_config
        self.preprocessor = preprocessor
        self.bigram = bigram
        self.trigram = trigram

    # gensim expects a re-entrant iterator
    def __iter__(self) -> Iterator[_Document]:
        df = _load_df(self.config, self.input_path)
        skipped = 0
        for _, row in df.iterrows():
            text = str(row[self.config.text_column])
            if not self.config.preprocessed and self.preprocessor:
                text = self.preprocessor.clean(text)
            if not text.strip():
                skipped += 1
                continue
            tokens: list = text.split()
            if self.bigram and self.trigram:
                tokens = list(self.trigram[self.bigram[tokens]])
            elif self.bigram:
                tokens = list(self.bigram[tokens])
            tags = _build_tags(row, self.config)
            yield _Document(tokens, tags)
        if skipped:
            logger.debug("Skipped %d empty documents.", skipped)


class PhraseIterator:
    """Lightweight iterator that yields only token lists for phrase detection."""

    def __init__(
        self,
        input_path: str,
        corpus_config: CorpusConfig,
        preprocessor: Optional[Preprocessor] = None,
    ):
        self.input_path = input_path
        self.config = corpus_config
        self.preprocessor = preprocessor

    def __iter__(self) -> Iterator[List[str]]:
        df = _load_df(self.config, self.input_path)
        for _, row in df.iterrows():
            text = str(row[self.config.text_column])
            if not self.config.preprocessed and self.preprocessor:
                text = self.preprocessor.clean(text)
            if text.strip():
                yield text.split()
