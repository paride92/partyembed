from __future__ import annotations

import string
import unicodedata
from functools import reduce
from typing import List, Optional, Set

from nltk.tokenize import ToktokTokenizer

from partyembed._contractions import CONTRACTIONS
from partyembed.config import PreprocessConfig

_SKLEARN_STOPWORDS: Optional[Set[str]] = None
_NLTK_CACHE: dict = {}


def _sklearn_stopwords() -> Set[str]:
    global _SKLEARN_STOPWORDS
    if _SKLEARN_STOPWORDS is None:
        from sklearn.feature_extraction import text
        _SKLEARN_STOPWORDS = set(text.ENGLISH_STOP_WORDS)
    return _SKLEARN_STOPWORDS


def _nltk_stopwords(language: str) -> Set[str]:
    if language not in _NLTK_CACHE:
        try:
            import nltk.corpus  # noqa: PLC0415
            _NLTK_CACHE[language] = set(nltk.corpus.stopwords.words(language))
        except LookupError:
            import nltk  # noqa: PLC0415
            nltk.download("stopwords", quiet=True)
            _NLTK_CACHE[language] = set(nltk.corpus.stopwords.words(language))
    return _NLTK_CACHE[language]


class Preprocessor:
    """Clean and tokenize raw text.

    Parameters
    ----------
    config:
        A :class:`~partyembed.config.PreprocessConfig` instance.
        Keyword arguments are forwarded to ``PreprocessConfig`` when *config* is
        ``None``.

    Examples
    --------
    >>> prep = Preprocessor(custom_stopwords=['member', 'speaker'])
    >>> prep.clean("The member won't oppose the bill!")
    'oppose bill'
    """

    def __init__(self, config: Optional[PreprocessConfig] = None, **kwargs):
        if config is None:
            config = PreprocessConfig(**kwargs)
        self.config = config
        self._tokenizer = ToktokTokenizer()
        self._stopwords: Set[str] = self._build_stopwords()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def clean(self, text: str) -> str:
        """Return a clean, space-separated token string."""
        if self.config.lowercase:
            text = text.lower()
        if self.config.expand_contractions:
            text = reduce(lambda a, kv: a.replace(*kv), CONTRACTIONS.items(), text)
        text = text.replace("\t", " ").replace("\n", " ").replace("\r", " ")
        if self.config.remove_accents:
            text = self._strip_accents(text)
        # Apostrophe handling: covers both ASCII ' (U+0027) and curly ' (U+2019).
        # Applied after contraction expansion so "won't" → "will not" first.
        apos = "'’"
        handling = self.config.apostrophe_handling
        if handling not in ("space", "remove", "keep"):
            raise ValueError("apostrophe_handling must be 'space', 'remove', or 'keep'.")
        if handling == "space":
            for ch in apos:
                text = text.replace(ch, " ")
        elif handling == "remove":
            for ch in apos:
                text = text.replace(ch, "")
        # "keep": leave apostrophes in place; exclude them from punctuation removal below
        punct = string.punctuation if handling != "keep" else string.punctuation.replace("'", "")
        text = text.translate(str.maketrans(punct, " " * len(punct)))
        tokens = self._tokenizer.tokenize(text)
        tokens = [
            w
            for w in tokens
            if w not in self._stopwords
            and len(w) >= self.config.min_token_length
            and w.strip()
            and not (self.config.remove_digits and w.isdigit())
        ]
        return " ".join(tokens)

    def preprocess_file(
        self,
        input_path: str,
        output_path: str,
        text_col: int = 0,
        sep: str = "\t",
        encoding: str = "utf-8",
        append_column: bool = True,
    ) -> None:
        """Preprocess every line of a file and write results.

        Parameters
        ----------
        text_col:
            Zero-based index of the column that contains raw text (when
            ``append_column=True``) or the entire line (when the file has
            a single column).
        append_column:
            If ``True``, append the preprocessed text as an extra column.
            If ``False``, replace the entire line with the preprocessed text.
        """
        with open(input_path, "r", encoding=encoding) as infile, \
             open(output_path, "w", encoding="utf-8") as outfile:
            for idx, line in enumerate(infile):
                parts = line.rstrip("\n").split(sep)
                raw = parts[text_col] if append_column else line.strip()
                cleaned = self.clean(raw)
                if cleaned:
                    if append_column:
                        outfile.write(line.rstrip("\n") + sep + cleaned + "\n")
                    else:
                        outfile.write(cleaned + "\n")
                if (idx + 1) % 100_000 == 0:
                    print(f"Processed {idx + 1:,} lines.")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_stopwords(self) -> Set[str]:
        words: Set[str] = set(self.config.custom_stopwords)
        if self.config.use_sklearn_stopwords:
            words.update(_sklearn_stopwords())
        for lang in self.config.nltk_languages:
            words.update(_nltk_stopwords(lang))
        return words

    @staticmethod
    def _strip_accents(text: str) -> str:
        text = unicodedata.normalize("NFD", text)
        text = text.encode("ascii", "ignore")
        return text.decode("utf-8")
