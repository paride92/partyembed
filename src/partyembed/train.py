from __future__ import annotations

import logging
from typing import Optional

from gensim.models.doc2vec import Doc2Vec
from gensim.models.phrases import Phraser, Phrases

from partyembed.config import CorpusConfig, ModelConfig, PreprocessConfig
from partyembed.corpus import CorpusIterator, PhraseIterator
from partyembed.preprocess import Preprocessor

logger = logging.getLogger(__name__)


class Trainer:
    """Orchestrate phrase detection and Doc2Vec training.

    Parameters
    ----------
    corpus_config:
        Describes file layout and tag scheme.
    model_config:
        Doc2Vec hyperparameters and phrase-detection settings.
    preprocess_config:
        Text cleaning options.  Pass ``None`` if the corpus is already
        preprocessed (set ``corpus_config.preprocessed = True``).

    Examples
    --------
    >>> from partyembed import Trainer, CorpusConfig, ModelConfig, PreprocessConfig
    >>> corpus_cfg = CorpusConfig(text_column='speech', tag_columns=['party', 'congress'])
    >>> model_cfg  = ModelConfig(vector_size=200, window=20, epochs=5)
    >>> prep_cfg   = PreprocessConfig(custom_stopwords=['member', 'speaker'])
    >>> trainer = Trainer(corpus_cfg, model_cfg, prep_cfg)
    >>> model   = trainer.train('data/speeches.tsv', save_path='models/mymodel')
    """

    def __init__(
        self,
        corpus_config: CorpusConfig,
        model_config: Optional[ModelConfig] = None,
        preprocess_config: Optional[PreprocessConfig | Preprocessor] = None,
    ):
        self.corpus_config = corpus_config
        self.model_config = model_config or ModelConfig()
        if isinstance(preprocess_config, Preprocessor):
            self.preprocessor = preprocess_config
            self.preprocess_config = preprocess_config.config
        elif isinstance(preprocess_config, PreprocessConfig):
            self.preprocess_config = preprocess_config
            self.preprocessor = Preprocessor(preprocess_config)
        else:
            self.preprocess_config = None
            self.preprocessor = None
        self.bigram: Optional[Phraser] = None
        self.trigram: Optional[Phraser] = None
        self.model: Optional[Doc2Vec] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_phrases(
        self, input_path: str, save_path: Optional[str] = None
    ) -> None:
        """Detect multi-word expressions in the corpus and store Phraser objects."""
        mc = self.model_config
        phrase_iter = PhraseIterator(input_path, self.corpus_config, self.preprocessor)

        logger.info("Fitting bigram phraser ...")
        phrases = Phrases(
            phrase_iter,
            min_count=mc.bigram_min_count,
            threshold=mc.bigram_threshold,
        )
        self.bigram = Phraser(phrases)

        if mc.use_trigrams:
            logger.info("Fitting trigram phraser ...")
            tphrases = Phrases(
                self.bigram[phrase_iter],
                min_count=mc.bigram_min_count,
                threshold=mc.bigram_threshold,
            )
            self.trigram = Phraser(tphrases)

        if save_path:
            self.bigram.save(save_path + "_bigram")
            logger.info("Bigram phraser saved to %s_bigram", save_path)
            if self.trigram:
                self.trigram.save(save_path + "_trigram")
                logger.info("Trigram phraser saved to %s_trigram", save_path)

    def load_phrases(self, bigram_path: str, trigram_path: Optional[str] = None) -> None:
        """Load previously saved Phraser objects."""
        self.bigram = Phraser.load(bigram_path)
        if trigram_path:
            self.trigram = Phraser.load(trigram_path)

    def train(self, input_path: str, save_path: Optional[str] = None) -> Doc2Vec:
        """Train a Doc2Vec model and optionally save it to disk."""
        mc = self.model_config

        if mc.use_bigrams and self.bigram is None:
            self.fit_phrases(input_path)

        corpus = CorpusIterator(
            input_path,
            self.corpus_config,
            self.preprocessor,
            self.bigram,
            self.trigram,
        )

        self.model = Doc2Vec(
            vector_size=mc.vector_size,
            window=mc.window,
            min_count=mc.min_count,
            workers=mc.workers,
            epochs=mc.epochs,
            dm=mc.dm,
        )

        logger.info("Building vocabulary ...")
        self.model.build_vocab(corpus)
        logger.info(
            "Vocabulary size: %d words, %d document tags",
            len(self.model.wv),
            len(self.model.dv),
        )
        logger.info("Training Doc2Vec ...")
        self.model.train(
            corpus,
            total_examples=self.model.corpus_count,
            epochs=self.model.epochs,
        )

        if save_path:
            self.model.save(save_path)
            logger.info("Model saved to %s", save_path)

        return self.model
