import typing as t
from dataclasses import dataclass, field
from typing import List
from langchain_core.callbacks import Callbacks
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics._faithfulness import HasSegmentMethod
from ragas.metrics.base import MetricType, SingleTurnMetric, get_segmenter
from ragas.run_config import RunConfig

def word_tokenizer_for_chinese(text: str) -> List[str]:
    """
    使用 jieba 分词对中文文本进行分词。

    Args:
        text (str): 待分词的中文文本。

    Returns:
        List[str]: 分词后的词语列表。
    """
    try:
        import jieba  # jieba 分词
    except ImportError:
        raise ImportError(
            "jieba is required for bleu score. Please install it using `pip install jieba`"
        )

    return jieba.lcut(text)


@dataclass
class BleuScoreForChinese(SingleTurnMetric):
    name: str = "bleu_score_zh"
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {MetricType.SINGLE_TURN: {"reference", "response"}}
    )
    weights: t.Tuple[float, ...] = (0.25, 0.25, 0.25, 0.25)

    def __post_init__(self) -> None:
        try:
            from nltk.translate.bleu_score import sentence_bleu
            from nltk.translate.bleu_score import SmoothingFunction
        except ImportError:
            raise ImportError(
                "nltk is required for bleu score. Please install it using `pip install nltk`"
            )

        self.segmenter = get_segmenter()
        self.word_tokenizer = word_tokenizer_for_chinese
        self.sentence_bleu = sentence_bleu
        self.smoothing_function = SmoothingFunction().method1

    def init(self, run_config: RunConfig) -> None:
        pass

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:

        reference = self.word_tokenizer(sample.reference)
        response = self.word_tokenizer(sample.response)

        score = self.sentence_bleu(
            [reference],
            response,
            weights=self.weights,
            smoothing_function=self.smoothing_function,
        )
        return float(score)

    async def _ascore(self, row: t.Dict[str, t.Any], callbacks: Callbacks) -> float:
        return await self._single_turn_ascore(SingleTurnSample(**row), callbacks)
