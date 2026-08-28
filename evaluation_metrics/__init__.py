import inspect
import sys
from ._answer_correctness import AnswerCorrectness
from ._bleu_score import BleuScoreForChinese

__all__ = [
    "BleuScoreForChinese",
    "AnswerCorrectness",
]

current_module = sys.modules[__name__]
ALL_MY_METRICS = [
    obj
    for name, obj in inspect.getmembers(current_module)
    if name in __all__ and not inspect.isclass(obj) and not inspect.isbuiltin(obj)
]