from __future__ import annotations

import logging
import typing as t
from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel

from ragas.dataset_schema import SingleTurnSample
from ragas.llms import BaseRagasLLM
from ragas.metrics._answer_similarity import AnswerSimilarity
from ragas.metrics._faithfulness import (
    FaithfulnessStatements,
    HasSegmentMethod,
    LongFormAnswerPrompt,
)
from ragas.metrics.base import (
    MetricType,
    MetricWithEmbeddings,
    MetricWithLLM,
    SingleTurnMetric,
    get_segmenter,
)
from ragas.prompt import PydanticPrompt
from ragas.run_config import RunConfig

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks

    from ragas.metrics._faithfulness import SentencesSimplified


logger = logging.getLogger(__name__)

class QuestionSentences(BaseModel):
    question: str
    sentences: t.List[str]


class RelevanceWithReason(BaseModel):
    sentence: str
    relevance: bool
    reason: str


class SentencesRelevanceOutput(BaseModel):
    relevances: t.List[RelevanceWithReason]


class RelevanceClassifier(PydanticPrompt[QuestionSentences, SentencesRelevanceOutput]):
    instruction = (
        "Given a question and a list of sentences, determine if each sentence is relevant to the question and provide a reason for the classification. "
        "The output should indicate for each sentence whether it is relevant or not and explain why."
    )
    input_model = QuestionSentences
    output_model = SentencesRelevanceOutput
    examples = [
        (
            QuestionSentences(
                question="What is the default IP address for a home router?",
                sentences=[
                    "The default IP address for a home router is usually 0.0.0.0.",
                    "This IP address allows users to configure their network settings.",
                    "Accessing this address via a web browser brings up the router's configuration page.",
                    "Some routers might have different default IP addresses such as 192.168.10.1 and 192.168.2.1.",
                ],
            ),
            SentencesRelevanceOutput(
                relevances=[
                    RelevanceWithReason(
                        sentence="The default IP address for a home router is usually 0.0.0.0.",
                        relevance=True,
                        reason="This sentence is relevant because it directly addresses the question about the default IP address, even though the information provided is incorrect.",
                    ),
                    RelevanceWithReason(
                        sentence="This IP address allows users to configure their network settings.",
                        relevance=False,
                        reason="This sentence provides additional details on what the IP address is used for, but does not answer the specific question about what the default IP address is.",
                    ),
                    RelevanceWithReason(
                        sentence="Accessing this address via a web browser brings up the router's configuration page.",
                        relevance=False,
                        reason="While informative about how to use the IP address, this does not address the question about the default IP address itself.",
                    ),
                    RelevanceWithReason(
                        sentence="Some routers might have different default IP addresses such as 192.168.10.1 and 192.168.2.1.",
                        relevance=True,
                        reason="This sentence is relevant because it directly addresses the question about the default IP address, no matter whether the information provided is correct or not.",
                    ),
                ]
            ),
        ),
        (
            QuestionSentences(
                question="What are the typical components of a pizza?",
                sentences=[
                    "A pizza typically consists of dough, sauce, cheese, and various toppings.",
                    "The dough is usually made from flour, water, and yeast.",
                    "The sauce can be tomato-based or other kinds of seasoning sauces.",
                    "Some pizzas use a white sauce or pesto as a base for sauce.",
                    "A pizza without cheese would lack creaminess and richness, resulting in a drier texture and less flavor.",
                    "Mozzarella is the most popular choice for pizza cheese.",
                ],
            ),
            SentencesRelevanceOutput(
                relevances=[
                    RelevanceWithReason(
                        sentence="A pizza typically consists of dough, sauce, cheese, and various toppings.",
                        relevance=True,
                        reason="This sentence directly answers the question by listing the main components of a pizza.",
                    ),
                    RelevanceWithReason(
                        sentence="The dough is usually made from flour, water, and yeast.",
                        relevance=False,
                        reason="This sentence, while informative about how pizza dough is made, does not directly answer the question regarding the components of a pizza.",
                    ),
                    RelevanceWithReason(
                        sentence="The sauce can be tomato-based or other kinds of seasoning sauces.",
                        relevance=False,
                        reason="This sentence provides additional details about pizza sauce options but does not address the primary components as asked in the question.",
                    ),
                    RelevanceWithReason(
                        sentence="Some pizzas use a white sauce or pesto as a base for sauce.",
                        relevance=False,
                        reason="This sentence provides additional details about pizza sauce options but does not address the primary components as asked in the question.",
                    ),
                    RelevanceWithReason(
                        sentence="A pizza without cheese would lack creaminess and richness, resulting in a drier texture and less flavor.",
                        relevance=True,
                        reason="This sentence addresses the absence of cheese in pizza, which is one of the key components of a traditional pizza.",
                    ),
                    RelevanceWithReason(
                        sentence="Mozzarella is the most popular choice for pizza cheese.",
                        relevance=False,
                        reason="This sentence provides additional details about pizza cheese options but does not address the primary components as asked in the question.",
                    ),
                ]
            ),
        ),
    ]


class QuestionAnswerGroundTruth(BaseModel):
    question: str
    answer: list[str]
    ground_truth: list[str]


class StatementsWithReason(BaseModel):
    statement: str
    reason: str


class ClassificationWithReason(BaseModel):
    Correct: list[StatementsWithReason]
    Wrong: list[StatementsWithReason]
    Missing: list[StatementsWithReason]
    Insignificant: list[StatementsWithReason]


class CorrectnessClassifier(
    PydanticPrompt[QuestionAnswerGroundTruth, ClassificationWithReason]
):
    instruction = (
        "Given a ground truth and a series of statements from an answer, analyze each statement and classify them in one of the following categories: "
        "Correct: statements that are present in answer that are also directly supported by the one or more statements in ground truth."
        "Wrong: statements present in the answer but not directly or indirectly supported by any statement in ground truth, or totally irrelevant to the question."
        "Missing: statements found in ground truth but not present in answer."
        "Insignificant: Statements that appear in the answer but not directly or indirectly supported by any statement in ground truth, but these statements **do not** affect the correctness of the answer when the answer is compared to the ground truth. These statements are usually relevant to the question or ground truth, but not an important part for the answer. For example, an extended introduction or explanation of certain objects in ground truth, or a description of their degree."
        "Note that Each statement can only belong to one of these categories. Provide a reason for each classification, considering how the statements in the answer interact rather than assessing each statement in isolation."
    )
    input_model = QuestionAnswerGroundTruth
    output_model = ClassificationWithReason
    examples = [
        (
            QuestionAnswerGroundTruth(
                question="What powers the sun and what is its primary function?",
                answer=[
                    "The sun is powered by nuclear fission, similar to nuclear reactors on Earth.",
                    "The primary function of the sun is to provide light to the solar system.",
                    "The sun's light plays a critical role in Earth's climate system.",
                    "The fusion process in the sun's core releases a tremendous amount of energy.",
                    "The energy from the sun provides heat and light, which are essential for life on Earth.",
                    "Sunlight helps to drive the weather and ocean currents.",
                    "The average distance from the Earth to the Sun is about 150 million kilometers.",
                    "The Sun formed from the gravitational collapse of a region within a large molecular cloud in space in 4.6 billion years ago.",
                    "Solar storms can disrupt communication systems, affect power grids, increase radiation exposure for astronauts, create stunning auroras, and damage satellites."
                ],
                ground_truth=[
                    "The sun is powered by nuclear fusion, where hydrogen atoms fuse to form helium.",
                    "This fusion process in the sun's core releases a tremendous amount of energy.",
                    "The energy from the sun provides heat and light, which are essential for life on Earth.",
                    "Sunlight helps to drive the weather and ocean currents.",
                ],
            ),
            ClassificationWithReason(
                Correct=[
                    StatementsWithReason(
                        statement="The primary function of the sun is to provide light to the solar system.",
                        reason="This statement is somewhat supported by the ground truth mentioning the sun providing light and its roles, though it focuses more broadly on the sun's energy.",
                    )
                ],
                Wrong=[
                    StatementsWithReason(
                        statement="The sun is powered by nuclear fission, similar to nuclear reactors on Earth.",
                        reason="This statement is incorrect and contradicts the ground truth which states that the sun is powered by nuclear fusion.",
                    ),
                    StatementsWithReason(
                        statement="Solar storms can disrupt communication systems, affect power grids, increase radiation exposure for astronauts, create stunning auroras, and damage satellites.",
                        reason="Solar storms and its effect is not mentioned in the ground truth, making this statement totally irrelevant to the question.",
                    ),
                ],
                Missing=[
                    StatementsWithReason(
                        statement="This fusion process in the sun's core releases a tremendous amount of energy.",
                        reason="This process and its significance are not mentioned in the answer.",
                    ),
                    StatementsWithReason(
                        statement="The energy from the sun provides heat and light, which are essential for life on Earth.",
                        reason="The answer only mentions light, omitting the essential aspects of heat and its necessity for life, which the ground truth covers.",
                    ),
                    StatementsWithReason(
                        statement="The sun's light plays a critical role in Earth's climate system.",
                        reason="The impact of the sun’s light on Earth's climate system is detailedly addressed in the answer, such as it helps to drive the weather and ocean currents..",
                    ),
                    StatementsWithReason(
                        statement="Sunlight helps to drive the weather and ocean currents.",
                        reason="The effect of sunlight on weather patterns and ocean currents is omitted in the answer.",
                    ),
                ],
                Insignificant=[
                    StatementsWithReason(
                        statement="The average distance from the Earth to the Sun is about 150 million kilometers.",
                        reason="The distance from Earth to the Sun is not mentioned in ground truth. This statement is relevant to the subject 'sun' but not important. It does not affect the correctness of the answer so it is considered insignificant.",
                    ),
                    StatementsWithReason(
                        statement="The Sun formed from the gravitational collapse of a region within a large molecular cloud in space in 4.6 billion years ago.",
                        reason="The formation of Sun is not mentioned in ground truth. This statement is relevant to the subject 'sun' but not important. It does not affect the correctness of the answer so it is considered insignificant.",
                    ),                    
                ],
            ),
        ),
        (
            QuestionAnswerGroundTruth(
                question="What is the temperature range for water to remain liquid under normal atmospheric pressure?",
                answer=[
                    "Water remains liquid above 0 degrees Celsius at sea level.",
                    "Water remains liquid below 100 degrees Celsius at sea level.",
                    "Water remains liquid below 105 degrees Celsius at sea level.",
                    "Water is essential for life.",
                    "The boiling point of water is approximately 120 degrees Celsius at two atmospheres of pressure",
                ],
                ground_truth=[
                    "Water is a liquid between 0 degrees Celsius and 100 degrees Celsius under standard atmospheric conditions."
                ],
            ),
            ClassificationWithReason(
                Correct=[
                    StatementsWithReason(
                        statement="Water remains liquid above 0 degrees Celsius at sea level.",
                        reason="This statement alone might mislead by implying that water remains liquid at any temperature above 0 degrees Celsius without an upper boundary. However, when combined with the statement about the upper limit, it contributes to a correct description of the temperature range within which water stays liquid.",
                    ),
                    StatementsWithReason(
                        statement="Water remains liquid below 100 degrees Celsius at sea level.",
                        reason="By itself, this statement might be interpreted as water remaining liquid below 100 degrees Celsius indefinitely downward, which is not correct. However when paired with the correct lower limit from the first statement, it helps form a complete and accurate depiction of the condition.",
                    ),
                ],
                Wrong=[
                    StatementsWithReason(
                        statement="Water remains liquid below 105 degrees Celsius at sea level.",
                        reason="According to the ground truth, water remains liquid below 100 degrees Celsius under standard atmospheric conditions. Therefore the answer is not correct.",
                    ),
                    StatementsWithReason(
                        statement="The boiling point of water is approximately 120 degrees Celsius at two atmospheres of pressure",
                        reason="This statement incorrectly interprets the conditional requirements stated in the question. The question is about the temperature range for water to remain liquid under normal atmospheric pressure, not under two atmospheres of pressure. Therefore the answer is wrong.",
                    ),                       
                ],
                Missing=[],
                Insignificant=[
                    StatementsWithReason(
                        statement="Water is essential for life.",
                        reason="While true, this statement does not directly relate to the temperature range for water to remain liquid and does not affect the correctness of the other statements. This statement is relevant to the subject 'water' but not important, it's just a description of some degree.",
                    ),                       
                ]             
            ),
        ),
        (
            QuestionAnswerGroundTruth(
                question="What's FBI? What's its primary function?",
                answer=[
                    "The FBI is the Federal Bureau of Investigation.",
                    "FBI's primary function is to enforce federal laws in the United States.",
                    "The FBI also investigates organized crime and terrorism.",
                    "The FBI was established in 1908 and is crutial to the United States.",
                    "The FBI focuses on international espionage.",
                    "The 'Feds' is a colloquial term used to refer to federal law enforcement officers, particularly those working for the FBI and other federal agencies."
                ],
                ground_truth=[
                    "The FBI, or Federal Bureau of Investigation, is a domestic intelligence and security service of the United States.",
                    "Its primary functions include enforcing federal laws, investigating major crimes, and protecting the U.S. from terrorism domesticly."
                ],
            ),
            ClassificationWithReason(
                Correct=[
                    StatementsWithReason(
                        statement="The FBI is the Federal Bureau of Investigation.",
                        reason="This statement accurately defines what the FBI stands for and is directly supported by the ground truth.",
                    ),
                    StatementsWithReason(
                        statement="FBI's primary function is to enforce federal laws in the United States.",
                        reason="This statement correctly identifies one of the primary functions of the FBI, aligning with the responsibilities outlined in the ground truth.",
                    ),
                    StatementsWithReason(
                        statement="The FBI also investigates organized crime and terrorism.",
                        reason="This statement describes additional functions of the FBI that are supported by the ground truth.",
                    ),
                ],
                Wrong=[
                    StatementsWithReason(
                        statement="The FBI focuses on international espionage.",
                        reason="The FBI primarily deals with domestic issues in the United States rather than international espionage, which is mentioned in the ground truth. Therefore the answer is wrong.",
                    ),
                ],
                Missing=[],
                Insignificant=[
                    StatementsWithReason(
                        statement="The FBI was established in 1908 and is crutial to the United States.",
                        reason="Although true, this statement does not affect the understanding of the FBI's primary function and is not relevant to the core question. This statement is an extended introduction and relevant to the subject 'FBI', therefore it is insignificant.",
                    ),
                    StatementsWithReason(
                        statement="The 'Feds' is a colloquial term used to refer to federal law enforcement officers, particularly those working for the FBI and other federal agencies.",
                        reason="This statement provides additional context about the FBI officers but does not directly relate to the primary function of the FBI as asked in the question. This statement is an extended introduction and relevant to the subject 'FBI', therefore it is insignificant.",
                    ),
                ],
            ),
        )

    ]


@dataclass
class AnswerCorrectness(MetricWithLLM, MetricWithEmbeddings, SingleTurnMetric):
    """
    Measures answer correctness compared to ground truth as a combination of
    factuality and semantic similarity.

    Attributes
    ----------
    name: string
        The name of the metrics
    weights:
        a list of two weights corresponding to factuality and semantic similarity
        Defaults [0.75, 0.25]
    answer_similarity:
        The AnswerSimilarity object
    """

    name: str = "answer_correctness"
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {"user_input", "response", "reference"}
        }
    )
    correctness_prompt: PydanticPrompt = field(default_factory=CorrectnessClassifier)
    long_form_answer_prompt: PydanticPrompt = field(
        default_factory=LongFormAnswerPrompt
    )
    relevance_prompt: PydanticPrompt = field(default_factory=RelevanceClassifier)
    weights: list[float] = field(default_factory=lambda: [0.9, 0.1])
    answer_similarity: t.Optional[AnswerSimilarity] = None
    sentence_segmenter: t.Optional[HasSegmentMethod] = None
    max_retries: int = 1

    def __post_init__(self: t.Self) -> None:
        if len(self.weights) != 2:
            raise ValueError(
                "Expects a list of two weights. First for factuality, second for semantic similarity"
            )
        if all([w == 0 for w in self.weights]):
            raise ValueError("At least one weight must be non-zero")
        if not all([w >= 0 for w in self.weights]):
            raise ValueError("Weights must be non-negative")

    def init(self, run_config: RunConfig) -> None:
        super().init(run_config)
        if self.answer_similarity is None and self.weights[1] != 0:
            self.answer_similarity = AnswerSimilarity(
                llm=self.llm, embeddings=self.embeddings
            )

    def _compute_statement_presence(
        self, prediction: ClassificationWithReason
    ) -> float:
        correct = len(prediction.Correct)
        wrong = len(prediction.Wrong)
        missing = len(prediction.Missing)
        insignificant = len(prediction.Insignificant)
        print(prediction)
        print(f"correct: {correct}, wrong: {wrong}, missing: {missing}, insignificant: {insignificant}")
        epsilon = 1e-10
        f = lambda x: x**2 if x > 0 else 0    # 使用平方来在两侧聚集分数
        # score = correct / (correct + 0.5 * wrong + 0.5 * missing) if correct > 0 else 0
        score = f(correct) / (f(correct) + 0.5 * f(wrong) + 0.5 * f(missing) + epsilon) if correct > 0 else 0
        return score

    async def _create_simplified_statements(
        self, question: str, text: str, callbacks: Callbacks
    ) -> SentencesSimplified:
        assert self.llm is not None, "llm is not set"

        # CHANGED: init sentence_segmenter here instead of in __post_init__
        if self.sentence_segmenter is None:
            language = self.long_form_answer_prompt.language
            self.sentence_segmenter = get_segmenter(language=language, clean=False)

        sentences = self.sentence_segmenter.segment(text)

        sentences_relevance = await self.relevance_prompt.generate(
            llm=self.llm,
            data=QuestionSentences(question=question, sentences=sentences),
            callbacks=callbacks,
        )

        if len(sentences) != len(sentences_relevance.relevances):
            logging.warning(
                "The number of sentences and the number of relevances are not equal."
            )

        relevant_sentences = []
        for relevancy in sentences_relevance.relevances:
            if relevancy.relevance:
                relevant_sentences.append(relevancy.sentence)

        # print("len(relevant_sentences): ", len(relevant_sentences))
        # print("relevant_sentences: ", relevant_sentences)

        sentences_with_index = {
            i: sentence
            for i, sentence in enumerate(relevant_sentences)
            if sentence.strip()
        }

        statements_simplified = await self.long_form_answer_prompt.generate(
            llm=self.llm,
            data=FaithfulnessStatements(
                question=question, answer=text, sentences=sentences_with_index
            ),
            callbacks=callbacks,
        )
        return statements_simplified

    async def _single_turn_ascore(
        self: t.Self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        row = sample.to_dict()
        score = await self._ascore(row, callbacks)
        return score

    async def _ascore(self, row: t.Dict[str, t.Any], callbacks: Callbacks) -> float:
        assert self.llm is not None, "LLM must be set"

        # extract the statements from the answer and the ground truth
        question = row["user_input"]
        statements: t.Dict[str, t.List[str]] = {}
        for item in ["response", "reference"]:
            simplified_statements = await self._create_simplified_statements(
                question, row[item], callbacks
            )
            _statements_unwrapped = []
            for component in simplified_statements.sentences:
                _statements_unwrapped.extend(component.simpler_statements)
            statements[item] = _statements_unwrapped

        if not all([val == [] for val in statements.values()]):
            ground_truth = [statement for statement in statements["reference"]]
            answer = [statement for statement in statements["response"]]
            answers = await self.correctness_prompt.generate(
                llm=self.llm,
                data=QuestionAnswerGroundTruth(
                    question=question,
                    answer=answer,
                    ground_truth=ground_truth,
                ),
                callbacks=callbacks,
            )
            if answers is None:
                return np.nan

            f1_score = self._compute_statement_presence(answers)
        else:
            f1_score = 1.0

        if self.weights[1] == 0:
            similarity_score = 0.0
        else:
            assert self.answer_similarity is not None, "AnswerSimilarity must be set"

            # row是一个字典，包含了原始user_input, response, reference键，没有被拆分短句
            # similarity_score是reference和response的嵌入向量相似度            
            # 实测发现这个similarity_score的值很大，通常在0.8-0.9以下，对最终得分的影响不大
            similarity_score = await self.answer_similarity.ascore(
                row, callbacks=callbacks
            )
        # 按照权重weights计算最终得分
        assert np.sum(self.weights) == 1.0
        score = np.average(
            [f1_score, similarity_score],
            weights=self.weights,
        )
        print(f"f1_score: {f1_score}, similarity_score: {similarity_score}, score: {score}")

        return float(score)
