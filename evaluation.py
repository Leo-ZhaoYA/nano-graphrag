from hashlib import md5
import numpy as np
import json
import asyncio
import os
import builtins
import logging
import re
import pandas as pd
from pathlib import Path
from pydantic import ValidationError
from ragas import EvaluationDataset, evaluate
from ragas.evaluation import MetricWithLLM, SingleTurnMetric
from ragas.metrics import AnswerRelevancy
from ragas.llms import BaseRagasLLM, LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.dataset_schema import SingleTurnSample
from pandas import DataFrame
from typing import Dict, List, Any
from contextlib import contextmanager
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from nano_graphrag._utils import read_yaml_config
from evaluation_metrics import AnswerCorrectness, BleuScoreForChinese



PROJECT_DIR = Path(os.getcwd())
yaml_config = read_yaml_config()
OPENAI_KEY = yaml_config.get("openai_api_key")
OPENAI_API_BASE = yaml_config.get("openai_api_url")
# # 将日志级别设置为 WARNING，忽略 INFO 级别的提示信息
# logging.getLogger("ragas").setLevel(logging.WARNING)


def create_embed_model(model: str = "text-embedding-ada-002") -> OpenAIEmbeddings:
    embed_model = OpenAIEmbeddings(
        model=model, api_key=OPENAI_KEY, base_url=OPENAI_API_BASE
    )
    return embed_model


def create_llm(model: str = "gpt-4o", temperature: float = 0) -> ChatOpenAI:
    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=OPENAI_KEY,
        base_url=OPENAI_API_BASE,
    )
    return llm

def remove_cache_prompts(prompts_path: Path, metric: MetricWithLLM) -> None:
    """删除缓存的prompt"""
    prompts = metric.get_prompts()
    for prompt_name, prompt in prompts.items():
        prompt_file_name = prompts_path / f"{prompt_name}_{prompt.language}.json"
        if os.path.exists(prompt_file_name):
            os.remove(prompt_file_name)


@contextmanager
def ensure_utf8_open():  # type: ignore
    """
    在函数调用期间，临时将内置的 open 函数默认编码设置为 utf-8。
    该上下文管理器会将所有的 open 调用修改为使用 utf-8 编码。

    在上下文结束时，会恢复原始的 open 函数。

    使用场景：
    - 用于需要在特定代码块内，强制使用 UTF-8 编码的文件读写操作。
    - 适用于库代码中无法直接修改 open 调用的情况。

    示例：
        with ensure_utf8_open():
            with open('file.txt', 'w') as f:
                f.write("这是UTF-8编码的文件内容")

            with open('file.txt', 'r') as f:
                content = f.read()

    注意：
    - 仅在上下文（with 语句块）内生效，退出后恢复原来的 open 行为。
    """

    original_open = builtins.open

    def open_utf8(filepath, mode, encoding=None, **kwargs):  # type: ignore
        if encoding is None:
            encoding = "utf-8"
        return original_open(filepath, mode, encoding=encoding, **kwargs)

    builtins.open = open_utf8

    try:
        yield  # 在这个上下文中执行代码
    finally:
        builtins.open = original_open



def convert_to_evaluation_dataset(
    test_samples: List[Dict[str, Any]]
) -> EvaluationDataset:
    """
    将测试样本列表转换为 EvaluationDataset。

    参数：
        test_samples (List[Dict[str, Any]]): 包含测试样本数据的字典列表。每个字典应包含以下键：
            - "question": 用户输入或问题。
            - "retrieved_contexts": 为问题检索到的上下文列表。
            - "response": 生成的响应。
            - "reference": 标准答案，用于评估响应。

    返回：
        EvaluationDataset: 包含从输入测试样本创建的 SingleTurnSample 实例的数据集。
    """
    single_turn_samples = [
        SingleTurnSample(
            user_input=sample["question"],
            # retrieved_contexts=sample["retrieved_contexts"],
            response=sample["response"], 
            reference=sample["reference"],
        )
        for sample in test_samples
    ]
    return EvaluationDataset(samples=single_turn_samples)


def load_or_adapt_prompts(
    prompts_path: Path,
    language: str,
    llm: BaseRagasLLM,
    metrics_with_llm: List[MetricWithLLM],
) -> None:
    """adapt prompt是指让metrics里的prompts适配到指定的语言。
    如果已经有了适配好的prompt，就直接加载，否则就适配并保存到指定路径"""

    os.makedirs(prompts_path, exist_ok=True)
    for metric in metrics_with_llm:
        try:
            with ensure_utf8_open():
                prompts = metric.load_prompts(str(prompts_path), language)
            metric.set_prompts(**prompts)
        except (FileNotFoundError, ValidationError):
            print(
                f"Adapting prompts for {metric.name} to {language} and saving to {prompts_path}"
            )
            prompts = asyncio.run(metric.adapt_prompts(language, llm))
            metric.set_prompts(**prompts)
            with ensure_utf8_open():
                remove_cache_prompts(prompts_path, metric)
                metric.save_prompts(str(prompts_path))


async def adapt_prompts(
    language: str, llm: BaseRagasLLM, metrics_with_llm: List[MetricWithLLM]
) -> None:
    """让metrics里的prompts适配到指定的语言"""
    for metric in metrics_with_llm:
        prompts = await metric.adapt_prompts(language, llm)
        metric.set_prompts(**prompts)


def rag_evaluate(samples: List[Dict[str, Any]], model: str = "gpt-4o") -> DataFrame:
    """samples 格式同 convert_to_evaluation_dataset 函数的输入"""
    # TODO： 增加metrics作为参数

    dataset = convert_to_evaluation_dataset(samples)
    llm = LangchainLLMWrapper(create_llm(model=model))
    embed_model = LangchainEmbeddingsWrapper(create_embed_model())
    metrics_with_llm: List[MetricWithLLM] = [
        AnswerRelevancy(),
        AnswerCorrectness(),
    ]
    metrics_without_llm: List[SingleTurnMetric] = [
        BleuScoreForChinese()
    ]

    language = "chinese"
    load_or_adapt_prompts(
        Path(PROJECT_DIR / "cache/.ragas_cache/prompts"), language, llm, metrics_with_llm
    )

    # TODO: dataset大的时候，容易出现问LLM超时的问题导致中断。现在暂时先建议每次输入15个问题以下。之后实现缓存机制
    result = None

    retry_time = 0
    while result is None:
        try:
            result = evaluate(
                dataset,
                metrics=metrics_with_llm + metrics_without_llm,
                # metrics=[AnswerCorrectness()],
                llm=llm,
                raise_exceptions=(retry_time <= 3),
                embeddings=embed_model,
            ) 
        except:
            retry_time += 1 
            print(f"Retrying... in time {retry_time}")

    result = result.to_pandas()
    list_result = [{key: r for (key, r) in zip(list(result.columns), rs)} for rs in result.values]

    return list_result


def evaluation_result_to_json(json_eval_results: List[Dict], output_path: Path) -> None:
    difficulties = list(set([i.get('difficulty', '') for i in json_eval_results]))
    difficulties += ['']

    eval_keys = list(i for i in json_eval_results[0].keys() if i not in ["user_input", "response", "reference", "difficulty"])

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_eval_results, f, ensure_ascii=False, indent=4)
    print(f"问答对数量：{len(json_eval_results)}")
    for dif in difficulties:
        print(f"难度：{dif}")
        for i in eval_keys:
            print(f"'{i}' 均值：{round(np.mean([j[i] for j in json_eval_results if dif in j.get('difficulty','')]), 3)}")


def batch_evaluate_and_save(data_samples: List[Dict[str, Any]], difficulty_list, output_path, batch_size: int = 10):
    all_results = []
    
    # 按批次处理数据
    for i in range(0, len(data_samples), batch_size):
        batch_samples = data_samples[i:i+batch_size]        # python 切片中即使结束索引超出了列表的实际长度，也不会报错
        # 调用评估函数处理当前批次
        # print(batch_samples)
        batch_result = rag_evaluate(batch_samples)
        all_results += batch_result
    
    # 将所有评估结果写入JSON文件
    for i, result in enumerate(all_results):
        result['difficulty'] = difficulty_list[i]

    evaluation_result_to_json(all_results, output_path)

def delete_citation(text):
    # 删除 <style> 到 </style> 之间的内容，包括这两个标签
    text = re.sub(r'<style>.*?</style>', '', text, flags=re.DOTALL)
    stack = []  # 存储未匹配的 '[' 的索引
    i = 0
    text_to_replace = []
    while i < len(text):
        if text[i] == "[":  # 检查是否是开始标记
            stack.append(i)  # 将开始标记的位置入栈
        elif text[i] == "]" and stack:  # 找到匹配的结束标记
            start = stack.pop()  # 弹出对应的开始标记位置
            if text[start:start+5] == "[数据来源":
                text_to_replace += [text[start:i+1]]
        i += 1  # 继续查找
    
    for item in text_to_replace:
        text = text.replace(item, "")
    return text.strip()

def eva(md5_dir, perfix=''):

    data_samples = json.load(open(PROJECT_DIR / f"graphrag_dir/{md5_dir}/{perfix}questions_and_answers.json", mode='r', encoding='utf-8'))
    labeled_samples = json.load(open(PROJECT_DIR / f"documents/labeled_queries.json", mode='r', encoding='utf-8'))
    difficulty_list = []
    for sample in data_samples:
        assert len([i["示例答案"] for i in labeled_samples if i["问题"] == sample["question"]])==1, f"问题：{sample['question']} 示例答案数量不为1而为{len([i['示例答案'] for i in labeled_samples if i['问题'] == sample['question']])}"
        sample["reference"] = [i["示例答案"] for i in labeled_samples if i["问题"] == sample["question"]][0]
        sample['response'] = delete_citation(sample['response'])
        difficulty_list += [i["难度"] for i in labeled_samples if i["问题"] == sample["question"]]
        del sample["mode"]

    # 分批次评估并汇总存储结果
    # output_path = Path(PROJECT_DIR / f"new_evaluated_questions_and_answers.json")
    output_path = Path(PROJECT_DIR / f"graphrag_dir/{md5_dir}/{perfix}evaluated_questions_and_answers.json")
    batch_evaluate_and_save(data_samples, difficulty_list, output_path, batch_size=5)


# def eva(md5_dir = '66db4cf7c1e9d925e1f2c2bd4b12861a', perfix=''):
    
#     data_samples = json.load(open(PROJECT_DIR / f"documents/rewrite_queries.json", mode='r', encoding='utf-8'))
#     # data_samples = json.load(open(PROJECT_DIR / f"graphrag_dir/{md5_dir}/questions_and_answers.json", mode='r', encoding='utf-8'))
#     labeled_samples = json.load(open(PROJECT_DIR / f"documents/labeled_queries.json", mode='r', encoding='utf-8'))
#     for sample in data_samples:
#         assert len([i["示例答案"] for i in labeled_samples if i["问题"] == sample["question"]])==1, f"问题：{sample['question']} 示例答案数量不为1"
#         sample["reference"] = [i["示例答案"] for i in labeled_samples if i["问题"] == sample["question"]][0]
#         sample['response'] = delete_citation(sample['response'])

#     # print(data_samples)
#     # 分批次评估并汇总存储结果
#     output_path = Path(PROJECT_DIR / f"new_evaluated_questions_and_answers.json")
#     # output_path = Path(PROJECT_DIR / f"graphrag_dir/{md5_dir}/{perfix}evaluated_questions_and_answers.json")
#     batch_evaluate_and_save(data_samples[:5], output_path, batch_size=5)

if __name__ == "__main__":
    eva(md5_dir = '5ce17013c611d6c0b529d0403b8dddd4', perfix = '')