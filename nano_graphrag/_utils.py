import asyncio
import html
import json
import logging
import os
import numpy as np
import tiktoken
import copy
import re
import difflib
import aiofiles
import networkx as nx
import numbers
import traceback
import yaml
from dataclasses import dataclass
from functools import wraps
from hashlib import md5
from datetime import datetime
from typing import Any, Union
from collections import defaultdict, Counter
from tqdm import tqdm
from pathlib import Path
from matplotlib import colormaps
from .prompt import GRAPH_FIELD_SEP

ENCODER = None


class CustomLogger:
    def __init__(self):
        # 设置外部logger的层级
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("neo4j").setLevel(logging.ERROR)  # 只显示ERROR级别以上的日志，忽略WARNING

        # 设置日志格式，包括时间戳 [yyyy-mm-dd_hh-mm-ss] 的格式
        self.log_format = "[%(asctime)s]: %(message)s"
        self.log_time_format = r"%Y-%m-%d_%H-%M-%S"
        self.log_filename_format = r"%Y-%m-%d_%H-%M-%S-%f"

        # 创建日志器
        self.logger = logging.getLogger("nano-graphrag")
        self.logger.setLevel(logging.INFO)

        # 删除之前的处理器（如果有）
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # 日志文件路径初始化为空
        self.log_path = None

    def set_logging_dir(self, log_path: str):
        """设置日志文件的输出目录"""
        self.log_path = log_path

        # 删除之前的文件处理器（如果有）
        self.remove_file_handlers()

        # 创建文件处理器，将日志写入文件

        if self.log_path is not None:
            log_file_path = os.path.join(
                self.log_path, f"nanoG_{datetime.now().strftime(self.log_filename_format)}.log"
            )
            file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(
                logging.Formatter(self.log_format, datefmt=self.log_time_format)
            )
            self.logger.addHandler(file_handler)

    def remove_file_handlers(self):
        """删除之前添加的文件处理器"""
        self.logger.handlers = [
            h for h in self.logger.handlers if not isinstance(h, logging.FileHandler)
        ]

    def info(self, message):
        """logger.info 方法"""
        self.logger.info(message)

    def warning(self, message):
        """logger.warning 方法"""
        self.logger.warning(message)

    def error(self, message):
        """logger.error 方法"""
        self.logger.error(message)

    def debug(self, message):
        """logger.debug 方法"""
        self.logger.debug(message)


# 创建一个全局的 logger 实例
logger = CustomLogger()

html_content = """<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta name="theme-color" content="#e22653" />
    <meta name="description" content="{}" />
    <title>{}</title>
    <script type="module" crossorigin src="../../rag_visualize/assets/index-CGWYNYx7.js"></script>
    <link rel="stylesheet" crossorigin href="../../rag_visualize/assets/index-BlFu1SzW.css">
</head>
<body>
    <noscript>You need to enable JavaScript to run this app.</noscript>
    <div id="root"></div>
</body>
</html>"""

def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
    try:
        # If there is already an event loop, use it.
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # If in a sub-thread, create a new event loop.
        logger.info("Creating a new event loop in a sub-thread.")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop

def read_yaml_config() -> dict:
    """Read a YAML file and return its content as a dictionary."""
    yaml_path = os.path.join(os.getcwd(), 'config', 'nanoG_config.yaml')
    assert os.path.exists(yaml_path), f"YAML file not found: {yaml_path}"
    with open(yaml_path, 'r', encoding='utf-8') as file:
        content = yaml.safe_load(file)

    assert content.get("graph_storage_cls","").lower().strip() in ["networkx", "neo4j"], "graph_storage_cls must be one of [networkx, neo4j]"
    assert content.get("use_embedding_func","").lower().strip() in ["openai", "local"], "use_embedding_func must be one of [openai, local]"
    assert content.get("use_conversation_func","").lower().strip() in ["openai", "local"], "use_conversation_func must be one of [openai, local]"
    service_port_number = content.get("websocket", {}).get("port")
    assert isinstance(service_port_number, int), "端口号必须是整数！"
    assert service_port_number > 0 and service_port_number < 65536, "端口号必须在1-65535之间！"
    assert content.get("neo4j_settings", {}).get("GENERATE_FROM_NETWORKX") in [True, False], "neo4j_settings/GENERATE_FROM_NETWORKX must be True or False"
    assert isinstance(content.get("llm_func_timeout"), int), "func_node_timeout/timeout must be an integer"
    assert content.get("llm_func_timeout") > 0, "func_node_timeout/timeout must be greater than 0"


    content["graph_storage_cls"] = content["graph_storage_cls"].lower().strip()
    content["use_embedding_func"] = content["use_embedding_func"].lower().strip()
    content["use_conversation_func"] = content["use_conversation_func"].lower().strip()
    return content

def validate_tiktoken_cache():    
    cache_key = "s9b5ad71b2ce5302211f9c61530b329a4922fc6a4"
    tiktoken_cache_dir = os.path.join(os.getcwd(), 'tiktoken_cache')
    os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir
    
    # validate
    return os.path.exists(os.path.join(tiktoken_cache_dir, cache_key)) 


def locate_json_string_body_from_string(content: str) -> Union[str, None]:
    """Locate the JSON string body from a string"""
    maybe_json_str = re.search(r"{.*}", content, re.DOTALL)
    if maybe_json_str is not None:
        return maybe_json_str.group(0)
    else:
        return None


def fix_json_format(json_str: str) -> dict:
    """Attempt to fix and re-parse the JSON string when initial parsing fails.
    这个函数必须确保返回的data不是None，否则会导致后续的代码出现问题
    """
    try:
        # from half_json.core import JSONFixer

        # json_fixer = JSONFixer()
        # json_str = json_fixer.fix(json_str).line
        # assert json_str is not None
        # data = json.loads(json_str)
        import pythonmonkey
        jsonrepair = pythonmonkey.require('jsonrepair').jsonrepair
        repaired_json = jsonrepair(json_str.replace('```json', '').replace('```', '').strip())
        data = json.loads(repaired_json)

        if list(data.keys()) == ['']:  # 修复有时候会产生多一层空字符串的键值对
            return data['']
        else:
            return data
    except Exception as e:
        return None


def convert_response_to_json(response: str) -> dict:
    json_str = locate_json_string_body_from_string(response)
    assert json_str is not None, f"Unable to parse JSON from response: {response}"
    try:
        data = json.loads(json_str)
        return data
    except json.JSONDecodeError as e:
        # logger.error(f"Failed to parse JSON: {json_str}")
        raise e from None

def record_question_and_answer(working_dir, query, response, mode):
    """记录问题和回答到文件"""
    if response is None:
        return
    assert mode in ["local", "global", "naive"]
    
    file_path = os.path.join(working_dir, "questions_and_answers.json")
    # 初始化一个空列表用于存储问题和回答
    qa_data = []
    response = re.sub(r'<style>.*?</style>', '', response, flags=re.DOTALL)

    # 如果文件已经存在，则读取已有内容
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                qa_data = json.load(f)
            except json.JSONDecodeError:
                # 如果文件格式有问题（如为空或内容损坏），跳过读取并继续
                pass

    # 添加新的问题和回答
    new_entry = {
        "mode": mode,
        "question": query,
        "response": response
    }
    qa_data.append(new_entry)

    # 保存回文件
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(qa_data, f, ensure_ascii=False, indent=4)


def encode_string_by_tiktoken(content: str, model_name: str = "gpt-4o"):
    global ENCODER
    if ENCODER is None:
        ENCODER = tiktoken.encoding_for_model(model_name)
    tokens = ENCODER.encode(content)
    return tokens


def decode_tokens_by_tiktoken(tokens: list[int], model_name: str = "gpt-4o"):
    global ENCODER
    if ENCODER is None:
        ENCODER = tiktoken.encoding_for_model(model_name)
    content = ENCODER.decode(tokens)
    return content


def truncate_list_by_token_size(list_data: list, key: callable, max_token_size: int):
    """Truncate a list of data by token size"""
    if max_token_size <= 0:
        return []
    tokens = 0
    for i, data in enumerate(list_data):
        tokens += len(encode_string_by_tiktoken(key(data)))
        if tokens > max_token_size:
            return list_data[:i]
    return list_data


def compute_mdhash_id(content, prefix: str = ""):
    return prefix + md5(content.encode()).hexdigest()


def write_json(json_obj, file_name):
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(json_obj, f, indent=2, ensure_ascii=False)


def load_json(file_name):
    if not os.path.exists(file_name):
        return None
    with open(file_name, encoding="utf-8") as f:
        return json.load(f)


# it's dirty to type, so it's a good way to have fun
def pack_user_ass_to_openai_messages(*args: str):
    roles = ["user", "assistant"]
    return [
        {"role": roles[i % 2], "content": content} for i, content in enumerate(args)
    ]


def is_float_regex(value):
    return bool(re.match(r"^[-+]?[0-9]*\.?[0-9]+$", value))


def compute_args_hash(*args):
    return md5(str(args).encode()).hexdigest()


def split_string_by_multi_markers(content: str, markers: list[str]) -> list[str]:
    """Split a string by multiple markers"""
    if not markers:
        return [content]
    results = re.split("|".join(re.escape(marker) for marker in markers), content)
    return [r.strip() for r in results if r.strip()]


def enclose_string_with_quotes(content: Any) -> str:
    """Enclose a string with quotes"""
    if isinstance(content, numbers.Number):
        return str(content)
    content = str(content)
    content = content.strip().strip("'").strip('"')
    return f'"{content}"'


def list_of_list_to_csv(data: list[list], markdown=False):
    if markdown:
        # 将表头作为第一行
        header = data[0]
        # 生成表头字符串，加上最左边和最右边的 "|"
        header_line = "| " + " | ".join([f"{col}" for col in header]) + " |"
        # 生成表头和内容的分隔符
        separator_line = "| " + " | ".join(["---" for _ in header]) + " |"
        # 生成内容部分
        content_lines = [
            "| " + " | ".join([f"{cell}" for cell in row]) + " |" for row in data[1:]
        ]
        # 将所有部分合并为最终的 Markdown 表格字符串
        return "\n".join([header_line, separator_line] + content_lines)
    else:
        return "\n".join(
            [
                ",\t".join(
                    [f"{enclose_string_with_quotes(data_dd)}" for data_dd in data_d]
                )
                for data_d in data
            ]
        )


# -----------------------------------------------------------------------------------
# Refer the utils functions of the official GraphRAG implementation:
# https://github.com/microsoft/graphrag
def clean_str(input: Any) -> str:
    """Clean an input string by removing HTML escapes, control characters, and other unwanted characters."""
    # If we get non-string input, just give it back
    if not isinstance(input, str):
        return input

    result = html.unescape(input.strip())
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python
    result = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", result)
    result = '''"{}"'''.format(result.strip('"\\ ')) if isinstance(result, str) else None
    return result


# Utils types -----------------------------------------------------------------------
@dataclass
class EmbeddingFunc:
    embedding_dim: int
    max_token_size: int
    func: callable

    async def __call__(self, *args, **kwargs) -> np.ndarray:
        return await self.func(*args, **kwargs)


# Decorators ------------------------------------------------------------------------
# 新添加超时抛出、限制最大并发任务数量、添加异常处理、不产生数据竞争问题，但不包含重试和APIConnectionError处理
# 重试和APIConnectionError处理需要放到_llm.py直接调用AsyncOpenAI的部分中，在装饰器这里无法正常捕获
def limit_async_func_call(
    max_size: int, waitting_time: float = 0.001
):
    """Add restriction of maximum async calling times for an async func, with a timeout"""

    yaml_config = read_yaml_config()
    timeout = int(yaml_config.get("llm_func_timeout"))
    semaphore = asyncio.Semaphore(max_size)

    def final_decro(func):
        """Using asyncio.Semaphore to control async concurrency"""

        @wraps(func)
        async def wait_func(*args, **kwargs):
            async with semaphore:  # Ensure semaphore is acquired and released properly
                try:
                    result = await asyncio.wait_for(
                        func(*args, **kwargs), timeout=timeout
                    )
                except asyncio.TimeoutError:
                    # 只记录简洁信息，不打印完整堆栈
                    logger.warning(f"Task timed out after {timeout}s (will be handled by retry mechanism)")
                    raise  # 继续抛出给retry处理
                except Exception as e:
                    raise
                return result

        return wait_func

    return final_decro


# # 原始的装饰器，不使用asyncio.Semaphore，避免使用nest-asyncio
# def limit_async_func_call(max_size: int, waitting_time: float = 0.0001):
#     """Add restriction of maximum async calling times for a async func"""

#     def final_decro(func):
#         """Not using async.Semaphore to aovid use nest-asyncio"""
#         __current_size = 0

#         @wraps(func)
#         async def wait_func(*args, **kwargs):
#             nonlocal __current_size
#             while __current_size >= max_size:
#                 await asyncio.sleep(waitting_time)
#             __current_size += 1
#             # 下面这步如果抛出异常，__current_size可能不会变化
#             # 导致程序一直停不下来、无法正常结束
#             result = await func(*args, **kwargs)
#             __current_size -= 1
#             return result

#         return wait_func

#     return final_decro


async def safe_gather(tasks, default_value=None, raise_on_error=True):
    """
    安全执行一组任务，处理可能的错误，尤其是正确处理调用llm时超时异常。
    为了安全起见，当await asycnio.gather(*[])内的任务包含llm调用时，需要在最内层将await asycnio.gather(*[])替换为await safe_gather([])。
    即替换函数名、删除*即可。
    return_exceptions=True可以让llm超时异常被limit_async_func_call捕获并由@retry装饰器重试，而不是直接抛出。
    具体原因可参考：https://www.notion.so/enoch2090/nanoG-LLM-1d5a48fa48b28020849cf5868ff1cb40?pvs=4
    
    Args:
        tasks: 要执行的任务列表
        default_value: 失败任务的默认返回值（仅在raise_on_error=False时才返回、否则直接raise）
        raise_on_error: 如果为True，任何错误都会导致引发第一个错误
        
    Returns:
        处理后的结果列表，或者在有错误且raise_on_error=True时引发异常
    """
    if not tasks:
        return []
        
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    errors = []
    processed_results = []
    
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            errors.append((i, result))
            processed_results.append(default_value)
        else:
            processed_results.append(result)
    
    if errors:
        error_details = []
        for i, err in errors:
            tb_str = "".join(traceback.format_exception(type(err), err, err.__traceback__))
            error_details.append(f"Task {i}: {type(err).__name__}: {err}\n{tb_str}")
            
        error_message = f"Encountered {len(errors)} errors:\n" + "\n".join(error_details)
        logger.error(error_message)
        
        if raise_on_error:
            # 抛出第一个错误，保留其原始堆栈
            first_error_idx, first_error = errors[0]
            logger.error(f"Raising first error from task {first_error_idx}")
            raise first_error
    
    return processed_results


def wrap_embedding_func_with_attrs(**kwargs):
    """Wrap a function with attributes"""

    def final_decro(func) -> EmbeddingFunc:
        new_func = EmbeddingFunc(**kwargs, func=func)
        return new_func

    return final_decro


def parse_json(text: str) -> str:
    '''
    通过正则表达式解析输入的字符串，若字符串中存在符合 JSON 格式（即 json.loads() 能读取的部分）则返回此部分，否则返回空字符串。
    '''
    json_pattern = re.compile(r'\{(?:[^{}]|(?R))*\}')
    match = json_pattern.search(text)
    return match.group(0) if match else ''


def compare_strings(str1, str2):
    # 使用 SequenceMatcher 进行相似度比较
    # 第一个参数用于设置自定义比较函数，此处使用默认方式（字符逐一比较），寻找最长公共子序列
    matcher = difflib.SequenceMatcher(None, str1, str2)
    return matcher.ratio()

def fix_citation_in_response(text: str, hash_dict: dict, websocket) -> str:
    """修复 OpenAI API 返回的引用问题。
    该函数将文本中的引用字符串转换为脚注格式，以便在 HTML 中显示。
    这个函数理论上不会出现任何exception
    """

    def find_data_references(text):
        # 创建一个字典，用于映射中文符号到英文符号
        # 编译正则表达式
        pattern = re.compile(r"""
            \[Data.*?\] |            # 匹配 [Data: ...]
            \【Data.*?\】 |           # 匹配 【Data: ...】
            \[数据.*?\] |             # 匹配 [数据: ...]
            \【数据.*?\】 |            # 匹配 【数据: ...】
            \[数据源.*?\] |           # 匹配 [数据源: ...]
            \【数据源.*?\】 |          # 匹配 【数据源: ...】
            \[数据来源.*?\] |         # 匹配 [数据来源: ...]
            \【数据来源.*?\】          # 匹配 【数据来源: ...】
        """, re.VERBOSE)
        # 在文本中查找所有匹配项
        return pattern.findall(text)
    
    # text = "根据现有数据，华为对当前工作的态度主要体现在对跌倒检测和步态身份识别两方面。\n\n在跌倒检测方面，华为的陈重指出了系统在精度和性能上的一些观点。具体来说，当前的跌倒识别精度已经达到99%，但在泛化性能方面，如处理反例、不同人群和不同场景方面，仍需要进一步提升[Data: 来源 (0), 报告 (0)]. WYM作为华为沟通中的重要参与者，也强调了跌倒检测比身份识别更容易实现精度提高，并提出需要在框架或输入特征层面进行优化[Data: 实体 (1); 关系 (0)]。\n\n对于步态身份识别，目前华为对其基本满意，认为不需要做进一步的优化实验。尽管如此，他们还指出，需要采集不同场景数据以优化和提高在不同时间段的测试准确率[Data: 来源 (0)]。\n\n总结来说，华为对当前的工作持支持和期待的态度，鼓励进一步的数据收集和算法优化，以提升系统的整体性能和精度[Data: 来源 (0), 实体 (0)]."
    try:
        punctuation_map = {
            "，": ",",
            "。": ".",
            "（": "(",
            "）": ")",
            "【": "[",
            "】": "]",
            "—": "-",
            "…": "...",
            "：": ":",
            "；": ";"
        }
        # 使用 translate 方法和 str.maketrans 函数创建一个转换表
        trans_table = str.maketrans(punctuation_map)
        original_cites = find_data_references(text)
        hash_dict = copy.deepcopy(hash_dict)
        footnote_style = """<style>
        /* 为具有 footnote-link 类的超链接设置蓝色和下划线 */
        .footnote-link {
            color: blue;
            text-decoration: underline;
        }
    </style>\n\n"""

        for key, value in hash_dict.items():
            hash_dict[key] = {k: {'id':v , 'type_index':-1, 'html_index':-1} for k, v in value.items()}     # 分别记录每个引用的类型、hash id、类型脚注、html脚注
            hash_dict[key]['total'] = 0      # 用于记录这个类型的引用的数量
        hash_dict['total'] = 0          # 用于记录所有类型的引用的数量

        for original_cite in original_cites:
            # 每个引用块
            use_cite_dict = defaultdict(list)
            o_cite = original_cite.translate(trans_table).strip()
            o_cite = o_cite[o_cite.find(":")+1:-1].strip()
            parts = o_cite.split(")")
            for part in parts:
                # 每种引用，得到同种类型的引用、转换为小写
                part = (part+')').lstrip(';, ').strip().lower()     # 处理分号和逗号错误的情况
                if any(part.startswith(i) for i in ["实体", "节点", "node", "nodes", "entity", "entities"]):
                    match_type = "实体"
                elif any(part.startswith(i) for i in ["关系", "边", "edge", "edges", "relation", "relations"]):
                    match_type = "关系"
                elif any(part.startswith(i) for i in ["社区报告", "报告", "社区", "report", "reports", "community", "communities"]):
                    match_type = "社区报告"
                elif any(part.startswith(i) for i in ["文本块", "来源", "source", "sources", "chunk", "chunks"]):
                    match_type = "文本块"
                else:
                    continue

                # 初始化结果列表
                index_list = []
                # 使用正则表达式找到所有括号内的内容
                matches = re.findall(r'\((.*?)\)', part)
                # 遍历所有匹配，分割和合并结果
                for match in matches:
                    for item in match.split(','):
                        item = item.strip()
                        if "more" in item:
                            index_list.append("+more")
                            item = item.replace("+more", "").replace("more", "").strip()        # 处理"more"、"+more"、"2+more"等情况
                        if item.isdigit() and int(item) in hash_dict[match_type]:               # 注意这里不是elif，因为上面可能会改变item的值
                            # 如果不是数字或者数字不在hash_dict里面，直接跳过，这个错误的item不会出现在最终返回的text中
                            if hash_dict[match_type][int(item)]['html_index'] == -1 and hash_dict[match_type][int(item)]['type_index'] == -1:
                                # 这个引用是第一次出现
                                hash_dict['total'] +=1
                                hash_dict[match_type]['total'] += 1
                                hash_dict[match_type][int(item)]['html_index'] = hash_dict['total']
                                hash_dict[match_type][int(item)]['type_index'] = hash_dict[match_type]['total']

                            index_list.append(f'{hash_dict[match_type][int(item)]["type_index"]}<sup><a href="#{hash_dict[match_type][int(item)]["id"]}" class="footnote-link">[{hash_dict[match_type][int(item)]["html_index"]}]</a></sup>')     # 这里加上hash id和html脚注，需要用html_index
                
                if len(index_list):
                    use_cite_dict[match_type] += index_list

            items = []
            for key, value in use_cite_dict.items():
                # 去重、如果+more在value里面，将其移动到最后面
                value = list(Counter(value).keys())
                value = [i for i in value if i != "+more"] + [i for i in value if i == "+more"]
                items.append(f"{key}({', '.join(map(str, value))})")
            # 将所有元素连接成一个字符串，以分号和空格分隔
            # 构建最终的输出格式，用分号分割不同类型的引用
            new_cite = f"[数据来源: {'; '.join(items)}]" if len(use_cite_dict) else ""
            text = text.replace(original_cite, new_cite)
    except:
        e = traceback.format_exc()
        logger.error(e)
        websocket.send(e)
        raise
    finally:
        return footnote_style + text



def get_node_cluster(attributes:dict):
    """获取节点的cluster信息"""
    clusters = attributes.get("clusters")
    if clusters is not None:
        cluster = json.loads(clusters)[0]["cluster"]
    else:
        cluster = None
    return cluster


def get_cluster_list(attributes:dict, cluster_info:dict)->list:
    """获取节点的cluster详细信息"""
    clusters = attributes.get("clusters")
    if clusters is not None:
        cluster = json.loads(clusters)
        for c in cluster:
            c["title"] = cluster_info.get(str(c["cluster"]), {}).get("title")
        cluster = sorted(cluster, key=lambda x: (x['level'], x['cluster']))
    else:
        cluster = [
            {
                "level":0,
                "cluster":-1,
                "title":"未分类",
            }
        ]
    return cluster

def rgb_to_hex(r, g, b):
    """将 RGB 颜色转换为十六进制颜色"""
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))

def generate_cluster_color_mapping(cluster_info: dict):
    """为每个簇生成十六进制颜色映射"""
    # 获取颜色映射（使用 'tab10' 颜色映射）
    cmap = colormaps.get_cmap('tab20')
    
    # 计算簇的数量并根据簇的数量生成对应颜色
    colors = [cmap(i % cmap.N) for i in range(len(cluster_info))]  # 获取颜色映射

    # 给每个簇分配颜色 (转换成十六进制)
    cluster_color_map = {
        cluster_id: rgb_to_hex(colors[index][0], colors[index][1], colors[index][2])
        for index, cluster_id in enumerate(cluster_info)
    }
    
    # 为没有簇信息的节点分配一个特殊颜色（灰色）
    cluster_color_map[None] = "#D3D3D3"  # 灰色的十六进制形式
    
    return cluster_color_map

async def create_graph_layout(G):
    """生成图节点与边的布局"""
    logger.info("Generating layout...")
    pos = nx.spring_layout(G)  # 生成坐标
    return pos

async def get_cluster_info(clusters) -> tuple[dict, dict]:
    """获取所有cluster的信息(title, level, key)，以及边的簇的信息(level, cluster, title)"""
    cluster_info = {}
    edges_info = {}
    for cluster_id, attributes in tqdm(clusters.items()):
        cluster_info[cluster_id] = {
            "title": attributes["report_json"]["title"], 
            "level": attributes["level"],
            "key":cluster_id,
        }
        
        for edge in attributes.get("edges", []):
            c = {"level": attributes["level"], "cluster": cluster_id, "title": attributes["report_json"]["title"]}
            edge = sorted(edge) # 无向边
            edges_info[tuple(edge)] = edges_info[tuple(edge)] + [c] if tuple(edge) in edges_info else [c]
            
    for edge in edges_info:
        edges_info[edge] = sorted(edges_info[edge], key=lambda x: (x['level'], x['cluster']))

    return cluster_info, edges_info

async def create_visualize_dataset(working_dir, chunk_entity_relation_graph, community_reports, text_chunks):
    # 并行执行图加载、布局生成和实体信息读取
    # 使用 copy.deepcopy 避免修改原始数据，因为这里是引用传递
    from ._storage import (
        JsonKVStorage,
        NanoVectorDBStorage,
        NetworkXStorage,
        Neo4jStorage,
    )
    if isinstance(chunk_entity_relation_graph, Neo4jStorage):
        # 如果是Neo4jStorage，直接从数据库中获取数据，不需要针对作图额外处理
        return
    else:
        G = copy.deepcopy(chunk_entity_relation_graph._graph)
        graph_layout_task = create_graph_layout(G)
        cluster_info_task = get_cluster_info(copy.deepcopy(community_reports._data))

        # 并行运行
        (pos),  (cluster_info, edges_info) = await asyncio.gather(
            graph_layout_task, cluster_info_task
        )
        entities_type = list(set([G.nodes[entity_name]["entity_type"].strip('\"') for entity_name in G.nodes]))

        sigma_data = defaultdict(list)

        """生成节点信息"""
        for node, coordinates in tqdm(pos.items()):
            # 获取节点的hash值，此处节点扔带双引号
            entity_hash = compute_mdhash_id(node, prefix="ent-")
            node_attributes = G.nodes(data=True)[node]
            
            node_attributes["clusters"] = get_cluster_list(node_attributes, cluster_info)  # 将簇信息添加到属性字典中

            sigma_data["nodes"].append({
                "id": node.strip('\"'),  # 节点的ID
                "label": node.strip('\"'),  # 节点标签
                "tag":node_attributes["entity_type"].strip('\"'),
                "description":node_attributes["description"].strip('\"'),
                "clusters":node_attributes["clusters"],
                "x": float(coordinates[0]),  # 节点的坐标
                "y": float(coordinates[1]), 
                "score": G.degree[node],
                "hash": entity_hash,
                "source_id": node_attributes["source_id"].split(GRAPH_FIELD_SEP)
            })


        """生成边信息"""
        for edge in tqdm(G.edges(data=True)):
            edge_attributes = edge[2]

            edge_cluster = edges_info.get(
                tuple(sorted((edge[0], edge[1]))), 
                [
                    {
                        "cluster":-1,
                        "level":0,
                        "title":"未分类"
                    }
                ],
                )
            edge0 = edge[0].strip('\"')
            edge1 = edge[1].strip('\"')
            sigma_data["edges"].append({
                "id": f"{edge0}-{edge1}",  # 边的 ID
                "source": edge0,  # 边的源节点
                "target": edge1,  # 边的目标节点
                "description": edge_attributes["description"].strip('\"'),
                "clusters": edge_cluster,
                "order": edge_attributes["order"],
            })

        """生成簇信息"""
        cluster_color_map = generate_cluster_color_mapping(cluster_info)
        for cluster_id, cluster_attr in cluster_info.items():
            sigma_data["clusters"].append({
                "key": cluster_id,
                "color": cluster_color_map[cluster_id],
                "clusterLabel": cluster_attr["title"],
            })
        sigma_data["clusters"].append(
            {
                "key":"-1",
                "color":"rgba(211, 211, 211, 1)",
                "clusterLabel":"未分类"

            }
        )
        """生成标签信息"""
        for tag in entities_type:
            sigma_data["tags"].append(
                {
                    "key":tag,
                }
            )

        sigma_data["chunks"] = copy.deepcopy(text_chunks._data)
    
        with open(os.path.join(working_dir, 'visualize_dataset.json'), 'w', encoding='utf-8') as json_file:
            json.dump(sigma_data, json_file, ensure_ascii=False, indent=4)
        with open(os.path.join(working_dir, 'last_build_graph.json'), 'r', encoding='utf-8') as f:
            g = json.load(f)['overall_hash']
            md5_dir, build_time = g[:32], g[32:]
        # 打开文件并写入
        description = f"工作区图谱可视化：{g}"
        with open(os.path.join(working_dir, "index.html"), "w", encoding="utf-8") as html_file:
            html_file.write(html_content.format(description, description))


async def save_graph_to_file(working_dir, chunk_entity_relation_graph, text_chunks):
    """将图谱数据保存到文件中"""
    data_for_vdb = {}
    entity_data_json = {}
    relation_data_json = {}
    from ._storage import (
        JsonKVStorage,
        NanoVectorDBStorage,
        NetworkXStorage,
        Neo4jStorage,
    )
    if isinstance(chunk_entity_relation_graph, Neo4jStorage):
        ns = await chunk_entity_relation_graph.get_all_nodes()
        es = await chunk_entity_relation_graph.get_all_edges()

        for entity in ns:
            entity_name = entity["id"]
            data_for_vdb[compute_mdhash_id(entity_name, prefix="ent-")] = {
                "content": entity_name + entity["description"],
                "entity_name": entity_name,
                "source_type": list(set([text_chunks._data[chunk_id]['type'] for chunk_id in entity["source_id"].split('<SEP>') if chunk_id.strip()])),
            }
            entity_data_json[compute_mdhash_id(entity_name, prefix="ent-")] = {
                "entity_name": entity_name,
                "description": entity["description"],
                "entity_type": entity["entity_type"],
                "source_id": entity["source_id"],
                "source_type": list(set([text_chunks._data[chunk_id]['type'] for chunk_id in entity["source_id"].split('<SEP>') if chunk_id.strip()])),
            }
        for relation in es:
            src_name = relation['src_name']
            tgt_name = relation['tgt_name']
            src_id = compute_mdhash_id(src_name, prefix="ent-")
            tgt_id = compute_mdhash_id(tgt_name, prefix="ent-")
            relation_data_json[f'rel-{src_id}_{tgt_id}'] = {
                "src_id": src_id,
                "src_name": src_name,
                "tgt_id": tgt_id,
                "tgt_name": tgt_name,
                "description": relation["description"],
                "weight": relation["weight"],
                "source_id": relation["source_id"],
                "source_type": list(set([text_chunks._data[chunk_id]['type'] for chunk_id in entity["source_id"].split('<SEP>') if chunk_id.strip()])),
                "order": relation["order"],
            }
    else:
        ns = chunk_entity_relation_graph._graph.nodes
        es = chunk_entity_relation_graph._graph.edges
        for entity_name in ns:
            entity = ns[entity_name]
            data_for_vdb[compute_mdhash_id(entity_name, prefix="ent-")] = {
                "content": entity_name + entity["description"],
                "entity_name": entity_name,
                "source_type": list(set([text_chunks._data[chunk_id]['type'] for chunk_id in entity["source_id"].split('<SEP>') if chunk_id.strip()])),
            }
            entity_data_json[compute_mdhash_id(entity_name, prefix="ent-")] = {
                "entity_name": entity_name,
                "description": entity["description"],
                "entity_type": entity["entity_type"],
                "source_id": entity["source_id"],
                "source_type": list(set([text_chunks._data[chunk_id]['type'] for chunk_id in entity["source_id"].split('<SEP>') if chunk_id.strip()])),
            }
        for src_name, tgt_name in es:
            relation = es[src_name, tgt_name]
            src_id = compute_mdhash_id(src_name, prefix="ent-")
            tgt_id = compute_mdhash_id(tgt_name, prefix="ent-")
            relation_data_json[f'rel-{src_id}_{tgt_id}'] = {
                "src_id": src_id,
                "src_name": src_name,
                "tgt_id": tgt_id,
                "tgt_name": tgt_name,
                "description": relation["description"],
                "weight": relation["weight"],
                "source_id": relation["source_id"],
                "source_type": list(set([text_chunks._data[chunk_id]['type'] for chunk_id in entity["source_id"].split('<SEP>') if chunk_id.strip()])),
                "order": relation["order"],
            }

    logger.info(f"Saving graph data to JSON files with {len(entity_data_json)} entities and {len(relation_data_json)} relationships...")
    # 使用 aiofiles 异步保存JSON文件
    async with aiofiles.open(
        os.path.join(working_dir, 'entities_info.json'), 'w', encoding='utf-8'
    ) as f:
        await f.write(json.dumps(entity_data_json, ensure_ascii=False, indent=4))

    async with aiofiles.open(
        os.path.join(working_dir, 'relationships_info.json'), 'w', encoding='utf-8'
    ) as f:
        await f.write(json.dumps(relation_data_json, ensure_ascii=False, indent=4))

    return data_for_vdb


async def resolve_from_json(md5_dir, knowledge_graph_inst):
    logger.warning(f"\n-*-*-*-*-*-*-*-*-*-*-\nLoading graph nodes and edges from JSON files in {md5_dir}...\n-*-*-*-*-*-*-*-*-*-*-\n")
    ns = json.load(open(os.path.join(os.getcwd(),'graphrag_dir',md5_dir,'entities_info.json'), 'r', encoding='utf-8'))
    es = json.load(open(os.path.join(os.getcwd(),'graphrag_dir',md5_dir,'relationships_info.json'), 'r', encoding='utf-8'))

    await knowledge_graph_inst.upsert_nodes_batch(
        [
            (
                info["entity_name"],
                {k: v for k, v in info.items() if k != "entity_name"}
            )
            for info in ns.values()
        ]
    )
    await knowledge_graph_inst.upsert_edges_batch(
        [
            (
                info["src_name"], 
                info["tgt_name"],
                {k: v for k, v in info.items() if k != "src_name" and k != "tgt_name"}
            )
            for info in es.values()
        ]
    )
    return knowledge_graph_inst


async def cluster2communityIds(nodes):
    ns = copy.deepcopy(nodes)
    ns = {k: v for k, v in ns.items()}
    non_count = 0
    # get v的clusters字段。如果没有这个字段，那么k的communityIds是[-1]
    # 如果有这个字段，用json.loads加载解析，比如'[{"level": 0, "cluster": 2}，{"level": 1, "cluster": 5}]'需要解析成[2, 5]
    for k, v in ns.items():
        if "clusters" in v:
            clusters = json.loads(v["clusters"])
            # 这里需要对clusters字典排序，level小的排前面
            clusters = sorted(clusters, key=lambda x: (x['level'], x['cluster']))
            ns[k]["communityIds"] = [c["cluster"] for c in clusters]
            del ns[k]["clusters"]
        else:
            ns[k]["communityIds"] = [-1]
            non_count += 1

    logger.info(f"Converting {len(ns)} Nodes with Community to Neo4j...\n{non_count} of which has no community info, defaulting to [-1]...")
    return ns