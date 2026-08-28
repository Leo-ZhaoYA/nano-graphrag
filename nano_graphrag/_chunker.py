import re
import tiktoken
import pickle
import asyncio
from bs4 import BeautifulSoup
from typing import Dict
from ._splitter import SeparatorSplitter
from ._utils import (
    logger,
    compute_mdhash_id,
    safe_gather
)
from .prompt import GRAPH_FIELD_SEP, PROMPTS
from .table_handler import TableHandler


def chunking_by_token_size(
    tokens_list: list[list[int]],
    doc_keys,
    tiktoken_model,
    overlap_token_size=128,
    max_token_size=1024,
):
    results = []
    for index, tokens in enumerate(tokens_list):
        chunk_token = []
        lengths = []
        for start in range(0, len(tokens), max_token_size - overlap_token_size):

            chunk_token.append(tokens[start : start + max_token_size])
            lengths.append(min(max_token_size, len(tokens) - start))

        # here somehow tricky, since the whole chunk tokens is list[list[list[int]]] for corpus(doc(chunk)),so it can't be decode entirely
        chunk_token = tiktoken_model.decode_batch(chunk_token)
        for i, chunk in enumerate(chunk_token):
            results.append(
                {
                    "tokens": lengths[i],
                    "content": chunk.strip(),
                    "chunk_order_index": i,
                    "full_doc_id": doc_keys[index],
                    "type": "text",
                }
            )

    return results


def chunking_by_seperators(
    tokens_list: list[list[int]],
    doc_keys,
    tiktoken_model,
    overlap_token_size=128,
    max_token_size=1024,
):

    splitter = SeparatorSplitter(
        separators=[
            tiktoken_model.encode(s) for s in PROMPTS["default_text_separator"]
        ],
        chunk_size=max_token_size,
        chunk_overlap=overlap_token_size,
    )
    results = []
    for index, tokens in enumerate(tokens_list):
        chunk_token = splitter.split_tokens(tokens)
        lengths = [len(c) for c in chunk_token]

        # here somehow tricky, since the whole chunk tokens is list[list[list[int]]] for corpus(doc(chunk)),so it can't be decode entirely
        chunk_token = tiktoken_model.decode_batch(chunk_token)
        for i, chunk in enumerate(chunk_token):

            results.append(
                {
                    "tokens": lengths[i],
                    "content": chunk.strip(),
                    "chunk_order_index": i,
                    "full_doc_id": doc_keys[index],
                }
            )

    return results


async def get_chunks(new_docs, chunk_func=chunking_by_token_size, **chunk_func_params):
    inserting_chunks = {}

    new_docs_list = list(new_docs.items())
    docs = [new_doc[1]["content"] for new_doc in new_docs_list]
    doc_keys = [new_doc[0] for new_doc in new_docs_list]

    ENCODER = tiktoken.encoding_for_model("gpt-4o")
    tokens = ENCODER.encode_batch(docs, num_threads=16)
    chunks = chunk_func(
        tokens, doc_keys=doc_keys, tiktoken_model=ENCODER, **chunk_func_params
    )

    for chunk in chunks:
        inserting_chunks.update(
            {compute_mdhash_id(chunk["content"], prefix="chunk-"): chunk}
        )

    return inserting_chunks, {}


async def get_tabular_chunks(
    new_docs, chunk_func=chunking_by_token_size, global_config=None, **chunk_func_params
):
    def process_table_chunk(part, doc_key):
        soup = BeautifulSoup(f"<td>{part.strip()}</td>", 'html.parser')
        assert len([str(t) for t in soup.find_all('table') if str(t)]) == 1, "The part should contain only one HTML table."
        # 使用 BeautifulSoup 解析 HTML 并替换 <br> 标签为 Markdown 的换行符号
        # 替换所有的 <br> 标签为 Markdown 中的换行符号 '  <br> '，以便后续转换为 Markdown 格式时pandas能够正确解析
        for br in soup.find_all("br"):
            br.replace_with("  <br>  ")
        part = [str(t) for t in soup.find_all('table')][0].strip()

        th = TableHandler(global_config=global_config)
        th.load_html_table(part)
        content = th.to_markdown()
        table_handlers[th.get_hash()] = th
        return {
            "tokens": len(ENCODER.encode(content)),
            "content": content,
            "chunk_order_index": len(chunks),
            "full_doc_id": doc_key,
            "type": "table",
        }

    def process_text_chunk(part, doc_key):
        part = clean_html(part)
        if not len(part):
            return None
        tokens = ENCODER.encode(part)
        return chunk_func(
            [tokens],
            doc_keys=[doc_key],
            tiktoken_model=ENCODER,
            **chunk_func_params,
        )

    def clean_html(part):
        # 清理文本中的 <td> 标签以及多余的 <table> 标签
        part = part.replace('<table>', '').replace('</table>', '').strip()
        for td in ['<td>', '</td>']:
            while part.startswith(td):
                part = part[len(td):]
            while part.endswith(td):
                part = part[:-len(td)]
        return part.strip()
    
    async def set_context_for_table_chunks(i):
        # 在这里就把table handler的context设置好
        # 向i的左右两侧寻找最近的text chunk
        nonlocal parse_table_count, parse_table_total
        pre_content = None
        post_content = None
        for j in range(i-1, -1, -1):
            if chunks[j]["type"] == "text":
                pre_content = chunks[j]["content"]
                break
        for j in range(i+1, len(chunks)):
            if chunks[j]["type"] == "text":
                post_content = chunks[j]["content"]
                break
        assert table_handlers[compute_mdhash_id(chunks[i]["content"], prefix="chunk-")].get_hash() == compute_mdhash_id(chunks[i]["content"], prefix="chunk-")
        assert pre_content is not None and post_content is not None
        await table_handlers[compute_mdhash_id(chunks[i]["content"], prefix="chunk-")].set_context(pre_content, post_content)
        parse_table_count += 1
        now_ticks = PROMPTS["process_tickers"][
            parse_table_count % len(PROMPTS["process_tickers"])
        ]        
        print(
            f"{now_ticks} Processed {parse_table_count}/{parse_table_total} table chunks...\r",
            end="",
            flush=True,
        )
        logger.info(f"Processed {parse_table_count}/{parse_table_total} table chunks...")

    inserting_chunks = {}
    new_docs_list = list(new_docs.items())
    docs = [new_doc[1]["content"] for new_doc in new_docs_list]
    doc_keys = [new_doc[0] for new_doc in new_docs_list]
    table_handlers: Dict[str, TableHandler] = dict()
    parse_table_count = 0
    parse_table_total = 0

    ENCODER = tiktoken.encoding_for_model("gpt-4o")

    for doc, doc_key in zip(docs, doc_keys):
        table_index = [0]
        pattern = r'(<table.*?>.*?</table>)'
        # re.finditer 返回一个匹配项迭代器，可以获取每个匹配的开始位置和结束位置
        for match in re.finditer(pattern, doc, flags=re.DOTALL):
            table_index.append(match.start())
            table_index.append(match.end())
        table_index.append(len(doc))
        # 一般认为不会出现<table>嵌套的情况
        assert table_index == sorted(table_index), "The table index is not sorted."
        doc_parts = [doc[table_index[i]:table_index[i+1]] for i in range(0, len(table_index)-1) if doc[table_index[i]:table_index[i+1]].strip()]

        # chunks存储有序的chunk
        chunks = []
        for part in doc_parts:
            if part.startswith('<table'):
                table_chunk = process_table_chunk(part, doc_key)
                if table_chunk:
                    chunks.append(table_chunk)
                    parse_table_total += 1
            else:
                text_chunks = process_text_chunk(part, doc_key)
                if text_chunks:
                    chunks.extend(text_chunks)        
        
        set_context_tasks = []
        for i in range(len(chunks)):
            inserting_chunks.update(
                {compute_mdhash_id(chunks[i]["content"], prefix="chunk-"): chunks[i]}
            )
            if chunks[i]["type"] == "table":
                set_context_tasks.append(set_context_for_table_chunks(i))

        await safe_gather(set_context_tasks)
        print()

    return inserting_chunks, table_handlers


