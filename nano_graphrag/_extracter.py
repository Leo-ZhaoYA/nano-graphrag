import re
import aiofiles
import os
import json
import copy
import asyncio
import pickle
from typing import Union
from collections import Counter, defaultdict

from sqlalchemy import exists
from ._utils import (
    logger,
    compute_mdhash_id,
    encode_string_by_tiktoken,
    pack_user_ass_to_openai_messages,
    split_string_by_multi_markers,
    clean_str,
    decode_tokens_by_tiktoken,
    encode_string_by_tiktoken,
    is_float_regex,
    resolve_from_json,
    safe_gather,
    )
from .base import (
    BaseGraphStorage,
    BaseVectorStorage,
    TextChunkSchema,
)
from .prompt import GRAPH_FIELD_SEP, PROMPTS
from .table_handler import TableHandler
from ._storage import (
    JsonKVStorage,
    NanoVectorDBStorage,
    NetworkXStorage,
    Neo4jStorage,
)

async def _handle_entity_relation_summary(
    entity_or_relation_name: str,
    description: str,
    global_config: dict,
) -> str:
    '''对于描述过长的实体或关系，使用LLM生成其总结'''
    use_llm_func: callable = global_config["cheap_model_func"]
    llm_max_tokens = global_config["cheap_model_max_token_size"]
    tiktoken_model_name = global_config["tiktoken_model_name"]
    summary_max_tokens = global_config["entity_summary_to_max_tokens"]

    tokens = encode_string_by_tiktoken(description, model_name=tiktoken_model_name)
    if len(tokens) < summary_max_tokens:  # No need for summary
        return description
    # 原来的描述太长了，那么重新生成实体描述的总结
    prompt_template = PROMPTS["summarize_entity_descriptions"]
    use_description = decode_tokens_by_tiktoken(
        tokens[:llm_max_tokens], model_name=tiktoken_model_name
    )
    context_base = dict(
        entity_name=entity_or_relation_name,
        description_list=use_description.split(GRAPH_FIELD_SEP),
        domain_in_specific=" ,".join(PROMPTS["DOMAIN_IN_SPECIFIC"]),
    )
    use_prompt = prompt_template.format(**context_base)
    logger.debug(f"Trigger summary: {entity_or_relation_name}")
    summary = await use_llm_func(use_prompt, max_tokens=summary_max_tokens)
    return summary


async def _handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 4 or not any(key in record_attributes[0] for key in ['entity', 'entities']):
        return None
    # add this record as a node in the G
    entity_name = clean_str(record_attributes[1].upper())
    if entity_name == '""':
        return None
    entity_type = clean_str(record_attributes[2].upper())
    entity_description = clean_str(record_attributes[3])
    entity_source_id = chunk_key
    return dict(
        entity_name=entity_name,
        entity_type=entity_type,
        description=entity_description,
        source_id=entity_source_id,
    )


async def _handle_single_relationship_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 5 or not any(key in record_attributes[0] for key in ['relation', 'relations', 'relationship', 'relationships']):
        return None
    # add this record as edge
    source = clean_str(record_attributes[1].upper())
    target = clean_str(record_attributes[2].upper())
    edge_description = clean_str(record_attributes[3])
    edge_source_id = chunk_key
    weight = (
        float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 1.0
    )
    return dict(
        src_name=source,
        tgt_name=target,
        weight=weight,
        description=edge_description,
        source_id=edge_source_id,
    )



async def _merge_nodes_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    already_node: dict,
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    '''合并同一个实体的多个数据（比如不同来源的描述、类型等），然后将合并后的实体插入或更新到知识图谱中'''
    already_entity_types = []
    already_source_ids = []
    already_description = []

    # already_node存储的是已经存在的同名节点的数据
    if already_node is not None:
        already_entity_types.append(already_node["entity_type"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_node["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_node["description"])
    
    # 通过统计每个来源提供的实体类型，选择出现次数最多的实体类型
    entity_type = sorted(
        Counter(
            [dp["entity_type"] for dp in nodes_data] + already_entity_types
        ).items(),
        key=lambda x: x[1],
        reverse=True,
    )[0][0]

    # 将所有描述合并为一个字符串，使用分隔符 GRAPH_FIELD_SEP 连接
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in nodes_data] + already_description))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in nodes_data] + already_source_ids)
    )
    # 使用LLM重新获取实体描述
    description = await _handle_entity_relation_summary(
        entity_name, description, global_config
    )
    # 最终通过 upsert_node 将合并后的实体插入到知识图谱中。如果实体已经存在，会更新其数据；如果不存在，则创建新节点
    node_data = dict(
        entity_type=entity_type,
        description=description,
        source_id=source_id,
    )
    if already_node is not None:
        # 如果图里面已经有这个点 那么直接插入更新
        # upsert代表update or insert
        await knowledge_graph_inst.upsert_node(
            entity_name,
            node_data=node_data,
        )
        return []
    else:
        # 如果图里面没有这个点 那么返回留待后面一起整体插入
        # 当前先不插入
        return [(entity_name, node_data)]

async def _merge_edges_then_upsert(
    src_name: str,
    tgt_name: str,
    edges_data: list[dict],
    already_edge: dict,
    already_edge_nodes: list[list],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    already_weights = []
    already_source_ids = []
    already_description = []
    # 检查知识图谱中是否已经存在这条边。如果存在，则提取已有的权重、描述和来源 ID，并准备合并。
    already_order = []
    # already_edge存储的是已经存在的同名边的数据
    if already_edge is not None:        
        already_weights.append(already_edge["weight"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_edge["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_edge["description"])
        already_order.append(already_edge.get("order", 1))

    # [numberchiffre]: `Relationship.order` is only returned from DSPy's predictions
    # order表示的是边的重要性，越小越重要、优先级越高
    # weight理解为边的权重，如果一对节点之间存在多条边或者多个来源支持这条边的存在，那么这条边的权重会增加
    order = min([dp.get("order", 1) for dp in edges_data] + already_order)
    weight = sum([dp["weight"] for dp in edges_data] + already_weights)
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in edges_data] + already_description))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in edges_data] + already_source_ids)
    )
    # 对于关系中的每个节点，如果不存在，则插入一个新节点
    for i in range(2):
        need_insert = src_name if i == 0 else tgt_name
        if already_edge_nodes[i] is None:
            node_data = dict(
                entity_type='"UNKNOWN"',
                description=description,
                source_id=source_id,
            )
            # upsert代表update or insert
            await knowledge_graph_inst.upsert_node(
                need_insert,
                node_data=node_data,
            )

    # 使用LLM重新获取关系描述
    description = await _handle_entity_relation_summary(
        (src_name, tgt_name), description, global_config
    )
    edge_data = dict(
        weight=weight, description=description, source_id=source_id, order=order
    )
    if already_edge is not None:
        # 如果图里面已经有这个边 那么直接插入更新
        # upsert代表update or insert
        await knowledge_graph_inst.upsert_edge(
            src_name,
            tgt_name,
            edge_data=edge_data,
        )
        return []
    else:
        # 如果图里面没有这个边 那么返回留待后面一起整体插入
        # 当前先不插入
        return [(src_name, tgt_name, edge_data)]


async def merge_nodes_and_edges_from_typed_chunks(results, chunk_type, knowledge_graph_inst, chunks, global_config):
    # TODO：这里针对表格类chunk的实体和关系合并逻辑需要进一步完善，比如in/相似度等
    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)
    for m_nodes, m_edges in results:
        for k, v in m_nodes.items():
            # 使用列表过滤符合条件的元素
            filtered_nodes = list(
                filter(lambda x: chunks[x['source_id']]['type'] == chunk_type, v)
            )
            # 仅当filtered_nodes非空时，才添加到maybe_nodes中
            if filtered_nodes:
                maybe_nodes[k].extend(filtered_nodes)
                
        for k, v in m_edges.items():
            # 使用列表过滤符合条件的元素
            filtered_edges = list(
                filter(lambda x: chunks[x['source_id']]['type'] == chunk_type, v)
            )
            # 仅当filtered_edges非空时，才添加到maybe_edges中
            if filtered_edges:
                maybe_edges[tuple(sorted(k))].extend(filtered_edges)
                
    logger.info(f'Merging nodes for {chunk_type} chunks')
    # 检查是否已经存在该节点
    already_nodes = await knowledge_graph_inst.get_nodes_batch(list(maybe_nodes.keys()))
    new_nodes_data = await safe_gather(
        [
            # 合并实体数据（类型、描述、来源等），插入到知识图谱中
            # 这里v是一个List[Dict]，是因为可能有同名实体
            _merge_nodes_then_upsert(k, v, a, knowledge_graph_inst, global_config)
            for i, (a, (k, v)) in enumerate(zip(already_nodes, maybe_nodes.items()))
        ]
    )
    # 批量插入原图中不存在的节点
    await knowledge_graph_inst.upsert_nodes_batch([i[0] for i in new_nodes_data if len(i)])
    logger.info(f'Merging edges for {chunk_type} chunks')
    
    # 检查是否已经存在该边
    already_edges = await knowledge_graph_inst.get_edges_batch(list(maybe_edges.keys()))
    # 批量检查新边的两侧节点是否存在
    edge_nodes = []
    for k in maybe_edges.keys():
        edge_nodes.append(k[0])
        edge_nodes.append(k[1])
    already_nodes_in_edge = await knowledge_graph_inst.get_nodes_batch(edge_nodes)
    new_edges_data = await safe_gather(
        [
            # 合并边（关系）数据，更新关系的权重、描述等，插入到知识图谱中
            _merge_edges_then_upsert(
                k[0], k[1], v, a, already_nodes_in_edge[2*i:2*i+2], knowledge_graph_inst, global_config
            )
            for i, (a, (k, v)) in enumerate(zip(already_edges, maybe_edges.items()))
        ]
    )
    # 批量插入原图中不存在的边
    await knowledge_graph_inst.upsert_edges_batch([i[0] for i in new_edges_data if len(i)])


async def extract_format_matcher(extract_results, chunk_key: str, context_base: dict):
    # 提取出来的文本经过split_string_by_multi_markers分割，分类为实体（maybe_nodes）或关系（maybe_edges）
    # 每个 record 都是模型返回的一条实体或关系信息
    records = split_string_by_multi_markers(
        extract_results,
        [context_base["record_delimiter"], context_base["completion_delimiter"]],
    )
    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)
    for record in records:
        # 提取括号内的内容
        record = record.replace('（','(').replace('）',')')     # 防止中文括号导致正则匹配失败
        record = re.search(r"\((.*)\)", record)
        if record is None:  # 没提取到指定格式的内容，跳过
            continue
        record = record.group(1)
        # 将提取出的 record 分割成多个属性，比如实体名称、实体类型、实体描述等
        record_attributes = split_string_by_multi_markers(
            record, [context_base["tuple_delimiter"]]
        )
        if_entities = await _handle_single_entity_extraction(
            record_attributes, chunk_key
        )
        if if_entities is not None:
            maybe_nodes[if_entities["entity_name"]].append(if_entities)
            continue

        if_relation = await _handle_single_relationship_extraction(
            record_attributes, chunk_key
        )
        if if_relation is not None:
            maybe_edges[(if_relation["src_name"], if_relation["tgt_name"])].append(
                if_relation
            )
    return (dict(maybe_nodes), dict(maybe_edges))


async def single_text_chunk_extracter(chunk_key_dp: tuple[str, TextChunkSchema], global_config: dict, mode: str = "seperate"):
    async def extracter(use_prompts: dict):
        continue_prompt = use_prompts["continue_prompt"]
        if_loop_prompt = use_prompts["if_loop_prompt"]
        hint_prompt = use_prompts["hint_prompt"]
        final_result = await use_llm_func(hint_prompt)

        history = pack_user_ass_to_openai_messages(hint_prompt, final_result)
        # 限制最大次数，继续提取
        for now_glean_index in range(entity_extract_max_gleaning):
            glean_result = await use_llm_func(continue_prompt, history_messages=history)

            history += pack_user_ass_to_openai_messages(continue_prompt, glean_result)
            final_result += glean_result
            if now_glean_index == entity_extract_max_gleaning - 1:
                break

            # 判断是否遗漏需要继续提取
            if_loop_result: str = await use_llm_func(
                if_loop_prompt, history_messages=history
            )
            if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
            if if_loop_result != "yes":
                break
        result = await extract_format_matcher(final_result, chunk_key, context_base)

        # existing_data = {}
        # os.makedirs('qa', exist_ok=True)
        # if os.path.exists(f'qa/{chunk_key}.pkl'):
        #     with open(f'qa/{chunk_key}.pkl', 'rb') as file:
        #         existing_data = pickle.load(file)
        #         assert existing_data['chunk_key'] == chunk_key, f"Chunk key mismatch: {existing_data['chunk_key']} != {chunk_key}"

        # existing_data['type'] = 'text'
        # existing_data['chunk_key'] = chunk_key
        # if "# 实体与关系提取任务" in hint_prompt:
        #     k = "e_r_extraction"
        # elif "# 实体提取任务" in hint_prompt:
        #     k = "e_extraction"
        # elif "# 关系提取任务" in hint_prompt:
        #     k = "r_extraction"
        # else:
        #     raise ValueError(f"Unknown prompt type")
        
        # existing_data[k] = {
        #     'prompt': hint_prompt,
        #     'qwen2.5-72b': {'response': final_result, 'result': result},
        # }
        # with open(f'qa/{chunk_key}.pkl', 'wb') as file:
        #     pickle.dump(existing_data, file)

        return result
    
    chunk_key = chunk_key_dp[0]
    chunk_dp = chunk_key_dp[1]    
    use_llm_func: callable = global_config["best_model_func"]
    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]
    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(PROMPTS["DEFAULT_ENTITY_TYPES"]),
    )
    if mode == "seperate":
        # 关系与实体分开提取
        extracted_entities, _ = await extracter(
            use_prompts = {
                "continue_prompt": PROMPTS["e_continue_extraction"],
                "if_loop_prompt": PROMPTS["e_if_loop_extraction"],
                "hint_prompt": PROMPTS["e_extraction"].format(**context_base, input_text=chunk_dp["content"]),
            }
        )

        extracted_entities_str = context_base["record_delimiter"].join(
            [
                f'("entity"{context_base["tuple_delimiter"]}{entity["entity_name"]}{context_base["tuple_delimiter"]}{entity["entity_type"]}{context_base["tuple_delimiter"]}{entity["description"]})' 
             for entity_list in extracted_entities.values() 
             for entity in entity_list
             ]
        )
        e_from_relationships, extracted_relationships = await extracter(
            use_prompts = {
                "continue_prompt": PROMPTS["r_continue_extraction"],
                "if_loop_prompt": PROMPTS["r_if_loop_extraction"],
                "hint_prompt": PROMPTS["r_extraction"].format(**context_base, input_text=chunk_dp["content"], entity_lists=extracted_entities_str),
            }
        )
        for e_name, e_data in e_from_relationships.items():
            # 补充关系提取中可能得到的新实体（若有）
            extracted_entities.get(e_name, []).extend(e_data)
        return (extracted_entities, extracted_relationships)
    elif mode == "one-off":
        # 关系与实体一起提取
        return await extracter(
            use_prompts = {
                "continue_prompt": PROMPTS["e_r_continue_extraction"],
                "if_loop_prompt": PROMPTS["e_r_if_loop_extraction"],
                "hint_prompt": PROMPTS["e_r_extraction"].format(**context_base, input_text=chunk_dp["content"]),
            }
        )
    else:
        raise ValueError(f"Unknown extraction mode: {mode}")


async def extract_entities(
    chunks: dict[str, TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
    table_handlers: dict[str, TableHandler],
    global_config: dict,
    websocket=None,
) -> Union[BaseGraphStorage, None]:
    
    # return await resolve_from_json(md5_dir='ec21afe0c1d471c8efd0faf59a4eb072', knowledge_graph_inst=knowledge_graph_inst)

    already_processed = 0
    already_entities = 0
    already_relations = 0

    async def _process_text_chunk(chunk_key_dp: tuple[str, TextChunkSchema]):
        nonlocal already_processed, already_entities, already_relations
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        result = await single_text_chunk_extracter(chunk_key_dp, global_config, mode=global_config["entity_extract_mode"])
        await save_result_to_file(chunk_key, result)
        already_processed += 1
        already_entities += len(result[0])
        already_relations += len(result[1])
        now_ticks = PROMPTS["process_tickers"][
            already_processed % len(PROMPTS["process_tickers"])
        ]
        print(
            f"{now_ticks} Processed {already_processed} chunks, {already_entities} entities(duplicated), {already_relations} relations(duplicated)\r",
            end="",
            flush=True,
        )
        logger.info(f"Processed {already_processed} chunks, {already_entities} entities(duplicated), {already_relations} relations(duplicated)")
        return result

    async def _process_table_chunk(
        chunk_key_dp: tuple[str, TextChunkSchema],
    ):
        nonlocal already_processed, already_entities, already_relations
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        assert chunk_key in table_handlers.keys(), f"Table handler for {chunk_key} not found!"
        maybe_nodes, maybe_edges = await table_handlers[chunk_key].parse_entities(step=5)  # type: ignore

        # 保存提取结果到文件并打印信息
        result = (dict(maybe_nodes), dict(maybe_edges))
        await save_result_to_file(chunk_key, result)
        already_processed += 1
        already_entities += len(result[0])
        already_relations += len(result[1])
        now_ticks = PROMPTS["process_tickers"][
            already_processed % len(PROMPTS["process_tickers"])
        ]
        print(
            f"{now_ticks} Processed {already_processed} chunks, {already_entities} entities(duplicated), {already_relations} relations(duplicated)\r",
            end="",
            flush=True,
        )
        logger.info(f"Processed {already_processed} chunks, {already_entities} entities(duplicated), {already_relations} relations(duplicated)")
        return result

    async def check_and_load_file(chunk_key):
        """异步检查文本块的实体文件是否存在，并读取文件内容"""
        file_path = os.path.join(chunks_dir, f"{chunk_key}.pkl")
        if os.path.exists(file_path):
            async with aiofiles.open(file_path, "rb") as f:
                return pickle.loads(await f.read())  # 读取并反序列化文件内容
        else:
            return None

    async def save_result_to_file(chunk_key, result):
        """每个chunk的提取结果出来后，立刻异步保存结果到文件"""
        file_path = os.path.join(chunks_dir, f"{chunk_key}.pkl")
        async with aiofiles.open(file_path, "wb") as f:
            await f.write(pickle.dumps(result))  # 将结果异步保存为.pkl文件

    async def process_chunks(chunks_with_index):
        """
        异步处理每一个 chunk
        chunks_with_index 是一个列表，每个元素是 (index, chunk) 的元组。
        结果保存到 results 中。
        """
        results = [None] * len(ordered_chunks)  # 初始化 results 列表
	    # 异步检查每个chunk是否有对应的文件存在
        file_results = await asyncio.gather(*[check_and_load_file(chunk[0]) for idx, chunk in chunks_with_index])

        failed_chunks = []  # 存储未成功处理的 chunks
        for idx, (file_result, (index, chunk)) in enumerate(
            zip(file_results, chunks_with_index)
        ):
            if file_result is not None:
                results[index] = file_result  # 文件存在，直接读取并存储结果
            else:
                failed_chunks.append((index, chunk))  # 文件不存在，加入待处理列表

        if len(failed_chunks):
            text_extract_results = await safe_gather(
                [_process_text_chunk(c[1]) for c in failed_chunks if c[1][1]['type'] == 'text']
            )

            table_extract_results = await safe_gather(
                [_process_table_chunk(c[1]) for c in failed_chunks if c[1][1]['type'] == 'table']
            )

            extract_results = text_extract_results + table_extract_results
            logger.info(f"\nTotal chunks processed: {len(extract_results)}; TEXT chunks: {len(text_extract_results)}; TABLE chunks: {len(table_extract_results)}")
            # 异步保存结果
            for result, (idx, chunk) in zip(extract_results, failed_chunks):
                results[idx] = result
        return results

    chunks_dir = os.path.join(global_config["working_dir"], "chunks")
    if not os.path.exists(chunks_dir):
        os.makedirs(chunks_dir)

    ordered_chunks = list(chunks.items())  # 包含字典键值对的元组列表


    # results是列表，长度是ordered_chunks，每个元素是元组 (maybe_nodes字典，maybe_edges字典)
    # 异步进行多个chunk的实体提取
    results = await process_chunks(
        [(idx, chunk) for idx, chunk in enumerate(ordered_chunks)]
    )
    print()  # clear the progress bar

    # 合并所有提取到的实体和关系数据
    # 对于每个chunk，表格实体提取可以和文本实体提取同时进行
    # 但合并多个chunk的实体的时候，就需要先合并所有文本实体，再合并所有表格实体；关系也是这样。
    # 不用担心knowledge_graph_inst，这东西是被引用传递的，所以不用另外加传参和返回
    # maybe_nodes和maybe_edges都是Dict[List[Dict]]
    # 先合并text chunk中提取到的实体和关系数据
    await merge_nodes_and_edges_from_typed_chunks(results, "text", knowledge_graph_inst, chunks, global_config)
    # 再合并table chunk中提取到的实体和关系数据
    await merge_nodes_and_edges_from_typed_chunks(results, "table", knowledge_graph_inst, chunks, global_config)

    return knowledge_graph_inst



async def merge_extractions(
    chunks: dict[str, TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
    knowledge_graph_merge: BaseGraphStorage,
    global_config: dict,
    websocket=None,
) -> Union[BaseGraphStorage, None]:
    """对两份建好的图合并实体和关系数据，把knowledge_graph_merge合并到knowledge_graph_inst中"""
    raise NotImplementedError("This function has not been correctly edited for use.")
    # 两张图需要是同一个类别
    assert knowledge_graph_inst.__class__ == knowledge_graph_merge.__class__, \
    f"The two graphs for merging should be of the same class. However, got {knowledge_graph_inst.__class__} and {knowledge_graph_merge.__class__}."

    chunks_dir = os.path.join(global_config["working_dir"], "chunks")
    if not os.path.exists(chunks_dir):
        os.makedirs(chunks_dir)

    # 从想要合并的图中提取实体和关系数据
    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)
    if isinstance(knowledge_graph_inst, Neo4jStorage):
        ns = await knowledge_graph_inst.get_all_nodes()
        es = await knowledge_graph_inst.get_all_edges()
        for entity_name in ns:
            # 修改取出的内容必须使用copy.deepcopy，否则会修改原图
            if_entities = copy.deepcopy(entity_name)
            if_entities["entity_name"] = entity_name["id"]
            if "clusters" in if_entities.keys():
                del if_entities["clusters"]
            maybe_nodes[if_entities["entity_name"]].append(if_entities)
        for e in es:
            if_relation = copy.deepcopy(e)
            if "order" in if_relation.keys():
                del if_relation["order"]
            maybe_edges[(src_name, tgt_name)].append(if_relation)
    else:
        ns = knowledge_graph_merge._graph.nodes
        es = knowledge_graph_merge._graph.edges
        for entity_name in ns:
            # 修改取出的内容必须使用copy.deepcopy，否则会修改原图
            if_entities = copy.deepcopy(ns[entity_name])
            if_entities["entity_name"] = entity_name
            if "clusters" in if_entities.keys():
                del if_entities["clusters"]
            maybe_nodes[if_entities["entity_name"]].append(if_entities)
        for src_name, tgt_name in es:
            if_relation = copy.deepcopy(es[src_name, tgt_name])
            if_relation["src_name"] = src_name
            if_relation["tgt_name"] = tgt_name
            if "order" in if_relation.keys():
                del if_relation["order"]
            maybe_edges[(src_name, tgt_name)].append(if_relation)

    # 合并实体和关系数据，这里不需要分开处理文本和表格源，因为建图时upsert会自动处理
    logger.info(f"Merging {len(ns)} entities and {len(es)} relationships to original graph...")

    await safe_gather(
        [
            # 合并实体数据（类型、描述、来源等），插入到知识图谱中
            # 这里v是一个List[Dict]，是因为可能有同名实体
            _merge_nodes_then_upsert(k, v, knowledge_graph_inst, chunks_dir, chunks, global_config)
            for k, v in maybe_nodes.items()
        ]
    )
    await safe_gather(
        [
            # 合并边（关系）数据，更新关系的权重、描述等，插入到知识图谱中
            _merge_edges_then_upsert(
                k[0], k[1], v, knowledge_graph_inst, chunks_dir, chunks, global_config
            )
            for k, v in maybe_edges.items()
        ]
    )
    logger.info(f"Merge finished with {len(knowledge_graph_inst._graph.nodes)} entities and {len(knowledge_graph_inst._graph.edges)} relationships in graph.")
    return knowledge_graph_inst