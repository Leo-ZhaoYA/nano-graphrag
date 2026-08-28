import json
import asyncio
import itertools
import copy
import re
from collections import Counter, defaultdict
from ._utils import (
    logger,
    compute_mdhash_id,
    list_of_list_to_csv,
    split_string_by_multi_markers,
    truncate_list_by_token_size,
    fix_json_format,
    fix_citation_in_response,
    safe_gather,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    CommunitySchema,
    TextChunkSchema,
    QueryParam,
)
from .prompt import GRAPH_FIELD_SEP, PROMPTS



async def _find_table_chunks_from_entities_and_relations(
    node_datas: list[dict],
    edge_datas: list[dict],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
):
    '''用于查找使用到的实体和关系相关的表格类chunk'''
    
    async def _get_chunk_type_by_ids(chunk_ids):
        chunks = await text_chunks_db.get_by_ids(chunk_ids)
        return [chunk["type"] for chunk in chunks]

    # 将 node_datas 和 edge_datas 中的 source_id 合并、平展并去重
    node_chunks = [
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
        for dp in node_datas
    ]
    edge_chunks = [
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
        for dp in edge_datas
    ]
    unique_chunks = sorted(set(itertools.chain(*node_chunks, *edge_chunks)))
    # 异步获取每个 chunk 的类型
    chunk_types = await _get_chunk_type_by_ids(unique_chunks)

    # 筛选出 chunk_type 为 'table' 的 chunk_id
    table_chunks = [
        chunk_id
        for chunk_id, chunk_type in zip(unique_chunks, chunk_types)
        if chunk_type == "table"
    ]

    return sorted(table_chunks)  # 返回表格类 chunk 的 ID 列表


async def _find_most_related_community_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    community_reports: BaseKVStorage[CommunitySchema],
):
    '''用于查找与给定实体列表 node_datas 最相关的社区，并返回其社区报告'''
    if not len(community_reports._data):
        return []
    related_communities = []
    # 从每个实体（node_d）的数据中提取它所属的社区（clusters）。如果实体没有社区信息，跳过该实体。
    for node_d in node_datas:
        if "clusters" not in node_d:
            continue
        related_communities.extend(json.loads(node_d["clusters"]))
    # 筛选出符合层级要求的社区
    related_community_dup_keys = [
        str(dp["cluster"])
        for dp in related_communities
        if dp["level"] <= query_param.level
    ]
    # 统计每个社区的出现次数
    related_community_keys_counts = dict(Counter(related_community_dup_keys))
    _related_community_datas = await community_reports.get_by_ids(list(related_community_keys_counts.keys()))

    related_community_datas = {
        k: v
        for k, v in zip(related_community_keys_counts.keys(), _related_community_datas)
        if v is not None
    }
    # 根据社区的出现次数和社区报告中的评分（rating）对社区进行排序
    related_community_keys = sorted(
        related_community_keys_counts.keys(),
        key=lambda k: (
            related_community_keys_counts[k],
            related_community_datas[k]["report_json"].get("rating", -1),
        ),
        reverse=True,
    )
    # 获取每个社区的详细信息，根据每个社区的频率和社区报告中的评分（rating），对社区进行排序。
    sorted_community_datas = [
        related_community_datas[k] for k in related_community_keys
    ]
    # 根据 token 限制截断社区报告列表
    use_community_reports = truncate_list_by_token_size(
        sorted_community_datas,
        key=lambda x: x["report_string"],
        max_token_size=query_param.local_max_token_for_community_report,
    )
    # 如果设置了 local_community_single_one 参数，则只返回最相关的一个社区，否则返回所有相关的社区。
    if query_param.local_community_single_one:
        use_community_reports = use_community_reports[:1]
    return use_community_reports


async def _find_most_related_text_unit_from_entities(
    node_datas: list[dict],
    use_relations: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
):
    # 取每个实体的source id
    # text_units 是输入实体的相关文本ID列表，列表每个元素是一个列表，表示一个输入实体的相关文本。
    text_units = [
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
        for dp in node_datas
    ]
    # 通过知识图谱获取每个实体相邻的节点存到all_one_hop_nodes_data
    edges = await knowledge_graph_inst.get_nodes_edges_batch([dp["entity_name"] for dp in node_datas])

    all_one_hop_nodes = set()
    for this_edges in edges:
        if not this_edges:
            continue
        all_one_hop_nodes.update([e[1] for e in this_edges])
    all_one_hop_nodes = sorted(all_one_hop_nodes)   # 对set调用sorted直接得到排序后的list
    all_one_hop_nodes_data = await knowledge_graph_inst.get_nodes_batch(all_one_hop_nodes)

    # 将每个相邻节点的 source_id 分割成文本单元 ID 列表，存储在 all_one_hop_text_units_lookup 字典中，字典的键是节点名称，值是该节点对应的文本单元 ID 集合。
    all_one_hop_text_units_lookup = {
        k: sorted(set(split_string_by_multi_markers(v["source_id"], [GRAPH_FIELD_SEP])))
        for k, v in zip(all_one_hop_nodes, all_one_hop_nodes_data)
        if v is not None
    }
    # 下面这个for循环就是检查text_units中每个chunk跟相邻节点的这些source chunk是否有关联，有则relation_counts+1
    all_text_units_lookup = {}
    for index, (this_text_units, this_edges) in enumerate(zip(text_units, edges)):
        # this_text_units表示一个输入实体的相关文本ID列表，this_edges表示一个输入实体的邻居节点
        for c_id in this_text_units:
            # 对于每个文本单元 c_id，计算它与一跳节点之间的关系计数 relation_counts。
            # 即检查该实体的该文本单元是否与该实体一跳节点中的文本单元存在关联（通过边 e[1] 查找）。
            if c_id in all_text_units_lookup:
                continue
            relation_counts = 0
            for e in this_edges:
                if (
                    e[1] in all_one_hop_text_units_lookup
                    and c_id in all_one_hop_text_units_lookup[e[1]]
                ):
                    relation_counts += 1
            # 获取每个文本单元的内容，并记录它的顺序和关系计数
            all_text_units_lookup[c_id] = {
                "data": await text_chunks_db.get_by_id(c_id),
                "order": index,
                "relation_counts": relation_counts,
            }
    if any([v is None for v in all_text_units_lookup.values()]):
        logger.warning("Text chunks are missing, maybe the storage is damaged")
    all_text_units = [
        {"id": k, **v} for k, v in all_text_units_lookup.items() if v is not None
    ]
    # 优先返回order较小且与其他节点关系计数较多的文本单元，并截断
    all_text_units = sorted(
        all_text_units, key=lambda x: (x["order"], -x["relation_counts"])
    )
    all_text_units = truncate_list_by_token_size(
        all_text_units,
        key=lambda x: x["data"]["content"],
        max_token_size=query_param.local_max_token_for_text_unit,
    )
    all_text_units: dict[TextChunkSchema] = {t["id"]: t["data"]["content"] for t in all_text_units}
    already_used_chunks = sorted(list(all_text_units.keys()))
    # 从用到的实体和关系中提取表格类source chunk
    use_table_chunks = await _find_table_chunks_from_entities_and_relations(
        node_datas, use_relations, text_chunks_db
    )
    # 这里把实体和关系挂载到的列表内容加入，暂时放在最后、不考虑超过token上限的问题
    # 在all_text_units后面追加TextChunkSchema类型的表格chunk数据，以markdown格式表格加入
    for chunk_id in use_table_chunks:
        if chunk_id not in already_used_chunks:
            # 添加还未出现过的table chunk
            all_text_units[chunk_id] = (await text_chunks_db.get_by_id(chunk_id))["content"]     # content 已经是markdown格式的表格
    return all_text_units


async def _find_most_related_edges_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
):
    '''
    给定一组实体，找到它们在知识图谱中的最相关的边（即关系），然后对这些边进行排序，基于它们的度（rank）和权重（weight）衡量相关性。
    函数会将边进行截断，以确保不会超过查询参数规定的最大 token 限制，并最终返回这些最相关的边数据。
    '''
    all_related_edges = await knowledge_graph_inst.get_nodes_edges_batch([dp["entity_name"] for dp in node_datas])
    # 收集并去重边数据，通过排序去重
    all_edges = set()
    for this_edges in all_related_edges:
        all_edges.update([tuple(sorted(e)) for e in this_edges])
    # 排序保证顺序稳定
    all_edges = sorted(all_edges)
    # 获取每条边的详细信息，包括边的权重和度数，然后根据度数和权重对边进行排序。
    all_edges_pack = await knowledge_graph_inst.get_edges_batch([(e[0], e[1]) for e in all_edges])

    all_edges_degree = await knowledge_graph_inst.edge_degrees_batch([(e[0], e[1]) for e in all_edges])
    all_edges_data = [
        {"src_tgt": k, "rank": d, **v}
        for k, v, d in zip(all_edges, all_edges_pack, all_edges_degree)
        if v is not None
    ]
    all_edges_data = sorted(
        all_edges_data, key=lambda x: (x["rank"], x["weight"]), reverse=True
    )
    all_edges_data = truncate_list_by_token_size(
        all_edges_data,
        key=lambda x: x["description"],
        max_token_size=query_param.local_max_token_for_local_context,
    )
    return all_edges_data


async def _build_local_query_context(
    query: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    community_reports: BaseKVStorage[CommunitySchema],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    websocket=None,
):
    # 通过向量比对找到与查询相关的前 top_k 个实体
    # 获取这些实体的节点数据（包括描述、类型等）和节点度信息（节点的连接数）
    text_results = []
    table_results = []
    tmp_results = await entities_vdb.query(query, top_k=2*query_param.top_k)
    if query_param.table_enhance_factor in [0, 1, 2, 3]:
        # 数字越大，表示越倾向于使用表格类chunk中提取到的实体
        # 每个results的内容内部都是有序的，从最相关到最不相关
        text_results = [r for r in tmp_results if "text" in r["source_type"]][:int((4-query_param.table_enhance_factor)*query_param.top_k/4)+1]
        table_results = [r for r in tmp_results if "table" in r["source_type"] if r not in text_results][:int(query_param.table_enhance_factor*query_param.top_k/4)+1]
        # 取向量相似度最高的前top_k个实体
        results = sorted(table_results + text_results, key=lambda x: x["__metrics__"], reverse=True)[:query_param.top_k]
    elif query_param.table_enhance_factor == -1:
        # 直接取前top_k个实体
        results = tmp_results[:query_param.top_k]

    if not len(results):
        # 没有查询到结果，直接返回
        return {}, {}, None
    node_datas = await knowledge_graph_inst.get_nodes_batch([r["entity_name"] for r in results])
    if not all([n is not None for n in node_datas]):
        logger.warning("Some nodes are missing, maybe the storage is damaged")
    node_degrees = await knowledge_graph_inst.node_degrees_batch([r["entity_name"] for r in results])

    node_datas = [
        {**n, "entity_name": k["entity_name"], "rank": d}
        for k, n, d in zip(results, node_datas, node_degrees)
        if n is not None
    ]
    # node_datas 是一个列表，包含了所有与查询相关的实体的节点数据，每个节点数据是一个字典。
    # 元素举例：{'entity_type': '"设备"', 'description': '"用于提升数据采集效率的辅助工具"', 'source_id': 'chunk-cbb5f457afdd748f40e4aedd9f092590', 'clusters': '[{"level": 0, "cluster": 1}]', 'entity_name': '"挡板"', 'rank': 1}
    # 从实体节点数据中提取最相关的社区、关系和文本内容
    use_communities = await _find_most_related_community_from_entities(
        node_datas, query_param, community_reports
    )
    use_relations = await _find_most_related_edges_from_entities(
        node_datas, query_param, knowledge_graph_inst
    )
    # 从实体节点数据中提取最相关的文本内容，同时将表格类chunk加入
    use_text_units = await _find_most_related_text_unit_from_entities(
        node_datas, use_relations, query_param, text_chunks_db, knowledge_graph_inst
    )

    logger.info(
        f"根据问题找到了 {len(node_datas)} 个实体，{len(use_communities)} 个社区，{len(use_relations)} 条关系和 {len(use_text_units)} 个文本内容..."
    )
    logger.info(f"相关实体: {[d.get('entity_name' , None) for d in node_datas]}")
    if websocket and query_param.show_query_process:
        await websocket.send(
            f"根据问题找到了 {len(node_datas)} 个实体，{len(use_communities)} 个社区，{len(use_relations)} 条关系和 {len(use_text_units)} 个文本内容..."
        )
        entities_str = '、'.join(
            sorted(
                set(
                    [
                        d.get('entity_name', '').strip("'").strip('"') for d in node_datas
                        ]
                    )
                )
            )
        await websocket.send(
            f"相关实体: {entities_str}"
        )

    # 以 CSV 格式列出所有相关实体及其描述、类型和排名信息。
    hash_dict = defaultdict(dict)      # 用于记录hash值和index的对应关系
    entites_section_list = [["id", "entity", "type", "description", "rank"]]
    for i, n in enumerate(node_datas):
        entites_section_list.append(
            [
                i,  # 这个i以及下面代码里面的i就是局部搜索时候的数据id
                n["entity_name"],
                n.get("entity_type", "UNKNOWN"),
                n.get("description", "UNKNOWN"),
                f'{n["rank"]:.2f}',
            ]
        )
        hash_dict['实体'][i] = f"""ent-{n['entity_name'].strip('"'.strip("'").strip())}"""
    entities_context = list_of_list_to_csv(entites_section_list)

    # 以 CSV 格式列出相关关系的描述、权重和排名信息。
    relations_section_list = [
        ["id", "source", "target", "description", "weight", "rank"]
    ]
    for i, e in enumerate(use_relations):
        relations_section_list.append(
            [
                i,
                e["src_tgt"][0],
                e["src_tgt"][1],
                e["description"],
                f'{e["weight"]:.2f}',
                f'{e["rank"]:.2f}',
            ]
        )
        hash_dict['关系'][i] = f"""rel-ent-{e["src_tgt"][0].strip('"').strip("'").strip()}_ent-{e["src_tgt"][1].strip('"').strip("'").strip()}"""
    relations_context = list_of_list_to_csv(relations_section_list)

    # 以 CSV 格式列出相关社区的报告内容。
    communities_section_list = [["id", "content"]]
    for i, c in enumerate(use_communities):
        communities_section_list.append([i, c["report_string"]])
        hash_dict['社区报告'][i] = c["title"].replace(" ", "-")
    communities_context = list_of_list_to_csv(communities_section_list)

    # 以 CSV 格式列出相关文本单元的内容。
    text_units_section_list = [["id", "content"]]
    for i, (hash_id, content) in enumerate(use_text_units.items()):
        text_units_section_list.append(
            [i, content]
        )
        hash_dict['文本块'][i] = hash_id        # 对于chunk这里直接使用已有的chunk id，避免用整个chunk重新计算hash出错
    text_units_context = list_of_list_to_csv(text_units_section_list)

    appendix = {
        "实体": entites_section_list,
        "关系": relations_section_list,
        "社区报告": [
            [i[0], "<br>".join(i[1].split("\n\n", 2)[:2]).replace('\n', '<br>')]
            for i in communities_section_list
        ],
        "文本块": [
            [
                i[0],
                (
                    f"{i[1][:50]}...".replace('\n', '<br>')
                    if len(i[1]) > 50
                    else i[1].replace('\n', '<br>')
                ),
            ]
            for i in text_units_section_list
        ],
    }

    # 最终将所有部分（实体、关系、社区、文本单元）组合成一个完整的上下文字符串，用于之后的查询响应生成。
    return (
        hash_dict,
        appendix,
        f"""
-----报告-----
```csv
{communities_context}
```
-----实体-----
```csv
{entities_context}
```
-----关系-----
```csv
{relations_context}
```
-----来源-----
```csv
{text_units_context}
```
""",
    )


async def local_query(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    community_reports: BaseKVStorage[CommunitySchema],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
    websocket=None,  # 添加 WebSocket 参数
) -> str:
    use_model_func = global_config["best_model_func"]
    # 构建上下文，基于嵌入向量比对找到最相关的实体、社区、关系和文本内容
    if websocket and query_param.show_query_process:
        await websocket.send("正在构建查询上下文...")
    hash_dict, appendix, context = await _build_local_query_context(
        query,
        knowledge_graph_inst,
        entities_vdb,
        community_reports,
        text_chunks_db,
        query_param,
        websocket=websocket,
    )
    if query_param.only_need_context:
        return context
    if context is None:
        logger.info(PROMPTS["fail_response"])
        if websocket:
            await websocket.send(PROMPTS["fail_response"])
        return PROMPTS["fail_response"]
    sys_prompt = PROMPTS["local_rag_response"].format(
        context_data=context, response_type=query_param.response_type
    )
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
    )
    response = fix_citation_in_response(response, hash_dict, websocket=websocket)
    appendix_str = "\n\n".join(
        [
            f"### 附录 {i+1}: {key}\n\n{list_of_list_to_csv(value, markdown=True)}"
            for i, (key, value) in enumerate(appendix.items())
        ]
    )
    msg = f"\n## 局部搜索结果: \n\n{response}\n\n"
    logger.info(re.sub(r'<style>.*?</style>', '', msg, flags=re.DOTALL))
    if websocket:
        await websocket.send(msg)
    return response


async def _map_global_communities(
    query: str,
    communities_data: list[CommunitySchema],
    query_param: QueryParam,
    global_config: dict,
    websocket=None,
):
    '''
    _map_global_communities 是 global_query 函数的一个辅助函数，
    用于从多个社区数据中提取与查询相关的“支持点”（support points）。
    这些支持点是最终形成回答的基础。
    '''

    use_model_func = global_config["best_model_func"]
    use_string_json_convert_func = global_config["convert_response_to_json_func"]
    community_groups = []
    hash_dict = defaultdict(dict)
    # 按 token 大小限制将社区数据分成若干组
    while len(communities_data):
        this_group = truncate_list_by_token_size(
            communities_data,
            key=lambda x: x["report_string"],
            max_token_size=query_param.global_max_token_for_community_report,
        )
        community_groups.append(this_group)
        communities_data = communities_data[len(this_group) :]

    async def _process(
        start_id: int, community_truncated_datas: list[CommunitySchema]
    ) -> dict:
        # 定义处理每个社区组的协程
        communities_section_list = [["id", "content", "rating", "importance"]]
        for i, c in enumerate(community_truncated_datas):
            communities_section_list.append(
                [
                    start_id + i,  # 这个就是全局搜索报告里面的数据来源id
                    c["report_string"],
                    f'{c["report_json"].get("rating", 0):.2f}',
                    f'{c["occurrence"]:.2f}',
                ]
            )
            hash_dict['社区报告'][start_id + i] = c["title"].replace(" ", "-")
        community_context = list_of_list_to_csv(communities_section_list)
        sys_prompt_temp = PROMPTS["global_map_rag_points"]
        sys_prompt = sys_prompt_temp.format(context_data=community_context)
        # 使用 LLM 处理这个上下文，生成初步的答案或支持点。返回的结果是从响应中提取的“points”字段
        data = None
        failed_json_count = 0
        while data is None and failed_json_count < 3:
            try:
                response = await use_model_func(
                    query,
                    system_prompt=sys_prompt,
                    **query_param.global_special_community_map_llm_kwargs,
                )
                data = use_string_json_convert_func(response)
            except (AssertionError, json.JSONDecodeError) as e:
                failed_json_count += 1
                data = None
        if data is None:
            # 同一个社区多次json格式都失败，那么尝试用最后一次response修复
            data = fix_json_format(response)
            data = {} if data is None else data
        return communities_section_list, data.get("points", [])

    logger.info(f"共有 {len(community_groups)} 个社区组需要处理...")
    if websocket and query_param.show_query_process:
        await websocket.send(f"共有 {len(community_groups)} 个社区组需要处理...")

    # 每个社区组都被异步处理，最终结果将并行生成
    tasks = []
    start_id = 0
    for community_group in community_groups:
        tasks.append(_process(start_id, community_group))
        start_id += len(community_group)

    results = await safe_gather(tasks)
    # 解包 results 来单独获取两个列表
    responses = [result[1] for result in results]
    # communities_section_list 是一个三维列表，要合并这个三维列表成为一个二维列表，并且去掉多余的表头
    c = [result[0] for result in results]
    communities_section_list = (
        [c[0][0]] + [item for sublist in c for item in sublist if item[0] != "id"]
        if len(c) > 0
        else []
    )

    appendix = {
        "社区报告": [
            [
                i[0],
                "<br>".join(i[1].split("\n\n", 2)[:2]).replace('\n', '<br>'),
                i[2],
                i[3],
            ]
            for i in communities_section_list
        ]
    }
    return hash_dict, appendix, responses


async def global_query(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    community_reports: BaseKVStorage[CommunitySchema],
    query_param: QueryParam,
    global_config: dict,
    websocket=None,  # 添加 WebSocket 参数
) -> str:
    # 进行社区筛选
    # leiden的level 大：社区内容更细化、划分更深入，意味着社区被进一步细分成了更多小的子社区。
    # leiden的level 小：社区内容更粗略，可能涵盖了更多的节点或子社区，社区更大。
    community_schema = await knowledge_graph_inst.community_schema()
    # 选取层级小于等于查询参数中的层级的社区，即包含内容更粗略的社区
    community_schema = {
        k: v for k, v in community_schema.items() if v["level"] <= query_param.level
    }
    if not len(community_schema) or not len(community_reports._data):
        logger.info(PROMPTS["fail_response"])
        if websocket:
            await websocket.send(PROMPTS["fail_response"])
        return PROMPTS["fail_response"]
    use_model_func = global_config["best_model_func"]

    sorted_community_schemas = sorted(
        community_schema.items(),
        key=lambda x: x[1]["occurrence"],
        reverse=True,
    )
    sorted_community_schemas = sorted_community_schemas[
        : query_param.global_max_consider_community
    ]
    community_datas = await community_reports.get_by_ids(
        [k[0] for k in sorted_community_schemas]
    )
    community_datas = [c for c in community_datas if c is not None]
    community_datas = [
        c
        for c in community_datas
        if c["report_json"].get("rating", 0) >= query_param.global_min_community_rating
    ]
    community_datas = sorted(
        community_datas,
        key=lambda x: (x["occurrence"], x["report_json"].get("rating", 0)),
        reverse=True,
    )
    logger.info(
        f"参考社区: {'、'.join([c.get('report_json',{}).get('title',None) for c in community_datas])}"
    )
    if websocket and query_param.show_query_process:
        await websocket.send(f"打分后剩余 {len(community_datas)} 个参考社区...")
        # await websocket.send(
        #     f"参考社区:{'、'.join([c.get('report_json',{}).get('title',None) for c in community_datas])}"
        # ) # 太多了

    # 并行处理这些社区数据，提取出所有相关的支持点
    hash_dict, appendix, map_communities_points = await _map_global_communities(
        query, community_datas, query_param, global_config, websocket=websocket
    )
    final_support_points = []
    for i, mc in enumerate(map_communities_points):
        for point in mc:
            if "description" not in point:
                continue
            final_support_points.append(
                {
                    "analyst": i,
                    "answer": point["description"],
                    "score": point.get("score", 1),
                }
            )

    final_support_points = [p for p in final_support_points if p["score"] > 0]
    if not len(final_support_points):
        logger.info(PROMPTS["fail_response"])
        if websocket:
            await websocket.send(PROMPTS["fail_response"])
        return PROMPTS["fail_response"]

    final_support_points = sorted(
        final_support_points, key=lambda x: x["score"], reverse=True
    )
    final_support_points = truncate_list_by_token_size(
        final_support_points,
        key=lambda x: x["answer"],
        max_token_size=query_param.global_max_token_for_community_report,
    )
    if websocket and query_param.show_query_process:
        await websocket.send(
            f"这些社区对本问题共提供了 {len(final_support_points)} 个支持点..."
        )
        # await websocket.send(
        #     f"最终支持点: {[p['answer'] for p in final_support_points]}\n"
        # )

    points_context = []
    for dp in final_support_points:
        points_context.append(
            f"""----分析师 {dp['analyst']}----
重要性打分: {dp['score']}
{dp['answer']}
"""
        )
    points_context = "\n".join(points_context)
    if query_param.only_need_context:
        return points_context
    sys_prompt_temp = PROMPTS["global_reduce_rag_response"]
    response = await use_model_func(
        query,
        sys_prompt_temp.format(
            report_data=points_context, response_type=query_param.response_type
        ),
    )
    response = fix_citation_in_response(response, hash_dict, websocket=websocket)
    appendix_str = "\n\n".join(
        [
            f"### 附录 {i+1}: {key}\n\n{list_of_list_to_csv(value, markdown=True)}"
            for i, (key, value) in enumerate(appendix.items())
        ]
    )
    msg = f"\n## 全局搜索结果: \n\n{response}\n\n"
    logger.info(re.sub(r'<style>.*?</style>', '', msg, flags=re.DOTALL))
    if websocket:
        await websocket.send(msg)
    return response


async def naive_query(
    query,
    chunks_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
    websocket=None,  # 添加 WebSocket 参数
):
    '''naive_query 函数用于执行简单的查询，通过向量化直接从文本数据库中提取相关chunk内容，然后使用 LLM 生成回答。'''
    if websocket:
        pass
        # await websocket.send(
        #     f"\n\n========================================\n\n\n用户执行NaiveRAG搜索: {query}"
        # )
    use_model_func = global_config["best_model_func"]
    # 返回与查询最相关的 top_k 个文本chunk
    results = await chunks_vdb.query(query, top_k=query_param.top_k)
    if not len(results):
        logger.info(PROMPTS["fail_response"])
        if websocket:
            await websocket.send(PROMPTS["fail_response"])
        return PROMPTS["fail_response"]
    chunks_ids = [r["id"] for r in results]
    chunks = await text_chunks_db.get_by_ids(chunks_ids)
    # 截断文本chunk，以确保不会超过查询参数规定的最大 token 限制
    maybe_trun_chunks = truncate_list_by_token_size(
        chunks,
        key=lambda x: x["content"],
        max_token_size=query_param.naive_max_token_for_text_unit,
    )
    logger.info(f"Truncate {len(chunks)} to {len(maybe_trun_chunks)} chunks")
    if websocket:
        await websocket.send(
            f"Truncate {len(chunks)} to {len(maybe_trun_chunks)} chunks"
        )
    # 将这些文本chunk的内容组合成一个完整的文本段落，之后用于LLM生成
    section = "--New Chunk--\n".join([c["content"] for c in maybe_trun_chunks])
    if query_param.only_need_context:
        return section
    sys_prompt_temp = PROMPTS["naive_rag_response"]
    sys_prompt = sys_prompt_temp.format(
        content_data=section, response_type=query_param.response_type
    )
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
    )
    appendix_str = "暂无。"
    msg = f"\n# NaiveRAG搜索结果: \n\n{response}\n\n"
    logger.info(msg)
    if websocket:
        await websocket.send(msg)
    return response
