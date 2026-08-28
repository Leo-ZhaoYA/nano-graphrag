
import json
import asyncio
from ._utils import (
    logger,
    encode_string_by_tiktoken,
    list_of_list_to_csv,
    truncate_list_by_token_size,
    fix_json_format,
    safe_gather,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    SingleCommunitySchema,
    CommunitySchema,
)
from .prompt import GRAPH_FIELD_SEP, PROMPTS


def _pack_single_community_by_sub_communities(
    community: SingleCommunitySchema,
    max_token_size: int,
    already_reports: dict[str, CommunitySchema],
) -> tuple[str, int]:
    # community: 表示当前的社区，它是一个包含多个子社区的对象，数据类型是 SingleCommunitySchema。
    # max_token_size: 最大的 token 数量限制，确保生成的报告不会超出这个限制。
    # already_reports: 一个字典，包含已经生成报告的社区。键是社区 ID，值是 CommunitySchema 类型的对象。
    # 获取当前社区中的所有子社区 ID，并在 already_reports 中查找这些子社区，筛选出已经生成过报告的子社区。
    all_sub_communities = [
        already_reports[k] for k in community["sub_communities"] if k in already_reports
    ]
    # 将最重要（出现次数最多）的子社区放在最前面，方便后续优先保留这些子社区。
    all_sub_communities = sorted(
        all_sub_communities, key=lambda x: x["occurrence"], reverse=True
    )
    may_trun_all_sub_communities = truncate_list_by_token_size(
        all_sub_communities,
        key=lambda x: x["report_string"],
        max_token_size=max_token_size,
    )
    sub_fields = ["id", "report", "rating", "importance"]
    sub_communities_describe = list_of_list_to_csv(
        [sub_fields]
        + [
            [
                i,
                c["report_string"],
                c["report_json"].get("rating", -1),
                c["occurrence"],
            ]
            for i, c in enumerate(may_trun_all_sub_communities)
        ]
    )
    # 对每个截断后的子社区 may_trun_all_sub_communities，提取返回的describe的社区的节点和边，说明这些节点和边是对当前社区较为重要的
    already_nodes = []
    already_edges = []
    for c in may_trun_all_sub_communities:
        already_nodes.extend(c["nodes"])
        already_edges.extend([tuple(e) for e in c["edges"]])

    # sub_communities_describe: 一个 CSV 格式的文本，描述了当前社区所有重要子社区的信息。
    # len(encode_string_by_tiktoken(sub_communities_describe)): 重要子社区描述的 token 数量。
    # set(already_nodes): 重要子社区中包含的所有节点。
    # set(already_edges): 重要子社区中包含的所有边。
    return (
        sub_communities_describe,
        len(encode_string_by_tiktoken(sub_communities_describe)),
        set(already_nodes),
        set(already_edges),
    )


async def _pack_single_community_describe(
    knowledge_graph_inst: BaseGraphStorage,
    community: SingleCommunitySchema,
    max_token_size: int = 12000,
    already_reports: dict[str, CommunitySchema] = {},
    global_config: dict = {},
) -> str:
    '''这个函数负责从知识图谱中提取单个社区的节点和边，并对其进行描述，生成适合作为输入的文本。'''
    # 将社区中的节点按字典顺序排序
    nodes_in_order = sorted(community["nodes"])
    # 社区中的边（edges）是按照源节点和目标节点的组合进行排序
    edges_in_order = sorted(community["edges"], key=lambda x: x[0] + x[1])

    nodes_data = await knowledge_graph_inst.get_nodes_batch(nodes_in_order)
    edges_data = await knowledge_graph_inst.get_edges_batch(edges_in_order)
    node_fields = ["id", "entity", "type", "description", "degree"]
    edge_fields = ["id", "source", "target", "description", "rank"]
    node_degrees = await knowledge_graph_inst.node_degrees_batch(nodes_in_order)
    nodes_list_data = [
        [
            i,
            node_name,
            node_data.get("entity_type", "UNKNOWN"),
            node_data.get("description", "UNKNOWN"),
            node_degrees[i],
        ]
        for i, (node_name, node_data) in enumerate(zip(nodes_in_order, nodes_data))
    ]
    # 对每个节点，提取节点的 ID、名称、实体类型、描述以及节点的度数，按度数降序排序。
    nodes_list_data = sorted(nodes_list_data, key=lambda x: x[-1], reverse=True)
    # 要根据每个节点的描述（x[3]）来计算 token 数量。如果超过了限制，就截断多余的节点，只保留满足限制的部分。
    nodes_may_truncate_list_data = truncate_list_by_token_size(
        nodes_list_data, key=lambda x: x[3], max_token_size=max_token_size // 2
    )
    edge_degrees = await knowledge_graph_inst.edge_degrees_batch(edges_in_order)
    edges_list_data = [
        [
            i,
            edge_name[0],
            edge_name[1],
            edge_data.get("description", "UNKNOWN"),
            edge_degrees[i],
        ]
        for i, (edge_name, edge_data) in enumerate(zip(edges_in_order, edges_data))
    ]

    # 对每条边，提取边的 ID、源节点、目标节点、描述以及边的度数（rank，即该边的连接重要性）。并按度数降序排序.
    edges_list_data = sorted(edges_list_data, key=lambda x: x[-1], reverse=True)
    # 同理，根据边的描述计算token后截断。
    edges_may_truncate_list_data = truncate_list_by_token_size(
        edges_list_data, key=lambda x: x[3], max_token_size=max_token_size // 2
    )

    # 上面的内容是直接使用原社区的节点和边生成报告，但是如果节点和边的描述超过了限制，就需要截断多余的部分，也就是丢失了信息。
    # truncated说明此时节点的描述或者边的描述超过了限制、产生了截断
    truncated = len(nodes_list_data) > len(nodes_may_truncate_list_data) or len(
        edges_list_data
    ) > len(edges_may_truncate_list_data)

    # If context is exceed the limit and have sub-communities:
    report_describe = ""
    need_to_use_sub_communities = (
        truncated and len(community["sub_communities"]) and len(already_reports)
    )
    force_to_use_sub_communities = global_config[
        "addon_params"
    ].get(
        "force_to_use_sub_communities", False
    )

    if need_to_use_sub_communities or force_to_use_sub_communities:
        # 直接使用原社区超过token限制了
        # 那么使用原社区的子社区报告+优先使用当前原社区中与子社区相关的重要节点和边
        logger.debug(
            f"Community {community['title']} exceeds the limit or you set force_to_use_sub_communities to True, using its sub-communities"
        )
        # 生成当前社区的重要子社区报告，用来当成当前社区的社区报告
        report_describe, report_size, contain_nodes, contain_edges = (
            _pack_single_community_by_sub_communities(
                community, max_token_size, already_reports
            )
        )

        # 包含或不包含在子社区中的节点和边
        report_exclude_nodes_list_data = [
            n for n in nodes_list_data if n[1] not in contain_nodes
        ]
        report_include_nodes_list_data = [
            n for n in nodes_list_data if n[1] in contain_nodes
        ]
        report_exclude_edges_list_data = [
            e for e in edges_list_data if (e[1], e[2]) not in contain_edges
        ]
        report_include_edges_list_data = [
            e for e in edges_list_data if (e[1], e[2]) in contain_edges
        ]
        # if report size is bigger than max_token_size, nodes and edges are []
        # truncate_list_by_token_size截断时是顺序读取的，这样拼接时保证了优先保留与子社区相关的重要节点和边
        # 优先保留与子社区相关的重要节点和边，而非子社区的内容可以作为备选项来截断，从而确保报告内容的核心信息不会丢失。
        nodes_may_truncate_list_data = truncate_list_by_token_size(
            report_exclude_nodes_list_data + report_include_nodes_list_data,
            key=lambda x: x[3],
            max_token_size=(max_token_size - report_size) // 2,
        )
        edges_may_truncate_list_data = truncate_list_by_token_size(
            report_exclude_edges_list_data + report_include_edges_list_data,
            key=lambda x: x[3],
            max_token_size=(max_token_size - report_size) // 2,
        )
    nodes_describe = list_of_list_to_csv([node_fields] + nodes_may_truncate_list_data)
    edges_describe = list_of_list_to_csv([edge_fields] + edges_may_truncate_list_data)
    return f"""-----报告-----
```csv
{report_describe}
```
-----实体-----
```csv
{nodes_describe}
```
-----关系-----
```csv
{edges_describe}
```"""


def _community_report_json_to_str(parsed_output: dict) -> str:
    """refer official graphrag: index/graph/extractors/community_reports"""
    title = parsed_output.get("title", "Report")
    summary = parsed_output.get("summary", "")
    findings = parsed_output.get("findings", [])

    def finding_summary(finding: dict):
        if isinstance(finding, str):
            return finding
        return finding.get("summary")

    def finding_explanation(finding: dict):
        if isinstance(finding, str):
            return ""
        return finding.get("explanation")

    report_sections = "\n\n".join(
        f"## {finding_summary(f)}\n\n{finding_explanation(f)}" for f in findings
    )
    return f"# {title}\n\n{summary}\n\n{report_sections}"


async def generate_community_report(
    community_report_kv: BaseKVStorage[CommunitySchema],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
    websocket=None,
):
    llm_extra_kwargs = global_config["special_community_report_llm_kwargs"]
    use_llm_func: callable = global_config["best_model_func"]
    community_report_prompt = PROMPTS[
        "community_report"
    ]  # 社区报告中应该不包含数据来源信息，因为其id并非真实id，而是enumerate出来的index
    use_string_json_convert_func: callable = global_config[
        "convert_response_to_json_func"
    ]
    domain_in_specific = " ,".join(PROMPTS["DOMAIN_IN_SPECIFIC"])

    communities_schema = (
        await knowledge_graph_inst.community_schema()
    )  # 返回字典，键是社区报告字符串，值是SingleCommunitySchema类型对象表示社区
    community_keys, community_values = list(communities_schema.keys()), list(
        communities_schema.values()
    )
    already_processed = 0

    async def _form_single_community_report(
        community: SingleCommunitySchema, already_reports: dict[str, CommunitySchema]
    ):
        '''为每个社区生成报告'''
        nonlocal already_processed
        # 将社区描述打包为文本
        describe = await _pack_single_community_describe(
            knowledge_graph_inst,
            community,
            max_token_size=global_config["best_model_max_token_size"],
            already_reports=already_reports,
            global_config=global_config,
        )
        # 调用 LLM 生成社区报告
        prompt = community_report_prompt.format(
            domain_in_specific=domain_in_specific, input_text=describe
        )
        data = None
        failed_json_count = 0
        while data is None and failed_json_count < 3:
            try:
                response = await use_llm_func(prompt, **llm_extra_kwargs)
                data = use_string_json_convert_func(response)
            except (AssertionError, json.JSONDecodeError) as e:
                failed_json_count += 1
                data = None
        if data is None:
            # 同一个社区多次json格式都失败，那么尝试用最后一次response修复
            data = fix_json_format(response)
            logger.warning(
                f"Community with title '{community['title']}' failed to convert response to JSON, using json_fixer to fix and get {data}...\n original_response: {response}"
            )
            if data is None:
                data = {
                    "title": "null",
                    "summary": "null",
                    "rating": 0,
                    "rating_explanation": "null",
                    "findings": [
                        {
                        "summary": "null",
                        "explanation": "null"
                        }
                    ]
                }

        already_processed += 1
        now_ticks = PROMPTS["process_tickers"][
            already_processed % len(PROMPTS["process_tickers"])
        ]
        print(
            f"{now_ticks} Processed {already_processed} communities\r",
            end="",
            flush=True,
        )
        logger.info(f"Processed {already_processed} communities")
        return data

    # level 值较大的社区会排在前面，即先生成较精细的底层社区的报告
    levels = sorted(set([c["level"] for c in community_values]), reverse=True)
    logger.info(f"Generating by levels: {levels}")
    if websocket:
        await websocket.send(f"Generating by levels: {levels}")
    community_datas = {}
    for level in levels:
        this_level_community_keys, this_level_community_values = zip(
            *[
                (k, v)
                for k, v in zip(community_keys, community_values)
                if v["level"] == level
            ]
        )
        logger.info(f"Generating {len(this_level_community_keys)} communities in level {level}")
        this_level_communities_reports = await safe_gather(
            [
                _form_single_community_report(c, community_datas)
                for c in this_level_community_values
            ]
        )
        # 将该层次社区的报告字符串和 JSON 格式存储到 community_datas 字典中
        # community_datas字典会在下次循环中被传入，然后再次更新
        # 即当满足特定条件时，低层次社区报告会用来生成更高层次社区的报告
        community_datas.update(
            {
                k: {
                    "report_string": _community_report_json_to_str(r),
                    "report_json": r,
                    **v,
                }
                for k, r, v in zip(
                    this_level_community_keys,
                    this_level_communities_reports,
                    this_level_community_values,
                )
            }
        )
    # 将生成的社区报告通过 community_report_kv.upsert 存储到键值存储中
    await community_report_kv.upsert(community_datas)
