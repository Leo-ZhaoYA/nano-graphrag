import os
from typing import Callable, Dict, List, Optional, Type, Union, cast
import websockets
import json
import aiofiles
from .prompt import GRAPH_FIELD_SEP, PROMPTS
from ._storage import (
    JsonKVStorage,
    NanoVectorDBStorage,
    NetworkXStorage,
)
from ._utils import (
    EmbeddingFunc,
    compute_mdhash_id,
    logger,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    StorageNameSpace,
    QueryParam,
)

async def save_json_data(file_path, data_to_save, logger, merge_existing=True):
    """
    通用 JSON 数据保存函数。

    :param file_path: 文件保存路径（包括文件名）。
    :param data_to_save: 需要保存的字典数据。
    :param logger: 日志记录器。
    :param merge_existing: 是否合并已存在的文件内容（默认为 False）。
    """
    try:
        # 如果需要合并数据并且文件存在，读取现有数据
        if merge_existing and os.path.exists(file_path):
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                try:
                    existing_data = json.loads(await f.read())
                except json.JSONDecodeError:
                    existing_data = {}
            # 合并数据
            existing_data.update(data_to_save)
        else:
            existing_data = data_to_save

        # 确保目标目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # 写入文件
        async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(existing_data, ensure_ascii=False, indent=4))

        logger.info(f"Data successfully saved to {file_path}.")
    except Exception as e:
        logger.error(f"Failed to save data to {file_path}: {e}")


async def _upsert_entity_of_chunks(
        text_chunks, 
        chunk_entity_relation_graph, 
        entities_vdb, 
        entity_name: str, 
        source_chunk_ids: List[str],
        global_config: Dict,
    ):
    """
    分析source_chunk_ids，并直接插入/更新实体
    """
    def add_quotes(s: str):
        return f'"{s}"' # entity_name、entity_description等几项的格式居然是要加引号的


    entity_descriptions = []
    source_types = []
    entity_types = []

    assert source_chunk_ids
    for chunk_id in source_chunk_ids:

        chunk = await text_chunks.get_by_id(chunk_id)

        if chunk is None:
            logger.error(f"Chunk {chunk_id} not found when inserting entity {entity_name}")
            continue

        prompt_context = dict(
            entity_name=entity_name,
            input_text=chunk["content"],
            entity_types=",".join(PROMPTS["DEFAULT_ENTITY_TYPES"]),
        )

        prompt = PROMPTS["get_entity_type_and_description"].format(**prompt_context)
        llm_result = await global_config["best_model_func"](prompt)
        llm_result = llm_result.replace("```json", "").strip().strip("`").strip()

        try:
            entity_info = dict(eval(llm_result))
            entity_type = entity_info.get("entity_type")
            entity_description = entity_info.get("entity_description")

            assert entity_name == entity_info.get("entity_name")
            assert entity_type is not None
            assert entity_description is not None
        except Exception as e:
            logger.error(f"Parse llm result '{llm_result}' error: {e}")
            return
        
        if chunk["type"] not in source_types:
            source_types.append(chunk["type"])
        
        if entity_type not in entity_types:
            entity_types.append(add_quotes(entity_type))
        
        entity_descriptions.append(add_quotes(entity_description))


    source_id_str = GRAPH_FIELD_SEP.join(source_chunk_ids)
    description_str = GRAPH_FIELD_SEP.join(entity_descriptions)
    entity_type_str = GRAPH_FIELD_SEP.join(entity_types)
    entity_name = add_quotes(entity_name) 

    node_data = dict(
        entity_type=entity_type_str,
        description=description_str,
        source_id=source_id_str,
    )
    
    await chunk_entity_relation_graph.upsert_node(
        entity_name,
        node_data=node_data,
    )

    await entities_vdb.upsert(
        {
            compute_mdhash_id(entity_name, prefix="ent-"): {
                "content": entity_name + description_str,
                "entity_name": entity_name,
                "source_type": source_types,
            }
        }
    )
    # 存储实体信息文件
    file_name = compute_mdhash_id(entity_name, prefix="ent-")
    data_to_save = {
        file_name: {
            "entity_name": entity_name,
            "description": description_str,
            "entity_type": entity_type_str,
            "source_id": source_id_str,
            "source_type": source_types
        }
    }
    
    logger.info(f"Saving entity data to {file_name}")
    await save_json_data(
        os.path.join(global_config["working_dir"], 'entities_info.json'),
        data_to_save,
        logger,
        merge_existing=True
    )

    return chunk_entity_relation_graph, entities_vdb


async def _upsert_relationship_of_chunks(
        text_chunks, 
        chunk_entity_relation_graph, 
        entities_vdb,
        source_entity: str, 
        target_entity: str, 
        source_chunk_ids: List[str],
        global_config: Dict,
    ):
    """
    分析source_chunk_ids，并直接插入/更新关系
    """
    if not isinstance(source_chunk_ids, list):
        raise ValueError("source_chunks must be a list of str")
    
    def add_quotes(s: str):
        return f'"{s}"' # entity_name、entity_description等几项的格式居然是要加引号的
    
    if not await chunk_entity_relation_graph.has_node(source_entity):
        chunk_entity_relation_graph, entities_vdb = await _upsert_entity_of_chunks(text_chunks, chunk_entity_relation_graph, entities_vdb, source_entity, source_chunk_ids, global_config)

    if not await chunk_entity_relation_graph.has_node(target_entity):
        chunk_entity_relation_graph, entities_vdb = await _upsert_entity_of_chunks(text_chunks, chunk_entity_relation_graph, entities_vdb, target_entity, source_chunk_ids, global_config)


    source_types = []
    relationship_description = []
    weights = []

    assert source_chunk_ids
    for chunk_id in source_chunk_ids:

        chunk = await text_chunks.get_by_id(chunk_id)

        if chunk is None:
            logger.error(f"Chunk {chunk_id} not found when inserting relationship between {source_entity} and {target_entity}")
            continue

        prompt_context = dict(
            source_entity=source_entity,
            target_entity=target_entity,
            input_text=chunk["content"],
        )

        prompt = PROMPTS["get_relationship_description_and_strength"].format(**prompt_context)
        llm_result = await global_config["best_model_func"](prompt)
        llm_result = llm_result.replace("```json", "").strip().strip("`").strip()

        try:
            relationship_info = dict(eval(llm_result))
            weight = relationship_info.get("relationship_strength")
            description = relationship_info.get("relationship_description")

            assert source_entity == relationship_info.get("source_entity")
            assert target_entity == relationship_info.get("target_entity")
            assert weight is not None
            assert description is not None

        except Exception as e:
            logger.error(f"Parse llm result '{llm_result}' error: {e}")
            return
        
        if chunk["type"] not in source_types:
            source_types.append(chunk["type"])
        
        relationship_description.append(add_quotes(description))
        weights.append(weight)


    
    source_ids_str = GRAPH_FIELD_SEP.join(source_chunk_ids)
    relationship_relation_str = GRAPH_FIELD_SEP.join(relationship_description)
    weight = max(weights)
    order = 1

    source_entity = add_quotes(source_entity)
    target_entity = add_quotes(target_entity)

    edge_data = dict(
        weight=weight,
        description=relationship_relation_str,
        source_id=source_ids_str,
        order=order
    )
    await chunk_entity_relation_graph.upsert_edge(
        source_entity,
        target_entity,
        edge_data=edge_data,
    )

    # 存储关系信息文件
    src_id = compute_mdhash_id(source_entity, prefix="ent-")
    tgt_id = compute_mdhash_id(target_entity, prefix="ent-")
    file_name = f'rel-{src_id}_{tgt_id}'

    data_to_save = {
        file_name: {
            "src_id": src_id,
            "src_name": source_entity,
            "tgt_id": tgt_id,
            "tgt_name": target_entity,
            "description": relationship_relation_str,
            "weight": weight,
            "source_id": source_ids_str,
            "source_type": source_types,
            "order": order,
        }
    }

    logger.info(f"Saving relationship data to {file_name}")
    await save_json_data(
        os.path.join(global_config["working_dir"], 'relationships_info.json'),
        data_to_save,
        logger,
        merge_existing=True
    )

    return chunk_entity_relation_graph, entities_vdb
