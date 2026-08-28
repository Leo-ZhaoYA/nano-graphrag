import asyncio
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import partial
import re
import copy
import traceback
from typing import Callable, Dict, List, Optional, Type, Union, cast
import websockets
import tiktoken
import json
import aiofiles

from ._llm import (
    local_model_complete,
    gpt_4o_complete,
    gpt_4o_mini_complete,
    openai_embedding,
    local_embedding,
    azure_gpt_4o_complete,
    azure_openai_embedding,
    azure_gpt_4o_mini_complete,
)
from ._chunker import (
    chunking_by_token_size,
    get_chunks,
    get_tabular_chunks,
)
from ._extracter import (
    extract_entities, 
    merge_extractions
)
from ._community_generator import generate_community_report
from ._answer_generator import (
    local_query,
    global_query,
    naive_query,
)
from ._storage import (
    JsonKVStorage,
    NanoVectorDBStorage,
    NetworkXStorage,
    Neo4jStorage,
)
from ._utils import (
    EmbeddingFunc,
    compute_mdhash_id,
    limit_async_func_call,
    convert_response_to_json,
    always_get_an_event_loop,
    record_question_and_answer,
    logger,
    create_visualize_dataset,
    save_graph_to_file,
    cluster2communityIds,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    StorageNameSpace,
    QueryParam,
)
from ._upserter import (
    _upsert_entity_of_chunks,
    _upsert_relationship_of_chunks,
)

@dataclass
class GraphRAG:
    working_dir: str = field(
        default_factory=lambda: f"./nano_graphrag_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )
    # graph mode
    enable_local: bool = True
    enable_naive_rag: bool = False

    # text chunking
    chunk_func: Callable[
        [
            list[list[int]],
            List[str],
            tiktoken.Encoding,
            Optional[int],
            Optional[int],
        ],
        List[Dict[str, Union[str, int]]],
    ] = chunking_by_token_size
    chunk_token_size: int = 1200
    chunk_overlap_token_size: int = 100
    tiktoken_model_name: str = "gpt-4o"

    # entity extraction
    entity_extract_max_gleaning: int = 1
    entity_summary_to_max_tokens: int = 500

    # graph clustering
    graph_cluster_algorithm: str = "leiden"
    max_graph_cluster_size: int = 10
    graph_cluster_seed: int = 0xDEADBEEF

    # node embedding
    node_embedding_algorithm: str = "node2vec"
    node2vec_params: dict = field(
        default_factory=lambda: {
            "dimensions": 1536,
            "num_walks": 10,
            "walk_length": 40,
            "num_walks": 10,
            "window_size": 2,
            "iterations": 3,
            "random_seed": 3,
        }
    )

    # community reports
    special_community_report_llm_kwargs: dict = field(
        default_factory=lambda: {"response_format": {"type": "json_object"}}
    )
    build_community_report: bool = True

    # text embedding
    use_embedding_func: str = "openai"
    embedding_func: EmbeddingFunc = field(default_factory=lambda: openai_embedding)
    embedding_batch_num: int = 32
    # embedding_func_max_async: int = 16
    embedding_func_max_async: int = 16
    query_better_than_threshold: float = 0.2

    # LLM
    use_conversation_func: str = "openai"
    best_model_func: callable = gpt_4o_complete
    cheap_model_func: callable = gpt_4o_mini_complete
    using_azure_openai: bool = False
    best_model_max_token_size: int = 8192
    best_model_max_async: int = 2
    cheap_model_max_token_size: int = 8192
    cheap_model_max_async: int = 2

    # entity extraction
    entity_extraction_func: callable = extract_entities

    # storage
    key_string_value_json_storage_cls: Type[BaseKVStorage] = JsonKVStorage
    vector_db_storage_cls: Type[BaseVectorStorage] = NanoVectorDBStorage
    vector_db_storage_cls_kwargs: dict = field(default_factory=dict)
    graph_storage_cls: Type[BaseGraphStorage] = NetworkXStorage
    enable_llm_cache: bool = True

    # extension
    always_create_working_dir: bool = True
    addon_params: dict = field(default_factory=dict)
    convert_response_to_json_func: callable = convert_response_to_json

    # entity_extract_mode可选"seperate"或"one-off"，指对于text chunk的实体和关系提取方式
    # "seperate"表示实体和关系分开提取，"one-off"表示一起提取
    entity_extract_mode: str = "seperate"

    def __post_init__(self):
        logger.info(f"Using embedding function: '{self.use_embedding_func}' and conversation function: '{self.use_conversation_func}'")
        if self.use_embedding_func == "openai":
            self.embedding_func = openai_embedding
        elif self.use_embedding_func == "local":
            self.embedding_func = local_embedding
        else:
            raise ValueError("use_embedding_func must be 'openai' or 'local'")
        
        if self.use_conversation_func == "openai":
            self.best_model_func = gpt_4o_complete
            self.cheap_model_func = gpt_4o_mini_complete
            self.best_model_max_async = 16
            self.cheap_model_max_async = 16
        elif self.use_conversation_func == "local":
            self.best_model_func = local_model_complete
            self.cheap_model_func = local_model_complete
            self.best_model_max_async = 4
            self.cheap_model_max_async = 4           
        else:
            raise ValueError("use_conversation_func must be 'openai' or 'local'")
        self.websocket = None  # 在 __post_init__ 中初始化 websocket 属性
        self.logging_dir = os.path.join(self.working_dir, "logs")
        if not os.path.exists(self.logging_dir):
            os.makedirs(self.logging_dir)
        logger.set_logging_dir(self.logging_dir)  # 设置日志文件路径

        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in asdict(self).items()])
        logger.debug(f"GraphRAG init with param:\n\n  {_print_config}\n")

        self.chunk_entity_relation_graph_neo4j = Neo4jStorage(
            namespace="chunk_entity_relation_networkx", global_config=asdict(self)
        ) if self.graph_storage_cls == NetworkXStorage and self.addon_params["to_neo4j"] else None

        if self.using_azure_openai:
            # If there's no OpenAI API key, use Azure OpenAI
            if self.best_model_func == gpt_4o_complete:
                self.best_model_func = azure_gpt_4o_complete
            if self.cheap_model_func == gpt_4o_mini_complete:
                self.cheap_model_func = azure_gpt_4o_mini_complete
            if self.embedding_func == openai_embedding:
                self.embedding_func = azure_openai_embedding
            logger.info(
                "Switched the default openai funcs to Azure OpenAI if you didn't set any of it"
            )

        if not os.path.exists(self.working_dir) and self.always_create_working_dir:
            logger.info(f"Creating working directory {self.working_dir}")
            os.makedirs(self.working_dir)

        self.full_docs = self.key_string_value_json_storage_cls(
            namespace="full_docs", global_config=asdict(self)
        )

        self.text_chunks = self.key_string_value_json_storage_cls(
            namespace="text_chunks", global_config=asdict(self)
        )

        self.llm_response_cache = (
            self.key_string_value_json_storage_cls(
                namespace="llm_response_cache", global_config=asdict(self)
            )
            if self.enable_llm_cache
            else None
        )
        # 当前的社区报告
        self.community_reports = self.key_string_value_json_storage_cls(
            namespace="community_reports", global_config=asdict(self)
        )
        # 当前的实体和关系图
        self.chunk_entity_relation_graph = self.graph_storage_cls(
            namespace="chunk_entity_relation", global_config=asdict(self)
        )

        self.embedding_func = limit_async_func_call(self.embedding_func_max_async)(
            self.embedding_func
        )
        # 实体向量数据库
        # 插入或更新使用upsert方法，传入list_data: List[Dict]，其中每个字典的键值对为：
        # __id__:str  实体的hash_id
        # __vector__:list[float]（实体名+实体描述）向量
        # entity_name:str  实体名
        # source_type:List[str]  实体来源
        self.entities_vdb = (
            self.vector_db_storage_cls(
                namespace=f"entities_{self.use_embedding_func}",
                global_config=asdict(self),
                embedding_func=self.embedding_func,
                meta_fields={"entity_name", "source_type"},
            )
            if self.enable_local
            else None
        )

        self.chunks_vdb = (
            self.vector_db_storage_cls(
                namespace="chunks",
                global_config=asdict(self),
                embedding_func=self.embedding_func,
            )
            if self.enable_naive_rag
            else None
        )

        self.best_model_func = limit_async_func_call(self.best_model_max_async)(
            partial(self.best_model_func, hashing_kv=self.llm_response_cache)
        )
        self.cheap_model_func = limit_async_func_call(self.cheap_model_max_async)(
            partial(self.cheap_model_func, hashing_kv=self.llm_response_cache)
        )

    async def setup_websocket(self, websocket_url="ws://localhost:8000/ws"):
        if websocket_url is None:
            return True
        self.websocket = await websockets.connect(uri=websocket_url)
        # await self.websocket.send("GraphRAG WebSocket Connected")
        return await self.end_process()

    async def flush_websocket(self):
        # 清空websocket接收缓存区
        # 尝试接收所有待处理的消息，直到超时
        try:
            while True:
                message = await asyncio.wait_for(self.websocket.recv(), timeout=0.1)
                # print(f"Flushed message: {message}")  # 可以选择如何处理这些消息
        except asyncio.TimeoutError:
            # print("No more messages to flush.")
            pass

    async def end_process(self, msg="---END OF STAGE---"):
        if self.websocket:
            await self.flush_websocket()
            # 等待服务器确认
            await self.websocket.send(msg)
            confirmation = await self.websocket.recv()
            return confirmation == f"[#-ack-#]{msg}[/#-ack-#]"
        else:
            return True

    async def graphrag_done(self):
        if self.websocket:
            await self.end_process("---END OF GRAPHRAG PROCESS---")
            self.websocket = None
        logger.remove_file_handlers()

    def insert(self, string_or_strings):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.ainsert(string_or_strings))

    def query(self, query: str, param: QueryParam = QueryParam()):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aquery(query, param))

    async def none_node_in_graph(self):
        return \
            (isinstance(self.chunk_entity_relation_graph, Neo4jStorage) and not len(await self.chunk_entity_relation_graph.get_all_nodes())) or \
            (not isinstance(self.chunk_entity_relation_graph, Neo4jStorage) and not len(self.chunk_entity_relation_graph._graph.nodes))

    async def upsert_entities_vdb(self):
        """构建用于匹配的实体向量数据库、本地保存实体和关系的info文件"""

        if self.enable_local:
            data_for_vdb = await save_graph_to_file(self.working_dir, self.chunk_entity_relation_graph, self.text_chunks)

            if len(data_for_vdb):
                # 删除旧的数据库文件并重建
                if os.path.exists(self.entities_vdb._client_file_name):
                    os.remove(self.entities_vdb._client_file_name)
                del self.entities_vdb
                self.entities_vdb = (
                    self.vector_db_storage_cls(
                        namespace=f"entities_{self.use_embedding_func}",
                        global_config=asdict(self),
                        embedding_func=self.embedding_func,
                        meta_fields={"entity_name", "source_type"},
                    )
                    if self.enable_local
                    else None
                )
                await self.entities_vdb.upsert(data_for_vdb)
                await cast(StorageNameSpace, self.entities_vdb).index_done_callback()

    async def merge_graphs(self):
        """合并多次针对相同内容的已有建图"""
        try:
            # 丢弃旧的社区报告
            await self.community_reports.drop()
            if not isinstance(self.chunk_entity_relation_graph, Neo4jStorage):
                await self.chunk_entity_relation_graph.drop_clusters()

            # 加载想要合并的图
            chunk_entity_relation_graph_merge = self.graph_storage_cls(
                namespace="chunk_entity_relation_merge", global_config=asdict(self)
            )
            if (await self.none_node_in_graph()):
                raise ValueError(f"No graph data to merge! In order to merge, rename a valid graph to name 'graph_chunk_entity_relation_merge.graphml' in {self.working_dir}!")
            # 开始合并
            logger.info("[Merging Entity Extraction]...")
            if self.websocket:
                await self.websocket.send("[Merging Entity Extraction]...")
            maybe_new_kg = await merge_extractions(
                self.text_chunks._data,
                knowledge_graph_inst=self.chunk_entity_relation_graph,
                knowledge_graph_merge=chunk_entity_relation_graph_merge,
                global_config=asdict(self),
                websocket=self.websocket,  # 将 websocket 对象传递给 entity_extraction_func 函数
            )
            await self.upsert_entities_vdb()

            # 存储阶段性结果，防止丢失
            # self.community_reports目前被整个drop掉了是空的，所以不需要存储
            self.chunk_entity_relation_graph = maybe_new_kg
            await cast(StorageNameSpace, self.entities_vdb).index_done_callback()
            logger.info(f"[Saving] entities_vdb done")
            if self.websocket:
                await self.websocket.send("[Saving] entities_vdb done")
            await cast(
                StorageNameSpace, self.chunk_entity_relation_graph
            ).index_done_callback()
            logger.info(f"[Saving] maybe_new_kg done")
            if self.websocket:
                await self.websocket.send("[Saving] maybe_new_kg done")
            await cast(StorageNameSpace, self.llm_response_cache).index_done_callback()
            logger.info(f"[Saving] partial llm_cache done")
            if self.websocket:
                await self.websocket.send("[Saving] partial llm_cache done")

            # ---------- update clusterings of graph
            # 更新图的聚类社区、生成社区报告
            # 使用leiden算法对实体关系图进行聚类
            # 使用_storage.py中的_leiden_clustering()函数
            # self.chunk_entity_relation_graph就是刚刚多次upsert过的实体关系图
            await self.chunk_entity_relation_graph.clustering(
                self.graph_cluster_algorithm
            )
            # leiden是基于优化的算法，对相同的图和相同的参数输入每次运行会得到不同的结果
            # 因此generate_community_report()函数中不进行社区中间数据的本地存储
            if self.build_community_report:
                logger.info("[Community Report]...")
                if self.websocket:
                    await self.websocket.send("[Community Report]...")
                await generate_community_report(
                    self.community_reports,
                    self.chunk_entity_relation_graph,
                    asdict(self),
                    self.websocket,
                )
            await self._insert_done()
            return await self.end_process()            
        except:
            logger.error(traceback.format_exc())
            if self.websocket:
                await self.websocket.send(traceback.format_exc())
            raise
        
    async def nkx2n4j(self):
        """将NetworkX图转换为Neo4j图"""
        if self.chunk_entity_relation_graph_neo4j is not None and isinstance(self.chunk_entity_relation_graph, NetworkXStorage):
            # assert isinstance(self.chunk_entity_relation_graph, NetworkXStorage), "chunk_entity_relation_graph must be NetworkXStorage"
            ns = self.chunk_entity_relation_graph._graph.nodes
            es = self.chunk_entity_relation_graph._graph.edges
            # if self.websocket:
            #     await self.websocket.send(f"Converting {len(ns)} Nodes and {len(es)} Edges to Neo4j...")
                    # 将NetworkX图转换为Neo4j图
            ns = await cluster2communityIds(ns)
            await self.chunk_entity_relation_graph_neo4j.upsert_nodes_batch([(k, v) for k, v in ns.items()])
            await self.chunk_entity_relation_graph_neo4j.upsert_edges_batch([(k[0], k[1] ,v) for k, v in es.items()])
            logger.info(f"Succeed in converting NetworkX to Neo4j")


    async def aquery(self, query: str, param: QueryParam = QueryParam()):
        response = None
        # await self.nkx2n4j()
        # await self.merge_graphs()
        # await create_visualize_dataset(self.working_dir, self.chunk_entity_relation_graph, self.community_reports, self.text_chunks)
        # await self.upsert_entities_vdb()
        # await cast(StorageNameSpace, self.entities_vdb).index_done_callback()
        logger.info(f"Mode: '{param.mode}' Querying: '{query}'")
        if param.mode == "local" and not self.enable_local:
            
            raise ValueError("enable_local is False, cannot query in local mode")
        if param.mode == "naive" and not self.enable_naive_rag:
            raise ValueError("enable_naive_rag is False, cannot query in naive mode")
        if param.mode == "local":
            response = await local_query(
                query,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.community_reports,
                self.text_chunks,
                param,
                asdict(self),
                websocket=self.websocket,  # 将 websocket 对象传递给 local_query 函数
            )
        elif param.mode == "global":
            response = await global_query(
                query,
                self.chunk_entity_relation_graph,
                self.community_reports,
                param,
                asdict(self),
                websocket=self.websocket,  # 将 websocket 对象传递给 global_query 函数
            )
        else:
            raise ValueError(f"Unknown mode {param.mode}")
        record_question_and_answer(self.working_dir, query, response, param.mode)
        await self._query_done()
        return await self.end_process()
        # return response

    async def ainsert(self, string_or_strings):
        '''插入文本数据并构建图谱'''
        if not isinstance(string_or_strings, (str, list)):
            raise ValueError("string_or_strings must be str or list of str")

        await self._insert_start()
        (from_fail, new_docs) = await self.from_fail_insert(string_or_strings)
        # ---------- chunking
        inserting_chunks, table_handlers = await get_tabular_chunks(
            new_docs=new_docs,
            chunk_func=self.chunk_func,
            global_config=asdict(self),
            overlap_token_size=self.chunk_overlap_token_size,
            max_token_size=self.chunk_token_size,
        )
        self.table_handlers = table_handlers

        if not len(new_docs):
            # 没有新文本
            if from_fail:
                # 从一次失败的插入中恢复，文本完全相同，不传递新文本
                await self.ainsert_from_fail()
            else:
                logger.info("所有文档都已在存储中，使用原有cache...")
                if self.websocket:
                    await self.websocket.send("所有文档都已在存储中，使用原有cache...")
            return await self.end_process()

        try:
            _add_chunk_keys = await self.text_chunks.filter_keys(
                list(inserting_chunks.keys())
            )
            inserting_chunks = {
                k: v for k, v in inserting_chunks.items() if k in _add_chunk_keys
            }
            if not len(inserting_chunks):
                logger.warning(f"All chunks are already in the storage")
                if self.websocket:
                    await self.websocket.send("所有分块都已在存储中，使用原有cache...")
                return
            logger.info(f"[New Chunks] inserting {len(inserting_chunks)} chunks")
            if self.websocket:
                await self.websocket.send(
                    f"[New Chunks] inserting {len(inserting_chunks)} chunks"
                )

            # 插入 new_docs 到 full_docs
            await self.full_docs.upsert(new_docs)
            # 插入后调用 full_docs 的 index_done_callback
            await cast(StorageNameSpace, self.full_docs).index_done_callback()
            # 插入 inserting_chunks 到 text_chunks
            await self.text_chunks.upsert(inserting_chunks)
            # 插入后调用 text_chunks 的 index_done_callback
            await cast(StorageNameSpace, self.text_chunks).index_done_callback()
            logger.info(f"[Saving] full_docs and text_chunks done")
            if self.websocket:
                await self.websocket.send(f"[Saving] full_docs and text_chunks done")

            # TODO: no incremental update for communities now, so just drop all
            # 现在没有针对社区的增量更新，直接删除原有的所有社区报告
            await self.community_reports.drop()
            if not isinstance(self.chunk_entity_relation_graph, Neo4jStorage):
                await self.chunk_entity_relation_graph.drop_clusters()
                
            # ---------- extract/summary entity and upsert to graph
            # 完成实体、关系的提取和摘要，并将其插入到图中
            # extract_entities() 函数会使用到这里self.best_model_func、self.cheap_model_func定义的模型
            # generate_community_report() 函数会使用到这里self.best_model_func定义的模型
            if self.enable_naive_rag:
                logger.info("Insert chunks for naive RAG")
                await self.chunks_vdb.upsert(inserting_chunks)

            # ---------- extract/summary entity and upsert to graph
            logger.info("[Entity Extraction]...")
            if self.websocket:
                await self.websocket.send("[Entity Extraction]...")
            maybe_new_kg = await self.entity_extraction_func(
                inserting_chunks,
                knowledge_graph_inst=self.chunk_entity_relation_graph,
                table_handlers=self.table_handlers,
                global_config=asdict(self),
                websocket=self.websocket,  # 将 websocket 对象传递给 entity_extraction_func 函数
            )
            await self.upsert_entities_vdb()
            if maybe_new_kg is None:
                logger.warning("No new entities found")
                if self.websocket:
                    await self.websocket.send("No new entities found")
                return

            # 存储阶段性结果，防止丢失
            # self.community_reports目前被整个drop掉了是空的，所以不需要存储
            self.chunk_entity_relation_graph = maybe_new_kg
            await cast(StorageNameSpace, self.entities_vdb).index_done_callback()
            logger.info(f"[Saving] entities_vdb done")
            if self.websocket:
                await self.websocket.send("[Saving] entities_vdb done")
            await cast(
                StorageNameSpace, self.chunk_entity_relation_graph
            ).index_done_callback()
            logger.info(f"[Saving] maybe_new_kg done")
            if self.websocket:
                await self.websocket.send("[Saving] maybe_new_kg done")
            await cast(StorageNameSpace, self.llm_response_cache).index_done_callback()
            logger.info(f"[Saving] partial llm_cache done")
            if self.websocket:
                await self.websocket.send("[Saving] partial llm_cache done")

            # ---------- update clusterings of graph
            # 更新图的聚类社区、生成社区报告

            # 使用leiden算法对实体关系图进行聚类
            # 使用_storage.py中的_leiden_clustering()函数
            # self.chunk_entity_relation_graph就是刚刚多次upsert过的实体关系图
            await self.chunk_entity_relation_graph.clustering(
                self.graph_cluster_algorithm
            )
            await self.nkx2n4j()
            # leiden是基于优化的算法，对相同的图和相同的参数输入每次运行会得到不同的结果
            # 因此generate_community_report()函数中不进行社区中间数据的本地存储
            if self.build_community_report:
                logger.info("[Community Report]...")
                if self.websocket:
                    await self.websocket.send("[Community Report]...")
                await generate_community_report(
                    self.community_reports,
                    self.chunk_entity_relation_graph,
                    asdict(self),
                    self.websocket,
                )
            await self._insert_done()
            return await self.end_process()            
        except:
            e = traceback.format_exc()
            logger.error(e)
            if self.websocket:
                await self.websocket.send(e)
            raise
        
    async def ainsert_from_fail(self):
        '''
        从一次失败的插入中恢复并继续插入。
        只有文本与之前尝试失败时完全相同才调用此函数，否则请用 ainsert(string_or_strings)函数。
        '''

        # 一般来说，full_docs 和 text_chunks 都不会为空、否则直接使用 ainsert(string_or_strings) 函数
        logger.info("[Recovering] from failed insert...")
        if self.websocket:
            await self.websocket.send("[Recovering] from failed insert...")
        try:
            # 直接取已有的工作路径中的 full_docs 和 text_chunks
            logger.info(f"[New Chunks] inserting {len(self.text_chunks._data)} chunks")
            if self.websocket:
                await self.websocket.send(
                    f"[New Chunks] inserting {len(self.text_chunks._data)} chunks"
                )

            # TODO: no incremental update for communities now, so just drop all
            # 直接删除原有的所有社区
            await self.community_reports.drop()
            if not isinstance(self.chunk_entity_relation_graph, Neo4jStorage):
                await self.chunk_entity_relation_graph.drop_clusters()

            # 完成实体、关系的提取和摘要，并将其插入到图中
            # Python 的属性名称重整机制（name mangling）
            # 在一个类中定义双下划线开头的变量（如 __storage），Python 会将其名称改为 _类名__属性名，以避免与子类的属性名发生冲突
            # 在 NanoVectorDB 类中，属性 __storage 实际上被重整为 _NanoVectorDB__storage
            if (await self.none_node_in_graph()):
                # 实体关系图记录为空，重新提取实体和关系
                # 得到标准字典对象，而非dict_items
                inserting_chunks = {
                    k: v for k, v in self.text_chunks._data.items()
                }

                logger.info(
                    f"Zero Entity Retrieved from Storage, Re-extracting {len(inserting_chunks)} Text Chunks..."
                )
                if self.websocket:
                    await self.websocket.send(
                        f"Zero Entity Retrieved from Storage, Re-extracting {len(inserting_chunks)} Text Chunks..."
                    )
                logger.info("[Entity Extraction]...")
                if self.websocket:
                    await self.websocket.send("[Entity Extraction]...")
                maybe_new_kg = await self.entity_extraction_func(
                    inserting_chunks,
                    knowledge_graph_inst=self.chunk_entity_relation_graph,
                    table_handlers=self.table_handlers,
                    global_config=asdict(self),
                    websocket=self.websocket,  # 将 websocket 对象传递给 extract_entities 函数
                )
                await self.upsert_entities_vdb()
                if maybe_new_kg is None:
                    logger.warning("No new entities found")
                    if self.websocket:
                        await self.websocket.send("No new entities found")
                    return

                # 存储阶段性结果，防止丢失
                # self.community_reports目前被整个drop掉了是空的，所以不需要存储
                self.chunk_entity_relation_graph = maybe_new_kg
                await cast(StorageNameSpace, self.entities_vdb).index_done_callback()
                logger.info(f"[Saving] entities_vdb done")
                if self.websocket:
                    await self.websocket.send("[Saving] entities_vdb done")
                await cast(
                    StorageNameSpace, self.chunk_entity_relation_graph
                ).index_done_callback()
                logger.info(f"[Saving] maybe_new_kg done")
                if self.websocket:
                    await self.websocket.send("[Saving] maybe_new_kg done")
                await cast(
                    StorageNameSpace, self.llm_response_cache
                ).index_done_callback()
                logger.info(f"[Saving] partial llm_cache done")
                if self.websocket:
                    await self.websocket.send("[Saving] partial llm_cache done")
            else:
                # 图不为空
                # 直接使用之前的实体和关系图
                ns = self.chunk_entity_relation_graph._graph.nodes if not isinstance(self.chunk_entity_relation_graph, Neo4jStorage) else await self.chunk_entity_relation_graph.get_all_nodes()
                es = self.chunk_entity_relation_graph._graph.edges if not isinstance(self.chunk_entity_relation_graph, Neo4jStorage) else await self.chunk_entity_relation_graph.get_all_edges()

                s = f"{len(ns)} Entities and {len(es)} Relations Retrieved from Storage, Re-using..."
                logger.info(s)
                if self.websocket:
                    await self.websocket.send(s)
            if not len(self.entities_vdb._client._NanoVectorDB__storage["data"]):
                # 实体向量数据库为空，重新插入实体向量
                await self.upsert_entities_vdb()

            # ---------- update clusterings of graph
            # 更新图的聚类社区、生成社区报告
            # 这一步无论实体有没有成功恢复都要重新做

            # 使用leiden算法对实体关系图进行聚类
            # 使用_storage.py中的_leiden_clustering()函数
            # self.chunk_entity_relation_graph就是刚刚多次upsert过的实体关系图
            await self.chunk_entity_relation_graph.clustering(
                self.graph_cluster_algorithm
            )
            await self.nkx2n4j()
            if self.build_community_report:
                logger.info("[Community Report]...")
                if self.websocket:
                    await self.websocket.send("[Community Report]...")                
                await generate_community_report(
                    self.community_reports,
                    self.chunk_entity_relation_graph,
                    asdict(self),
                    self.websocket,
                )
            await self._insert_done()
            return await self.end_process()            
        except:
            e = traceback.format_exc()
            logger.error(e)
            if self.websocket:
                await self.websocket.send(e)     
            raise  


    async def _insert_start(self):
        tasks = []
        for storage_inst in [
            self.chunk_entity_relation_graph,
            self.chunk_entity_relation_graph_neo4j
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_start_callback())
        await asyncio.gather(*tasks)

    async def _insert_done(self):
        '''完整建图后进行存储操作'''
        tasks = []
        for storage_inst in [
            self.llm_response_cache,
            self.community_reports,
            self.chunk_entity_relation_graph,
            self.chunks_vdb,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
            logger.info(f"[Saving] {storage_inst.namespace} done")
            if self.websocket:
                await self.websocket.send(f"[Saving] {storage_inst.namespace} done")
        await asyncio.gather(*tasks)
        # await create_visualize_dataset(self.working_dir, self.chunk_entity_relation_graph, self.community_reports, self.text_chunks)

    async def _query_done(self):
        '''查询后进行存储操作'''
        tasks = []
        for storage_inst in [self.llm_response_cache]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

    async def filter_new_docs(self, string_or_strings):
        '''过滤出新的文档'''
        if isinstance(string_or_strings, str):
            string_or_strings = [string_or_strings]
        # ---------- new docs
        new_docs = {
            compute_mdhash_id(c.strip(), prefix="doc-"): {"content": c.strip()}
            for c in string_or_strings
        }
        _add_doc_keys = await self.full_docs.filter_keys(
            list(new_docs.keys())
        )  # 返回 new_docs 中不存在于 full_docs 的键准备插入
        new_docs = {k: v for k, v in new_docs.items() if k in _add_doc_keys}
        if not len(new_docs):
            # logger.warning(f"All docs are already in the storage")
            return new_docs
        logger.info(f"[New Docs] inserting {len(new_docs)} docs")
        if self.websocket:
            await self.websocket.send(f"[New Docs] inserting {len(new_docs)} docs")

        return new_docs

    async def from_fail_insert(self, string_or_strings):
        """
        检查上一次插入是否失败
        """
        # 异步获取新的文档
        new_docs = await self.filter_new_docs(string_or_strings)

        # 需要检查的数据集合
        data_sets = [
            self.full_docs._data,
            self.text_chunks._data,
            self.entities_vdb._client._NanoVectorDB__storage.get("data", []),
            self.community_reports._data,
            self.llm_response_cache._data,
        ]
        if not isinstance(self.chunk_entity_relation_graph, Neo4jStorage):
            data_sets.append(self.chunk_entity_relation_graph._graph.nodes)

        # 如果任意数据集为空且没有新文档，说明上一次插入失败，可以执行失败插入逻辑
        if any(len(data) == 0 for data in data_sets) and not new_docs:
            return True, new_docs

        # 其余情况，认为需要重新插入
        return False, new_docs

    async def search_chunks_with_keywords(self, keywords: List[str]) -> Dict[str, str]:
        """
        搜索同时包含所有keywords的chunks
        return: Dict[chunk_id, chunk_content]
        """

        if not isinstance(keywords, list):
            raise ValueError("keywords must be a list of str")

        search_results = {}
        all_chunks = list(await self.text_chunks.all_keys())
        for chunk_id in all_chunks:
            chunk = await self.text_chunks.get_by_id(chunk_id)
            chunk_text: str = chunk["content"]
            if all(keyword in chunk_text for keyword in keywords):
                search_results[chunk_id] = chunk_text
        
        return search_results

    async def upsert_entity(self, entity_name: str, source_chunk_ids: Optional[List[str]] = None):
        """
        插入/更新实体
        param entity_name: 实体名
        param source_chunks: 实体来源的chunks id (以"chunk-"开头)（如果为None则遍历所有chunks）
        """
        if await self.chunk_entity_relation_graph.has_node(entity_name):
            logger.info(f"Entity {entity_name} already exists, updating...")

        # 只对包含实体名的chunk进行分析和实体插入
        if not source_chunk_ids:
            search_results = await self.search_chunks_with_keywords([entity_name])
            source_chunk_ids = list(search_results.keys())
            if len(source_chunk_ids) == 0:
                logger.error(f"Could not find any source chunk containing entity '{entity_name}'")
                return
            
        chunk_entity_relation_graph, entities_vdb = await _upsert_entity_of_chunks(
            self.text_chunks, 
            self.chunk_entity_relation_graph, 
            self.entities_vdb, 
            entity_name, 
            source_chunk_ids, 
            asdict(self),
        )
        self.chunk_entity_relation_graph = chunk_entity_relation_graph
        self.entities_vdb = entities_vdb
        await cast(StorageNameSpace, self.chunk_entity_relation_graph).index_done_callback()
        await cast(StorageNameSpace, self.entities_vdb).index_done_callback()

        return await self.end_process()
    
    async def upsert_relationship(self, 
                                  src_entity_name: str,
                                  dst_entity_name:str,
                                  source_chunks: Optional[List[str]] = None):
        """
        插入/更新关系
        param src_entity_name: 源实体名
        param dst_entity_name: 目标实体名
        param source_chunks: 关系来源的chunks id (以"chunk-"开头)（如果为None则遍历所有chunks）
        """

        if not isinstance(source_chunks, list):
            raise ValueError("source_chunks must be a list of str")

        # 只选取同时包含src_entity_name和dst_entity_name的chunk进行分析和关系插入
        if not source_chunks:
            search_results = await self.search_chunks_with_keywords([src_entity_name, dst_entity_name])
            source_chunks = list(search_results.keys())
            if len(source_chunks) == 0:
                logger.error(f"Could not find any source chunk containing both '{src_entity_name}' and '{dst_entity_name}'")
                return
        
        chunk_entity_relation_graph, entities_vdb = await _upsert_relationship_of_chunks(
            self.text_chunks, 
            self.chunk_entity_relation_graph, 
            self.entities_vdb,
            src_entity_name, 
            dst_entity_name, 
            source_chunks,
            asdict(self),
        )
        self.chunk_entity_relation_graph = chunk_entity_relation_graph
        self.entities_vdb = entities_vdb
        await cast(StorageNameSpace, self.chunk_entity_relation_graph).index_done_callback()
        await cast(StorageNameSpace, self.entities_vdb).index_done_callback()

        return await self.end_process()