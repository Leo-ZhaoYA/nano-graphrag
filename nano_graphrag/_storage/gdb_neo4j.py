import json
import asyncio
from collections import defaultdict
from typing import List, Dict
from neo4j import AsyncGraphDatabase
from dataclasses import dataclass
from typing import Union
from ..base import BaseGraphStorage, SingleCommunitySchema
from .._utils import logger
from ..prompt import GRAPH_FIELD_SEP

neo4j_lock = asyncio.Lock()


def make_path_idable(path):
    return path.split('graphrag_dir')[-1].strip('\\/.').strip().replace(".", "_").replace("/", "__").replace("-", "_").replace(":", "_").replace("\\", "__")


@dataclass
class Neo4jStorage(BaseGraphStorage):
    def __post_init__(self):
        self.neo4j_url = self.global_config["addon_params"].get("neo4j_url", None)
        self.neo4j_auth = self.global_config["addon_params"].get("neo4j_auth", None)
        self.namespace = (
            f"{make_path_idable(self.global_config['working_dir'])}__{self.namespace}"
        )
        if self.namespace[0].isalpha():
            # 确保Windows磁盘符号是小写的
            self.namespace = self.namespace[0].lower() + self.namespace[1:]
        logger.info(f"Using the label {self.namespace} for Neo4j as identifier")
        if self.neo4j_url is None or self.neo4j_auth is None:
            raise ValueError("Missing neo4j_url or neo4j_auth in addon_params")
        self.async_driver = AsyncGraphDatabase.driver(
            self.neo4j_url, auth=self.neo4j_auth, max_connection_pool_size=50,      # 调整连接池大小
        )

    # async def create_database(self):
    #     async with self.async_driver.session() as session:
    #         try:
    #             constraints = await session.run("SHOW CONSTRAINTS")
    #             # TODO I don't know why CREATE CONSTRAINT IF NOT EXISTS still trigger error
    #             # so have to check if the constrain exists
    #             constrain_exists = False

    #             async for record in constraints:
    #                 if (
    #                     self.namespace in record["labelsOrTypes"]
    #                     and "id" in record["properties"]
    #                     and record["type"] == "UNIQUENESS"
    #                 ):
    #                     constrain_exists = True
    #                     break
    #             if not constrain_exists:
    #                 await session.run(
    #                     f"CREATE CONSTRAINT FOR (n:{self.namespace}) REQUIRE n.id IS UNIQUE"
    #                 )
    #                 logger.info(f"Add constraint for namespace: {self.namespace}")

    #         except Exception as e:
    #             logger.error(f"Error accessing or setting up the database: {str(e)}")
    #             raise

    async def _init_workspace(self):
        await self.async_driver.verify_authentication()
        await self.async_driver.verify_connectivity()
        # TODOLater: create database if not exists always cause an error when async
        # await self.create_database()

    async def ensure_indexes(self):
        """
        创建必要的索引以提高查询性能。
        针对 Neo4j Community Edition 创建单属性索引。
        """
        try:
            async with self.async_driver.session() as session:
                # 1. 为节点ID创建索引 - 这是最重要的索引，用于节点查找
                await session.run(
                    f"CREATE INDEX IF NOT EXISTS FOR (n:`{self.namespace}`) ON (n.id)"
                )
                
                # 2. 为实体类型创建索引 - 优化按类型查询
                await session.run(
                    f"CREATE INDEX IF NOT EXISTS FOR (n:`{self.namespace}`) ON (n.entity_type)"
                )
                
                # 3. 为社区ID创建索引 - 优化社区查询
                await session.run(
                    f"CREATE INDEX IF NOT EXISTS FOR (n:`{self.namespace}`) ON (n.communityIds)"
                )
                
                # 4. 为源ID创建索引 - 优化按源文档查询
                await session.run(
                    f"CREATE INDEX IF NOT EXISTS FOR (n:`{self.namespace}`) ON (n.source_id)"
                )          
                logger.info("Neo4j indexes created successfully")                
        except Exception as e:
            logger.error(f"Failed to create indexes: {e}")


    async def index_start_callback(self):
        logger.info("Init Neo4j workspace")
        await self._init_workspace()
        # 确保索引存在
        await self.ensure_indexes()

    async def has_node(self, node_id: str) -> bool:
        async with self.async_driver.session() as session:
            result = await session.run(
                f"MATCH (n:`{self.namespace}`) WHERE n.id = $node_id RETURN COUNT(n) > 0 AS exists",
                node_id=node_id,
            )
            record = await result.single()
            return record["exists"] if record else False

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        async with self.async_driver.session() as session:
            result = await session.run(
                f"""
                MATCH (s:`{self.namespace}`)
                WHERE s.id = $source_id
                MATCH (t:`{self.namespace}`)
                WHERE t.id = $target_id
                RETURN EXISTS((s)-[]->(t)) AS exists
                """,
                source_id=source_node_id,
                target_id=target_node_id,
            )
    
            record = await result.single()
            return record["exists"] if record else False

    async def node_degree(self, node_id: str) -> int:
        results = await self.node_degrees_batch([node_id])
        return results[0] if results else 0
        
    async def node_degrees_batch(self, node_ids: List[str]) -> List[str]:
        """批量获取节点度数"""
        if not node_ids:
            return {}
                    
        result_dict = {node_id: 0 for node_id in node_ids}
        async with self.async_driver.session() as session:
            result = await session.run(
                f"""
                UNWIND $node_ids AS node_id
                MATCH (n:`{self.namespace}`)
                WHERE n.id = node_id
                OPTIONAL MATCH (n)-[]-(m:`{self.namespace}`)
                RETURN node_id, COUNT(m) AS degree
                """,
                node_ids=node_ids
            )
                
            async for record in result:
                result_dict[record["node_id"]] = record["degree"]
                
        return [result_dict[node_id] for node_id in node_ids]
    
    async def edge_degrees_batch(self, edge_pairs: list[tuple[str, str]]) -> list[int]:
        """批量获取边的度数，减少网络往返次数"""
        if not edge_pairs:
            return []
        
        # 创建有序字典，保持边对的顺序
        result_dict = {tuple(edge_pair): 0 for edge_pair in edge_pairs}
        
        # 转换为Neo4j可处理的格式
        edges_params = [{"src_id": src, "tgt_id": tgt} for src, tgt in edge_pairs]
        
        try:
            async with self.async_driver.session() as session:
                result = await session.run(
                    f"""
                    UNWIND $edges AS edge
                    
                    // 查询源节点度数
                    MATCH (s:`{self.namespace}`)
                    WHERE s.id = edge.src_id
                    WITH edge, s
                    OPTIONAL MATCH (s)-[]-(n1:`{self.namespace}`)
                    WITH edge, COUNT(n1) AS src_degree
                    
                    // 查询目标节点度数
                    MATCH (t:`{self.namespace}`)
                    WHERE t.id = edge.tgt_id
                    WITH edge, src_degree, t
                    OPTIONAL MATCH (t)-[]-(n2:`{self.namespace}`)
                    WITH edge.src_id AS src_id, edge.tgt_id AS tgt_id, src_degree, COUNT(n2) AS tgt_degree
                    
                    // 返回两节点度数之和
                    RETURN src_id, tgt_id, src_degree + tgt_degree AS degree
                    """,
                    edges=edges_params
                )
                
                async for record in result:
                    src_id = record["src_id"]
                    tgt_id = record["tgt_id"]
                    degree = record["degree"]
                    
                    # 更新结果字典
                    edge_pair = (src_id, tgt_id)
                    result_dict[edge_pair] = degree
            
            # 返回字典值的列表，保持输入顺序
            return [result_dict[tuple(edge_pair)] for edge_pair in edge_pairs]
        except Exception as e:
            logger.error(f"Error in batch edge degree calculation: {e}")
            return [0] * len(edge_pairs)  # 出错时返回0度

    # 保留原始的单一查询函数，但内部调用批量函数
    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        """获取单个边的度数（内部调用批量函数）"""
        results = await self.edge_degrees_batch([(src_id, tgt_id)])
        return results[0] if results else 0

    async def get_node(self, node_id: str) -> Union[dict, None]:
        result = await self.get_nodes_batch([node_id])
        return result[0] if result else None

    async def get_nodes_batch(self, node_ids: list[str]) -> dict[str, Union[dict, None]]:
        """批量获取节点，减少网络往返次数"""
        if not node_ids:
            return {}
            
        result_dict = {node_id: None for node_id in node_ids}

        try:
            async with self.async_driver.session() as session:
                result = await session.run(
                    f"""
                    UNWIND $node_ids AS node_id
                    MATCH (n:`{self.namespace}`)
                    WHERE n.id = node_id
                    RETURN node_id, properties(n) AS node_data
                    """,
                    node_ids=node_ids
                )
                
                async for record in result:
                    node_id = record["node_id"]
                    raw_node_data = record["node_data"]
                    
                    if raw_node_data:
                        raw_node_data["clusters"] = json.dumps(
                            [
                                {
                                    "level": index,
                                    "cluster": cluster_id,
                                }
                                for index, cluster_id in enumerate(
                                    raw_node_data.get("communityIds", [])
                                )
                            ]
                        )
                        result_dict[node_id] = raw_node_data
            # 返回字典值的列表，保持与输入node_ids相同的顺序
            return [result_dict[node_id] for node_id in node_ids]
        except Exception as e:
            logger.error(f"Error in batch node retrieval: {e}")
            raise e

    async def get_edges_batch(
        self, edge_pairs: list[tuple[str, str]]
    ) -> list[Union[dict, None]]:
        """批量获取边属性，减少网络往返次数"""
        if not edge_pairs:
            return []
            
        # 创建有序字典，保持边对的顺序
        result_dict = {tuple(edge_pair): None for edge_pair in edge_pairs}
        
        # 转换为Neo4j可处理的格式
        edges_params = [{"source_id": src, "target_id": tgt} for src, tgt in edge_pairs]
        
        try:
            async with self.async_driver.session() as session:
                result = await session.run(
                    f"""
                    UNWIND $edges AS edge
                    MATCH (s:`{self.namespace}`)-[r]->(t:`{self.namespace}`)
                    WHERE s.id = edge.source_id AND t.id = edge.target_id
                    RETURN edge.source_id AS source_id, edge.target_id AS target_id, properties(r) AS edge_data
                    """,
                    edges=edges_params
                )
                
                async for record in result:
                    source_id = record["source_id"]
                    target_id = record["target_id"]
                    edge_data = record["edge_data"]
                    
                    # 更新结果字典
                    edge_pair = (source_id, target_id)
                    result_dict[edge_pair] = edge_data
            
            # 返回字典值的列表，保持输入顺序
            return [result_dict[tuple(edge_pair)] for edge_pair in edge_pairs]
        except Exception as e:
            logger.error(f"Error in batch edge retrieval: {e}")
            return [None] * len(edge_pairs)

    async def get_nodes_edges_batch(
        self, node_ids: list[str]
    ) -> list[list[tuple[str, str]]]:
        """批量获取多个节点的出边，减少网络往返次数"""
        if not node_ids:
            return []
            
        # 创建有序字典，保持节点ID的顺序
        result_dict = {node_id: [] for node_id in node_ids}
        
        try:
            async with self.async_driver.session() as session:
                result = await session.run(
                    f"""
                    UNWIND $node_ids AS node_id
                    MATCH (s:`{self.namespace}`)-[r]->(t:`{self.namespace}`)
                    WHERE s.id = node_id
                    RETURN s.id AS source_id, t.id AS target_id
                    """,
                    node_ids=node_ids
                )
                
                async for record in result:
                    source_id = record["source_id"]
                    target_id = record["target_id"]
                    
                    # 将边添加到相应的节点列表中
                    if source_id in result_dict:
                        result_dict[source_id].append((source_id, target_id))
            
            # 返回字典值的列表，保持输入顺序
            return [result_dict[node_id] for node_id in node_ids]
        except Exception as e:
            logger.error(f"Error in batch node edges retrieval: {e}")
            return [[] for _ in node_ids]  # 返回空列表列表

    # 保留原始的单一查询函数，但内部调用批量函数
    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> Union[dict, None]:
        """获取单个边的属性（内部调用批量函数）"""
        results = await self.get_edges_batch([(source_node_id, target_node_id)])
        return results[0] if results else None

    async def get_node_edges(
        self, source_node_id: str
    ) -> list[tuple[str, str]]:
        """获取单个节点的所有出边（内部调用批量函数）"""
        results = await self.get_nodes_edges_batch([source_node_id])
        return results[0] if results else []

    async def upsert_nodes_batch(self, nodes_data: list[tuple[str, dict[str, str]]]):
        """批量更新/创建节点，减少网络往返次数
        
        参数:
            nodes_data: 节点数据列表，每项为 (node_id, node_data) 元组
        """
        if not nodes_data:
            return []
        
        # 按节点类型分组，以避免动态标签问题
        nodes_by_type = {}
        for node_id, node_data in nodes_data:
            node_type = node_data.get("entity_type", "UNKNOWN").strip('"')
            if node_type not in nodes_by_type:
                nodes_by_type[node_type] = []
            nodes_by_type[node_type].append((node_id, node_data))
        
        # 对每种类型的节点分别执行批量操作
        async with self.async_driver.session() as session:
            for node_type, type_nodes in nodes_by_type.items():
                # 构建该类型的参数
                params = [{"id": node_id, "data": node_data} for node_id, node_data in type_nodes]
                
                # 执行批量更新
                await session.run(
                    f"""
                    UNWIND $nodes AS node
                    MERGE (n:`{self.namespace}`:`{node_type}` {{id: node.id}})
                    SET n += node.data
                    """,
                    nodes=params
                )
        

    # 保留原始的单一插入函数，但内部调用批量函数
    async def upsert_node(self, node_id: str, node_data: dict[str, str]):
        """插入或更新单个节点（内部调用批量函数）"""
        await self.upsert_nodes_batch([(node_id, node_data)])

    async def upsert_edges_batch(
        self, edges_data: list[tuple[str, str, dict[str, str]]]
    ):
        """批量更新/创建边，减少网络往返次数
        
        参数:
            edges_data: 边数据列表，每项为 (source_node_id, target_node_id, edge_data) 元组
        """
        if not edges_data:
            return
        
        # 处理边数据
        edges_params = []
        for source_id, target_id, edge_data in edges_data:
            # 确保每条边都有权重属性
            edge_data_copy = edge_data.copy()  # 创建副本避免修改原始数据
            edge_data_copy.setdefault("weight", 0.0)
            
            edges_params.append({
                "source_id": source_id,
                "target_id": target_id,
                "edge_data": edge_data_copy
            })
        
        # 批量执行边更新/创建
        async with self.async_driver.session() as session:
            await session.run(
                f"""
                UNWIND $edges AS edge
                MATCH (s:`{self.namespace}`)
                WHERE s.id = edge.source_id
                WITH edge, s
                MATCH (t:`{self.namespace}`)
                WHERE t.id = edge.target_id
                MERGE (s)-[r:RELATED]->(t)
                SET r += edge.edge_data
                """,
                edges=edges_params
            )
        

    # 保留原始的单一插入函数，内部调用批量函数
    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ):
        """插入或更新单个边（内部调用批量函数）"""
        await self.upsert_edges_batch([(source_node_id, target_node_id, edge_data)])


    # async def upsert_node(self, node_id: str, node_data: dict[str, str]):
    #     node_type = node_data.get("entity_type", "UNKNOWN").strip('"')
    #     async with self.async_driver.session() as session:
    #         query = f"""
    #             MERGE (n:`{self.namespace}`:`{node_type}` {{id: $node_id}})
    #             SET n += $node_data
    #         """
    #         await session.run(query, node_id=node_id, node_data=node_data)

    # async def upsert_edge(
    #     self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    # ):
    #     edge_data.setdefault("weight", 0.0)
    #     async with self.async_driver.session() as session:
    #         await session.run(
    #             f"""
    #             MATCH (s:`{self.namespace}`)
    #             WHERE s.id = $source_id
    #             WITH s
    #             MATCH (t:`{self.namespace}`)
    #             WHERE t.id = $target_id
    #             MERGE (s)-[r:RELATED]->(t)
    #             SET r += $edge_data
    #             """,
    #             source_id=source_node_id,
    #             target_id=target_node_id,
    #             edge_data=edge_data,
    #         )

    async def get_all_nodes(self) -> list[dict]:
        """
        获取所有 `self.namespace` 标签下的节点并返回。
        """
        async with self.async_driver.session() as session:
            result = await session.run(
                f"""
                MATCH (n:`{self.namespace}`)
                RETURN properties(n) AS node_data
                """
            )
            all_nodes = []
            async for record in result:
                all_nodes.append(record["node_data"])
        return all_nodes


    async def get_all_edges(self) -> list[dict]:
        """
        获取所有 `self.namespace` 标签下的边，并以列表的形式返回边的属性字典，并附带起始/结束节点ID。
        """
        async with self.async_driver.session() as session:
            result = await session.run(
                f"""
                MATCH (s:`{self.namespace}`)-[r]->(t:`{self.namespace}`)
                RETURN s.id AS src_name, t.id AS tgt_name, properties(r) AS edge_data
                """
            )
            all_edges = []
            async for record in result:
                edge_info = {
                    "src_name": record["src_name"],
                    "tgt_name": record["tgt_name"],
                    **record["edge_data"],
                }
                all_edges.append(edge_info)
        return all_edges


    async def clustering(self, algorithm: str):
        if algorithm != "leiden":
            raise ValueError(
                f"Clustering algorithm {algorithm} not supported in Neo4j implementation"
            )

        random_seed = self.global_config["graph_cluster_seed"]
        max_level = self.global_config["max_graph_cluster_size"]
        async with self.async_driver.session() as session:
            try:
                # Project the graph with undirected relationships
                # 在 Neo4j Graph Data Science (GDS) 库中创建一个内存中的图投影
                await session.run(
                    f"""
                    CALL gds.graph.project(
                        'graph_{self.namespace}',
                        ['{self.namespace}'],
                        {{
                            RELATED: {{
                                orientation: 'UNDIRECTED',
                                properties: ['weight']
                            }}
                        }}
                    )
                    """
                )

                # Run Leiden algorithm
                # 较小的gamma和较小的theta会产生较大的社区
                result = await session.run(
                    f"""
                    CALL gds.leiden.write(
                        'graph_{self.namespace}',
                        {{
                            writeProperty: 'communityIds',
                            includeIntermediateCommunities: True,
                            relationshipWeightProperty: "weight",
                            maxLevels: {max_level},
                            tolerance: 0.0001,
                            gamma: 1.0,
                            theta: 0.01,
                            randomSeed: {random_seed}
                        }}
                    )
                    YIELD communityCount, modularities;
                    """
                )
                result = await result.single()
                # community_count: int = result["communityCount"]
                modularities = result["modularities"]
            finally:
                # Drop the projected graph
                # 从内存中删除之前创建的图投影
                await session.run(f"CALL gds.graph.drop('graph_{self.namespace}')")
                result = await self.community_schema()
                levels = {}
                for c in result.values():
                    levels[c["level"]] = levels.get(c["level"], 0) + 1
                # 按key值排序levels
                levels = dict(sorted(levels.items(), key=lambda item: item[0]))
                logger.info(f"Community levels: {levels} with modularities {modularities}")

    async def community_schema(self) -> dict[str, SingleCommunitySchema]:
        results = defaultdict(
            lambda: dict(
                level=None,
                title=None,
                edges=set(),
                nodes=set(),
                chunk_ids=set(),
                occurrence=0.0,
                sub_communities=[],
            )
        )
        # 提取每个节点的节点ID、源ID（文档/区块来源）、社区ID数组（之前由 Leiden 算法生成）、与该节点相连的所有节点
        async with self.async_driver.session() as session:
            # Fetch community data
            result = await session.run(
                f"""
                MATCH (n:`{self.namespace}`)
                WITH n, n.communityIds AS communityIds, [(n)-[]-(m:`{self.namespace}`) | m.id] AS connected_nodes
                RETURN n.id AS node_id, n.source_id AS source_id, 
                       communityIds AS cluster_key,
                       connected_nodes
                """
            )

            # records = await result.fetch()
            # max_num_ids是所有社区中拥有最多不同区块 ID 的那个社区的区块数量
            max_num_ids = 0
            async for record in result:
                for index, c_id in enumerate(record["cluster_key"]):
                    node_id = str(record["node_id"])
                    source_id = record["source_id"]
                    level = index
                    cluster_key = str(c_id)
                    connected_nodes = record["connected_nodes"]

                    results[cluster_key]["level"] = level
                    results[cluster_key]["title"] = f"Cluster {cluster_key}"
                    results[cluster_key]["nodes"].add(node_id)
                    results[cluster_key]["edges"].update(
                        [
                            tuple(sorted([node_id, str(connected)]))
                            for connected in connected_nodes
                            if connected != node_id
                        ]
                    )
                    chunk_ids = source_id.split(GRAPH_FIELD_SEP)
                    results[cluster_key]["chunk_ids"].update(chunk_ids)
                    max_num_ids = max(
                        max_num_ids, len(results[cluster_key]["chunk_ids"])
                    )

            # Process results
            for k, v in results.items():
                v["edges"] = [list(e) for e in v["edges"]]
                v["nodes"] = list(v["nodes"])
                v["chunk_ids"] = list(v["chunk_ids"])
                # occurence表示社区覆盖度，描述当前社区覆盖多少不同的区块 ID，即社区的“全面性”
                v["occurrence"] = len(v["chunk_ids"]) / max_num_ids

            # Compute sub-communities (this is a simplified approach)
            # 父社区的level更小，同时子社区的节点集合是父社区的子集
            for cluster in results.values():
                cluster["sub_communities"] = [
                    sub_key
                    for sub_key, sub_cluster in results.items()
                    if sub_cluster["level"] > cluster["level"]
                    and set(sub_cluster["nodes"]).issubset(set(cluster["nodes"]))
                ]

        return dict(results)

    async def index_done_callback(self):
        await self.async_driver.close()

    async def _debug_delete_all_node_edges(self):
        async with self.async_driver.session() as session:
            try:
                # Delete all relationships in the namespace
                await session.run(f"MATCH (n:`{self.namespace}`)-[r]-() DELETE r")

                # Delete all nodes in the namespace
                await session.run(f"MATCH (n:`{self.namespace}`) DELETE n")

                logger.info(
                    f"All nodes and edges in namespace '{self.namespace}' have been deleted."
                )
            except Exception as e:
                logger.error(f"Error deleting nodes and edges: {str(e)}")
                raise

    async def drop_clusters(self):
        pass