
import os
import zipfile
import shutil
import asyncio
import json
import re
import chardet
import logging
import traceback
import importlib
import hashlib
import uvicorn
from pydantic import BaseModel, Field
from websockets.legacy.client import WebSocketClientProtocol
from starlette.websockets import WebSocket as StarletteWebSocket
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, WebSocketDisconnect, WebSocket
from fastapi.responses import FileResponse
from typing import List, Dict, Optional

from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag._storage import NetworkXStorage, Neo4jStorage
from nano_graphrag._utils import read_yaml_config, validate_tiktoken_cache

# 用户输入类，定义需要提取的键和默认值
# 通过继承BaseModel类，可以使用Pydantic库的数据验证功能
class UserInput(BaseModel):
    rebuild_graph: bool = False
    file_path: List[str] = Field(default_factory=list)
    md5_dir: Optional[str] = None
    global_query_text: List[str] = Field(default_factory=list)
    local_query_text: List[str] = Field(default_factory=list)
    table_enhance_factor: int = 2
    show_query_process: bool = False
    enable_websocket: bool = True
    build_community_report: bool = True


app = FastAPI()
graph_func = None

# 设置日志配置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebSocketManager:
    def __init__(self):
        # 这里不用加异步锁了，fastapi会自动处理并发问题
        self.clients = set()

    async def connect(self, websocket: WebSocket):
        if websocket not in self.clients:
            await websocket.accept()
            self.clients.add(websocket)
        logger.info("WebSocket client connected")

    async def disconnect(self, websocket):
        if websocket in self.clients:
            self.clients.remove(websocket)
        logger.info("WebSocket client disconnected")

    async def send(self, message: str):
        for client in self.clients:
            await client.send_text(message)

websocket_manager = WebSocketManager()

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    # 接受txt、json、zip文件
    # 可以命令行调用：curl -F "file=@C:\LabProject\fresh_pack\nano-GRAG-merge\pdf5.md" http://localhost:8004/upload/

    upload_dir = './uploaded_files/'
    os.makedirs(upload_dir, exist_ok=True)
    
    file_location = f"{upload_dir}{file.filename}"
    file_extension = file.filename.split('.')[-1].lower()  # 获取文件扩展名，转换为小写

    # 写入文件到上传目录
    with open(file_location, "wb+") as f:
        shutil.copyfileobj(file.file, f)
    new_uploaded = []
    if file_extension in ['json', 'txt', 'md']:
        new_uploaded.append(file_location)
    elif file_extension == 'zip':
        try:
            # 解压缩文件，忽略ZIP内部结构
            with zipfile.ZipFile(file_location, 'r') as zip_ref:
                # 提取所有文件，忽略目录结构
                for member in zip_ref.infolist():
                    # 检查文件是否在zip文件的根目录
                    if member.filename[-1] == '/':
                        continue  # 这是一个目录，跳过
                    source = zip_ref.open(member)
                    target_file_path = os.path.join(upload_dir, os.path.basename(member.filename))
                    with open(target_file_path, "wb") as target:
                        shutil.copyfileobj(source, target)
                    source.close()
                    new_uploaded.append(target_file_path)
            logger.info(f"File {file.filename} uploaded and extracted successfully.")

        except zipfile.BadZipFile:
            os.remove(file_location)  # 如果解压失败，删除损坏的zip文件
            return HTTPException(status_code=400, detail="Invalid uploaded zip file.")
    else:
        os.remove(file_location)  # 删除不支持的文件类型
        return HTTPException(status_code=400, detail=f"Unsupported file type '{file_extension}'.")

    # 清除zip源文件，如果是zip类型
    if file_extension == 'zip':
        os.remove(file_location)

    path_dict =  {"message":"本次成功上传文件：", "file_paths": [os.path.join(upload_dir, f) for f in os.listdir(upload_dir) if os.path.isfile(os.path.join(upload_dir, f)) and os.path.join(upload_dir, f) in new_uploaded]}
    # logger.info(path_dict)
    return path_dict

# 压缩并返回文件夹
@app.get("/download/")
async def download_files(download_filename: str = Query(default="graphrag_return.zip", description="Name of the file to download")):    
    # 可以命令行调用
    # curl -o "graphrag_return.zip" -J "http://localhost:8000/download/?filename=graphrag_return.zip" && powershell -command "Expand-Archive -Path 'graphrag_return.zip' -DestinationPath 'temp_graphrag'; if (-Not (Test-Path 'graphrag_return')) { New-Item -Path 'graphrag_return' -ItemType Directory }; Get-ChildItem -Path 'temp_graphrag\*' -Recurse | Move-Item -Destination 'graphrag_return' -Force; Remove-Item 'temp_graphrag' -Recurse -Force; Remove-Item 'graphrag_return.zip'"

    folder_path = './graphrag_dir'
    zip_filename = './graphrag_dir.zip'
    
    if not os.path.exists(folder_path):
        return HTTPException(status_code=400, detail=f"GraphRAG建图目录不存在 '{folder_path}'.")
    
    # 创建 ZIP 文件并添加目录下的所有文件
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # os.walk 生成目录树下的文件名
        for dirpath, dirnames, filenames in os.walk(folder_path):
            for filename in filenames:
                # 文件的完整路径
                file_path = os.path.join(dirpath, filename)
                # 为文件在 ZIP 文件中定义的路径（保留了文件的层级结构）
                arcname = os.path.relpath(file_path, os.path.dirname(folder_path))
                # 将文件写入到 ZIP 文件中
                zipf.write(file_path, arcname)
    
    return FileResponse(zip_filename, media_type='application/zip', filename=download_filename)

@app.get("/print_md5_dir/")
async def print_md5_dir():
    def is_md5(md5_string):
        return (re.compile(r'^[a-fA-F0-9]{32}$')).match(md5_string) is not None
    
    directory = './graphrag_dir/'
    md5_dict = {}
    if os.path.exists(directory):
        md5_dir_list = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f)) and is_md5(f)]
        for i in md5_dir_list:
            with open(os.path.join(directory, i, 'last_build_graph.json'), 'r', encoding='utf-8') as f:
                # 时间戳处理，从第33位开始是时间戳
                md5_dict[i] = {k: (v[33:].strip() if v[33:].strip() != "" else None) for k, v in json.load(f).items()}

    return {
        "message": f"{directory}下的MD5文件夹及其建图使用文件",
        "md5_dir": md5_dict,
    }

@app.get("/list_uploaded_files/")
async def list_uploaded_files():
    directory = './uploaded_files/'
    files_info = {'message':'当前已上传文件及其MD5码', 'files':{}}
    # 确保目录存在
    if not os.path.exists(directory):
        return files_info

    # 遍历目录
    for filename in os.listdir(directory):
        if filename.endswith('.txt') or filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'rb') as file:
                file_data = file.read()
                md5_hash = hashlib.md5(file_data).hexdigest()
            files_info['files'][f"@container://{filename}"] = md5_hash

    return files_info

@app.delete("/delete_uploaded_file/")
async def delete_uploaded_file(filename: str):
    directory = './uploaded_files/'
    file_path = os.path.join(directory, filename)

    # 检查文件是否存在
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found")

    # 删除文件
    os.remove(file_path)
    return {"message": f"File '{filename}' has been deleted."}

@app.delete("/delete_all_uploads/")
async def delete_all_uploads():
    directory = './uploaded_files/'

    # 检查目录是否存在
    if not os.path.exists(directory):
        return {"message": "No files to delete."}

    # 删除目录下的所有文件
    shutil.rmtree(directory)
    return {"message": "All uploaded files have been deleted."}

@app.delete("/delete_server_zip/")
async def delete_server_zip():
    zip_filename = './graphrag_dir.zip'
    # 检查文件是否存在
    if not os.path.isfile(zip_filename):
        return {"message": "No server zip file to delete."}
    # 删除文件
    os.remove(zip_filename)
    return {"message": "Server zip file has been deleted."}


class SearchChunkRequest(BaseModel):
    keywords: List[str]
@app.post("/search_chunks/")
async def search_chunks(request: SearchChunkRequest) -> Dict[str, str]:
    try:
        global graph_func
        if graph_func is None:
            raise HTTPException(status_code=400, detail="GraphRAG instance does not exist. Please use process_query to .")
            # logger.info("GraphRAG instance does not exist. Creating new instance.")        
            # rebuild_graph = False
            # file_paths = request.file_paths
            # md5_dir = request.md5_dir
            # if md5_dir is None and file_paths is None:
            #     raise HTTPException(status_code=400, detail="必须提供 md5_dir 或 file_paths 之一！")
            # table_enhance_factor: int = 2
            # working_dir = os.path.join(os.getcwd(), 'graphrag_dir')

            # contents, overall_hash = await load_files(working_dir, rebuild_graph, file_paths, md5_dir)
            # graph_func = GraphRAG(working_dir=os.path.join(working_dir, overall_hash), use_embedding_func='openai', use_conversation_func='openai', table_enhance_factor=table_enhance_factor)
        search_result = await graph_func.search_chunks_with_keywords(request.keywords)
        return search_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class UpsertEntityRequest(BaseModel):
    entity: str
    chunk_ids: Optional[List[str]] = None
@app.post("/upsert/entity/")
async def upsert_entity(request: UpsertEntityRequest) -> Dict[str, str]:
    try:
        global graph_func
        if graph_func is None:
            raise HTTPException(status_code=400, detail="GraphRAG instance does not exist. Please use process_query to .")
        await graph_func.upsert_entity(request.entity, request.chunk_ids)
        return {"message": "Entity upserted successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class UpsertRelationshipRequest(BaseModel):
    src_entity: str
    dst_entity: str
    chunk_ids: Optional[List[str]] = None
@app.post("/upsert/relationship/")
async def upsert_relationship(request: UpsertRelationshipRequest) -> Dict[str, str]:
    try:
        global graph_func
        if graph_func is None:
            raise HTTPException(status_code=400, detail="GraphRAG instance does not exist. Please use process_query to .")
        
        # 调用 graph_func 的 upsert_relationship 方法
        await graph_func.upsert_relationship(
            src_entity_name=request.src_entity,
            dst_entity_name=request.dst_entity,
            source_chunks=request.chunk_ids
        )
        return {"message": "Relationship upserted successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 处理查询并通过 WebSocket 返回结果
async def process_query(input_params: UserInput, yaml_config: Dict):
    overall_hash = None
    try:
        if isinstance(input_params.global_query_text, str):
            input_params.global_query_text = [input_params.global_query_text]
        if isinstance(input_params.local_query_text, str):
            input_params.local_query_text = [input_params.local_query_text]
        input_params.global_query_text = [q.strip() for q in input_params.global_query_text if len(q.strip())]
        input_params.local_query_text = [q.strip() for q in input_params.local_query_text if len(q.strip())]
        working_dir = os.path.join(os.getcwd(), 'graphrag_dir')
        os.makedirs(working_dir, exist_ok=True)
        
        websocket_url = f"ws://localhost:{service_port_number}/ws" if input_params.enable_websocket else None
        # 使用 FastAPI WebSocket 地址，默认端口8000
        # await 确保你的程序会等待前一个异步调用完成后才继续执行，但是它并不保证立即得到响应或处理结果
        # 尽管使用了 await，WebSocket 服务器处理连接和响应可能存在延迟，导致发送的消息顺序和接收的消息顺序不一致
        # 通过if xxx: pass，使得外部程序等待GraphRAG的处理完成、输出返回值后再继续运行下一步
        # 这样可以保证外部程序的message等到GraphRAG一整个阶段完成后再发送
        # 这样内外部消息顺序就不会错乱了

        # 加载文件内容并计算总哈希值
        # 下面的两个func可以选择“local”或者“openai”
        # 注意建图和查询需要使用同一个embedding_func
        contents, overall_hash = await load_files(working_dir, input_params.rebuild_graph, input_params.file_path, input_params.md5_dir)
        global graph_func

        neo4j_config = {
            "neo4j_url": yaml_config['neo4j_settings']['NEO4J_URL'],
            "neo4j_auth": (
                yaml_config['neo4j_settings']['NEO4J_USER'],
                str(yaml_config['neo4j_settings']['NEO4J_PASSWORD']),
            )
        }
        
        if yaml_config['graph_storage_cls'] == 'networkx':
            graph_storage_cls = NetworkXStorage
            addon_params = {"to_neo4j": False}
        else:
            if yaml_config['neo4j_settings']['GENERATE_FROM_NETWORKX']:
                # neo4j只是用于可视化、不用于建图以及查询
                graph_storage_cls = NetworkXStorage
                addon_params = {**neo4j_config, "to_neo4j": True}
            else:
                # neo4j用于建图和查询
                graph_storage_cls = Neo4jStorage
                addon_params = {**neo4j_config, "to_neo4j": False}

        graph_func = GraphRAG(
            working_dir=os.path.join(working_dir, overall_hash), 
            use_embedding_func=yaml_config['use_embedding_func'], 
            use_conversation_func=yaml_config['use_conversation_func'],
            graph_storage_cls=graph_storage_cls,
            addon_params=addon_params,
            build_community_report=input_params.build_community_report
        )
        if await graph_func.setup_websocket(websocket_url=websocket_url):
            pass
        if len(contents):
            if await graph_func.ainsert(contents):
                pass
        if input_params.show_query_process:
            await websocket_manager.send(f"已完成建图！")
        logger.info(f"已完成建图！")

        # # 如果需要单独做networkx2neo4j的可视化
        # if await graph_func.nkx2n4j():
        #     pass

        # del graph_func
        # # 重新创建 GraphRAG 实例，这样可以在查询时更换不同的conversation_func
        # graph_func = GraphRAG(working_dir=os.path.join(working_dir, overall_hash), use_embedding_func='openai', use_conversation_func='openai', table_enhance_factor=table_enhance_factor)
        # # 执行查询
        if len(input_params.global_query_text):
            for query in input_params.global_query_text:
                if await graph_func.aquery(query, param=QueryParam(show_query_process=input_params.show_query_process)):pass
        if len(input_params.local_query_text):
            for query in input_params.local_query_text:
                if await graph_func.aquery(query, param=QueryParam(mode="local", top_k=30, table_enhance_factor=int(input_params.table_enhance_factor), show_query_process=input_params.show_query_process)):pass
        
    except Exception as e:
        try:
            # 刷新websocket缓冲区，确保客户端能够看到错误信息
            if await graph_func.end_process():pass
        except:
            pass
        e = traceback.format_exc()
        logger.error(e)
        await websocket_manager.send(e)
    finally:
        try:
            await asyncio.sleep(2)
            await graph_func.graphrag_done()
            await asyncio.sleep(2)
            # del graph_func
        except:
            pass
        finally:
            return overall_hash

async def get_graphrag_instance():
    global graph_func
    if graph_func is None:
        raise HTTPException(status_code=400, detail="GraphRAG instance does not exist. Please use process_query to .")
    return graph_func

async def get_fastapi_app():
    return app

async def load_files(working_dir, rebuild_graph, file_path, md5_dir=None):
    """读取文件内容并计算当前使用文件的总哈希值"""
    def extract_txt_md_content(file_path):
        """读取txt或md文件内容"""
        with open(file_path, 'rb') as f:
            raw_data = f.read()
        encoding = chardet.detect(raw_data)['encoding']
        encoding = 'utf-8' if encoding is None else encoding
        file_content = raw_data.decode(encoding)
        return file_content

    def extract_json_content(data):
        """递归提取 JSON 中的所有 content 字段并拼接"""
        text_output = ""
        # 提取当前层级的 content 字段
        if 'content' in data and data['content']:
            text_output += data['content'].strip() + "\n"  # 拼接并添加换行
        # 递归处理子数据
        if 'subdata' in data and data['subdata']:
            for subitem in data['subdata']:
                text_output += extract_json_content(subitem)  # 递归调用处理子数据
        return text_output
    
    def file_hash(file):
        """计算单个文件的MD5哈希值"""
        with open(file, 'rb') as f:
            file_data = f.read()
            return hashlib.md5(file_data).hexdigest()
    
    def load_file_contents(file_hash_dict, force_content=False):
        """加载文件内容并更新file_hash_dict，返回需要新建图的文件内容列表和更新后的哈希字典"""
        file_contents = []
        # 逐个读取本次插入文件内容
        for file in file_path:
            if file.startswith('@container://'):
                file = file.replace('@container://', './uploaded_files/')
            assert os.path.exists(file), f"文件 {file} 不存在！"
            if file.endswith('.txt') or file.endswith('.md'):
                file_content = extract_txt_md_content(file)
            elif file.endswith('.json'):
                with open(file, "r", encoding="utf-8") as f:
                    json_data = json.load(f)
                    file_content = extract_json_content(json_data)
            else:
                raise Exception(f"Unsupported file type: {file}")
            
            file_hash_value = file_hash(file)
            already_hash = [i[:32] for i in file_hash_dict.values()]
            if file_hash_value not in already_hash:
                # 哈希值不同的文件被加入
                file_hash_dict[file] = f"{file_hash_value}@{current_time}"
                file_contents.append(file_content)
            else:
                # 某个文件的哈希值已经存在，说明这个文件已经被处理过了，更新其文件名
                logger.warning(f"文件 {file} 的哈希值已存在！")
                if force_content:
                    file_contents.append(file_content)
                # 这里使用list(file_hash_dict.items())是为了避免在遍历时直接修改字典
                for k, v in list(file_hash_dict.items()):
                    if v[:32] == file_hash_value:
                        file_hash_dict.pop(k)  # 删除当前的键值对
                        file_hash_dict[file] = f"{file_hash_value}@{current_time}"
                        # 同步更新总体overall_hash
                        file_hash_dict['overall_hash'] = f"{file_hash_dict['overall_hash'][:32]}@{current_time}"
        return file_contents, file_hash_dict

    file_hash_dict = {}     # 文件哈希值字典. key: 文件路径, value: 文件哈希值@时间戳
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    if rebuild_graph:
        # 重建图时，重新计算哈希
        assert md5_dir is None, "重建图时不应指定MD5工作区！请设置md5_dir=None。"
        md5_dir = hashlib.md5(current_time.encode()).hexdigest()
        file_contents, file_hash_dict = load_file_contents(file_hash_dict)
        file_hash_dict['overall_hash'] = f"{md5_dir}@{current_time}"
        await websocket_manager.send(f"Working in MD5_DIR == {md5_dir}")
    else:
        # 不重新建图
        if md5_dir is not None:
            if os.path.exists(os.path.join(working_dir, md5_dir)) and os.path.exists(os.path.join(working_dir, md5_dir, 'last_build_graph.json')):
                # 指定了MD5工作区且存在，读取MD5工作区的记录文件
                # 且不改变MD5工作区名称
                with open(os.path.join(working_dir, md5_dir, 'last_build_graph.json'), 'r', encoding='utf-8') as old_f:
                    file_hash_dict = json.load(old_f)
                # 这里设置force_content=True是因为rebuild_graph=False、md5_dir指定且存在、file_path指定且存在时，可能存在上次建图失败、需要insert_from_fail的情况
                # 所以这里强制返回文件内容，让graphrag自行判断是否需要insert_from_fail
                file_contents, file_hash_dict = load_file_contents(file_hash_dict, force_content=True)
            else:
                if not os.path.exists(os.path.join(working_dir, md5_dir)):
                    raise ValueError(f"指定的MD5工作区{md5_dir}不存在！")
                if not os.path.exists(os.path.join(working_dir, md5_dir, 'last_build_graph.json')):
                    raise ValueError(f"指定的MD5工作区{md5_dir}中的记录文件不存在！")
        else:
            # 未指定MD5工作区
            raise ValueError(f"未指定md5_dir工作区！必须指定一个存在的md5_dir工作区以供增量插入或查询！\n如果需要对当前输入文件单独创建工作区，请指定参数rebuild_graph=True。")
    file_hash_dict = {k: v for k, v in sorted(file_hash_dict.items(), key=lambda item: item[0])}

    # 新建MD5工作区（若需要）
    os.makedirs(os.path.join(working_dir, md5_dir), exist_ok=True)

    # 保存文件哈希值记录
    with open(os.path.join(working_dir, md5_dir, 'last_build_graph.json'), 'w', encoding='utf-8') as new_f:
        json.dump(file_hash_dict, new_f, indent=4, ensure_ascii=False)

    logger.info(f"Working in MD5_DIR == {md5_dir}")

    return file_contents, md5_dir

def extract_and_log(data, input_class: UserInput):
    # 通过类初始化，将输入字典映射为类实例并传参
    extracted_data = input_class(**{key: data.get(key, getattr(input_class(), key)) for key in input_class.__annotations__})
    logger.info(
        "接收到用户输入: " +
        ", ".join(f"{key}: {value}" for key, value in extracted_data.__dict__.items()) + "\n\n"
    )
    return extracted_data



@app.websocket("/ws")
async def websocket_handler(websocket: WebSocket):
    """用于注册、管理、注销客户端连接，并处理客户端消息"""
    # Websocket客户端逻辑
    # 每个客户端连接到 WebSocket 服务器时，websockets.serve 自动接受连接，并将这个连接封装成一个 websocket 对象传给 websocket_handler 函数
    # 对于每个客户端连接，服务器都会在后台运行一个独立的 websocket_handler 协程。这允许服务器与每个客户端保持独立的、持续的双向通信
    # 可以在 websocket_handler 函数中通过设定初始化消息或检查指定标志的方式，以不同方式处理不同客户端的消息
    # 在本项目中，客户端暂时就两个：GraphRAG类内的 self.websocket 和外部程序 client.py
    
    # 出于安全考虑，一般来说每个客户端只能看到自己和服务器的私聊，不应该看到其他客户端的消息
    # 即客户端连接到服务器后，只能看到服务器发给自己的消息，不应该看到其他任何消息
    # 在 WebSocket 协议中，通信是双向的，但是每个消息的传递必须由一端显式地发送到另一端
    # 这意味着如果服务器没有明确发送消息给特定客户端，那么该客户端将不会接收到任何信息
    # 所以如果client.py想看到GraphRAG发的消息，就需要通过websocket_manager做消息广播

    client_address = f"{websocket.client.host}:{websocket.client.port}"
    logger.info(f"New WebSocket connection: {client_address}")
    await websocket_manager.connect(websocket)
    try:
        while True: # 持续监听消息
            message = (await websocket.receive_text()).strip()
            if message.startswith("[user_input]"):
                # 解析消息以提取参数
                match = re.search(r'\[user_input\](.*?)\[/user_input\]', message)
                if match:
                    try:
                        data = json.loads(match.group(1))
                        current_md5_dir = await process_query(extract_and_log(data, UserInput), yaml_config=yaml_config)
                    except WebSocketDisconnect:
                        logger.info(f"WebSocket connection closed by client: {client_address}")
                        await websocket_manager.disconnect(websocket)
                    except:
                        e = traceback.format_exc()
                        logger.error(e)
                        await websocket_manager.send(e)
                    finally:
                        break
                else:
                    logger.error(f"Error: 无法解析用户输入！{message}\n")
                    await websocket_manager.send(f"Error: 无法解析用户输入！{message}\n")
            elif (message == "---END OF STAGE---"):
                # GraphRAG 一个阶段完成
                # 这里只需要给websocket(GraphRAG侧)发送一个消息，告诉GraphRAG可以继续下一个阶段了
                # 不需要给websocket_manager发送消息，不需要让客户端看到这个消息
                # 通过这个ack，可以让外部程序的message等到GraphRAG一整个阶段完成后再发送
                # 这里只回给websocket而不是websocket_manager，就是只回给了这条消息的原始发送者
                # FastAPI/Starlette库的发送的方法是send_text，websockets库的发送方法是send
                if isinstance(websocket, WebSocketClientProtocol):
                    await websocket.send(f"[#-ack-#]{message.strip()}[/#-ack-#]")  # 使用 websockets 库的方法
                elif isinstance(websocket, StarletteWebSocket):
                    await websocket.send_text(f"[#-ack-#]{message.strip()}[/#-ack-#]")  # 使用 FastAPI/Starlette 的方法
            else:
                await websocket_manager.send(f'{message}\n')
    except WebSocketDisconnect:
        logger.info(f"WebSocket connection closed by client: {client_address}")
        await websocket_manager.disconnect(websocket)
    except Exception as e:
        if 'WebSocket is not connected. Need to call "accept" first' in str(e) or 'once a close message has been sent' in str(e):
            logger.error(f"WebSocket connection closed by client: {client_address}")
            await websocket_manager.disconnect(websocket)
        else:
            e = traceback.format_exc()
            logger.error(e)
            await websocket_manager.send(e)
    finally:
        await asyncio.sleep(3)
        if websocket.client_state != "Disconnected":
            await websocket_manager.disconnect(websocket)

if __name__ == "__main__":

    # 也可以使用启动命令
    # uvicorn app_new:app --host 0.0.0.0 --port 8000 --reload --log-level info
    # 当前提供的服务：上传文件、列出已上传文件、下载文件、WebSocket服务、重置用户API、打印用户API、删除所有上传文件、删除指定上传文件

    # 检查与配置读取
    assert validate_tiktoken_cache(), "tiktoken缓存目录不存在或不可读，请检查！"
    yaml_config = read_yaml_config()

    # # 设置端口号
    service_port_number = yaml_config['websocket']['port']

    # 用服务器 + websocket启动服务
    # host设置0.0.0.0，可以本地访问、局域网访问、公网访问、容器访问
    # 如果host设置localhost或127.0.0.1，那么只能本地访问，局域网、公网都无法访问
    uvicorn.run(app, host="0.0.0.0", port=service_port_number, log_level="info")

    # 直接启动服务
    # # #TODO: 每次跑之前注意这里rebuild_graph和md5_dir的设置！！！
    # rebuild_graph = False
    # global_query_text = []
    # file_path = []
    # local_query_text = [q['问题'] for q in json.load(open('documents/labeled_queries.json', 'r', encoding='utf-8'))]
    # # local_query_text = []
    # # print(local_query_text)
    # md5_dir = '5ce17013c611d6c0b529d0403b8dddd4'
    # table_enhance_factor: int = 2
    # show_query_process = False
    # current_md5_dir = asyncio.run(process_query(rebuild_graph, file_path, global_query_text, local_query_text, table_enhance_factor=table_enhance_factor, md5_dir=md5_dir, show_query_process=show_query_process, enable_websocket=False))

    # # # 执行评估
    # from evaluation import eva
    # eva(current_md5_dir)
