import asyncio
import shutil
import traceback
from typing import List, Optional
import websockets
import json
import requests
import zipfile
import os
import re

# 为了使用websocket进行流式输出打印，还是要用python开发环境
# 但要额外安装的库可以很少，只需要安装websockets、requests、asyncio即可
# 用curl命令建图和查询的话，无法实现流式输出打印，只能等到任务完成后一次性输出，因此还是采用了python开发环境做客户端
# python 3.11.9
# pip install websockets requests asyncio jsonfixer

async def print_with_pause(message:str, pause:int=0.005):
    """逐字符打印消息，模拟实时打字效果"""
    pause = 0
    if pause > 0:
        for char in message:
            print(char, end='', flush=True)  # 模拟实时打字效果
            await asyncio.sleep(pause)
            print()  # 换行
    else:
        print(message)

async def upload_files(file_path_list):
    """上传文件至容器内，返回容器内的文件解压路径"""
    async def create_zip_with_file_list(file_path_list):
        """
        将 file_path_list 中的所有文件压缩到 zip_file 中。
        
        :param file_path_list: 文件路径列表
        :param zip_file: 生成的 zip 文件路径
        """
        with zipfile.ZipFile(zip_file, 'w') as zipf:
            for file in file_path_list:
                assert os.path.exists(file), f"文件路径 {file} 不存在，请检查路径是否正确"
                if os.path.isfile(file):
                    zipf.write(file, os.path.basename(file))
                else:
                    await print_with_pause(f"Warning: {file} 不是一个有效的文件路径，跳过其压缩...")

    file_path_list = [file_path.strip() for file_path in file_path_list if not file_path.startswith("@container://")]
    if not len(file_path_list):
        await print_with_pause("没有本地文件需要上传")
        return []
    zip_file = "inputs.zip"  # 压缩后的本地文件
    await create_zip_with_file_list(file_path_list)

    url = f"http://{service_ip}:{service_port_number}/upload/"
    with open(zip_file, 'rb') as file_handle:
        files = [('file', (zip_file.split('/')[-1], file_handle, 'application/zip'))]
        response = requests.post(url, files=files)
    if response.status_code == 200:
        # 返回容器内的文件解压路径
        msg = json.dumps(response.json(), indent=4, ensure_ascii=False)
        await print_with_pause(msg)
        os.remove(zip_file)  # 删除压缩文件
        return response.json()['file_paths']
    else:
        await print_with_pause(f"文件上传失败：{response.json()['detail']}")
        return []

async def download_results(zip_path="graphrag_return.zip"):
    """下载GraphRAG输出文件"""

    url = f"http://{service_ip}:{service_port_number}/download/?download_filename={zip_path}"
    response = requests.get(url)
    if response.status_code == 200:
        with open(zip_path, "wb") as f:
            f.write(response.content)
        # 解压缩文件
        target_dir = "./graphrag_return"
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        os.makedirs(target_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)  # 先解压到指定文件夹
        # 移动文件，删除中间目录
        middle_dir = os.path.join(target_dir, "graphrag_dir")
        if os.path.exists(middle_dir):
            for filename in os.listdir(middle_dir):
                shutil.move(os.path.join(middle_dir, filename), target_dir)
            os.rmdir(middle_dir)  # 删除现在空的中间目录
        await print_with_pause(f"GraphRAG输出文件已下载并解压至 {os.path.abspath(target_dir)}")
        if os.path.exists(zip_path):
            os.remove(zip_path)
    else:
        await print_with_pause(f"GraphRAG输出文件下载失败：{response.json()['detail']}")


async def graphrag_service(file_path_list=[], md5_dir=None, global_query_text=[], local_query_text=[], rebuild_graph=False, build_community_report=True):
    """GraphRAG建图与查询服务"""

    assert isinstance(file_path_list, list), "file_path_list必须是一个列表"
    assert isinstance(rebuild_graph, bool), "rebuild_graph必须是一个布尔值"
    if isinstance(global_query_text, str):
        global_query_text = [global_query_text.strip()]
    if isinstance(local_query_text, str):
        local_query_text = [local_query_text.strip()]
    assert global_query_text is None or isinstance(global_query_text, list), "global_query_text必须是一个字符串或字符串列表"
    assert local_query_text is None or isinstance(local_query_text, list), "local_query_text必须是一个字符串或字符串列表"
    if md5_dir == "":
        md5_dir = None
    if md5_dir is not None:
        md5_dir = md5_dir.strip()
        pattern = re.compile(r'^[a-fA-F0-9]{32}$')
        assert isinstance(md5_dir, str) and pattern.match(md5_dir) is not None, "md5_dir必须是一个32位的合法MD5字符串"
        assert md5_dir in (await print_md5_dir(print_info=False)), f"MD5工作区 {md5_dir} 不存在，请检查容器中该工作区是否存在"
    print("开始执行任务...")

    # Step 1: 上传压缩文件
    # uploaded_files是文件上传后在容器内的解压路径
    # uploaded_files = await upload_files(file_path_list=file_path_list)
    # container_files = await list_uploaded_files(print_info=False)
    # for file in file_path_list:
    #     if file.startswith("@container://"):
    #        assert file in container_files, f"容器内文件路径{file}不存在，请检查路径是否正确"
    # build_graph_files = uploaded_files + [file_path.strip() for file_path in file_path_list if file_path.strip().startswith("@container://")]
    # build_graph_files = []
    # rebuild_graph = False
    # global_query_text = []
    # local_query_text = [q['问题'] for q in json.load(open('documents/labeled_queries.json', 'r', encoding='utf-8'))]
    # # print(local_query_text)
    # md5_dir = '66db4cf7c1e9d925e1f2c2bd4b12861a'        # 这是test_cn的建图md5工作区
    # await print_with_pause(f"使用的建图文件路径: {build_graph_files}")

    
    # Step 2: 与WebSocket服务器建立连接，发送用户输入
    input_dict = {
        "rebuild_graph": rebuild_graph,
        "file_path": file_path_list,
        "md5_dir": md5_dir,
        "global_query_text": global_query_text,
        "local_query_text": local_query_text,
        "build_community_report": build_community_report,
    }
    input_json_text = f'[user_input]{json.dumps(input_dict, ensure_ascii = False)}[/user_input]'
    # print(input_json_text)

    url = f"ws://{service_ip}:{service_port_number}/ws"
    send_input = False
    error_orrured = False
    while True:
        try:
            await print_with_pause("尝试连接 WebSocket 服务器...")
            # 建立与 WebSocket 服务器的连接
            async with websockets.connect(url,open_timeout=60,ping_timeout=30, close_timeout=30) as websocket:
                await print_with_pause("已连接到 WebSocket 服务器")
                if not send_input:
                    await websocket.send(input_json_text)
                    send_input = True
                async for message in websocket:
                    if not error_orrured and "---END OF GRAPHRAG PROCESS---" in message:
                        # await print_with_pause("\n任务完成，准备下载GraphRAG输出文件...")
                        await print_with_pause("\n任务完成！")
                        # await download_results(zip_path="graphrag_return.zip")
                        # await delete_server_zip()
                        return  # 任务完成后退出循环
                    else:
                        # 每次接收到完整的消息后，逐字符打印
                        await print_with_pause(message)
                        if "Traceback (most recent call last)" in message:
                            error_orrured = True
        except (websockets.ConnectionClosed, ConnectionRefusedError) as e:
            await print_with_pause(f"连接WebSocket服务器报错: {e}")
            await print_with_pause("尝试重连...请确认WebSocket服务器是否已启动...")
            await asyncio.sleep(5)  # 等待5秒后尝试重新连接

async def delete_server_zip():
    url = f"http://{service_ip}:{service_port_number}/delete_server_zip/"
    response = requests.delete(url)

async def search_chunks(keywords: List[str]):
    """在chunk_list中搜索query_text"""
    url = f"http://{service_ip}:{service_port_number}/search_chunks/"
    payload = {"keywords": keywords}
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()  # 如果状态码不是 200-299，则抛出异常
        return response.json()
    except requests.RequestException as exc:
        print(f"网络请求出错: {exc}")
        return {"error": str(exc)}

async def upsert_entity(entity_name: str, chunk_ids:  Optional[List[str]] = None):
    url = f"http://{service_ip}:{service_port_number}/upsert/entity/"
    payload = {"entity": entity_name, "chunk_ids": chunk_ids}
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()  # 如果状态码不是 200-299，则抛出异常
        return response.json()
    except requests.RequestException as exc:
        print(f"网络请求出错: {exc}")
        return {"error": str(exc)}

async def upsert_relationship(src_entity_name: str, dst_entity_name: str, chunk_ids:  Optional[List[str]] = None):
    url = f"http://{service_ip}:{service_port_number}/upsert/relationship/"
    payload = {"src_entity" : src_entity_name, "dst_entity" : dst_entity_name, "chunk_ids": chunk_ids}
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()  # 如果状态码不是 200-299，则抛出异常
        return response.json()
    except requests.RequestException as exc:
        print(f"网络请求出错: {exc}")
        return {"error": str(exc)}

async def reset_user_api(my_api_url = "", my_api_key = ""):
    '''更改并刷新API URL和KEY'''

    assert my_api_url and my_api_key, "请填写OPENAI API的URL和KEY"
    assert my_api_url.startswith("https://") or my_api_url.startswith("http://"), "API URL必须以'http://'或'https://'开头"
    assert my_api_key.startswith("sk-"), "API KEY必须以'sk-'开头"

    url = f"http://{service_ip}:{service_port_number}/reset_user_api/?api_url={my_api_url}&api_key={my_api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        await print_with_pause(f"新的API URL和KEY已生效！\n当前API URL: {my_api_url}\n当前API KEY: {my_api_key}")
    else:
        await print_with_pause(f"API重置失败，状态码：{response.status_code}， 请检查URL和KEY是否正确后重试")

async def print_user_api():
    '''获取当前API URL和KEY'''

    url = f"http://{service_ip}:{service_port_number}/print_user_api/"
    response = requests.get(url)
    if response.status_code == 200:
        user_api = response.json()
        await print_with_pause(f"当前API URL: {user_api['api_url']}\n当前API KEY: {user_api['api_key']}")

async def print_md5_dir(print_info=True):
    '''获取当前已有MD5工作区'''

    url = f"http://{service_ip}:{service_port_number}/print_md5_dir/"
    response = requests.get(url)
    if response.status_code == 200:
        msg = json.dumps(response.json(), indent=4, ensure_ascii=False) 
        if print_info: await print_with_pause(msg)
        return response.json()["md5_dir"]
    else:
        if print_info: await print_with_pause(f"获取MD5工作区失败，状态码: {response.status_code}")

async def list_uploaded_files(print_info=True):
    '''获取上传过的json和txt文件信息'''

    url = f"http://{service_ip}:{service_port_number}/list_uploaded_files/"
    response = requests.get(url)
    if response.status_code == 200:
        msg = json.dumps(response.json(), indent=4, ensure_ascii=False) 
        if print_info: await print_with_pause(msg)
        return list(response.json()["files"].keys())
    else:
        if print_info: await print_with_pause(f"获取上传文件信息失败，状态码: {response.status_code}")

# 删除指定文件
async def delete_uploaded_file(filename):
    url = f"http://{service_ip}:{service_port_number}/delete_uploaded_file/?filename={filename}"
    response = requests.delete(url)
    if response.status_code == 200:
        msg = json.dumps(response.json(), ensure_ascii=False) 
        await print_with_pause(msg)
    else:
        await print_with_pause(f"删除失败: {response.json()['detail']}")

# 删除所有上传的文件
async def delete_all_files():
    url = f"http://{service_ip}:{service_port_number}/delete_all_uploads/"
    response = requests.delete(url)
    msg = json.dumps(response.json(), ensure_ascii=False) 
    await print_with_pause(msg)


if __name__ == "__main__":
    # 环境配置
    # python 3.11.9
    # pip install websockets requests asyncio jsonfixer

    tasks = {
        "graphrag_service": {
            "description": "GraphRAG建图与查询服务",
        #     "task": lambda: graphrag_service(
        #         # 可以是本地文件或容器内文件，容器内文件路径以@container://开头，如'@container://专利研究.txt'
        #         file_path_list=[],  
        #         md5_dir='5ce17013c611d6c0b529d0403b8dddd4',   # ad77f09ef99c584e519a4ea149df8971
        #         global_query_text=[], 
        #         local_query_text=['简易型MLS 配置跟基本型MLS相比，有什么区别？'], 
        #         # 当rebuild_graph为True时，重新建图；为False时，使用已有图增量插入，并可以通过md5_dir指定工作区来增量插入
        #         # 可选的md5_dir可以通过print_md5_dir查询
        #         rebuild_graph=False,
        #         build_community_report=False,
        #     )
        # },
            "task": lambda: graphrag_service(
                # 可以是本地文件或容器内文件，容器内文件路径以@container://开头，如'@container://专利研究.txt'
                file_path_list=[], 
                # 可选的md5_dir可以通过print_md5_dir查询
                md5_dir='b01636031f7523c0d95208d566fabf87',
                global_query_text=['简易型MLS 配置跟基本型MLS相比，有什么区别？'], 
                # local_query_text=[],
                local_query_text=['简易型MLS 配置跟基本型MLS相比，有什么区别？'], 
                # 当rebuild_graph为True时，重新建图；为False时，使用已有图增量插入，并可以通过md5_dir指定工作区来增量插入
                rebuild_graph=False,
                build_community_report=False
            )
        },
        "print_user_api": {
            "description": "查看当前API URL和KEY服务",
            "task": print_user_api
        },
        "reset_user_api": {
            "description": "更改并刷新API URL和KEY服务",
            "task": lambda: reset_user_api(
                my_api_url="your_api_url_here", 
                my_api_key="your_api_key_here"
            )
        },
        "print_md5_dir": {
            "description": "查看当前已有MD5工作区服务",
            "task": print_md5_dir
        },
        "upload_files": {
            "description": "上传文件至容器内服务",
            "task": lambda: upload_files(file_path_list=['./pdf5.txt'])
        },
        "list_uploaded_files": {
            "description": "查看上传过的json和txt文件信息服务",
            "task": list_uploaded_files
        },
        "delete_uploaded_file": {
            "description": "删除已上传的指定文件服务",
            "task": lambda: delete_uploaded_file(filename="your_file_to_be_deleted.txt")
        },
        "delete_all_files": {
            "description": "删除所有上传的文件服务",
            "task": delete_all_files
        },
        "search_chunks": {
            "description": "在chunk_list中搜索query_text服务",
            "task": lambda: search_chunks(keywords=["query_text"])
        },
        "upsert_entity": {
            "description": "插入/更新实体信息服务",
            "task": lambda: upsert_entity(entity_name="entity_name", chunk_ids=["chunk_id1", "chunk_id2"])
        },
        "upsert_relationship": {
            "description": "插入/更新关系信息服务",
            "task": lambda: upsert_relationship(src_entity_name="src_entity_name", dst_entity_name="dst_entity_name", chunk_ids=["chunk_id1", "chunk_id2"])
        }
    }

    # 请从上面多个任务中选择一个任务，每次仅运行一个任务
    # 可选任务名为tasks字典的键，并修改上面字典中对应任务的参数输入
    selected_task = tasks["graphrag_service"]  # 替换为所需的任务

    # 设置容器服务IP地址
    service_ip = "localhost"
    # service_ip = "127.0.0.1"
    # 设置容器服务映射在当前宿主机的端口号
    service_port_number = 8004
    # 打印任务描述
    asyncio.run(print_with_pause(f"当前指定的宿主机侧端口号为：{service_port_number}\n执行任务: {selected_task['description']}\n"))

    # 执行任务
    try:
        asyncio.run(selected_task["task"]() if callable(selected_task["task"]) else selected_task["task"])
    except requests.exceptions.ConnectionError as e:
        asyncio.run(print_with_pause(f"\n无法连接到服务器 'http://{service_ip}:{service_port_number}/' ...\n请检查服务器（或容器）是否已启动，或者检查服务端口号重试..."))
        asyncio.run(print_with_pause(f"错误信息: {traceback.format_exc()}"))
    except:
        asyncio.run(print_with_pause(traceback.format_exc()))