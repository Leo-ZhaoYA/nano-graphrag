import time
import numpy as np
from openai import AsyncOpenAI, AsyncAzureOpenAI, APIConnectionError, RateLimitError, AuthenticationError, InternalServerError
import asyncio
import aiohttp
import json
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from ._utils import compute_args_hash, wrap_embedding_func_with_attrs, logger, read_yaml_config
from .base import BaseKVStorage

global_openai_async_client = None
global_azure_openai_async_client = None

retry_exceptions = (RateLimitError, APIConnectionError, AuthenticationError, InternalServerError, asyncio.TimeoutError)

yaml_config = read_yaml_config()
openai_api_key = yaml_config.get("openai_api_key")
openai_api_url = yaml_config.get("openai_api_url")
local_conversation_url = yaml_config.get("local_conversation_url")
local_embedding_url = yaml_config.get("local_embedding_url")
llm_func_timeout = int(yaml_config.get("llm_func_timeout"))

def get_openai_async_client_instance():
    global global_openai_async_client
    if global_openai_async_client is None:
        global_openai_async_client = AsyncOpenAI(api_key=openai_api_key, base_url=openai_api_url, timeout=llm_func_timeout)
    return global_openai_async_client


def get_azure_openai_async_client_instance():
    global global_azure_openai_async_client
    if global_azure_openai_async_client is None:
        global_azure_openai_async_client = AsyncAzureOpenAI()
    return global_azure_openai_async_client

# 装饰器
@retry(
    # 表示在最多尝试 5 次之后停止重试
    stop=stop_after_attempt(5),     
    # 表示重试之间的等待时间按照指数增长，最少 4 秒，最多 10 秒。指数的初始值由 multiplier 控制
    wait=wait_exponential(multiplier=1, min=4, max=10),         
    # 表示只有在遇到这些异常时才重试
    retry=retry_if_exception_type((asyncio.TimeoutError)),
)
async def ollama_complete_if_cache(
    model, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    async def local_chat(text, **kwargs):
        url = local_conversation_url
            # 构造请求数据
        payload = {
            "prompt": json.dumps(text, ensure_ascii=False),
            "temperature": 0.3,
            "max_tokens": 2048,
            "timeout": llm_func_timeout
        }
        payload.update(kwargs)
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f'Error in achieving embedding from local url :{url}, retrying...')
                    raise ValueError
        
    # remove kwargs that are not supported by ollama
    kwargs.pop("max_tokens", None)
    kwargs.pop("response_format", None)
    
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    # Get the cached response if having-------------------
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    if hashing_kv is not None:
        args_hash = compute_args_hash(model, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]

    response = await local_chat(messages, **kwargs)
    # response = await ollama_client.chat(
    #     model=model, messages=[str(i) for i in messages], **kwargs
    # )
    # Cache the response if having-------------------
    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": response['choices'][0]['text'], "model": response['model']}}
        )
        await hashing_kv.index_done_callback()
    return response['choices'][0]['text']


# 装饰器
@retry(
    # 表示在最多尝试 5 次之后停止重试
    stop=stop_after_attempt(5),     
    # 表示重试之间的等待时间按照指数增长，最少 4 秒，最多 10 秒。指数的初始值由 multiplier 控制
    wait=wait_exponential(multiplier=1, min=4, max=10),         
    # 表示只有在遇到这些异常时才重试
    retry=retry_if_exception_type(retry_exceptions),
)
async def openai_complete_if_cache(
    model, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    openai_async_client = get_openai_async_client_instance()
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    if hashing_kv is not None:
        args_hash = compute_args_hash(model, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]

    response = await openai_async_client.chat.completions.create(
        model=model, messages=messages, **kwargs
    )

    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": response.choices[0].message.content, "model": model}}
        )
        await hashing_kv.index_done_callback()
    return response.choices[0].message.content

async def local_model_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await ollama_complete_if_cache(
        "qwen2.5:72b",
        # "llama3.3:70b-instruct-q5_K_M",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )

async def gpt_4o_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await openai_complete_if_cache(
        "gpt-4o",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def gpt_4o_mini_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await openai_complete_if_cache(
        "gpt-4o-mini",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


@wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=8192)
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(retry_exceptions),
)
async def openai_embedding(texts: list[str]) -> np.ndarray:
    openai_async_client = get_openai_async_client_instance()
    response = await openai_async_client.embeddings.create(
        model="text-embedding-3-small", input=texts, encoding_format="float"
    )
    return np.array([dp.embedding for dp in response.data])


@wrap_embedding_func_with_attrs(embedding_dim=1024, max_token_size=512) 
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((asyncio.TimeoutError, ValueError)),
)
async def local_embedding(texts: list[str]) -> np.ndarray:
    url = local_embedding_url
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json={"input": texts}) as response:
            if response.status == 200:
                results = await response.json()
                embedding = np.array([i['embedding'] for i in results['data']])
                return embedding
            else:
                logger.error(f'Error in achieving embedding from local url :{url}, retrying...')
                raise ValueError

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(retry_exceptions),
)
async def azure_openai_complete_if_cache(
    deployment_name, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    azure_openai_client = get_azure_openai_async_client_instance()
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    if hashing_kv is not None:
        args_hash = compute_args_hash(deployment_name, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]

    response = await azure_openai_client.chat.completions.create(
        model=deployment_name, messages=messages, **kwargs
    )

    if hashing_kv is not None:
        await hashing_kv.upsert(
            {
                args_hash: {
                    "return": response.choices[0].message.content,
                    "model": deployment_name,
                }
            }
        )
        await hashing_kv.index_done_callback()
    return response.choices[0].message.content


async def azure_gpt_4o_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await azure_openai_complete_if_cache(
        "gpt-4o",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def azure_gpt_4o_mini_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await azure_openai_complete_if_cache(
        "gpt-4o-mini",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


@wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=8192)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(retry_exceptions),
)
async def azure_openai_embedding(texts: list[str]) -> np.ndarray:
    azure_openai_client = get_azure_openai_async_client_instance()
    response = await azure_openai_client.embeddings.create(
        model="text-embedding-3-small", input=texts, encoding_format="float"
    )
    return np.array([dp.embedding for dp in response.data])
