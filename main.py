import asyncio
import os

import numpy as np

from lightrag import LightRAG, QueryParam
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

WORKING_DIR = "./kongyiji"
# 创建工作目录
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


async def llm_model_func(
        prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    # 自定义LLM接口
    return await openai_complete_if_cache(
        "qwen-plus",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        **kwargs,
    )


async def embedding_func(texts: list[str]) -> np.ndarray:
    # 自定义embedding接口
    return await openai_embed(
        texts,
        model="text-embedding-v3",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )


async def get_embedding_dim():
    """
    test_text = ["This is a test sentence."]
    embedding = await embedding_func(test_text)
    embedding_dim = embedding.shape[1]
    """
    # 阿里云text-embedding-v3的embedding_dimension是1024
    # 这里直接返回
    return 1024


# function for test
async def funcs_test():
    # 测试LLM接口
    result = await llm_model_func("你是谁？")
    print("llm_model_func: ", result)
    # 测试embedding接口
    result = await embedding_func(["你是谁？"])
    print("embedding_func: ", result, result.shape)


async def initialize_rag():
    # 获取embedding_dimension
    embedding_dimension = await get_embedding_dim()
    print(f"Detected embedding dimension: {embedding_dimension}")

    # 实例化LightRAG
    # 许多初始化参数可以用.env文件进行配置。这里为了统一进行注释，所以在代码中显式配置
    rag = LightRAG(
        kv_storage="JsonKVStorage",
        # key-value格式的数据的存储类型，默认为"JsonKVStorage"。该类型用于存储LLM问答历史llm_response_cache、原始文档full_docs、文本片段text_chunks
        vector_storage="NanoVectorDBStorage",
        # 向量数据库类型，默认为"NanoVectorDBStorage"，一个基于numpy的简单向量数据库实现。该类型用于存储实体entities_vdb、关系relationships_vdb、文本片段chunks_vdb及其向量
        graph_storage="NetworkXStorage",
        # 知识图谱存储类型，默认为"NetworkXStorage"，基于networkx的简单实现。该类型用于存储知识图谱chunk_entity_relation_graph
        doc_status_storage="JsonDocStatusStorage",  # 文档状态存储类型，默认为"JsonDocStatusStorage"。该类型用于实时记录各文档的处理进度与状态doc_status
        working_dir=WORKING_DIR,  # 工作目录
        chunk_token_size=600,  # 切分文本的token长度，默认为1200。为了充分提取实体，设置为600
        chunk_overlap_token_size=100,  # 切分文本的token重叠长度，默认为100
        embedding_batch_num=10,  # 调用embedding模型接口的batch size，默认为16。text-embedding-v3允许的batch size最大为10，所以这里设置为10
        tiktoken_model_name="gpt-4",
        # tokenizer对应的LLM名称，默认为"gpt-4o-mini"，对应的tokenizer为"o200k_base"。这里换成更常用的"cl100k_base"，对应的LLM名称为"gpt-4"
        max_parallel_insert=1,  # 向知识库中插入新文档的最大并发数，默认为2。为了便于调试协程，这里设置为1
        addon_params={'language': '中文'},  # 额外参数。默认的PROMPT都是英文，但测试文档是中文。这里的language参数会被插入PROMPT中，向LLM强调应该用中文回答
        llm_model_func=llm_model_func,  # LLM接口函数
        llm_model_name="qwen-plus",  # LLM模型名称，默认为"gpt-4o-mini"，这里使用qwen-plus
        llm_model_max_async=1,  # LLM接口的最大并发数，默认为4。为了便于调试协程，这里设置为1
        embedding_func=EmbeddingFunc(
            embedding_dim=embedding_dimension,
            max_token_size=8192,
            func=embedding_func,
        ),  # 实例化之后的embedding模型接口类，使用接口函数和向量维度作为参数
        embedding_func_max_async=1,  # embedding接口的最大并发数，默认为16。为了便于调试协程，这里设置为1
        embedding_cache_config={ # 调用LLM与embedding接口的缓存配置
            "enabled": True, # 是否启用embedding缓存
            "similarity_threshold": 0.95, # 缓存命中的相似度阈值，默认为0.95
            "use_llm_check": False, # 是否启用LLM判断缓存有没有命中，默认为False
        },
    )

    # 基于协程并发调用各存储对象的初始化函数
    await rag.initialize_storages()
    # 初始化pipeline，用于确保多进程的数据一致性，以及进程间通信
    await initialize_pipeline_status()

    return rag


async def main():
    try:
        # Initialize RAG instance
        rag = await initialize_rag()

        # 采用《孔乙己》作为测试文档
        with open("./kongyiji/kongyiji.txt", "r", encoding="utf-8") as f:
            # 基于协程并发，插入新文档，并将相关索引增量更新到知识库中
            # LightRAG新增文档，只需要在知识库中新增该文档的相关内容。不需要像GraphRAG那样，更新整个知识库
            # 插入时，会根据文档内容进行去重，并根据文档ID（默认为MD5值）过滤掉数据库中已有的文档
            # ainsert依次调用了apipeline_enqueue_documents和apipeline_process_enqueue_documents两个函数
            # apipeline_enqueue_documents是对文档进行预处理，功能较为简单，注释已经非常详细。
            # apipeline_process_enqueue_documents是对新增文档进行切片、实体与关系提取、索引构建
            await rag.ainsert(input=f.read(), file_paths="./kongyiji/kongyiji.txt")

        # 以下展示不同模式下的检索与生成结果
        # aquery函数会根据 检索模式参数mode 分别调用不同的检索函数，并返回检索结果
        # aquery的参数
        # query: 用户输入的query
        # param: QueryParam对象，用于汇总检索参数。其中包括大量参数及其默认值，具体见类注释。
        # system_prompt: 系统提示词，默认为None，即使用预设的默认系统提示词
        # 如果只调试检索功能，可以在param中设置only_need_context=True或only_need_prompt=True，aquery函数此时会直接返回检索结果或提示词

        # Perform naive search
        print(
            await rag.aquery(
                "孔乙己与丁举人之间发生过什么？",
                param=QueryParam(mode="naive", top_k=3),
                system_prompt=None,
            )
        )

        # Perform local search
        print(
            await rag.aquery(
                "孔乙己与丁举人之间发生过什么？",
                param=QueryParam(mode="local", top_k=10),
                system_prompt=None,
            )
        )

        # Perform global search
        print(
            await rag.aquery(
                "孔乙己与丁举人之间发生过什么？",
                param=QueryParam(mode="global", top_k=10),
                system_prompt=None,
            )
        )

        # Perform hybrid search
        print(
            await rag.aquery(
                "孔乙己与丁举人之间发生过什么？",
                param=QueryParam(mode="hybrid", top_k=10),
                system_prompt=None,
            )
        )

        # Perform mix search
        print(
            await rag.aquery(
                "孔乙己与丁举人之间发生过什么？",
                param=QueryParam(mode="mix", top_k=10),
                system_prompt=None,
            )
        )
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    # 主函数
    asyncio.run(main())

    # API接口测试
    # asyncio.run(funcs_test())
