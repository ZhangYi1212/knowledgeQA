import os

# 在以下字典中修改属性值，以指定本地embedding模型存储位置
embedding_model_dict = {
    "ernie-tiny": "nghuyong/ernie-3.0-nano-zh",
    "ernie-base": "nghuyong/ernie-3.0-base-zh",
    "text2vec-base": "shibing624/text2vec-base-chinese",
    "text2vec": "GanymedeNil/text2vec-large-chinese",
    "text2vec-paraphrase": "shibing624/text2vec-base-chinese-paraphrase",
    "text2vec-sentence": "shibing624/text2vec-base-chinese-sentence",
    "text2vec-multilingual": "shibing624/text2vec-base-multilingual",
    "m3e-small": "moka-ai/m3e-small",
    "m3e-base": "/home/zy/workspace/llm/pretrained_models/m3e-base",
    "m3e-large": "moka-ai/m3e-large",
    "bge-small-zh": "BAAI/bge-small-zh",
    "bge-base-zh": "BAAI/bge-base-zh",
    "bge-large-zh": "BAAI/bge-large-zh",
    "bge-large-zh-noinstruct": "BAAI/bge-large-zh-noinstruct",
    "text-embedding-ada-002": os.environ.get("OPENAI_API_KEY")
}
# 选用的 Embedding 名称
EMBEDDING_MODEL = "m3e-base"

# 知识库默认存储路径/当前目录下新建一个文件夹:knowledge_base
KB_ROOT_PATH = os.path.join(os.path.dirname(__file__), "knowledge_base")

# 要新添加的文档名称
ADD_DCO = "sunshine.txt"

# 可选向量库类型及对应配置
kbs_config = {
    "faiss": {
    },
    "milvus": {
        "host": "127.0.0.1",
        "port": "19530",
        "user": "",
        "password": "",
        "secure": False,
    }
}

# 默认向量库类型。可选：faiss, milvus.
DEFAULT_VS_TYPE = "faiss"

# 知识库中单段文本长度
CHUNK_SIZE = 100

# 知识库中相邻文本重合长度
OVERLAP_SIZE = 50

# 知识库匹配向量数量
VECTOR_SEARCH_TOP_K = 3

# 知识库匹配相关度阈值，取值范围在0-1之间，SCORE越小，相关度越高，取到1相当于不筛选，建议设置在0.5左右
SCORE_THRESHOLD = 1



