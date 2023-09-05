import os
from typing import List, Union, Dict
from abc import ABC, abstractmethod
from langchain.embeddings.base import Embeddings
from langchain.docstore.document import Document
from utils import (
    get_kb_path, 
    get_doc_path,
    KnowledgeFile,
    load_embeddings
)
from kb_config import (
    EMBEDDING_MODEL,
    SCORE_THRESHOLD,
    VECTOR_SEARCH_TOP_K
)

class SupportedVSType:
    FAISS = 'faiss'
    MILVUS = 'milvus'
    DEFAULT = 'default'

class KBService(ABC):

    def __init__(self,
                 knowledge_base_name: str,
                 embed_model: str = EMBEDDING_MODEL,
                 ):
        self.kb_name = knowledge_base_name
        self.embed_model = embed_model
        self.kb_path = get_kb_path(self.kb_name)
        self.doc_path = get_doc_path(self.kb_name)
        self.do_init()

    def create_kb(self):
        """
        创建知识库
        """
        if not os.path.exists(self.doc_path):
            os.makedirs(self.doc_path)
        self.do_create_kb()
        return True
        # status = add_kb_to_db(self.kb_name, self.vs_type(), self.embed_model)
        # return status

    def clear_vs(self):
        """
        删除向量库中所有内容
        """
        self.do_clear_vs()
        # status = delete_files_from_db(self.kb_name)
        # return status

    def drop_kb(self):
        """
        删除知识库
        """
        self.do_drop_kb()
        # status = delete_kb_from_db(self.kb_name)
        # return status

    def add_doc(self, kb_file: KnowledgeFile, **kwargs):
        """
        向知识库添加文件
        """
        pass
        # docs = kb_file.file2text()
        # if docs:
        #     self.delete_doc(kb_file)
        #     embeddings = load_embeddings(embed_model = EMBEDDING_MODEL)
        #     self.do_add_doc(docs, embeddings, **kwargs)
        #     status = add_doc_to_db(kb_file)
        # else:
        #     status = False
        # return status

    def delete_doc(self, kb_file: KnowledgeFile, delete_content: bool = False, **kwargs):
        """
        从知识库删除文件
        """
        pass
        # self.do_delete_doc(kb_file, **kwargs)
        # status = delete_file_from_db(kb_file)
        # if delete_content and os.path.exists(kb_file.filepath):
        #     os.remove(kb_file.filepath)
        # return status

    def update_doc(self, kb_file: KnowledgeFile, **kwargs):
        """
        使用content中的文件更新向量库
        """
        pass
        # if os.path.exists(kb_file.filepath):
        #     self.delete_doc(kb_file, **kwargs)
        #     return self.add_doc(kb_file, **kwargs)

    def search_docs(self,
                    query: str,
                    top_k: int = VECTOR_SEARCH_TOP_K,
                    score_threshold: float = SCORE_THRESHOLD,
                    ):
        docs = self.do_search(query, top_k, score_threshold)
        return docs

    @abstractmethod
    def do_create_kb(self):
        """
        创建知识库子类实现自己逻辑
        """
        pass

    @abstractmethod
    def vs_type(self) -> str:
        pass

    @abstractmethod
    def do_init(self):
        pass

    @abstractmethod
    def do_drop_kb(self):
        """
        删除知识库子类实现自己逻辑
        """
        pass

    @abstractmethod
    def do_search(self,
                  query: str,
                  top_k: int,
                  score_threshold: float,
                  ) -> List[Document]:
        """
        搜索知识库子类实现自己逻辑
        """
        pass

    @abstractmethod
    def do_add_doc(self,
                   docs: List[Document],
                   embeddings: Embeddings,
                   ):
        """
        向知识库添加文档子类实现自己逻辑
        """
        pass

    @abstractmethod
    def do_delete_doc(self,
                      kb_file: KnowledgeFile):
        """
        从知识库删除文档子类实现自己逻辑
        """
        pass

    @abstractmethod
    def do_clear_vs(self):
        """
        从知识库删除全部向量子类实现自己逻辑
        """
        pass


