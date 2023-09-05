import os
import shutil
from typing import List
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from base import KBService, SupportedVSType
from utils import (
    get_vs_path, 
    get_doc_path, 
    load_embeddings,
    KnowledgeFile
)
from kb_config import (
    KB_ROOT_PATH,
    EMBEDDING_MODEL,
    SCORE_THRESHOLD,
    VECTOR_SEARCH_TOP_K
)

def load_vector_store(knowledge_base_name: str, embed_model: str = EMBEDDING_MODEL):
    print(f"loading vector store in '{knowledge_base_name}'.")
    vs_path = get_vs_path(knowledge_base_name)
    doc_path = get_doc_path(knowledge_base_name = knowledge_base_name)

    if not os.path.exists(vs_path):
        os.makedirs(vs_path)
    if not os.path.exists(doc_path):
        os.makedirs(doc_path)

    embeddings = load_embeddings(embed_model)

    if "index.faiss" in os.listdir(vs_path):
        search_index = FAISS.load_local(vs_path, embeddings, normalize_L2=True)
    else:  # create an empty vector store
        doc = Document(page_content="init", metadata={})
        search_index = FAISS.from_documents([doc], embeddings, normalize_L2=True)
        ids = [k for k, v in search_index.docstore._dict.items()]
        search_index.delete(ids)
        search_index.save_local(vs_path)
    
    return search_index

def refresh_vs_cache(kb_name: str):
    """
    make vector store cache refreshed when next loading
    """
    print(f"知识库 {kb_name} 缓存刷新")

class FaissKBService(KBService):
    vs_path: str
    kb_path: str

    def vs_type(self) -> str:
        return SupportedVSType.FAISS

    @staticmethod
    def get_vs_path(knowledge_base_name: str):
        return os.path.join(FaissKBService.get_kb_path(knowledge_base_name), "vector_store")

    @staticmethod
    def get_kb_path(knowledge_base_name: str):
        return os.path.join(KB_ROOT_PATH, knowledge_base_name)

    def do_init(self):
        self.kb_path = FaissKBService.get_kb_path(self.kb_name)
        self.vs_path = FaissKBService.get_vs_path(self.kb_name)

    def do_create_kb(self):
        if not os.path.exists(self.vs_path):
            os.makedirs(self.vs_path)
        load_vector_store(self.kb_name)
    
    def do_clear_vs(self):
        """
        仅删除vector store
        """
        shutil.rmtree(self.vs_path)
        os.makedirs(self.vs_path)
        refresh_vs_cache(self.kb_name)

    def do_drop_kb(self):
        """
        删除整个知识库KB
        """
        self.clear_vs()
        shutil.rmtree(self.kb_path)
        refresh_vs_cache(self.kb_name)
        
    def do_search(self,
                  query: str,
                  top_k: int = VECTOR_SEARCH_TOP_K,
                  score_threshold: float = SCORE_THRESHOLD,
                  ) -> List[Document]:
        search_index = load_vector_store(self.kb_name)
        docs = search_index.similarity_search_with_score(query, k=top_k, score_threshold=score_threshold)
        return docs

    def do_add_doc(self,
                   docs: List[Document],
                   **kwargs,
                   ):
        vector_store = load_vector_store(self.kb_name)
        vector_store.add_documents(docs)
        if not kwargs.get("not_refresh_vs_cache"):
            vector_store.save_local(self.vs_path)
            refresh_vs_cache(self.kb_name)

    def do_delete_doc(self,
                      kb_file: KnowledgeFile,
                      **kwargs):
        vector_store = load_vector_store(self.kb_name)
        ids = [k for k, v in vector_store.docstore._dict.items() if v.metadata["source"] == kb_file.filepath]
        if len(ids) == 0:
            return None

        vector_store.delete(ids)
        if not kwargs.get("not_refresh_vs_cache"):
            vector_store.save_local(self.vs_path)
            refresh_vs_cache(self.kb_name)

        return True






