import os
from utils import KnowledgeFile
from faiss_kb_service import FaissKBService
from kb_config import ADD_DCO

if __name__ == "__main__":
    ## 指定名称初始化一个知识库
    knowledge_base_name = "sample"
    faiss = FaissKBService(knowledge_base_name = knowledge_base_name)
    faiss.do_init()
    faiss.do_create_kb()

    ## 删除
    # faiss.do_clear_vs()  # 删除vector_store 
    # faiss.do_drop_kb()   # 删除整个kb

    ## 为知识库添加文档
    kf = KnowledgeFile(filename = ADD_DCO, knowledge_base_name = "sample")
    faiss.do_add_doc(docs = kf.file2text())

    ## 查询知识库
    res = faiss.do_search(query = "一、阳光人寿阳光升B款终身寿险产品介绍")
    print(res)

    ## 删除知识库中的指定文档
    fi = KnowledgeFile(filename = ADD_DCO, knowledge_base_name = "sample")
    res = faiss.do_delete_doc(fi)
    print(res)



