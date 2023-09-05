import os
import importlib
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from kb_config import (
    KB_ROOT_PATH,
    embedding_model_dict,
    CHUNK_SIZE,
    OVERLAP_SIZE
)

def get_kb_path(knowledge_base_name: str):
    return os.path.join(KB_ROOT_PATH, knowledge_base_name)

def get_vs_path(knowledge_base_name: str):
    return os.path.join(get_kb_path(knowledge_base_name), "vector_store")

def get_doc_path(knowledge_base_name: str):
    return os.path.join(get_kb_path(knowledge_base_name), "content")

def get_file_path(knowledge_base_name: str, doc_name: str):
    return os.path.join(get_doc_path(knowledge_base_name), doc_name)

def load_embeddings(model: str):
    if model == "text-embedding-ada-002":  
        embeddings = OpenAIEmbeddings(openai_api_key=model, chunk_size=CHUNK_SIZE)
    else:
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model_dict[model],)
    return embeddings

def get_LoaderClass(file_extension):
    for LoaderClass, extensions in LOADER_DICT.items():
        if file_extension in extensions:
            return LoaderClass

LOADER_DICT = {"UnstructuredFileLoader": ['.eml', '.html', '.json', '.md', '.msg', '.rst',
                                          '.rtf', '.txt', '.xml',
                                          '.doc', '.docx', '.epub', '.odt', '.pdf',
                                          '.ppt', '.pptx', '.tsv'],  
               "CSVLoader": [".csv"],
               "PyPDFLoader": [".pdf"],
               }
SUPPORTED_EXTS = [ext for sublist in LOADER_DICT.values() for ext in sublist]

class KnowledgeFile:
    def __init__(
            self,
            filename: str,
            knowledge_base_name: str
    ):
        self.kb_name = knowledge_base_name
        self.filename = filename
        self.ext = os.path.splitext(filename)[-1].lower()
        if self.ext not in SUPPORTED_EXTS:
            raise ValueError(f"暂未支持的文件格式 {self.ext}")
        self.filepath = get_file_path(knowledge_base_name, filename)
        self.docs = None
        self.document_loader_name = get_LoaderClass(self.ext)

        # 指定text_splitter
        self.text_splitter_name = None

    def file2text(self):
        print(self.document_loader_name)
        try:
            document_loaders_module = importlib.import_module('langchain.document_loaders')
            DocumentLoader = getattr(document_loaders_module, self.document_loader_name)
        except Exception as e:
            print(e)
            document_loaders_module = importlib.import_module('langchain.document_loaders')
            DocumentLoader = getattr(document_loaders_module, "UnstructuredFileLoader")
        if self.document_loader_name == "UnstructuredFileLoader":
            loader = DocumentLoader(self.filepath, autodetect_encoding=True)
        else:
            loader = DocumentLoader(self.filepath)

        try:
            if self.text_splitter_name is None:
                text_splitter_module = importlib.import_module('langchain.text_splitter')
                TextSplitter = getattr(text_splitter_module, "SpacyTextSplitter")
                text_splitter = TextSplitter(
                    pipeline="zh_core_web_sm",
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=OVERLAP_SIZE,
                )
                self.text_splitter_name = "SpacyTextSplitter"
            else:
                text_splitter_module = importlib.import_module('langchain.text_splitter')
                TextSplitter = getattr(text_splitter_module, self.text_splitter_name)
                text_splitter = TextSplitter(
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=OVERLAP_SIZE)
        except Exception as e:
            print(e)
            text_splitter_module = importlib.import_module('langchain.text_splitter')
            TextSplitter = getattr(text_splitter_module, "RecursiveCharacterTextSplitter")  # CharacterTextSplitter
            text_splitter = TextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=OVERLAP_SIZE,
            )
        docs = loader.load_and_split(text_splitter)

        return docs
