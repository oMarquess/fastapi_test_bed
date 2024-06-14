import os
import uuid
import logging

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv, find_dotenv
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter,CharacterTextSplitter
from langchain_chroma import Chroma as Chroma_db
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

host = os.getenv("CHROMA_IP")
port = os.getenv("CHROMA_PORT")


class Data_Embedder:
    def __init__(self) -> None:
        self.docs = None
        logger.info("🚀 Data_Embedder instance created")

    '''
    this function is used to collect the data from the websites for embedding
    the embedding is stored in a local store for temporal use.
    the embedding funcion is openAI "text-embedding-ada-002"
    '''
    def temp_embed(self,data):

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.create_documents([data])

        # documents_content = [doc.page_content for doc in chunks]

        #  embedding function
        embeddings_funk = OpenAIEmbeddings(model="text-embedding-ada-002")
       
        # load it into Chroma
        db = Chroma_db.from_documents(docs, embeddings_funk)

        return db


    def do_embed(
        self,
        docs,
        chunk_size=1000,
        chunk_overlap=200,
        user_id=None,
        project_id=None,
        mongo_context=False,
        s3_context=False,
        project_files_context=False,
    ):
        try:
            logger.info("🔍 Starting embedding process")
            chroma_client = chromadb.HttpClient(host=host, port=port)
            logger.info("🌐 ChromaDB client initialized")

            embeddings_funk = embedding_functions.OpenAIEmbeddingFunction(
                api_key=OPENAI_API_KEY, model_name="text-embedding-ada-002"
            )
            logger.info("🧠 Embedding function set up")

            text_splitter = CharacterTextSplitter(chunk_size=chunk_size,
                                                  chunk_overlap=chunk_overlap)
            logger.info("🔪 Text splitter initialized")

            chunks = text_splitter.split_documents(docs)
            logger.info(f"📄 Documents split into {len(chunks)} chunks")

            if mongo_context:
                collection_name = "embed_mongo_profile"
            elif s3_context:
                collection_name = "embed_s3_context"
            elif project_files_context:
                collection_name = "embed_project_files_context"
            else:
                collection_name = "embed_user_data"
            logger.info(f"📦 Collection name set to {collection_name}")

            if collection_name in [c.name for c in chroma_client.list_collections()]:
                chroma_client.delete_collection(name=collection_name)
                logger.info(f"🗑️ Collection {collection_name} deleted")

            collection = chroma_client.get_or_create_collection(
                name=collection_name, embedding_function=embeddings_funk
            )
            logger.info(f"🆕 Collection {collection_name} created or retrieved")

            documents_content = [doc.page_content for doc in chunks]
            metadatas = []
            for doc in chunks:
                metadata_entry = doc.metadata.copy()
                if user_id:
                    metadata_entry.update({"user_id": user_id})
                elif project_id:
                    metadata_entry.update({"project_id": project_id})
                metadatas.append(metadata_entry)
            logger.info("📊 Metadata prepared")

            ids = [str(uuid.uuid4()) for _ in range(len(chunks))]
            embedded = embeddings_funk(documents_content)
            logger.info("🔗 Documents embedded")

            collection.add(
                embeddings=embedded,
                metadatas=metadatas,
                documents=documents_content,
                ids=ids,
            )
            logger.info("📤 Embedded documents added to collection")

            db = Chroma(
                client=chroma_client,
                collection_name=collection_name,
                embedding_function=OpenAIEmbeddings(),
            )
            logger.info("✅ Embedding Successful")

            return db
        except Exception as e:
            logger.error(f"❌ Embedding failed: {e}")
            raise e
