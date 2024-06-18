from fastapi import APIRouter
import logging
import asyncio
import json
import os
import re
import tempfile
import traceback
import uuid
from functools import partial
from io import BytesIO
from typing import Annotated, Any, List, Optional, Union
from langchain import hub
from langchain.chat_models import ChatOpenAI
from docx2pdf import convert
from docx import Document
from dotenv import load_dotenv
from fastapi import (APIRouter, BackgroundTasks, Depends, File, Form,
                     HTTPException, Request, Response, UploadFile)
from fastapi.datastructures import UploadFile as FastAPIUploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from langchain.document_loaders.mongodb import MongodbLoader
from langchain_community.chat_models import BedrockChat
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import (DirectoryLoader,
                                                  S3DirectoryLoader)
from embeddings.embed_data import Data_Embedder
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_community.document_loaders.csv_loader import UnstructuredCSVLoader
from langchain_community.document_loaders import UnstructuredFileLoader
#from pdfplumber import open as open_pdf

neta_prompt = hub.pull("sredeemer/friiimind")
file_load_restructure_prompt = hub.pull("sredeemer/file_load_restructure")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")
AWS_DEFAULT_REGION = os.getenv("AWS_REGION")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
# initialize LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2, openai_api_key = OPENAI_API_KEY) 
# llm = BedrockChat(
#     model_id="anthropic.claude-3-sonnet-20240229-v1:0",
#     model_kwargs={"temperature": 0.2, "max_tokens": 200000 },
#     region_name = AWS_DEFAULT_REGION
# )
# Make sure to replace with your actual API key



router = APIRouter(
     prefix="/files",
    responses={404: {"description": "Not found"}},
    #dependencies=[Depends(JWTBearer())],
    tags=["files"],
)

@router.get("/users/", tags=["users"])
async def read_user():
    return [{"Hello world": 23}]


#         #TODO: Make LLM map headings to standard headings
#         # Extract content of relevant headings
#         # Re-format content into one doc well structured
#         # Embed structured content
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@router.post("/matching")
async def generate_matching_criteria(files: List[UploadFile] = File(...)):
    try:
        results = []
        file_count = 0
        #embedder = Data_Embedder() 
         # Create an instance of Data_Embedder

        for file in files:
            file_count += 1
            content = await file.read()

            suffix = ".csv" if file.filename.endswith('.csv') else (".xlsx" if file.filename.endswith('.xlsx') else None)
            if suffix:
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                    temp_file.write(content)
                    temp_file.flush()
                    temp_file_path = temp_file.name

                    try:
                        loader = UnstructuredCSVLoader(temp_file_path, unstructured_kwargs={"encoding": "latin1", "delimiter": ","}) if suffix == ".csv" else UnstructuredExcelLoader(temp_file_path)
                        docs = loader.load()
                        document_content = ' '.join([str(doc) for doc in docs])

                        # First invocation: Structuring the content
                        formatted = (file_load_restructure_prompt | llm).invoke({"table": document_content})
                        logging.info(f"Formatted content from LangChain for file {file.filename}: {formatted}")

                        # Second invocation: Analysis using neta_prompt
                        response = (neta_prompt | llm).invoke({"proposal": formatted})
                        logging.info(f"Analysis response from LangChain for file {file.filename}: {response}")

                        # Write response to a file
                        response_file_path = os.path.splitext(temp_file_path)[0] + "_response.txt"
                        with open(response_file_path, "w") as response_file:
                            response_file.write(str(response))
                            logging.info(f"Response written to file: {response_file_path}")

                        # Embedding the response
                        raw_documents = TextLoader(response_file_path).load()
                        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
                        documents = text_splitter.split_documents(raw_documents)
                        embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
                        db = Chroma.from_documents(docs, embedding_function, ids=None, collection_name="langchain", persist_directory="./chroma_db")
                        logging.info("Completed embedding process")

                    except Exception as e:
                        logging.error(f"Failed to process file {file.filename}: {e}")
                        continue  # Skip processing this file
            else:
                logging.error(f"Unsupported file format for file {file.filename}")
                continue  # Skip processing this file

        logging.info(f"Total number of files processed: {file_count}, Names of files: {[file.filename for file in files]}")
        return {"status": "success", "data": results, "file_count": file_count}
    except HTTPException as e:
        return JSONResponse(content={"error": str(e.detail)}, status_code=e.status_code)
    except Exception as e:
        logging.error(f"Unexpected error during processing: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
