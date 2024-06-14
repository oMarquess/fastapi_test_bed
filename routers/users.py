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

from langchain_community.document_loaders import (DirectoryLoader,
                                                  S3DirectoryLoader)
from embeddings.embed_data import Data_Embedder
#from pdfplumber import open as open_pdf
prompt = hub.pull("sredeemer/friiimind")
# initialize LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2, openai_api_key = '') 
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

# @router.get("/users/me", tags=["users"])
# async def read_user():
#     return [{"Hello world": 24}]

# @router.get("/users/{username}", tags=["users"])
# async def read_user(username: str):
#     return {"Hello world": 21233}


# @router.post("/matching")
# async def generate_matching_criteria(files: List[UploadFile] = File(...)):
#     try:
#         results = []
#         for file in files:
#             # Read the content of the CSV file
#             content = await file.read()
#             # Save the content to a temporary file
#             with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
#                 temp_file.write(content)
#                 temp_file_path = temp_file.name

#             # Use CSVLoader to load the CSV file
#             loader = CSVLoader(file_path=temp_file_path)
#             docs = loader.load()

#             # Process the loaded documents
#             for doc in docs:
#                 results.append(doc)

#             # Clean up the temporary file
#             os.remove(temp_file_path)

#         #TODO: Make LLM map headings to standard headings
#         # Extract content of relevant headings
#         # Re-format content into one doc well structured
#         # Embed structured content
#         # 

#         return {"status": "success", "data": results}
#     except Exception as e:
#         return JSONResponse(content={"error": str(e)}, status_code=500)



# @router.post("/matching")
# async def generate_matching_criteria(files: List[UploadFile] = File(...)):
#     try:
#         results = []
#         for file in files:
#             # Read and temporarily store the file
#             content = await file.read()
#             with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
#                 temp_file.write(content)
#                 temp_file_path = temp_file.name

#             # Load documents from CSV
#             loader = CSVLoader(file_path=temp_file_path)  # Assuming CSVLoader is setup properly
#             docs = loader.load()

#             # Process each document
#             for doc in docs:
#                 # Create a DOC file to store the LLM's response
#                 doc_filename = f"{temp_file_path.split('.')[0]}.docx"
#                 document = Document()
#                 document.add_heading('LLM Structured Response', level=1)

#                 # Prepare the prompt
#                 prompt_text = f"Classify this data: {doc}"
#                 response = llm.complete(prompt_text)

#                 # Add response to the DOC file
#                 document.add_paragraph(str(response))
#                 document.save(doc_filename)
#                 logging.info(f"DOC file saved: {doc_filename}")

#                 # Embed the document (Placeholder)
#                 embedded_doc_filename = data_embedder.do_embed(
#                 docs=doc_filename, 
#                 #mongo_context=True, 
#                 #user_id= user_id
#                 )
#                 logging.info(f"Document embeded")

#                 # Add the file path to results
#                 results.append(doc_filename)

#                 # Logging the structured response from LLM
#                 logging.info(f"Structured response from LLM: {response}")

#             # Clean up the temporary CSV file
#             os.remove(temp_file_path)

#         return {"status": "success", "data": results}
#     except Exception as e:
#         logging.error(f"Error processing files: {e}")
#         return JSONResponse(content={"error": str(e)}, status_code=500)

# from fastapi import FastAPI, File, UploadFile
# from typing import List

# app = FastAPI()

# @router.post("/uploadfiles/")
# async def generate_matching_criteria(files: List[UploadFile] = File(...)):
#     try:
#         print({"file_sizes": [len(await file.read()) for file in files]})
#         # your code to handle file uploads
#     except Exception as e:
#         logging.error(f"Error processing files: {e}")
#         return JSONResponse(content={"error": str(e)}, status_code=500)
    
# import logging
# from fastapi import FastAPI, File, UploadFile
# from typing import List
# from fastapi.responses import JSONResponse
# import tempfile
# import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#app = FastAPI()

@router.post("/matching")
# async def generate_matching_criteria(files: List[UploadFile] = File(...)):
#     try:
#         results = []
#         for file in files:
#             # Read and temporarily store the file
#             content = await file.read()
#             with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
#                 temp_file.write(content)
#                 temp_file_path = temp_file.name

#             # Assume some processing happens here
#             processed_result = "Processed content here"
#             results.append(processed_result)

#             logging.info(f"Processed file saved: {processed_result}")

#             # Clean up the temporary CSV file
#             os.remove(temp_file_path)

#         return {"status": "success", "data": results}
#     except Exception as e:
#         logging.error(f"Error processing files: {e}")
#         return JSONResponse(content={"error": str(e)}, status_code=500)

# import logging
# from fastapi import FastAPI, File, UploadFile
# from typing import List
# from fastapi.responses import JSONResponse
# import tempfile
# import os
# from langchain.loaders import CSVLoader  # Import the CSVLoader from LangChain

# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# app = FastAPI()

# @app.post("/matching")
# async def generate_matching_criteria(files: List[UploadFile] = File(...)):
#     try:
#         results = []
#         for file in files:
#             # Read and temporarily store the file
#             content = await file.read()
#             with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
#                 temp_file.write(content)
#                 temp_file_path = temp_file.name

#             # Load the CSV using LangChain's CSVLoader
#             loader = CSVLoader(temp_file_path)
#             docs = loader.load()

#             # Assume some processing happens here
#             processed_results = [f"Processed content from doc: {doc}" for doc in docs]
#             results.extend(processed_results)

#             for result in processed_results:
#                 logging.info(f"Processed result saved: {result}")

#             # Clean up the temporary CSV file
#             os.remove(temp_file_path)

#         return {"status": "success", "data": results}
#     except Exception as e:
#         logging.error(f"Error processing files: {e}")
#         return JSONResponse(content={"error": str(e)}, status_code=500)

# import logging
# from fastapi import FastAPI, File, UploadFile
# from typing import List
# from fastapi.responses import JSONResponse
# import tempfile
# import os
# from langchain.loaders import CSVLoader  # Import the CSVLoader from LangChain

# Configure logging
#logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#app = FastAPI()


# async def generate_matching_criteria(files: List[UploadFile] = File(...)):
#     try:
#         results = []
#         file_count = 0  # Counter for files processed

#         for file in files:
#             file_count += 1  # Increment counter for each file processed
#             # Read and temporarily store the file
#             content = await file.read()
#             with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
#                 temp_file.write(content)
#                 temp_file_path = temp_file.name

#             # Load the CSV using LangChain's CSVLoader
#             loader = CSVLoader(temp_file_path)
#             docs = loader.load()

#             # Assume some processing happens here
#             processed_results = [f"Processed content from doc: {doc}" for doc in docs]
#             results.extend(processed_results)

#             # Logging the results along with the file name
#             for result in processed_results:
#                 logging.info(f"Processed result saved: {result} from file: {temp_file_path}")

#             # Clean up the temporary CSV file
#             os.remove(temp_file_path)

#         # Log the count of files processed
#         logging.info(f"Total number of files processed: {file_count}, Names of files: {[file.filename for file in files]}")

#         return {"status": "success", "data": results, "file_count": file_count}
#     except Exception as e:
#         logging.error(f"Error processing files: {e}")
#         return JSONResponse(content={"error": str(e)}, status_code=500)

# @router.post("/matching")
# async def generate_matching_criteria(files: List[UploadFile] = File(...)):
#     try:
#         results = []
#         file_count = 0

#         for file in files:
#             # Check if the file is a CSV by content type or extension
#             if not file.filename.endswith('.csv'):
#                 raise HTTPException(status_code=400, detail=f"File {file.filename} is not a CSV file")

#             file_count += 1
#             content = await file.read()
#             with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
#                 temp_file.write(content)
#                 temp_file_path = temp_file.name

#             loader = CSVLoader(temp_file_path)
#             docs = loader.load()
#             processed_results = [f"Processed content from doc: {doc}" for doc in docs]
#             results.extend(processed_results)

#             for result in processed_results:
#                 logging.info(f"Processed result saved: {result} from file: {temp_file_path}")

#             os.remove(temp_file_path)

#         logging.info(f"Total number of files processed: {file_count}, Names of files: {[file.filename for file in files]}")
#         return {"status": "success", "data": results, "file_count": file_count}
#     except HTTPException as e:
#         return JSONResponse(content={"error": str(e.detail)}, status_code=e.status_code)
#     except Exception as e:
#         logging.error(f"Error processing files: {e}")
#         return JSONResponse(content={"error": str(e)}, status_code=500)


# @router.post("/matching")
# async def generate_matching_criteria(files: List[UploadFile] = File(...)):
#     try:
#         results = []
#         file_count = 0

#         for file in files:
#             if not file.filename.endswith('.csv'):
#                 raise HTTPException(status_code=400, detail=f"File {file.filename} is not a CSV file")
            
#             file_count += 1
#             content = await file.read()

#             # Assuming content is now the CSV content stored in a variable
#             runnable = prompt | llm
#             response = runnable.invoke({"proposal": content.decode()})  # Decoding may be necessary depending on your content
            
#             # Log and add the response to results
#             logging.info(f"Response from LangChain for file {file.filename}: {response}")
#             results.append(response)

#             # No need to save temporary files in this scenario

#         logging.info(f"Total number of files processed: {file_count}, Names of files: {[file.filename for file in files]}")
#         return {"status": "success", "data": results, "file_count": file_count}
#     except HTTPException as e:
#         return JSONResponse(content={"error": str(e.detail)}, status_code=e.status_code)
#     except Exception as e:
#         logging.error(f"Error processing files: {e}")
#         return JSONResponse(content={"error": str(e)}, status_code=500)

#INTERNAL SERVER ERROR

# @router.post("/matching")
# async def generate_matching_criteria(files: List[UploadFile] = File(...)):
#     try:
#         results = []
#         file_count = 0
#         #embedder = Data_Embedder()  # Instance of your Data_Embedder

#         for file in files:
#             if not file.filename.endswith('.csv'):
#                 raise HTTPException(status_code=400, detail=f"File {file.filename} is not a CSV file")
            
#             file_count += 1
#             content = await file.read()

#             # Use CSVLoader to load the content from the CSV file
#             with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
#                 temp_file.write(content)
#                 temp_file_path = temp_file.name

#             loader = CSVLoader(temp_file_path)
#             docs = loader.load()

#             # Convert documents into a string for processing
#             document_content = ' '.join([str(doc) for doc in docs])

#             runnable = prompt | llm
#             response = runnable.invoke({"proposal": document_content})  # Processing the concatenated document content

#             # Write the response to a temporary file for embedding
#             with tempfile.NamedTemporaryFile(delete=False, mode='w+', suffix=".txt") as temp_file:
#                 temp_file.write(str(response))
#                 temp_file_path = temp_file.name

#             # Embed the content of the temporary file
#             with open(temp_file_path, 'r') as content_file:
#                 file_content = content_file.read()
#                 #db = embedder.temp_embed(file_content)  # Assuming temp_embed method can handle this directly

#             # Log the process
#             logging.info(f"Response from LangChain for file {file.filename}: {response}")
#             logging.info(f"Embedded data stored for file {file.filename}")

#             results.append(response)
#             os.remove(temp_file_path)  # Clean up the temporary file

#         logging.info(f"Total number of files processed: {file_count}, Names of files: {[file.filename for file in files]}")
#         return {"status": "success", "data": results, "file_count": file_count}
#     except HTTPException as e:
#         return JSONResponse(content={"error": str(e.detail)}, status_code=e.status_code)
#     except Exception as e:
#         logging.error(f"Error processing files: {e}")
#         return JSONResponse(content={"error": str(e)}, status_code=500)

## PROBLEMATIC CSV_LOADER

# @router.post("/matching")
# async def generate_matching_criteria(files: List[UploadFile] = File(...)):
#     try:
#         results = []
#         file_count = 0
#         embedder = Data_Embedder()

#         for file in files:
#             if not file.filename.endswith('.csv'):
#                 raise HTTPException(status_code=400, detail=f"File {file.filename} is not a CSV file")
            
#             file_count += 1
#             content = await file.read()

#             with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
#                 temp_file.write(content)
#                 temp_file_path = temp_file.name

#             try:
#                 loader = CSVLoader(temp_file_path)
#                 docs = loader.load()
#                 document_content = ' '.join([str(doc) for doc in docs])
#             except Exception as e:
#                 logging.error(f"Failed to load or process CSV file {temp_file_path}: {e}")
#                 continue  # Skip processing this file

#             runnable = prompt | llm
#             try:
#                 response = runnable.invoke({"proposal": document_content})
#             except Exception as e:
#                 logging.error(f"LangChain model invocation failed for {temp_file_path}: {e}")
#                 continue  # Skip embedding this response

#             with tempfile.NamedTemporaryFile(delete=False, mode='w+', suffix=".txt") as temp_file:
#                 temp_file.write(str(response))
#                 response_file_path = temp_file.name

#             try:
#                 with open(response_file_path, 'r') as content_file:
#                     file_content = content_file.read()
#                     db = embedder.temp_embed(file_content)
#             except Exception as e:
#                 logging.error(f"Embedding failed for file {response_file_path}: {e}")
#                 continue  # Skip logging this response

#             logging.info(f"Response from LangChain for file {file.filename}: {response}")
#             logging.info(f"Embedded data stored for file {file.filename}")

#             results.append(response)
#             os.remove(temp_file_path)
#             os.remove(response_file_path)

#         logging.info(f"Total number of files processed: {file_count}, Names of files: {[file.filename for file in files]}")
#         return {"status": "success", "data": results, "file_count": file_count}
#     except HTTPException as e:
#         return JSONResponse(content={"error": str(e.detail)}, status_code=e.status_code)
#     except Exception as e:
#         logging.error(f"Unexpected error: {e}")
#         return JSONResponse(content={"error": str(e)}, status_code=500)

@router.post("/matching")
async def generate_matching_criteria(files: List[UploadFile] = File(...)):
    try:
        results = []
        file_count = 0
        embedder = Data_Embedder()  # Create an instance of your Data_Embedder

        for file in files:
            if not file.filename.endswith('.csv'):
                raise HTTPException(status_code=400, detail=f"File {file.filename} is not a CSV file")
            
            file_count += 1
            content = await file.read()

            # Process the CSV content using LangChain LLM
            runnable = prompt | llm
            response = runnable.invoke({"proposal": content.decode()})  # Decoding may be necessary depending on your content
            
            # Log the LLM response
            logging.info(f"Response from LangChain for file {file.filename}: {response}")

            # Embed the response content using Data_Embedder
            try:
                embedding_result = embedder.temp_embed(str(response))
                logging.info(f"Embedding successful for file {file.filename}")
            except Exception as e:
                logging.error(f"Failed to embed data for file {file.filename}: {e}")
                continue  # Continue to next file if embedding fails

            results.append(response)

        logging.info(f"Total number of files processed: {file_count}, Names of files: {[file.filename for file in files]}")
        return {"status": "success", "data": results, "file_count": file_count}
    except HTTPException as e:
        return JSONResponse(content={"error": str(e.detail)}, status_code=e.status_code)
    except Exception as e:
        logging.error(f"Unexpected error during processing: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)