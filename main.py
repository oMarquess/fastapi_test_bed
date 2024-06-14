from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from pydantic import BaseModel
from typing import List, Optional
from uuid import UUID, uuid4
from starlette.formparsers import MultiPartParser

MultiPartParser.max_file_size = 2 * 1024 * 1024
app = FastAPI()

class Task(BaseModel):
    id: Optional[UUID] = None
    title: str
    description: Optional[str] = None
    completed: bool = False

tasks = []

from routers.users import router

app.include_router(router)

@app.post("/tasks/", response_model= Task)
def create_task(task: Task):
    task.id = uuid4()
    tasks.append(task)
    return task

@app.get("/tasks/", response_model = List[Task])
def read_tasks():
    return tasks

@app.get("/tasks/{task_id}", response_model=Task)
def read_task(task_id: UUID):
    for task in tasks:
        if task.id == task_id:
            return task
    raise HTTPException(status_code=404, detail= "Task not found")

@app.put("/tasks/{task_id}", response_model=Task)
def update_task(task_id: UUID, task_update: Task):
    for idx, task in enumerate(tasks):
        if task.id == task_id:
            updated_task = task.copy(update = task_update.dict(exclude_unset=True))
            tasks [idx] = updated_task
            return updated_task        
    raise HTTPException(status_code=404, detail= "Task not found")


@app.delete("/tasks/{task_id: UUID}", response_model=Task)
def delete_task(task_id):
    for idx, task in enumerate(tasks):
        if task.id == task_id:
            return tasks.pop(idx)
    raise HTTPException(status_code = 404, detail="Task not found")      

#faster
# @app.post("/upload/")
# async def file_endpoint(uploaded_file: UploadFile):
#     content = await uploaded_file.read()
#     print(content)


# @app.post("/upload/")
# async def file_endpoint(uploaded_file: UploadFile):
#     print(uploaded_file.file)
#     print(uploaded_file._in_memory)
#     """
#     if file_size > max_size store on disk in a temp file
#     else store in memory

#     """
#     chunk_size = 3
#     while True:
        
#         chunk = await uploaded_file.read(chunk_size)
#         if not chunk:
#             break
#         print(chunk)

#     return Response("OK")


# @app.post("/upload/")
# async def file_endpoint(uploaded_file: bytes = File()):
#     content = uploaded_file
#     print(content)

@app.post("/upload")
async def endpoint(request: Request):
    async for data in request.stream():
        print(data)



if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
