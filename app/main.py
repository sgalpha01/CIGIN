from fastapi import FastAPI, BackgroundTasks
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from app.predictions import api_names, predict_dum

from celery import Celery
from app.worker import create_task, create_task_two
from celery.result import AsyncResult
from fastapi.responses import JSONResponse

# from mangum import Mangum
import os

# stage = os.environ.get('STAGE', None)
# openapi_prefix = f"/{stage}" if stage else "/"
# openapi_prefix=openapi_prefix

DEVICE = "cuda"
app = FastAPI()
origins = ["*"]

# handler = Mangum(app)


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get(api_names[0])
async def predict(solute,solvent):
    task = create_task.delay(solute,solvent)
    return JSONResponse({'task_id': task.id})

@app.get('/predict_solubility/{task_id}')
async def post(task_id):
    task_result = AsyncResult(task_id)
    result = {
        'task_id': task_id,
        'task_status': task_result.status,
        'task_result' : task_result.result
    }
    return result
  
@app.get(api_names[2])
async def predict_two(solute):
    task = create_task_two.delay(solute)
    return JSONResponse({'task_id': task.id})

@app.get('/predict_solubility_json/{task_id}')
async def post(task_id):
    task_result = AsyncResult(task_id)
    result = {
        'task_id': task_id,
        'task_status': task_result.status,
        'task_result' : task_result.result
    }
    return result


# {'result': attach_drug_name()}
if __name__ == "__main__":
    uvicorn.run("app.main:app", port=8000)