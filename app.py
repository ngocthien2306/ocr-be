import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from src.utils.project_config import project_config
from fastapi.middleware.cors import CORSMiddleware
from src.utils.utils import *
from src.router import ocr_router
from src.router import home_router

app = FastAPI(
    title=project_config.DOCS_TITLE,
    debug=project_config.DEBUG,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ocr_router.router, tags=["OCR"])
app.include_router(home_router.router, tags=["HOME"])
app.mount("/", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=project_config.SERVER_BE_IP,
        port=project_config.BE_PORT,
        ssl_keyfile="./key.pem",
        ssl_certfile="./cert.pem",
    )
