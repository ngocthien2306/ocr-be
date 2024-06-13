from fastapi import APIRouter, HTTPException, Request
from src.utils import model
from fastapi.responses import FileResponse
from src.service import ocr_service
import json
import cv2
import traceback

router = APIRouter(prefix="")

@router.get("/")
async def home_intr():
    return FileResponse("static/index.html")