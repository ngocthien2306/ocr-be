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

@router.get("/home")
async def home():
    return FileResponse("static/index.html")

@router.get("/404")
async def error():
    return FileResponse("static/404.html")

@router.get("/about")
async def error():
    return FileResponse("static/about.html")

@router.get("/contact")
async def contact():
    return FileResponse("static/contact.html")

@router.get("/faq")
async def faq():
    return FileResponse("static/faq.html")

@router.get("/feature")
async def feature():
    return FileResponse("static/feature.html")

@router.get("/project")
async def project():
    return FileResponse("static/project.html")

@router.get("/service")
async def service():
    return FileResponse("static/service.html")

@router.get("/team")
async def team():
    return FileResponse("static/team.html")

@router.get("/testimonial")
async def testimonial():
    return FileResponse("static/testimonial.html")
