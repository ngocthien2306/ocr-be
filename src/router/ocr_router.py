from fastapi import APIRouter, HTTPException, Request
from src.utils import model
from src.service import ocr_service
import json
import cv2
import traceback

router = APIRouter(prefix="/ocr")

@router.get("/intr")
async def intr():
    return {"message": "OCR Powerfull", "author": "I-SOFT AI Team"}

@router.post("/ocr-image")
async def process_image_endpoint(image_data: model.ImageBase64):
    try:
        image = ocr_service.base64_to_image(image_data.image_base64)
        cv2.imwrite(r".\public\output\API_raw_img.jpg", image)

        image_data.point1 = [
            int(image_data.point1[0] * image.shape[1]),
            int(image_data.point1[1] * image.shape[0]),
        ]
        image_data.point2 = [
            int(image_data.point2[0] * image.shape[1]),
            int(image_data.point2[1] * image.shape[0]),
        ]
        print(image_data.point1, image_data.point2)
        img_result, ocr_result = ocr_service.OCR_crop_image(
            image,
            scan_method=2,
            point1=image_data.point1,
            point2=image_data.point2,
        )
        cv2.imwrite(r".\public\output\API_result_img.jpg", img_result)
        # Chuyển đổi hình ảnh đã xử lý sang base64 string
        img_result_base64 = ocr_service.image_to_base64(img_result)

        # Trả về hình ảnh đã xử lý và kết quả
        if ocr_result is None:
            return {
                "base64_image": img_result_base64,
                "rpm": "",
                "oilt": "",
            }
        else:
            return {
                "base64_image": img_result_base64,
                "rpm": ocr_result[0],
                "oilt": ocr_result[1],
            }
    except Exception as e:
        print(traceback.print_exc())
        raise HTTPException(status_code=400, detail=str(e))


