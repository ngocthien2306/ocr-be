from src.PaddleOCR.paddleocr import draw_ocr, draw_ocr2
from fastapi import HTTPException
from src.utils.utils import *
import numpy as np
import base64
import gc
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as torchvision_T
from src.PaddleOCR.paddleocr import PaddleOCR
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large


def order_points(pts):
    """Rearrange coordinates to order:
    top-left, top-right, bottom-right, bottom-left"""
    rect = np.zeros((4, 2), dtype="float32")
    pts = np.array(pts)
    s = pts.sum(axis=1)
    # Top-left point will have the smallest sum.
    rect[0] = pts[np.argmin(s)]
    # Bottom-right point will have the largest sum.
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    # Top-right point will have the smallest difference.
    rect[1] = pts[np.argmin(diff)]
    # Bottom-left will have the largest difference.
    rect[3] = pts[np.argmax(diff)]
    # Return the ordered coordinates.
    return rect.astype("int").tolist()


def find_dest(pts):
    (tl, tr, br, bl) = pts
    # Finding the maximum width.
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # Finding the maximum height.
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # Final destination co-ordinates.
    destination_corners = [[0, 0], [maxWidth, 0], [maxWidth, maxHeight], [0, maxHeight]]

    return order_points(destination_corners)


def image_preproces_transforms(
    mean=(0.4611, 0.4359, 0.3905),
    std=(0.2193, 0.2150, 0.2109),
):
    common_transforms = torchvision_T.Compose(
        [
            torchvision_T.ToTensor(),
            torchvision_T.Normalize(mean, std),
        ]
    )

    return common_transforms


def load_model(num_classes=2, model_name="mbv3", checkpoint_path=None, device=None):

    if model_name == "mbv3":
        model = deeplabv3_mobilenet_v3_large(num_classes=num_classes)

    else:
        model = deeplabv3_resnet50(num_classes=num_classes)

    model.to(device)
    checkpoints = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoints, strict=False)
    model.eval()

    _ = model(torch.randn((2, 3, 384, 384)).to(device))

    return model


def extract(
    image_true=None,
    trained_model=None,
    preprocess_transforms=None,
    device=None,
    image_size=384,
    BUFFER=10,
):

    IMAGE_SIZE = image_size
    half = IMAGE_SIZE // 2

    imH, imW, C = image_true.shape

    image_model = cv2.resize(
        image_true, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST
    )

    scale_x = imW / IMAGE_SIZE
    scale_y = imH / IMAGE_SIZE

    image_model = preprocess_transforms(image_model)
    image_model = torch.unsqueeze(image_model, dim=0).to(device)

    with torch.no_grad():
        out = trained_model(image_model)["out"].cpu()

    del image_model
    gc.collect()

    out = (
        torch.argmax(out, dim=1, keepdims=True)
        .permute(0, 2, 3, 1)[0]
        .numpy()
        .squeeze()
        .astype(np.int32)
    )
    r_H, r_W = out.shape

    _out_extended = np.zeros((IMAGE_SIZE + r_H, IMAGE_SIZE + r_W), dtype=out.dtype)
    _out_extended[half : half + IMAGE_SIZE, half : half + IMAGE_SIZE] = out * 255
    out = _out_extended.copy()

    del _out_extended
    gc.collect()

    # Edge Detection.
    canny = cv2.Canny(out.astype(np.uint8), 225, 255)
    canny = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    contours, _ = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    page = sorted(contours, key=cv2.contourArea, reverse=True)[0]

    # ==========================================
    epsilon = 0.02 * cv2.arcLength(page, True)
    corners = cv2.approxPolyDP(page, epsilon, True)

    corners = np.concatenate(corners).astype(np.float32)

    corners[:, 0] -= half
    corners[:, 1] -= half

    corners[:, 0] *= scale_x
    corners[:, 1] *= scale_y

    # check if corners are inside.
    # if not find smallest enclosing box, expand_image then extract document
    # else extract document

    if not (
        np.all(corners.min(axis=0) >= (0, 0))
        and np.all(corners.max(axis=0) <= (imW, imH))
    ):

        left_pad, top_pad, right_pad, bottom_pad = 0, 0, 0, 0

        rect = cv2.minAreaRect(corners.reshape((-1, 1, 2)))
        box = cv2.boxPoints(rect)
        box_corners = np.int32(box)
        #     box_corners = minimum_bounding_rectangle(corners)

        box_x_min = np.min(box_corners[:, 0])
        box_x_max = np.max(box_corners[:, 0])
        box_y_min = np.min(box_corners[:, 1])
        box_y_max = np.max(box_corners[:, 1])

        # Find corner point which doesn't satify the image constraint
        # and record the amount of shift required to make the box
        # corner satisfy the constraint
        if box_x_min <= 0:
            left_pad = abs(box_x_min) + BUFFER

        if box_x_max >= imW:
            right_pad = (box_x_max - imW) + BUFFER

        if box_y_min <= 0:
            top_pad = abs(box_y_min) + BUFFER

        if box_y_max >= imH:
            bottom_pad = (box_y_max - imH) + BUFFER

        # new image with additional zeros pixels
        image_extended = np.zeros(
            (top_pad + bottom_pad + imH, left_pad + right_pad + imW, C),
            dtype=image_true.dtype,
        )

        # adjust original image within the new 'image_extended'
        image_extended[top_pad : top_pad + imH, left_pad : left_pad + imW, :] = (
            image_true
        )
        image_extended = image_extended.astype(np.float32)

        # shifting 'box_corners' the required amount
        box_corners[:, 0] += left_pad
        box_corners[:, 1] += top_pad

        corners = box_corners
        image_true = image_extended

    corners = sorted(corners.tolist())
    corners = order_points(corners)
    destination_corners = find_dest(corners)
    M = cv2.getPerspectiveTransform(
        np.float32(corners), np.float32(destination_corners)
    )

    final = cv2.warpPerspective(
        image_true,
        M,
        (destination_corners[2][0], destination_corners[2][1]),
        flags=cv2.INTER_LANCZOS4,
    )
    final = np.clip(final, a_min=0.0, a_max=255.0)

    return final


def scan_document1(img, document_rect=(70, 85, 400, 565)):
    orig_img = img.copy()
    # Repeated Closing operation to remove text from the document.
    kernel = np.ones((5, 5), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=1)

    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    cv2.grabCut(img, mask, document_rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
    grab_cut_img = img * mask2[:, :, np.newaxis]

    gray = cv2.cvtColor(grab_cut_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (11, 11), 0)
    # Edge Detection.
    canny = cv2.Canny(gray, 75, 200)
    canny = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    # Blank canvas.
    con = np.zeros_like(img)
    # Finding contours for the detected edges.
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # Keeping only the largest detected contour.
    page = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    con = cv2.drawContours(con, page, -1, (0, 255, 255), 3)

    # Loop over the contours.
    if len(page) == 0:
        print("Khong tim dc contour")
        raise HTTPException(status_code=400, detail="Not found contour")
    for c in page:
        # Approximate the contour.
        epsilon = 0.02 * cv2.arcLength(c, True)
        corners = cv2.approxPolyDP(c, epsilon, True)
        # If our approximated contour has four points
        if len(corners) == 4:
            break

    # Sorting the corners and converting them to desired shape.
    corners = sorted(np.concatenate(corners).tolist())

    # For 4 corner points being detected.
    corners = order_points(corners)
    destination_corners = find_dest(corners)

    h, w = img.shape[:2]
    # Getting the homography.
    M = cv2.getPerspectiveTransform(
        np.float32(corners), np.float32(destination_corners)
    )
    # Perspective transform using homography.
    final = cv2.warpPerspective(
        orig_img, M, (destination_corners[2][0], destination_corners[2][1])
    )

    return final


CHECKPOINT_MODEL_PATH_MBv3 = "model/model_mbv3_iou_mix_2C049.pth"
CHECKPOINT_MODEL_PATH_R50 = "model/model_mbv3_iou_mix_2C049.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_mb = load_model(
    num_classes=2,
    model_name="mbv3",
    checkpoint_path=CHECKPOINT_MODEL_PATH_MBv3,
    device=device,
)
model_res = load_model(
    num_classes=2,
    model_name="r50",
    checkpoint_path=CHECKPOINT_MODEL_PATH_R50,
    device=device,
)
preprocess_transforms = image_preproces_transforms()
ocr_model = PaddleOCR(use_angle_cls=True, lang="en")


def scan_document2(image, model_name="r50"):
    image_rgb = image[:, :, ::-1]
    if model_name == "mbv3":
        trained_model = model_mb
    elif model_name == "r50":
        trained_model = model_res

    document = extract(
        image_true=image_rgb,
        trained_model=trained_model,
        preprocess_transforms=preprocess_transforms,
        device=device,
    )
    return document


def OCR(image):
    try:
        result = ocr_model.ocr(image, cls=True)
        # Assuming result, img_path, and font_path are defined elsewhere
        result = result[0]

        boxes = [line[0] for line in result]
        txts = [line[1][0] for line in result]
        scores = [line[1][1] for line in result]
    except:
        boxes = []
        txts = []
        scores = []

    return boxes, txts, scores


def base64_to_image(base64_str: str) -> np.ndarray:
    # Chuyển base64 string sang hình ảnh dạng numpy array
    image_data = base64.b64decode(base64_str)
    np_arr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image data")
    return image


def image_to_base64(image: np.ndarray) -> str:
    # Chuyển hình ảnh dạng numpy array sang base64 string
    _, buffer = cv2.imencode(".png", image)
    base64_str = base64.b64encode(buffer).decode("utf-8")
    return base64_str


def filter_txts(txts):
    new_txts = []
    ocr_result = []
    for txt in txts:
        if len(txt) > 2:
            new_txts.append(txt)
    print("ocr_results: ", new_txts)
    if len(new_txts) == 4:
        ocr_result = [new_txts[1], new_txts[3]]
    else:
        for new_text in new_txts:
            if "rpm" in new_text:
                ocr_result[0] = new_text
            elif "C" in new_text:
                ocr_result[1] = new_text
    print("ocr results after filter: ", ocr_result)
    return ocr_result


# def OCR_image(image: np.ndarray) -> (np.ndarray, str):
def OCR_crop_image(original_image, scan_method, point1, point2):
    """
    Function use to OCR an image

    Args:
        - image : an numpy image
        - scan_method : method for scanning document (1:using image processing and 2:using Deep Learning )

    Returns:
        img_result, ocr_result
    """
    image = crop_image(original_image, point1, point2, margin=(5, 5))
    cv2.imwrite("public\output\croped_for_scan.jpg", image)
    if scan_method == 1:
        scan_image = scan_document1(image, (0, 0, image.shape[0], image.shape[1]))
    elif scan_method == 2:
        scan_image = scan_document2(image, "mbv3")
    else:
        raise ValueError(
            "Wrong scan method passed. Must be one of '1' or '2'. (1:using image processing and 2:using Deep Learning ) "
        )

    scan_image = cv2.resize(scan_image, (480, 640))
    cv2.imwrite("public\output\scan.jpg", scan_image)
    cropped_image = crop_image(scan_image, (0, 435), (480, 510))
    cv2.imwrite("public\output\croped_text.jpg", cropped_image)
    cv2.rectangle(scan_image, (0, 435), (480, 510), (0, 255, 0), 2)
    try:
        boxes, txts, scores = OCR(cropped_image)
        # Update bounding box coordinates to match original image

        updated_boxes = update_box_coordination((0, 435), boxes)
        # Draw OCR results
        # im_show = draw_ocr(img, boxes, txts, scores,font_path=r"PaddleOCR\doc\fonts\simfang.ttf", drop_score=0.5)
        img_result = draw_ocr2(
            scan_image,
            updated_boxes,
            txts,
            scores,
            text_size=0.3,
            text_color=(255, 255, 255),
        )
        ocr_result = filter_txts(txts)
    except:
        img_result = scan_image
        ocr_result = None
    return img_result, ocr_result
