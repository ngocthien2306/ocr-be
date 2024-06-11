from pydantic import BaseModel, root_validator
from typing import List


class ImageBase64(BaseModel):
    image_base64: str
    point1: List[float]
    point2: List[float]

    @root_validator(pre=True)
    def validate_points(cls, values):
        # Ép các phần tử trong point1 và point2 sang kiểu float (trường hợp input là float)
        values["point1"] = list(map(float, values["point1"]))
        values["point2"] = list(map(float, values["point2"]))
        return values

    @root_validator
    def check_points_length(cls, values):
        if len(values["point1"]) != 2:
            raise ValueError("point1 must have exactly 2 elements")
        if len(values["point2"]) != 2:
            raise ValueError("point2 must have exactly 2 elements")
        return values
