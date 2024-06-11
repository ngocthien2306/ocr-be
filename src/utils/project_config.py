import os
from dotenv import load_dotenv
from pydantic import BaseSettings

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
load_dotenv(os.path.join(BASE_DIR, ".env"))


class ProjectConfig(BaseSettings):
    DEBUG: bool = True
    DOCS_TITLE: str = os.getenv("DOCS_TITLE", "Backend OCR")
    BE_PORT: int = os.getenv("BE_PORT", 8080)
    SERVER_BE_IP: str = os.getenv("SERVER_BE_IP")


project_config = ProjectConfig()

print(project_config)
