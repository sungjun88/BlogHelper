from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

try:
    from backend.image_classifier import (
        ALLOWED_IMAGE_EXTENSIONS,
        classify_image,
        get_category_metadata,
        get_enabled_categories,
        list_uploaded_images,
    )
except ModuleNotFoundError:
    from image_classifier import (
        ALLOWED_IMAGE_EXTENSIONS,
        classify_image,
        get_category_metadata,
        get_enabled_categories,
        list_uploaded_images,
    )


app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
UPLOAD_DIR = BASE_DIR / "uploads"
FRONTEND_DIR = PROJECT_ROOT / "frontend"
UPLOAD_DIR.mkdir(exist_ok=True)
FRONTEND_DIR.mkdir(exist_ok=True)

app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
app.mount("/assets", StaticFiles(directory=FRONTEND_DIR), name="assets")


def _build_image_url(filename: str) -> str:
    return f"/uploads/{quote(filename)}"


def _serialize_analysis(analysis: dict) -> dict:
    return {
        **analysis,
        "image_url": _build_image_url(analysis["filename"]),
    }


def _clear_upload_dir() -> None:
    for path in UPLOAD_DIR.iterdir():
        if path.is_file():
            path.unlink()


@app.get("/")
def root():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/results")
def results_page():
    return FileResponse(FRONTEND_DIR / "results.html")


@app.get("/health")
def health():
    return {"message": "API running"}


@app.get("/categories")
def get_categories():
    return {
        "categories": get_category_metadata(),
        "enabled_category_keys": [category.key for category in get_enabled_categories()],
    }


@app.get("/test-upload", response_class=HTMLResponse)
async def test_upload_page():
    return """
    <html>
        <head>
            <meta charset="utf-8">
            <title>파일 업로드 테스트</title>
        </head>
        <body>
            <h2>여러 이미지 업로드</h2>
            <form action="/upload" enctype="multipart/form-data" method="post">
                <input name="files" type="file" multiple>
                <button type="submit">업로드</button>
            </form>
        </body>
    </html>
    """


@app.post("/upload")
async def upload_images(files: list[UploadFile] = File(...)):
    uploaded_files = []
    ignored_files = []
    failed_files = []

    _clear_upload_dir()

    for file in files:
        file_extension = Path(file.filename or "").suffix.lower()
        if file_extension not in ALLOWED_IMAGE_EXTENSIONS:
            ignored_files.append(
                {
                    "filename": file.filename,
                    "reason": "이미지 파일만 업로드할 수 있습니다.",
                }
            )
            continue

        file_path = UPLOAD_DIR / file.filename
        content = await file.read()
        file_path.write_bytes(content)
        try:
            uploaded_files.append(_serialize_analysis(classify_image(file_path)))
        except Exception:
            if file_path.exists():
                file_path.unlink()
            failed_files.append(
                {
                    "filename": file.filename,
                    "reason": "이미지 분석에 실패했습니다.",
                }
            )

    return {
        "uploaded_count": len(uploaded_files),
        "ignored_count": len(ignored_files),
        "failed_count": len(failed_files),
        "categories": get_category_metadata(),
        "files": uploaded_files,
        "ignored_files": ignored_files,
        "failed_files": failed_files,
    }


@app.get("/images")
def get_uploaded_images():
    files = [path.name for path in list_uploaded_images(UPLOAD_DIR)]
    return {"images": files}


@app.get("/images/categorized")
def get_categorized_images():
    analyses = [
        _serialize_analysis(classify_image(path))
        for path in list_uploaded_images(UPLOAD_DIR)
    ]
    grouped_images = {category.key: [] for category in get_enabled_categories()}
    for analysis in analyses:
        grouped_images[analysis["category"]].append(analysis)

    return {
        "total_count": len(analyses),
        "categories": get_category_metadata(),
        "grouped_images": grouped_images,
    }
