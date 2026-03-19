# BlogHelper

네이버 블로그용 음식점 사진을 업로드하고 카테고리별로 정리하는 웹앱입니다.

## 현재 기능

- 여러 장 이미지 업로드
- 업로드 후 결과 페이지에서 카테고리별 사진 확인
- 카테고리: 외부전경, 주차장, 내부, 메뉴, 음식
- 로컬 CLIP 분류기 구조 포함
  - `torch`, `transformers`가 설치되어 있으면 CLIP 사용
  - 설치되지 않았으면 임시 휴리스틱 분류기로 fallback

## 실행 방법

### 1. 가상환경 생성 및 패키지 설치

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 서버 실행

프로젝트 루트에서:

```powershell
python -m uvicorn backend.main:app --reload
```

또는 `backend` 폴더에서:

```powershell
python -m uvicorn main:app --reload
```

브라우저에서 `http://127.0.0.1:8000/` 접속

## 로컬 CLIP 분류기

데스크톱 GPU 환경에서 로컬 CLIP 분류기를 쓰려면 `torch`, `torchvision`, `transformers`가 필요합니다.

환경 변수:

```powershell
$env:BLOGHELPER_CLASSIFIER_MODE="auto"
```

강제로 휴리스틱 분류기만 쓸 때:

```powershell
$env:BLOGHELPER_CLASSIFIER_MODE="heuristic"
```

모델 ID 변경:

```powershell
$env:BLOGHELPER_CLIP_MODEL_ID="openai/clip-vit-base-patch32"
```

## Git에 포함되지 않는 항목

- `backend/venv`
- `backend/uploads`
- `backend/tuning_labels.json`
- `__pycache__`
