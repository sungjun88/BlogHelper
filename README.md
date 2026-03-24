# BlogHelper

네이버 블로그용 사진과 동영상을 업로드해서 카테고리별로 정리하고, 결과를 직접 수정하면서 분류기를 점점 개선할 수 있는 도구입니다.

## 주요 기능

- 이미지 파일 업로드
- 동영상 파일 업로드
  - 동영상 내용 전체를 분석하지 않고, 대표 프레임 썸네일 1장을 추출해 그 이미지 기준으로 분류
- 업로드 진행률, 경과 시간, 예상 남은 시간 표시
- 결과 페이지에서 카테고리별 분류 결과 확인
- 드래그앤드롭으로 카테고리 수정
- `학습` 버튼으로 현재 결과를 라벨로 저장하고 즉시 학습
- `ETC` 카테고리 지원
  - 보관은 하되 학습과 평가에서는 제외
- 로컬 CLIP 분류 지원
- 학습된 임베딩 분류기 우선 사용
- hover 미리보기
  - 결과 화면에는 100x100 썸네일만 표시
  - 마우스를 올리면 최대 400px 크기의 상세 미리보기와 파일 정보 표시

## 카테고리

- `exterior`
- `parking`
- `interior`
- `menu`
- `food`
- `etc`

참고:

- `thumbnail` 카테고리 정의는 코드에 남아 있지만 기본 UI 분류 흐름에서는 사용하지 않는 상태입니다.
- `etc`는 학습 제외용 카테고리입니다.

## 실행 방법

### 1. 가상환경 생성 및 패키지 설치

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 서버 실행

프로젝트 루트에서 실행:

```powershell
python -m uvicorn backend.main:app --reload
```

브라우저에서 아래 주소로 접속:

```text
http://127.0.0.1:8000/
```

## 기본 사용 흐름

### 1. 업로드

- 이미지 또는 동영상 파일을 업로드합니다.
- 지원 예시:
  - 이미지: `png`, `jpg`, `jpeg`, `webp`, `bmp`
  - 동영상: `mp4`, `mov`, `avi`, `mkv`, `webm`, `m4v`

### 2. 자동 분류 확인

- 업로드가 끝나면 결과 페이지로 이동합니다.
- 각 항목은 카테고리별로 정리되어 보입니다.
- 동영상은 썸네일 기준으로 분류되며 hover 정보에도 표시됩니다.

### 3. 분류 수정

- 결과 페이지에서 썸네일 카드를 다른 카테고리로 드래그앤드롭합니다.
- 애매하거나 학습에서 제외하고 싶은 항목은 `ETC`로 이동합니다.

### 4. 학습

- `학습` 버튼을 누르면 현재 화면의 분류 결과를 라벨로 저장합니다.
- 저장된 라벨을 기준으로 분류기를 다시 학습합니다.
- 즉, 자동 분류만 봤다고 바로 학습되지는 않고, 사용자가 `학습` 버튼을 눌렀을 때만 반영됩니다.

## 동영상 처리 방식

동영상은 대표 프레임 한 장만 뽑아서 분류합니다.

현재 썸네일 추출 기준:

1. 첫 프레임
2. 전체 프레임 수의 1/3 지점
3. 전체 프레임 수의 1/2 지점

이 순서로 읽기를 시도해서 처음 성공한 프레임을 사용합니다.

## 분류기 동작 방식

현재 분류 우선순위는 다음과 같습니다.

1. 학습된 CLIP 임베딩 분류기
2. 로컬 CLIP zero-shot 분류기
3. 휴리스틱 fallback 분류기

즉, 학습된 모델 파일이 있으면 그 모델을 먼저 사용합니다.

## 환경 변수

### 기본 모드

```powershell
$env:BLOGHELPER_CLASSIFIER_MODE="auto"
```

### 휴리스틱 강제

```powershell
$env:BLOGHELPER_CLASSIFIER_MODE="heuristic"
```

### CLIP 모델 ID 변경

```powershell
$env:BLOGHELPER_CLIP_MODEL_ID="openai/clip-vit-base-patch32"
```

### 학습 모델 저장 경로 변경

```powershell
$env:BLOGHELPER_TRAINED_MODEL_PATH="D:\path\to\trained_classifier.npz"
```

## 튜닝 스크립트

수동으로 라벨 파일을 만들고 평가/학습할 수도 있습니다.

### 라벨 템플릿 생성

```powershell
.venv\Scripts\python.exe backend\tune_classifier.py --init-labels
```

### 현재 성능 평가

```powershell
.venv\Scripts\python.exe backend\tune_classifier.py
```

### 라벨 기반 학습

```powershell
.venv\Scripts\python.exe backend\tune_classifier.py --train
```

### 저장된 모델 요약 확인

```powershell
.venv\Scripts\python.exe backend\tune_classifier.py --show-model
```

## 생성되는 주요 파일

- `backend/uploads/`
  - 업로드된 원본 파일 저장
- `backend/tuning_labels.json`
  - 파일별 정답 라벨 저장
- `backend/trained_classifier.npz`
  - 학습된 분류기 저장 파일

## Git에 포함하지 않는 항목

- `backend/uploads/`
- `backend/tuning_labels.json`
- `backend/trained_classifier.npz`
- `.venv/`
- `backend/venv/`
- `__pycache__/`

## 주의 사항

- `학습` 버튼은 현재 화면의 결과 전체를 정답으로 간주합니다.
- 잘못 분류된 항목만 수정하는 것도 가능하지만, 수정 없이 `학습`을 누르면 현재 자동 분류 결과가 그대로 학습에 반영됩니다.
- `ETC`에 들어간 항목은 저장은 되지만 학습 데이터에서는 제외됩니다.
