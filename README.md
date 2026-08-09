# PetFit Backend

영상 기반 AI 반려동물 생활환경 분석 서비스의 백엔드.

설계 문서는 `markdown/` 에 있다. 구현과 문서가 어긋나면 **문서를 먼저 고친다.**

| 문서 | 정본으로 삼는 범위 |
| --- | --- |
| AI 분석 정의서 | 점수 계산식, 감점 규칙, 프레임 추출 기준, 탐지 대상 12종 |
| AI 설계서 | 파이프라인 12단계 구성과 순서 |
| API 명세서 | 엔드포인트, 요청·응답 형식, 오류 코드 |
| DB 명세서 · DB 설계서 | 테이블, 제약, 인덱스 |
| 프롬프트 설계서 | System Prompt, 그룹별 지시문, Few-shot |
| AI 성능 평가서 · 성능평가 진행 가이드 | 모델 선정 기준과 측정 절차 |
| 작업 분담 | 담당 범위와 남은 작업 |
| 요구사항 정의서 · 기능 명세서 · MVP 기획안 | 서비스 범위 |

---

## 요구 사항

| 항목 | 버전 | 비고 |
| --- | --- | --- |
| Python | **3.10 고정** | 상위 버전으로 개발하지 않는다 |
| MySQL | **8.0.16 이상** | 미만은 CHECK 제약을 무시한다 |

### Python 3.10을 고정하는 이유

3.11 이상에서 개발하면 `asyncio.timeout` · `TaskGroup` · `StrEnum` · `datetime.UTC` 같은 상위 버전 전용 기능을 써도 **본인 환경에서는 통과한다.** 그 코드는 3.10을 쓰는 다른 팀원 환경에서만 깨지므로 발견이 늦어진다.

`python -m pytest` 는 다른 버전에서 실행하면 경고를 낸다.

---

## 설치

```bash
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1        # Windows PowerShell
source .venv/bin/activate         # macOS / Linux

pip install -r requirements.txt
```

버전 확인.

```bash
python -V     # Python 3.10.x
```

`3.10` 이 아니면 가상환경을 지우고 다시 만든다.

```bash
deactivate
Remove-Item -Recurse -Force .venv    # Windows
rm -rf .venv                         # macOS / Linux
```

### 객체 탐지 모델을 돌리는 경우에만

```bash
pip install -r requirements-ai.txt
```

**모델을 실행하는 장비에만 설치한다.** `torch` 가 1~2GB 를 차지한다.
탐지기는 `Detector` 규약 뒤에 있고 나머지 단계는 `StubDetector` 로 검증되므로,
API 개발·서비스 개발·테스트 전부 이것 없이 돌아간다.

---

## 환경 변수

`.env` 는 저장소에 없다. 각자 만든다.

```
APP_NAME=PetFit
DEBUG=false

DB_USER=petfit
DB_PASSWORD=
DB_HOST=localhost
DB_PORT=3306
DB_NAME=petfit

STORAGE_ROOT=./storage
YOLO_MODEL_PATH=./models/yolo26m.pt
YOLO_DEVICE=                      # mps | cuda | cpu. 비우면 자동 선택
LLM_PROVIDER=qwen
LLM_API_KEY=
```

`DATABASE_URL` 을 지정하면 위 `DB_*` 보다 우선한다. 컨테이너 배포용이며 보통은 비워둔다.

---

## 데이터베이스

```bash
python -m scripts.init_db          # 스키마·테이블 생성 후 검증
python -m scripts.init_db --check  # 생성하지 않고 상태만 확인
python -m scripts.init_db --drop   # 전체 삭제 후 재생성 (개발 전용)
```

검증 항목은 MySQL 버전, 세션 타임존, 테이블·인덱스·외래키다.

### 시각은 KST로 저장한다

`DATETIME(6)` 에 KST(UTC+9) 기준으로 저장하고, API 응답에 `+09:00` 오프셋을 붙인다.

`CURRENT_TIMESTAMP(6)` 는 세션 타임존을 따르므로 연결 시점에 `+09:00` 으로 고정한다. 고정하지 않으면 서버 OS 설정에 따라 값이 달라지고, **오류 없이 조용히 9시간 어긋난다.**

### 스키마 정본은 모델이다

`migrations/001_initial.sql` 은 파생물이므로 직접 수정하지 않는다.

```bash
python -m scripts.gen_ddl > migrations/001_initial.sql
```

---

## 테스트

```bash
python -m pytest
```

MySQL 서버가 없어도 실행된다. 서비스 계층 테스트는 SQLite 인메모리 DB를 쓰되 **모델의 CHECK 제약을 그대로 적용**한다.

---

## 구조

| 디렉터리 | 내용 | 담당 |
| --- | --- | --- |
| `app/utils/` | 반올림, 좌표 압축 면적, 시각 처리 | 공용 |
| `app/rules/` | 중요도, 공간별 적용, 감점률, 위험도 판정 | 공용 |
| `app/schemas/` | Enum, API 요청·응답 DTO | 공용 |
| `app/models/` `app/db/` | 테이블, 제약, 세션 | 백엔드 B |
| `app/services/` | 저장소, 상태 전이, 큐 | 백엔드 B |
| `app/api/` `app/main.py` | 라우터, 예외 핸들러 | 백엔드 A |
| `app/ai/` | 파이프라인 계약, 점수 산출(11단계), 스텁, 조립 | 공용 |
| `app/ai/vision/` | 영상 처리·탐지·추적·시각화 (1~10단계) | AI C |
| `app/ai/prompts.py` `validation.py` `environment_analysis.py` `llm/` | 환경 분석 (12단계) | AI D |

### 경계가 세 곳 있다

경계마다 규약이 있고, 양쪽은 서로의 내부를 모른다. 구현이 바뀌어도 반대편은 수정하지 않는다.

| 규약 | 가르는 것 |
| --- | --- |
| `ai/pipeline.py` 의 `Pipeline` | 서비스 계층 ↔ AI 계층 |
| `ai/vision/detector.py` 의 `Detector` | Vision ↔ 탐지 모델 |
| `ai/llm/base.py` 의 `VisionLLM` | 환경 분석 ↔ 생성 모델 |

`app/ai/real_pipeline.py` 가 셋을 엮어 `Pipeline` 을 만족시킨다. **세 담당의 산출물이 만나는 유일한 지점이다.**

모델 없이 전 구간이 돌아간다. `StubDetector` 와 `FakeLLM` 을 끼우면 영상 파일에서 최종 결과까지 검증된다. 실제 모델은 성능평가 후에 교체한다.

---

## 규칙

| 항목 | 내용 |
| --- | --- |
| 브랜치 | 개인 브랜치에서 작업 후 `main` 으로 병합 |
| 커밋 전 | `git status` 에 `.venv/` 가 뜨면 안 된다 |
| `requirements.txt` | **ASCII만 사용한다.** 구버전 pip가 한국어 Windows에서 비ASCII 문자를 읽지 못한다 |
| 점수 산출 | 규칙 기반이다. 생성형 AI는 점수를 만들지 않는다 |
| `progress` | `stage` 에서 파생된다. `progress_for()` 를 거치고 직접 대입하지 않는다 |
