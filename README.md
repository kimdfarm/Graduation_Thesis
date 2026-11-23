# GitHub 사용자 활동 분석 및 청크 기반 데이터 가치 검증기

졸업작품의 GitHub 활동 기반 문서 생성 파이프라인을 독립 모듈로 분리하여, **특정 사용자(author)의 저장소 기여 데이터**를 수집·정제하고 **청크(Chunk) 분할 및 압축 전후의 데이터 가치(토큰 효율, 정보 보존율, 생성 품질)**를 검증하기 위한 실험용 프로젝트입니다.

## 연구 및 검증 목적
장문/다량 로그(커밋, PR, 이슈)를 LLM에 그대로 입력할 때 발생하는 **컨텍스트 과부하 및 노이즈 희석** 문제를 완화하기 위해, 속성(커밋/PR·이슈) 및 토큰 예산 단위로 청크를 나누고 정제하는 방식이 실제로 유효한지 비교·검증합니다.

## 주요 기능
1. **GitHub 사용자 기반 저장소 탐색 (`/api/develop/github/repositories`)**
   - GitHub 토큰과 사용자 ID(`github_id`), 기간 필터를 바탕으로 소유·기여 저장소를 조회합니다.
   - 포크(Fork) 저장소의 경우 해당 사용자의 실제 커밋 이력이 있을 때만 포함합니다.
2. **특정 저장소 내 사용자 활동 수집 (`/api/github/analyze`)**
   - 선택한 저장소의 언어 분포, 커밋 이력(페이지네이션 포함), 이슈, PR 데이터를 GraphQL로 수집합니다.
3. **사용자 중심 전처리 및 청크 분할 (`/api/github/evaluate-chunks`)**
   - 지정한 작성자(`target_author`)와 기간 기준으로 커밋·PR·이슈를 필터링하고 중복 커밋을 병합합니다.
   - 원본 정제 데이터 대비 청크 분할 후의 **추정 토큰 수, 토큰 절감률(%), 항목 보존율(%), 청크 수**를 산출합니다.
4. **구조화 결과 생성 (`/api/resumes/github-generate`)**
   - 청크 적용 여부(`force_chunking`)와 토큰 예산을 달리하며 구조화 생성 결과를 비교할 수 있습니다.

## 빠른 실행
```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. 환경 변수 설정 (.env)
# GROQ_API_KEY=...
# PRIMARY_MODEL=openai/gpt-oss-120b
# FALLBACK_MODEL=openai/gpt-oss-20b

# 3. 백엔드 실행
uvicorn backend.main:app --reload --port 8000

# 4. 프론트엔드 실행
streamlit run frontend/app.py
```