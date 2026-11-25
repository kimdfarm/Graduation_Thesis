import json
from pathlib import Path
from typing import Any

import tiktoken

FINAL_BASE_PROMPT_TEMPLATE = """You are an expert developer resume formatting and structuring engine.

CRITICAL GOALS:
1. Analyze the aggregated GitHub repository data and organize them into the requested template sections (`template_sections`).
2. **Each Section must define its own unified `ai_columns` array at the section level** matching the provided template columns.
3. **DO NOT OMIT OR OVERSIMPLIFY DETAILS.** For each project or card, write rich, comprehensive resume descriptions covering architecture, technical challenges solved, specific features implemented, and concrete outcomes based *strictly* on the provided commit/PR/issue data. Avoid vague one-line summaries or raw log dumps.
4. All detail cards within that section **must share the exact same `ai_columns`**, acting like a uniform table where each card holds corresponding `column_values`.
5. Never invent or hallucinate missing details; stick strictly to provided facts.
6. Write clear, polished developer resume descriptions rather than copying raw commit headlines verbatim.
7. DETERMINE CARD COUNT BY REPOSITORY / DISTINCT PROJECT: Evaluate the aggregated repository data holistically and create ONE cohesive card per repository unless the data clearly contains multiple distinct standalone sub-projects. Do NOT split a single repository into multiple cards by chunk order.
8. CLEAN BULLET FORMATTING: Format multi-point fields (`핵심 구현 및 커밋 성과`, `트러블슈팅 및 개선` 등) as clean newline-separated bullet lines (`- ...`) rather than long run-on sentences."""

GITHUB_TEMPLATE_PRESETS: dict[str, dict[str, Any]] = {
    "CUSTOM_SINGLE": {
        "name": "커스텀 템플릿 (사용자 직접 컬럼 설정)",
        "desc": "원하는 추출 컬럼을 자유롭게 추가/삭제하여 맞춤 스키마로 구성합니다.",
        "section_title": "GitHub 활동 분석 및 추출 컬럼",
        "columns": ["프로젝트 개요 및 역할", "사용 기술 및 라이브러리", "핵심 구현 및 커밋 성과", "트러블슈팅 및 개선"],
    },
    "GITHUB_DEEP_DIVE": {
        "name": "GitHub 심층 분석 프로필",
        "desc": "아키텍처, 기술 스택, 커밋 및 문제 해결 성과를 중심으로 구성합니다.",
        "section_title": "GitHub 핵심 프로젝트 심층 분석",
        "columns": [
            "프로젝트 개요 및 역할 (Overview & Role)",
            "사용 기술 및 라이브러리 (Tech Stack)",
            "핵심 구현 내용 및 커밋 성과 (Key Features & Commits)",
            "트러블슈팅 및 아키텍처 개선 (Troubleshooting & Impact)",
        ],
    },
    "GITHUB_SYS_ARCHITECTURE": {
        "name": "시스템 아키텍처 및 백엔드 심층 분석",
        "desc": "시스템 구조, 데이터 흐름, 핵심 구현 및 성능/확장성 관점으로 구성합니다.",
        "section_title": "시스템 설계 및 아키텍처 심층 분석",
        "columns": [
            "System Overview & Purpose (시스템 개요 및 목적)",
            "Architecture & Data Flow (아키텍처 및 데이터 흐름)",
            "Core Backend Implementation (핵심 로직 및 API 구현)",
            "Performance Tuning & Scalability (성능 최적화 및 확장성)",
        ],
    },
    "GITHUB_TROUBLESHOOTING": {
        "name": "문제 해결 및 성능 최적화 중심",
        "desc": "병목/이슈 해결 과정과 정량적·기술적 개선 사항 위주로 구성합니다.",
        "section_title": "트러블슈팅 및 성능 개선 리포트",
        "columns": [
            "Project Background (배경 및 목표)",
            "Technical Challenges & Bottlenecks (기술적 한계 및 병목)",
            "Troubleshooting Steps (해결 과정 및 접근)",
            "Results & Improvements (개선 결과 및 성과)",
        ],
    },
    "GITHUB_RESUME_ALIGNED": {
        "name": "프로젝트 성과 요약형",
        "desc": "주요 기여, 기술 스택, 성과 지표를 간결하게 뽑아내기 위한 구성입니다.",
        "section_title": "GitHub 핵심 프로젝트 성과 요약",
        "columns": [
            "Project Title & Overview (프로젝트 명 및 소개)",
            "Tech Stack & Architecture (기술 스택 및 구조)",
            "Key Contributions & Features (주요 기여 및 구현)",
            "Quantitative Impact & Troubleshooting (성과 및 문제 해결)",
        ],
    },
}

BASE_PROMPT_TEMPLATE = """You are an objective software engineering activity analyst.
Analyze the provided GitHub activity records strictly grounded in source facts without hallucination.
Preserve key metrics, technical stacks, commits, PRs, issues, and troubleshooting outcomes.
Merge duplicate findings across chunks cleanly and organize the final output strictly into the configured template sections and columns in JSON format."""

SUB_BASE_PROMPT_TEMPLATE = """You are a strict technical data compression engine for developer activity analysis.
STRICT RULES:
1. ZERO FABRICATION: Every fact, commit message, PR/issue title, line count, repetition count, or tech stack MUST come strictly from the input chunk.
2. DO NOT CREATE RESUME SECTIONS OR CARDS: Only compress raw activity evidence into the flat JSON schema.
3. PRESERVE HIGH-SIGNAL EVIDENCE: Retain concrete technical actions, tech stacks (`tech_stacks`), key commits (`key_commits`), and PRs/issues (`key_prs_and_issues`).
4. DROP TRIVIAL NOISE: Remove meaningless commits/PRs/issues (e.g., typo, formatting) unless tied to concrete technical changes."""

CACHE_DIR = Path(__file__).resolve().parent / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
TRACE_FILE = CACHE_DIR / "experiment_trace.json"

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent / "experiments"
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_CSV_PATH = EXPERIMENT_DIR / "runs_summary.csv"

ALL_EXPLORE_TYPES = ["commits", "prs", "issues", "languages", "repo_meta"]
EXPLORE_LABELS = {
    "commits": "커밋 (Commits)",
    "prs": "Pull Requests (PRs)",
    "issues": "이슈 (Issues)",
    "languages": "언어/기술스택 (Languages)",
    "repo_meta": "저장소 메타정보 (Meta)",
}

try:
    _ENC = tiktoken.get_encoding("o200k_base")
except Exception:
    _ENC = None


def count_ui_tokens(text: str) -> int:
    if not text:
        return 0
    if _ENC is not None:
        return len(_ENC.encode(text))
    return max(1, len(text) // 2)


def calc_pure_prompt_tokens(template_config: dict[str, Any] | None = None) -> int:
    schema_part = json.dumps(template_config, ensure_ascii=False) if template_config else ""
    return count_ui_tokens(f"{BASE_PROMPT_TEMPLATE}\n{schema_part}".strip())


def compute_ui_sub_budgets(
    chunk_tokens_list: list[int],
    max_total_tokens: int,
    base_prompt_tokens: int,
    allocation_mode: str,
    total_sub_cap_input: int,
) -> tuple[list[int], int]:
    n = max(1, len(chunk_tokens_list))
    auto_pool = max(200, int((max_total_tokens - base_prompt_tokens) * 0.9))
    total_pool = int(total_sub_cap_input) if total_sub_cap_input > 0 else auto_pool

    if allocation_mode == "equal":
        per_chunk = max(50, total_pool // n)
        return [per_chunk] * n, total_pool

    sum_tok = sum(chunk_tokens_list) or 1
    budgets = [max(50, int(round(total_pool * (tok / sum_tok)))) for tok in chunk_tokens_list]
    return budgets, total_pool