import datetime
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import pandas as pd
import requests
import streamlit as st
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


def _format_card_value_html(val: Any) -> str:
    if isinstance(val, list):
        lines = [str(x).strip() for x in val if str(x).strip()]
    else:
        raw = str(val or "").strip()
        normalized = raw.replace(" · ", "\n• ")
        lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    if not lines:
        return "-"
    formatted_lines = []
    for line in lines:
        clean_line = line.lstrip("-• ").strip()
        formatted_lines.append(f"• {clean_line}")
    return "<br>".join(formatted_lines)


def _flatten_generated_sections_to_df(generated_sections: list[dict[str, Any]]) -> pd.DataFrame:
    """종합 LLM이 생성한 JSON(sections -> details -> column_values)을 표(DataFrame)로 평탄화"""
    rows: list[dict[str, Any]] = []
    for s_idx, sec in enumerate(generated_sections or []):
        if not isinstance(sec, dict):
            continue
        sec_title = sec.get("section_title") or sec.get("title") or f"섹션 {s_idx + 1}"
        sec_type = sec.get("section_type", "-")
        details = sec.get("details") or []
        if not details:
            rows.append({"섹션유형": sec_type, "섹션제목": sec_title, "항목제목": "-"})
            continue
        for d_idx, item in enumerate(details):
            if not isinstance(item, dict):
                continue
            row: dict[str, Any] = {
                "섹션유형": sec_type,
                "섹션제목": sec_title,
                "카드ID": item.get("id", f"card_{d_idx + 1}"),
                "항목제목": item.get("title", "-"),
            }
            col_vals = item.get("column_values") or {}
            if isinstance(col_vals, dict):
                for k, v in col_vals.items():
                    row[str(k)] = "\n".join(map(str, v)) if isinstance(v, list) else str(v)
            rows.append(row)
    return pd.DataFrame(rows)


def _flatten_sub_outputs_to_df(sub_prompts: list[dict[str, Any]]) -> pd.DataFrame:
    """서브 LLM 청크별 압축 JSON 결과를 표(DataFrame)로 변환"""
    rows: list[dict[str, Any]] = []
    for idx, sp in enumerate(sub_prompts or []):
        sub_out = sp.get("sub_llm_output") or {}
        if not isinstance(sub_out, dict):
            continue
        ret = sp.get("retention") or {}
        commits = sub_out.get("key_commits") or []
        prs_issues = sub_out.get("key_prs_and_issues") or []
        stacks = sub_out.get("tech_stacks") or []
        col_ev = sub_out.get("column_evidence") or {}

        rows.append({
            "청크번호": sp.get("chunk_index", idx + 1),
            "저장소": sub_out.get("repo_name") or sp.get("repo_name", "-"),
            "작성자": sub_out.get("target_author", "-"),
            "할당한도(tok)": sp.get("sub_llm_max_tokens", 0),
            "압축전(tok)": sp.get("tokens", 0),
            "압축후(tok)": sp.get("final_input_tokens", 0),
            "항목보존율(%)": ret.get("retention_rate_pct", 0.0),
            "기술스택": ", ".join(map(str, stacks)),
            "압축된_커밋_수": len(commits),
            "압축된_PR이슈_수": len(prs_issues),
            "압축된_핵심커밋": " | ".join(map(str, commits)),
            "압축된_PR이슈": " | ".join(map(str, prs_issues)),
            "템플릿_컬럼근거": json.dumps(col_ev, ensure_ascii=False) if col_ev else "-",
        })
    return pd.DataFrame(rows)


def _save_run_to_csvs(
    pipe: dict,
    chunk_limit_mode: str,
    run_label: str = "",
) -> str:
    """3-3 실행 결과를 runs_summary.csv(누적) + chunks_{run_id}.csv + items_{run_id}.csv + generated_{run_id}.csv 로 저장"""
    now_dt = datetime.datetime.now()
    run_id = now_dt.strftime("%Y%m%d_%H%M%S")
    run_sum = pipe.get("sub_run_summary") or {}
    u = pipe.get("token_usage") or {}
    comp = pipe.get("comparison") or {}
    sub_prompts = pipe.get("sub_prompts") or []
    cleaned_data = pipe.get("cleaned_data") or []

    repos = [p.get("repo_name", "-") for p in cleaned_data if isinstance(p, dict)]
    repo_str = ", ".join(repos) if repos else "-"

    summary_row = {
        "run_id": run_id,
        "실험라벨": run_label.strip() or f"실험_{run_id}",
        "실행시각": now_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "대상저장소": repo_str,
        "분할기준": chunk_limit_mode,
        "청크수": len(sub_prompts),
        "배분방식": "비례 배분" if run_sum.get("allocation_mode") == "proportional" else "공통 배분",
        "서브템플릿반영": "ON" if run_sum.get("apply_template_in_sub", True) else "OFF",
        "서브한도합계": run_sum.get("total_sub_pool", 0),
        "압축전토큰": run_sum.get("raw_sub_tokens", 0),
        "압축후토큰": run_sum.get("compressed_sub_tokens", 0),
        "토큰압축률(%)": run_sum.get("token_compression_pct", 0.0),
        "원본항목수": run_sum.get("total_raw_items", 0),
        "서브보존항목수": run_sum.get("retained_items", 0),
        "서브보존율(%)": run_sum.get("sub_item_retention_pct", 0.0),
        "종합근거반영률(%)": comp.get("direct_evidence_mention_pct", 0.0),
        "실측입력토큰": u.get("prompt_tokens", 0),
        "실측출력토큰": u.get("completion_tokens", 0),
        "실측총API토큰": u.get("total_tokens", 0),
    }
    sum_df_new = pd.DataFrame([summary_row])
    if SUMMARY_CSV_PATH.exists():
        sum_df_prev = pd.read_csv(SUMMARY_CSV_PATH, dtype={"run_id": str})
        sum_df_all = pd.concat([sum_df_prev, sum_df_new], ignore_index=True)
    else:
        sum_df_all = sum_df_new
    sum_df_all.to_csv(SUMMARY_CSV_PATH, index=False, encoding="utf-8-sig")

    chunk_rows = []
    item_rows = []
    final_blob = json.dumps(pipe.get("generated_sections") or [], ensure_ascii=False)

    for idx, sp in enumerate(sub_prompts):
        c_idx = sp.get("chunk_index", idx + 1)
        r_name = sp.get("repo_name", "-")
        ret = sp.get("retention") or {}
        raw_ch = sp.get("raw_chunk_data") or {}
        sub_out_blob = json.dumps(sp.get("sub_llm_output") or {}, ensure_ascii=False)

        chunk_rows.append({
            "run_id": run_id,
            "청크번호": c_idx,
            "저장소": r_name,
            "할당한도(tokens)": sp.get("sub_llm_max_tokens", 0),
            "압축전토큰": sp.get("tokens", 0),
            "압축후토큰": sp.get("final_input_tokens", 0),
            "원본항목수": ret.get("raw_item_count", 0),
            "보존항목수": ret.get("retained_item_count", 0),
            "항목보존율(%)": ret.get("retention_rate_pct", 0.0),
        })

        for item_type, key_name in [("Commit", "key_commits"), ("PR/Issue", "key_prs_and_issues")]:
            for raw_line in raw_ch.get(key_name, []) or []:
                core = raw_line.lstrip("- ").split(" (반복")[0].split(" (상태:")[0].strip()
                sub_kept = "O" if (core and core in sub_out_blob) else "X"
                final_kept = "O" if (core and core in final_blob) else "X"
                item_rows.append({
                    "run_id": run_id,
                    "청크번호": c_idx,
                    "저장소": r_name,
                    "유형": item_type,
                    "원문항목": raw_line,
                    "핵심문자열(core)": core,
                    "서브보존여부": sub_kept,
                    "종합반영여부": final_kept,
                })

    gen_df = _flatten_generated_sections_to_df(pipe.get("generated_sections") or [])
    if not gen_df.empty:
        gen_df.insert(0, "run_id", run_id)
    pd.DataFrame(chunk_rows).to_csv(EXPERIMENT_DIR / f"chunks_{run_id}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(item_rows).to_csv(EXPERIMENT_DIR / f"items_{run_id}.csv", index=False, encoding="utf-8-sig")
    gen_df.to_csv(EXPERIMENT_DIR / f"generated_{run_id}.csv", index=False, encoding="utf-8-sig")
    return run_id


def _load_trace() -> dict[str, Any]:
    if not TRACE_FILE.exists():
        return {}
    try:
        return json.loads(TRACE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_trace(data: dict[str, Any]) -> None:
    try:
        TRACE_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def _clear_streamlit_caches() -> None:
    st.cache_data.clear()
    st.cache_resource.clear()


def _clear_trace(scope: str) -> None:
    _clear_streamlit_caches()
    trace = _load_trace()

    if scope == "step3":
        trace.pop("pipeline_res", None)
        trace.pop("pipeline_step_stage", None)
        for k in ("pipeline_res", "pipeline_step_stage", "sub_tpl_mode_history"):
            st.session_state.pop(k, None)
        _save_trace(trace)

    elif scope == "step4":
        for fp in EXPERIMENT_DIR.glob("*.csv"):
            fp.unlink(missing_ok=True)
        for k in ("s4_tpl", "s4_alloc", "s4_run_pick", "s4_item_mode"):
            st.session_state.pop(k, None)

    elif scope == "stage2":
        for k in ("repos", "analysis_data", "pipeline_res", "pipeline_step_stage"):
            trace.pop(k, None)
            st.session_state.pop(k, None)
        st.session_state.pop("sub_tpl_mode_history", None)
        _save_trace(trace)

    elif scope == "all":
        if TRACE_FILE.exists():
            TRACE_FILE.unlink()
        for fp in EXPERIMENT_DIR.glob("*.csv"):
            fp.unlink(missing_ok=True)
        for k in (
            "repos",
            "analysis_data",
            "pipeline_res",
            "pipeline_step_stage",
            "sub_tpl_mode_history",
            "sb_gh_id",
            "sb_start_date",
            "sb_end_date",
            "s4_tpl",
            "s4_alloc",
            "s4_run_pick",
            "s4_item_mode",
        ):
            st.session_state.pop(k, None)


load_dotenv()
ENV_GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "").strip()

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


def _count_ui_tokens(text: str) -> int:
    if not text:
        return 0
    if _ENC is not None:
        return len(_ENC.encode(text))
    return max(1, len(text) // 2)


def _calc_pure_prompt_tokens(template_config: dict[str, Any] | None = None) -> int:
    schema_part = ""
    if template_config:
        schema_part = json.dumps(template_config, ensure_ascii=False)
    return _count_ui_tokens(f"{BASE_PROMPT_TEMPLATE}\n{schema_part}".strip())


def _calc_auto_sub_llm_tokens(max_total_tokens: int, chunk_count: int, base_prompt_tokens: int) -> int:
    available_for_merged = max(200, int(max_total_tokens) - int(base_prompt_tokens))
    safe_chunks = max(1, int(chunk_count))
    per_chunk_budget = int((available_for_merged * 0.9) // safe_chunks)
    return max(100, per_chunk_budget)


st.set_page_config(page_title="GitHub 활동 탐색 및 청크 가치 검증기", layout="wide")

st.markdown(
    """
    <style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 18px;
        font-weight: 600;
    }
    .resume-card {
        border: 1px solid rgba(128, 128, 128, 0.28);
        border-radius: 10px;
        padding: 18px 20px;
        margin-bottom: 16px;
        background-color: rgba(255, 255, 255, 0.02);
    }
    .resume-card-title {
        font-size: 1.08rem;
        font-weight: 700;
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 1px solid rgba(128, 128, 128, 0.2);
    }
    .resume-field-label {
        font-weight: 600;
        margin-top: 10px;
        margin-bottom: 4px;
        font-size: 0.95rem;
    }
    .resume-field-body {
        line-height: 1.6;
        font-size: 0.92rem;
        padding-left: 8px;
        margin-bottom: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "_trace_restored" not in st.session_state:
    saved = _load_trace()
    for k in ("repos", "analysis_data", "pipeline_res", "pipeline_step_stage"):
        if k in saved and k not in st.session_state:
            st.session_state[k] = saved[k]
    if "github_id" in saved and "sb_gh_id" not in st.session_state:
        st.session_state["sb_gh_id"] = saved["github_id"]
    if "start_date" in saved and saved["start_date"] and "sb_start_date" not in st.session_state:
        st.session_state["sb_start_date"] = datetime.date.fromisoformat(saved["start_date"][:10])
    if "end_date" in saved and saved["end_date"] and "sb_end_date" not in st.session_state:
        st.session_state["sb_end_date"] = datetime.date.fromisoformat(saved["end_date"][:10])
    st.session_state["_trace_restored"] = True

st.title("GitHub 사용자 활동 탐색 & 청크 기반 데이터 가치 검증")

# =========================================================
# 상단 캐시 초기화 영역
# =========================================================
top_cache_info, top_btn1, top_btn2, top_btn3, top_btn4 = st.columns([2.6, 1, 1, 1, 1])
with top_cache_info:
    has_analysis = bool(st.session_state.get("analysis_data"))
    has_pipe = bool(st.session_state.get("pipeline_res"))
    csv_count = len(list(EXPERIMENT_DIR.glob("*.csv"))) if EXPERIMENT_DIR.exists() else 0
    st.caption(
        f"캐시·흔적 상태: `{TRACE_FILE.name}` ({'저장됨' if TRACE_FILE.exists() else '없음'}) · "
        f"2단계: {'완료' if has_analysis else '없음'} · "
        f"3단계: {'완료' if has_pipe else '없음'} · "
        f"4단계 CSV: `{csv_count}개`"
    )
if top_btn1.button("캐시 초기화 (3단계)", width="stretch", key="top_btn_reset_step3"):
    _clear_trace("step3")
    st.rerun()
if top_btn2.button("캐시 초기화 (4단계 CSV)", width="stretch", key="top_btn_reset_step4"):
    _clear_trace("step4")
    st.rerun()
if top_btn3.button("캐시 초기화 (2·3단계)", width="stretch", key="top_btn_reset_stage2"):
    _clear_trace("stage2")
    st.rerun()
if top_btn4.button("전체 초기화 (1~4단계)", width="stretch", key="top_btn_reset_all"):
    _clear_trace("all")
    st.rerun()

# =========================================================
# 1. 사이드바: 인증 및 조회 대상 설정
# =========================================================
with st.sidebar:
    st.header("1. 인증 및 대상 설정")
    backend_url = st.text_input("백엔드 URL", value="http://localhost:8000", key="sb_backend_url")

    if ENV_GITHUB_TOKEN:
        st.success(".env에서 GITHUB_TOKEN을 불러왔습니다. (입력 생략 가능)")
        override_token = st.text_input(
            "다른 GitHub 토큰으로 덮어쓰기 (선택)",
            type="password",
            value="",
            placeholder="비워두면 .env 토큰 사용",
            key="sb_override_token",
        )
        github_token = override_token.strip() or ENV_GITHUB_TOKEN
    else:
        st.info(".env에 GITHUB_TOKEN이 없어 화면에서 직접 입력합니다.")
        github_token = st.text_input("GitHub Personal Access Token", type="password", key="sb_gh_token").strip()

    github_id = st.text_input("GitHub 사용자 ID", placeholder="예: kimdfarm", key="sb_gh_id").strip()
    col_d1, col_d2 = st.columns(2)
    start_date = col_d1.date_input("시작일", value=None, key="sb_start_date")
    end_date = col_d2.date_input("종료일", value=None, key="sb_end_date")

    auth_headers = {"Authorization": f"Bearer {github_token}"} if github_token else {}

    if st.button("저장소 검색", type="primary", width="stretch", key="btn_search_repos"):
        if not github_id:
            st.error("GitHub 사용자 ID를 입력하세요.")
        else:
            params: dict[str, Any] = {"github_id": github_id}
            if start_date:
                params["start_date"] = str(start_date)
            if end_date:
                params["end_date"] = str(end_date)
            resp = requests.get(
                f"{backend_url}/api/develop/github/repositories",
                params=params,
                headers=auth_headers,
                timeout=30,
            )
            if resp.status_code == 200:
                repos_data = resp.json().get("repositories", [])
                st.session_state["repos"] = repos_data
                _save_trace(
                    {
                        **_load_trace(),
                        "github_id": github_id,
                        "start_date": str(start_date) if start_date else None,
                        "end_date": str(end_date) if end_date else None,
                        "repos": repos_data,
                    }
                )
            else:
                st.error(f"저장소 조회 실패: {resp.text}")

    st.divider()
    st.markdown("**사용 흔적 및 초기화**")
    csv_count_sb = len(list(EXPERIMENT_DIR.glob("*.csv"))) if EXPERIMENT_DIR.exists() else 0
    st.caption(
        f"흔적 파일: `{TRACE_FILE.name}` ({'저장됨' if TRACE_FILE.exists() else '없음'}) · "
        f"CSV: `{csv_count_sb}개`"
    )

    rst_c1, rst_c2 = st.columns(2)
    if rst_c1.button("3단계 초기화", width="stretch", key="btn_reset_step3"):
        _clear_trace("step3")
        st.rerun()
    if rst_c2.button("4단계 CSV 초기화", width="stretch", key="btn_reset_step4"):
        _clear_trace("step4")
        st.rerun()

    rst_c3, rst_c4 = st.columns(2)
    if rst_c3.button("2·3단계 초기화", width="stretch", key="btn_reset_stage2"):
        _clear_trace("stage2")
        st.rerun()
    if rst_c4.button("전체 초기화 (1~4단계)", width="stretch", key="btn_reset_all"):
        _clear_trace("all")
        st.rerun()


# =========================================================
# 2단계 & 3단계 & 4단계 탭 분리
# =========================================================
tab_stage2, tab_stage3, tab_stage4 = st.tabs(
    [
        "2단계 · 저장소 선택 및 사용자 활동 분석",
        "3단계 · 청크 분할 가치 평가 및 생성 테스트",
        "4단계 · 해석 및 분석 (CSV 표 검증)",
    ]
)

# =========================================================
# TAB 1 (2단계)
# =========================================================
with tab_stage2:
    st.subheader("2. 저장소 선택(다중 가능), 탐색 항목·템플릿 설정 및 사용자 활동 분석")

    repos = st.session_state.get("repos", [])
    selected_repos: list[dict[str, Any]] = []

    if repos:
        repo_options = {
            f"{r['full_name']} (★ {r.get('stargazers_count', 0)} | {r.get('language') or '-'})": r
            for r in repos
        }
        default_repo_labels = [list(repo_options.keys())[0]] if repo_options else []
        picked_labels = st.multiselect(
            "검색된 저장소 중 분석할 저장소를 1개 이상 선택하세요",
            options=list(repo_options.keys()),
            default=default_repo_labels,
            key="ms_selected_repos",
        )
        selected_repos = [repo_options[lbl] for lbl in picked_labels]

    for _k, _default_val in [
        ("chk_commits", True),
        ("chk_prs", True),
        ("chk_issues", True),
        ("chk_languages", True),
        ("chk_repo_meta", False),
    ]:
        if _k not in st.session_state:
            st.session_state[_k] = _default_val

    st.markdown("**① 탐색 대상 항목 선택** (원하는 원천 데이터 체크박스를 선택하세요)")
    preset_col1, preset_col2, _ = st.columns([1, 1, 2])
    if preset_col1.button("커밋만 선택", key="btn_only_commits"):
        st.session_state["chk_commits"] = True
        st.session_state["chk_prs"] = False
        st.session_state["chk_issues"] = False
        st.session_state["chk_languages"] = False
        st.session_state["chk_repo_meta"] = False
        st.rerun()

    if preset_col2.button("전체 선택", key="btn_all_explore"):
        st.session_state["chk_commits"] = True
        st.session_state["chk_prs"] = True
        st.session_state["chk_issues"] = True
        st.session_state["chk_languages"] = True
        st.session_state["chk_repo_meta"] = True
        st.rerun()

    chk_cols = st.columns(5)
    chk_state_map = {
        "commits": chk_cols[0].checkbox(EXPLORE_LABELS["commits"], key="chk_commits"),
        "prs": chk_cols[1].checkbox(EXPLORE_LABELS["prs"], key="chk_prs"),
        "issues": chk_cols[2].checkbox(EXPLORE_LABELS["issues"], key="chk_issues"),
        "languages": chk_cols[3].checkbox(EXPLORE_LABELS["languages"], key="chk_languages"),
        "repo_meta": chk_cols[4].checkbox(EXPLORE_LABELS["repo_meta"], key="chk_repo_meta"),
    }
    selected_explore_types = [k for k, checked in chk_state_map.items() if checked]

    st.markdown("**② 추출 데이터 컬럼 & 템플릿 커스텀 설정**")
    tpl_keys = list(GITHUB_TEMPLATE_PRESETS.keys())
    selected_tpl_key = st.selectbox(
        "템플릿 프리셋 또는 커스텀 모드 선택",
        options=tpl_keys,
        index=0,
        format_func=lambda k: GITHUB_TEMPLATE_PRESETS[k]["name"],
        key="sb_tpl_preset",
    )
    active_tpl_preset = GITHUB_TEMPLATE_PRESETS[selected_tpl_key]
    st.caption(active_tpl_preset["desc"])

    if "custom_extract_columns" not in st.session_state:
        st.session_state["custom_extract_columns"] = list(active_tpl_preset["columns"])
    if "last_tpl_key" not in st.session_state:
        st.session_state["last_tpl_key"] = selected_tpl_key

    if st.session_state["last_tpl_key"] != selected_tpl_key:
        st.session_state["custom_extract_columns"] = list(active_tpl_preset["columns"])
        st.session_state["custom_section_title"] = active_tpl_preset["section_title"]
        st.session_state["last_tpl_key"] = selected_tpl_key

    custom_section_title = st.text_input(
        "템플릿 섹션 제목",
        value=active_tpl_preset["section_title"],
        key="custom_section_title",
    )

    col_add_in, col_add_btn = st.columns([3, 1])
    new_col_name = col_add_in.text_input(
        "추출/정리할 컬럼 항목명 추가",
        placeholder="예: 작성자/커밋횟수, 변경라인수, 주요기술스택, 트러블슈팅",
        key="ti_new_extract_col",
    )
    if col_add_btn.button("컬럼 추가", width="stretch", key="btn_add_extract_col"):
        trimmed = (new_col_name or "").strip()
        if trimmed and trimmed not in st.session_state["custom_extract_columns"]:
            st.session_state["custom_extract_columns"].append(trimmed)
            st.rerun()

    current_cols: list[str] = st.session_state.get("custom_extract_columns", [])
    active_extract_columns: list[str] = []

    if current_cols:
        st.caption("현재 설정된 추출 컬럼 (체크 유지 시 반영, ✕ 클릭 시 제거):")
        tag_cols = st.columns(min(4, max(1, len(current_cols))))
        remove_idx: int | None = None
        for idx, col_label in enumerate(current_cols):
            with tag_cols[idx % len(tag_cols)]:
                c_chk, c_del = st.columns([5, 1])
                keep = c_chk.checkbox(col_label, value=True, key=f"chk_ext_col_{idx}_{col_label}")
                if keep:
                    active_extract_columns.append(col_label)
                if c_del.button("✕", key=f"btn_del_ext_col_{idx}_{col_label}"):
                    remove_idx = idx
        if remove_idx is not None:
            st.session_state["custom_extract_columns"].pop(remove_idx)
            st.rerun()

    custom_template_payload = {
        "template_id": selected_tpl_key,
        "name": active_tpl_preset["name"],
        "sections": [
            {
                "section_type": "CUSTOM" if selected_tpl_key == "CUSTOM_SINGLE" else selected_tpl_key,
                "title": (custom_section_title or active_tpl_preset["section_title"]).strip(),
                "columns": active_extract_columns or list(active_tpl_preset["columns"]),
                "display_order": 1,
            }
        ],
    }

    target_author = st.text_input("분석할 사용자 활동 ID(author)", value=github_id, key="ti_target_author")

    if st.button("선택 저장소 활동 분석 실행", type="primary", key="btn_run_analyze"):
        if not github_token or not selected_repos:
            st.warning("GitHub 토큰과 1개 이상의 저장소를 선택/입력하세요.")
        else:
            active_types = selected_explore_types or ["commits"]
            collected_projects: list[dict[str, Any]] = []
            errors: list[str] = []

            progress = st.progress(0)
            for idx, repo_obj in enumerate(selected_repos):
                payload = {
                    "github_id": target_author or github_id,
                    "repo_name": repo_obj["full_name"],
                    "start_date": str(start_date) if start_date else None,
                    "end_date": str(end_date) if end_date else None,
                    "include_types": active_types,
                }
                resp = requests.post(
                    f"{backend_url}/api/github/analyze",
                    json=payload,
                    headers={"Authorization": f"Bearer {github_token}"},
                    timeout=120,
                )
                if resp.status_code == 200:
                    collected_projects.append(resp.json())
                else:
                    errors.append(f"{repo_obj['full_name']}: {resp.text}")
                progress.progress((idx + 1) / len(selected_repos))

            if collected_projects:
                analysis_payload = {
                    "projects_data": collected_projects,
                    "include_types": active_types,
                    "target_author": target_author or github_id,
                    "template_config": custom_template_payload,
                }
                st.session_state["analysis_data"] = analysis_payload
                _save_trace({**_load_trace(), "analysis_data": analysis_payload})
                names_str = ", ".join(p.get("full_name") or p.get("repo_name", "") for p in collected_projects)
                types_str = ", ".join(EXPLORE_LABELS[t] for t in active_types)
                cols_str = ", ".join(custom_template_payload["sections"][0]["columns"])
                st.success(
                    f"분석 완료 ({len(collected_projects)}개 저장소): {names_str} | "
                    f"탐색 항목: {types_str} | 컬럼: {cols_str} — 상단의 '3단계' 탭으로 이동해 청크 검증을 진행하세요."
                )

            for err_msg in errors:
                st.error(f"분석 실패 — {err_msg}")


def _compute_ui_sub_budgets(
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


# =========================================================
# TAB 2 (3단계)
# =========================================================
with tab_stage3:
    analysis_data = st.session_state.get("analysis_data")
    target_author_val = st.session_state.get("ti_target_author", github_id)

    if not analysis_data:
        st.info("먼저 **2단계** 탭에서 저장소를 선택하고 **선택 저장소 활동 분석 실행**을 완료해 주세요.")
    else:
        st.header("3. 청크 분할 가치 평가 및 백엔드 생성 기능 테스트")

        st.subheader("3-1. 토큰 예산 및 청크 분할 한도 설정")

        raw_proj_list = analysis_data.get("projects_data") or [analysis_data]
        active_inc = set(analysis_data.get("include_types") or ALL_EXPLORE_TYPES)

        raw_commits_n = 0
        raw_prs_n = 0
        raw_issues_n = 0
        raw_langs_set: set[str] = set()
        meta_repos_n = 0

        for proj in raw_proj_list:
            if not isinstance(proj, dict):
                continue
            if "commits" in active_inc:
                raw_commits_n += len(
                    ((((proj.get("defaultBranchRef") or {}).get("target") or {}).get("history") or {}).get("nodes")) or []
                )
            if "prs" in active_inc:
                raw_prs_n += len(((proj.get("pullRequests") or {}).get("nodes")) or [])
            if "issues" in active_inc:
                raw_issues_n += len(((proj.get("issues") or {}).get("nodes")) or [])
            if "languages" in active_inc:
                langs = proj.get("languages") or {}
                if isinstance(langs, dict):
                    for e in langs.get("edges") or []:
                        name = ((e or {}).get("node") or {}).get("name")
                        if name:
                            raw_langs_set.add(name)
                    for n in langs.get("nodes") or []:
                        name = (n or {}).get("name")
                        if name:
                            raw_langs_set.add(name)
            if "repo_meta" in active_inc:
                if any(k in proj for k in ("description", "stargazerCount", "forkCount", "repo_meta")):
                    meta_repos_n += 1

        raw_total_items = raw_commits_n + raw_prs_n + raw_issues_n
        raw_json_str = json.dumps(analysis_data, ensure_ascii=False)
        raw_tokens = _count_ui_tokens(raw_json_str)
        base_prompt_tokens = _calc_pure_prompt_tokens(analysis_data.get("template_config"))

        s1, s2, s3, s4 = st.columns(4)
        s1.metric(
            "원본 활동 항목 수",
            f"{raw_total_items:,}개",
            help=f"커밋 {raw_commits_n} / PR {raw_prs_n} / 이슈 {raw_issues_n}",
        )
        s2.metric("원본 생 데이터 토큰", f"{raw_tokens:,}", help=f"문자 수: {len(raw_json_str):,}자")
        s3.metric(
            "기본 프롬프트 토큰",
            f"{base_prompt_tokens:,}",
            help="원본 데이터를 제외한 순수 프롬프트 지시문·템플릿 스키마 고정 토큰 수",
        )
        s4.metric(
            "선택 부가정보 요약",
            f"언어 {len(raw_langs_set)}종 · 메타 {meta_repos_n}건",
            help=", ".join(sorted(raw_langs_set)) if raw_langs_set else "선택된 언어 없음",
        )

        chunk_limit_mode = st.radio(
            "청크 분할 기준 선택 (둘 중 하나만 적용)",
            options=["max_chunks", "chunk_tokens"],
            index=0,
            horizontal=True,
            format_func=lambda m: (
                "제한 청크 수 (최대 개수 기준)"
                if m == "max_chunks"
                else "청크당 최대 토큰 한도 기준"
            ),
            key="ui_chunk_limit_mode",
        )

        max_chunks_val: int | None = None
        chunk_token_limit_val: int | None = None

        c1, c2 = st.columns(2)
        max_total_tokens = c1.number_input(
            "전체 토큰 예산",
            min_value=500,
            max_value=50000,
            value=5000,
            step=500,
            key="ui_max_total_tokens",
        )

        if chunk_limit_mode == "max_chunks":
            max_chunks_val = int(
                c2.number_input(
                    "제한 청크 수 (최대 개수)",
                    min_value=1,
                    max_value=50,
                    value=6,
                    step=1,
                    key="ui_max_chunks",
                )
            )
        else:
            chunk_token_limit_val = int(
                c2.number_input(
                    "청크당 최대 토큰 한도",
                    min_value=200,
                    max_value=20000,
                    value=4000,
                    step=100,
                    key="ui_chunk_token_limit",
                )
            )

        col_opt1, col_opt2, col_opt3 = st.columns(3)
        force_chunking = col_opt1.checkbox("단일 청크일 때도 분할 파이프라인 강제 적용", value=True, key="ui_force_chunking")
        run_generation = col_opt2.checkbox("종합 프롬프트로 구조화 생성까지 실행", value=True, key="ui_run_generation")
        language = col_opt3.selectbox("출력 언어", options=["ko", "en"], index=0, key="ui_language")

        if st.button("① 3-2 생 데이터 청크 분할 실행 (AI 미사용 · 즉시 확인)", type="primary", width="stretch", key="btn_run_chunk_only"):
            req_payload = {
                "analysis_data": analysis_data,
                "target_author": target_author_val or github_id,
                "start_date": str(start_date) if start_date else None,
                "end_date": str(end_date) if end_date else None,
                "max_total_tokens": int(max_total_tokens),
                "chunk_limit_mode": chunk_limit_mode,
                "max_chunks": int(max_chunks_val) if max_chunks_val is not None else None,
                "chunk_token_limit": int(chunk_token_limit_val) if chunk_token_limit_val is not None else None,
                "force_chunking": force_chunking,
                "template_structure": analysis_data.get("template_config"),
                "language": language,
                "run_generation": False,
            }
            try:
                resp = requests.post(f"{backend_url}/api/github/chunk-preview", json=req_payload, timeout=60)
                if resp.status_code == 200:
                    chunk_data = resp.json()
                    st.session_state["pipeline_res"] = chunk_data
                    st.session_state["pipeline_step_stage"] = 1
                    _save_trace({**_load_trace(), "pipeline_res": chunk_data, "pipeline_step_stage": 1})
                    st.rerun()
                else:
                    st.error(f"청크 분할 실행 실패: {resp.text}")
            except requests.exceptions.ConnectionError:
                st.error(f"백엔드 서버({backend_url})에 연결할 수 없습니다. FastAPI 서버가 실행 중인지 확인해 주세요.")

        pipe = st.session_state.get("pipeline_res")

        if pipe:
            sub_prompts = pipe.get("sub_prompts", [])
            limits = pipe.get("chunk_limits", {})
            eval_rows = pipe.get("chunk_evaluations", [])
            has_ai_result = bool(pipe.get("generated_sections")) or any(
                sp.get("sub_llm_output") is not None for sp in sub_prompts
            )

            # =====================================================
            # [3-2] AI 미사용: 생 데이터 -> 전처리 -> 청크 분해
            # =====================================================
            st.divider()
            st.subheader("3-2. 생 데이터 정보 · 전처리 되는 정보 · 청크 분해된 정보 (AI 미사용)")

            raw_src_projects = analysis_data.get("projects_data") or [analysis_data]
            raw_src_tokens = _count_ui_tokens(json.dumps(analysis_data, ensure_ascii=False))
            cleaned_all = pipe.get("cleaned_data", [])
            cleaned_total_tokens = sum(_count_ui_tokens(json.dumps(c, ensure_ascii=False)) for c in cleaned_all)
            chunked_total_tokens = sum(int(sp.get("tokens", 0) or 0) for sp in sub_prompts)
            total_sub_prompts = sum(1 for sp in sub_prompts if sp.get("sub_prompt"))

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("생 데이터 토큰", f"{raw_src_tokens:,}")
            m2.metric("전처리 전체 토큰", f"{cleaned_total_tokens:,}")
            m3.metric("청크 분해 합산 토큰", f"{chunked_total_tokens:,}")
            m4.metric("실제 분할 청크 수", f"{limits.get('actual_chunk_count', len(sub_prompts))}개")
            m5.metric("생성된 서브 프롬프트 수", f"{total_sub_prompts}개")

            st.markdown("### ① 생 데이터 정보 (`analysis_data`)")
            raw_info_rows = []
            for p in raw_src_projects:
                if not isinstance(p, dict):
                    continue
                hist_nodes = (
                    ((((p.get("defaultBranchRef") or {}).get("target") or {}).get("history") or {}).get("nodes"))
                    or []
                )
                pr_nodes = ((p.get("pullRequests") or {}).get("nodes")) or []
                issue_nodes = ((p.get("issues") or {}).get("nodes")) or []
                raw_info_rows.append({
                    "저장소": p.get("full_name") or p.get("repo_name") or p.get("name") or "-",
                    "대상 작성자": p.get("target_author") or target_author_val or "-",
                    "원본 커밋 수": len(hist_nodes),
                    "원본 PR 수": len(pr_nodes),
                    "원본 이슈 수": len(issue_nodes),
                    "생 데이터 토큰": _count_ui_tokens(json.dumps(p, ensure_ascii=False)),
                })
            if raw_info_rows:
                st.dataframe(pd.DataFrame(raw_info_rows), width="stretch")
            with st.expander("생 데이터 원본 JSON 보기"):
                st.json(analysis_data)

            st.markdown("### ② 전처리 되는 정보 (`cleaned_data`)")
            st.caption(
                f"작성자 필터(`{target_author_val or github_id or 'ALL'}`), 동일 커밋 메시지 병합, "
                f"선택 항목(`{', '.join(sorted(active_inc))}`) 반영 후 정제된 데이터입니다."
            )
            raw_df = pd.DataFrame(pipe.get("raw_table_rows", []))
            if not raw_df.empty:
                st.dataframe(raw_df, width="stretch", height=220)
            else:
                st.info("표시할 전처리 활동 항목이 없습니다.")
            with st.expander("전처리된 데이터 JSON 보기"):
                st.json(cleaned_all)

            st.markdown("### ③ 청크 분해된 정보 (`chunks` / `sub_prompts`)")
            simplified_eval_rows = []
            for idx, sp in enumerate(sub_prompts):
                row_src = eval_rows[idx] if idx < len(eval_rows) and isinstance(eval_rows[idx], dict) else {}
                raw_chunk_tok = int(sp.get("tokens", row_src.get("청크_데이터_토큰수", 0)) or 0)
                sub_llm_tok = int(sp.get("fixed_sub_cmd_tokens", 0) or 0)

                simplified_eval_rows.append({
                    "청크번호": sp.get("chunk_index", row_src.get("청크번호", idx + 1)),
                    "저장소/구분": sp.get("repo_name", row_src.get("저장소/구분", "-")),
                    "커밋수": row_src.get("커밋수", len((sp.get("raw_chunk_data") or {}).get("key_commits", []))),
                    "PR/이슈수": row_src.get("PR/이슈수", len((sp.get("raw_chunk_data") or {}).get("key_prs_and_issues", []))),
                    "총항목수": row_src.get("총항목수", 0),
                    "한도초과여부": row_src.get("한도초과여부", "초과" if sp.get("exceeded") else "정상"),
                    "생 데이터 토큰 수": raw_chunk_tok,
                    "서브 LLM 토큰 수": sub_llm_tok,
                    "종합": sub_llm_tok + raw_chunk_tok,
                })

            if simplified_eval_rows:
                st.dataframe(pd.DataFrame(simplified_eval_rows), width="stretch")
            else:
                st.info("청크 평가 데이터가 없습니다.")

            with st.expander("청크별 분해 데이터 상세 JSON 보기"):
                for idx, sp in enumerate(sub_prompts):
                    c_idx = sp.get("chunk_index", idx + 1)
                    r_name = sp.get("repo_name", "-")
                    st.markdown(f"**청크 #{c_idx} — {r_name}** (`{sp.get('tokens', 0)} tokens`)")
                    st.json(sp.get("raw_chunk_data"))

            # =====================================================
            # [3-3] AI 실행 영역
            # =====================================================
            st.divider()
            st.subheader("3-3. 서브/종합 LLM 커스텀 · 템플릿 반영 설정 및 실시간 압축 실행")

            linked_tpl = analysis_data.get("template_config") or {
                "name": "기본 템플릿",
                "sections": [
                    {
                        "section_type": "CUSTOM",
                        "title": "사용자 기여 분석 및 구현 성과",
                        "columns": ["구현 및 문제 해결", "기술 스택 및 활동 근거"],
                    }
                ],
            }
            linked_sec = (linked_tpl.get("sections") or [{}])[0]
            linked_cols = linked_sec.get("columns") or []

            with st.container(border=True):
                st.markdown("**① 2단계 템플릿 연동 & 서브 LLM 반영 여부**")
                t_col1, t_col2 = st.columns([2, 1])
                with t_col1:
                    st.caption(
                        f"연동된 템플릿: **{linked_tpl.get('name', '-')}** · "
                        f"섹션: **{linked_sec.get('title') or linked_sec.get('section_title', '-')}**"
                    )
                    st.write("추출 컬럼:", ", ".join(f"`{c}`" for c in linked_cols) if linked_cols else "-")
                with t_col2:
                    apply_template_in_sub = st.toggle(
                        "서브 LLM에 템플릿 반영",
                        value=True,
                        help=(
                            "ON: 서브 LLM 단계에서 템플릿 컬럼명(target_columns)에 맞춰 근거를 압축합니다. (카드/섹션 미생성)\n"
                            "OFF: 서브 LLM은 템플릿 없이 순수 기술 사실만 압축하고, 최종 종합 LLM에서만 템플릿 섹션과 카드를 생성합니다."
                        ),
                        key="ui_apply_tpl_in_sub",
                    )
                    st.caption(
                        "현재 모드: **컬럼 기준 압축 (ON)**"
                        if apply_template_in_sub
                        else "현재 모드: **순수 사실 보존 압축 (OFF)**"
                    )

            if "saved_sub_base_prompt" not in st.session_state:
                st.session_state["saved_sub_base_prompt"] = SUB_BASE_PROMPT_TEMPLATE
            if "saved_final_base_prompt" not in st.session_state:
                st.session_state["saved_final_base_prompt"] = FINAL_BASE_PROMPT_TEMPLATE
            if "prompt_saved_at" not in st.session_state:
                st.session_state["prompt_saved_at"] = "초기 기본값"

            # 서브 LLM에는 카드/섹션이 아닌 순수 컬럼명 목록과 플랫 스키마만 표시
            sub_struct_with_tpl = json.dumps(
                {
                    "target_columns": linked_cols,
                    "schema_on": {
                        "repo_name": "",
                        "target_author": "",
                        "tech_stacks": [],
                        "key_commits": [],
                        "key_prs_and_issues": [],
                        "column_evidence": {c: [] for c in linked_cols},
                    },
                },
                ensure_ascii=False,
            )
            sub_struct_without_tpl = json.dumps(
                {
                    "mode": "template_agnostic_pure_fact_compression",
                    "schema_off": {
                        "repo_name": "",
                        "target_author": "",
                        "tech_stacks": [],
                        "key_commits": [],
                        "key_prs_and_issues": [],
                    },
                },
                ensure_ascii=False,
            )
            active_sub_struct_str = sub_struct_with_tpl if apply_template_in_sub else sub_struct_without_tpl
            sub_struct_tokens = _count_ui_tokens(active_sub_struct_str)

            primary_sec_preview = (linked_tpl.get("sections") or [{}])[0]
            example_cols_preview = primary_sec_preview.get("columns") or ["역할 및 구현", "기술 스택 및 근거"]
            final_struct_str = json.dumps(
                {
                    "doc_type": "RESUME",
                    "sections": [
                        {
                            "section_type": primary_sec_preview.get("section_type", "CUSTOM"),
                            "section_title": primary_sec_preview.get("section_title") or primary_sec_preview.get("title") or "섹션 제목",
                            "display_order": 1,
                            "ai_columns": example_cols_preview,
                            "details": [{"id": "card_1", "title": "항목 제목", "column_values": {c: "" for c in example_cols_preview}}],
                        }
                    ],
                    "template_sections": linked_tpl.get("sections") or [],
                },
                ensure_ascii=False,
            )
            final_struct_tokens = _count_ui_tokens(final_struct_str)

            sub_cmd_saved_tokens = _count_ui_tokens(st.session_state["saved_sub_base_prompt"])
            final_cmd_saved_tokens = _count_ui_tokens(st.session_state["saved_final_base_prompt"])
            sub_total_fixed_tokens = sub_cmd_saved_tokens + sub_struct_tokens
            final_total_fixed_tokens = final_cmd_saved_tokens + final_struct_tokens

            sub_is_custom = st.session_state["saved_sub_base_prompt"].strip() != SUB_BASE_PROMPT_TEMPLATE.strip()
            final_is_custom = st.session_state["saved_final_base_prompt"].strip() != FINAL_BASE_PROMPT_TEMPLATE.strip()

            expander_title = (
                f"② 프롬프트 설정 — 명령(커스텀) vs 구조(고정 · 서브 {sub_total_fixed_tokens:,}tok / 종합 {final_total_fixed_tokens:,}tok) "
                f"[{'저장됨' if (sub_is_custom or final_is_custom) else '기본값'} · {st.session_state['prompt_saved_at']}]"
            )

            with st.expander(expander_title, expanded=False):
                st.caption(
                    "**명령 프롬프트**는 커스텀 수정 후 저장할 수 있고, **구조 프롬프트**는 출력 파싱을 위해 고정으로 들어가는 기본 토큰입니다."
                )

                p_col1, p_col2 = st.columns(2)

                with p_col1:
                    st.markdown("#### 서브 LLM 프롬프트 (컬럼 기준 데이터 압축 전용)")
                    draft_sub_prompt = st.text_area(
                        "서브 명령 프롬프트 (수정 가능)",
                        value=st.session_state["saved_sub_base_prompt"],
                        height=150,
                        key="ui_draft_sub_base_prompt",
                    )
                    draft_sub_cmd_tok = _count_ui_tokens(draft_sub_prompt)
                    sub_has_unsaved = draft_sub_prompt.strip() != st.session_state["saved_sub_base_prompt"].strip()

                    st.markdown("**🔒 서브 구조 프롬프트 (변경 불가 · 카드/섹션 미포함)**")
                    st.code(active_sub_struct_str, language="json")

                    st.markdown(
                        f"- **명령 프롬프트 토큰(저장 기준)**: `{sub_cmd_saved_tokens:,} tokens` "
                        f"(입력창 `{draft_sub_cmd_tok:,} tokens`)\n"
                        f"- **구조 프롬프트 토큰(고정 기본값)**: `{sub_struct_tokens:,} tokens` "
                        f"({'컬럼 반영 ON 스키마' if apply_template_in_sub else '템플릿 OFF 스키마'})\n"
                        f"- **서브 프롬프트 합계 토큰**: **`{sub_total_fixed_tokens:,} tokens`**\n"
                        f"- **상태**: `{'⚠️ 미저장 변경 있음' if sub_has_unsaved else '✅ 저장 완료'}` · "
                        f"`{'커스텀' if sub_is_custom else '기본값'}`"
                    )

                with p_col2:
                    st.markdown("#### 종합 LLM 프롬프트 (저장소별 통합 & 카드 생성)")
                    draft_final_prompt = st.text_area(
                        "종합 명령 프롬프트 (수정 가능)",
                        value=st.session_state["saved_final_base_prompt"],
                        height=150,
                        key="ui_draft_final_base_prompt",
                    )
                    draft_final_cmd_tok = _count_ui_tokens(draft_final_prompt)
                    final_has_unsaved = draft_final_prompt.strip() != st.session_state["saved_final_base_prompt"].strip()

                    st.markdown("**🔒 종합 구조 프롬프트 (변경 불가 · 섹션/카드 스키마)**")
                    st.code(final_struct_str, language="json")

                    st.markdown(
                        f"- **명령 프롬프트 토큰(저장 기준)**: `{final_cmd_saved_tokens:,} tokens` "
                        f"(입력창 `{draft_final_cmd_tok:,} tokens`)\n"
                        f"- **구조 프롬프트 토큰(고정 기본값)**: `{final_struct_tokens:,} tokens` (JSON 스키마 + 템플릿 구조)\n"
                        f"- **종합 프롬프트 합계 토큰**: **`{final_total_fixed_tokens:,} tokens`**\n"
                        f"- **상태**: `{'⚠️ 미저장 변경 있음' if final_has_unsaved else '✅ 저장 완료'}` · "
                        f"`{'커스텀' if final_is_custom else '기본값'}`"
                    )

                st.divider()
                st.markdown("**토큰 소요 요약표 (명령 + 구조)**")
                st.dataframe(
                    pd.DataFrame([
                        {
                            "구분": "서브 LLM",
                            "명령 프롬프트(저장)": f"{sub_cmd_saved_tokens:,} tok",
                            "구조 프롬프트(고정)": f"{sub_struct_tokens:,} tok",
                            "프롬프트 소계": f"{sub_total_fixed_tokens:,} tok",
                            "저장 상태": "미저장 변경" if sub_has_unsaved else ("커스텀 저장됨" if sub_is_custom else "기본값"),
                        },
                        {
                            "구분": "종합 LLM",
                            "명령 프롬프트(저장)": f"{final_cmd_saved_tokens:,} tok",
                            "구조 프롬프트(고정)": f"{final_struct_tokens:,} tok",
                            "프롬프트 소계": f"{final_total_fixed_tokens:,} tok",
                            "저장 상태": "미저장 변경" if final_has_unsaved else ("커스텀 저장됨" if final_is_custom else "기본값"),
                        },
                    ]),
                    width="stretch",
                )

                b_col1, b_col2, b_col3 = st.columns([2, 1, 1])
                if b_col1.button("💾 커스텀 명령 프롬프트 저장 (실행에 반영)", type="primary", width="stretch", key="btn_save_custom_prompts"):
                    st.session_state["saved_sub_base_prompt"] = draft_sub_prompt
                    st.session_state["saved_final_base_prompt"] = draft_final_prompt
                    st.session_state["prompt_saved_at"] = datetime.datetime.now().strftime("%H:%M:%S")
                    st.success(f"명령 프롬프트가 저장되었습니다. ({st.session_state['prompt_saved_at']})")
                    st.rerun()

                if b_col2.button("서브 명령 기본값 복원", width="stretch", key="btn_reset_sub_prompt"):
                    st.session_state["saved_sub_base_prompt"] = SUB_BASE_PROMPT_TEMPLATE
                    st.session_state["ui_draft_sub_base_prompt"] = SUB_BASE_PROMPT_TEMPLATE
                    st.session_state["prompt_saved_at"] = datetime.datetime.now().strftime("%H:%M:%S")
                    st.rerun()

                if b_col3.button("종합 명령 기본값 복원", width="stretch", key="btn_reset_final_prompt"):
                    st.session_state["saved_final_base_prompt"] = FINAL_BASE_PROMPT_TEMPLATE
                    st.session_state["ui_draft_final_base_prompt"] = FINAL_BASE_PROMPT_TEMPLATE
                    st.session_state["prompt_saved_at"] = datetime.datetime.now().strftime("%H:%M:%S")
                    st.rerun()

            custom_sub_base_prompt = st.session_state["saved_sub_base_prompt"]
            custom_final_base_prompt = st.session_state["saved_final_base_prompt"]

            with st.container(border=True):
                st.markdown("**③ 서브 LLM 전체 출력 한도 및 청크별 배분 설정**")
                cfg_col1, cfg_col2 = st.columns(2)
                sub_alloc_mode = cfg_col1.radio(
                    "청크별 서브 LLM 출력 한도 배분 방식",
                    options=["proportional", "equal"],
                    index=0,
                    horizontal=True,
                    format_func=lambda m: (
                        "비례 배분 (청크 데이터 토큰 비율대로 분배)"
                        if m == "proportional"
                        else "공통 배분 (전체 한도 ÷ 청크 수 N)"
                    ),
                    key="ui_sub_alloc_mode",
                )
                sub_llm_total_cap_input = int(
                    cfg_col2.number_input(
                        "서브 LLM 전체 출력 한도 합계 (0=자동)",
                        min_value=0,
                        max_value=50000,
                        value=0,
                        step=100,
                        help=(
                            "모든 청크의 서브 LLM 압축 결과가 합산될 최대 토큰 총량입니다. "
                            "0=자동 시 전체 토큰 예산에서 기본 프롬프트를 제외한 가용 예산의 90%가 설정됩니다."
                        ),
                        key="ui_sub_llm_max_tokens",
                    )
                )

                chunk_tok_list = [int(sp.get("tokens", 0) or 0) for sp in sub_prompts]
                n_chunks = max(1, len(sub_prompts))
                per_chunk_budgets, effective_total_sub_pool = _compute_ui_sub_budgets(
                    chunk_tokens_list=chunk_tok_list,
                    max_total_tokens=int(max_total_tokens),
                    base_prompt_tokens=base_prompt_tokens,
                    allocation_mode=sub_alloc_mode,
                    total_sub_cap_input=sub_llm_total_cap_input,
                )

                if sub_alloc_mode == "equal":
                    st.caption(
                        f"공통 배분 계산식: 전체 한도 **{effective_total_sub_pool:,} tokens** ÷ 청크 수 **{n_chunks}개** "
                        f"= 청크당 **{per_chunk_budgets[0] if per_chunk_budgets else 0:,} tokens**"
                    )
                else:
                    st.caption(
                        f"비례 배분 계산식: 전체 한도 **{effective_total_sub_pool:,} tokens** × 각 청크 데이터 토큰 비율(%)"
                    )

                budget_preview_rows = []
                sum_chunk_tok = sum(chunk_tok_list) or 1
                sum_budget_tok = sum(per_chunk_budgets) or 1
                for idx, sp in enumerate(sub_prompts):
                    c_tok = chunk_tok_list[idx]
                    b_tok = per_chunk_budgets[idx] if idx < len(per_chunk_budgets) else 50
                    data_ratio = round((c_tok / sum_chunk_tok) * 100, 1)
                    alloc_ratio = round((b_tok / sum_budget_tok) * 100, 1)
                    budget_preview_rows.append({
                        "청크": f"#{sp.get('chunk_index', idx + 1)} {sp.get('repo_name', '-')}",
                        "청크 데이터 토큰": c_tok,
                        "데이터 비율(%)": data_ratio,
                        "배분 비율(%)": alloc_ratio,
                        "할당 출력 한도(tokens)": b_tok,
                    })

                if budget_preview_rows:
                    st.dataframe(pd.DataFrame(budget_preview_rows), width="stretch")

            if st.button("② 3-3 AI 실행 (청크별 실시간 서브 LLM 압축 + 종합 LLM 생성)", type="primary", width="stretch", key="btn_run_ai_stage"):
                tpl_sections = linked_tpl.get("sections") or []
                normalized_tpl_sections = []
                for s_i, sec in enumerate(tpl_sections):
                    normalized_tpl_sections.append({
                        "section_type": sec.get("section_type", "CUSTOM"),
                        "section_title": sec.get("section_title") or sec.get("title") or "사용자 기여 분석 및 구현 성과",
                        "display_order": sec.get("display_order", s_i + 1),
                        "columns": sec.get("columns") or ["구현 및 문제 해결", "기술 스택 및 활동 근거"],
                    })

                total_chunks = len(sub_prompts)
                progress_bar = st.progress(0.0, text="서브 LLM 청크 압축 준비 중...")
                live_log_box = st.empty()
                live_logs: list[str] = []

                total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                updated_sub_prompts = []
                compressed_chunks_list = []
                chunk_retentions = []

                try:
                    with st.status(
                        f"총 {total_chunks}개 청크 실시간 서브 LLM 압축 중 (템플릿 반영: {'ON' if apply_template_in_sub else 'OFF'})...",
                        expanded=True,
                    ) as status_box:
                        for idx, sp in enumerate(sub_prompts):
                            c_idx = sp.get("chunk_index", idx + 1)
                            r_name = sp.get("repo_name", "-")
                            assigned_budget = per_chunk_budgets[idx] if idx < len(per_chunk_budgets) else 400
                            raw_ch = sp.get("raw_chunk_data", {})

                            progress_bar.progress(
                                idx / max(1, total_chunks + 1),
                                text=f"[{c_idx}/{total_chunks}] 청크 압축 중: {r_name} (한도 {assigned_budget} tokens)...",
                            )
                            status_box.write(f"⏳ **청크 #{c_idx} (`{r_name}`)** 서브 LLM 압축 실행 중... (상한 `{assigned_budget}` tokens)")

                            c_resp = requests.post(
                                f"{backend_url}/api/github/compress-chunk",
                                json={
                                    "chunk": raw_ch,
                                    "chunk_index": c_idx,
                                    "template_sections": normalized_tpl_sections,
                                    "sub_budget_tokens": assigned_budget,
                                    "apply_template_in_sub": apply_template_in_sub,
                                    "custom_sub_base_prompt": custom_sub_base_prompt,
                                },
                                timeout=120,
                            )
                            if c_resp.status_code != 200:
                                st.error(f"청크 #{c_idx} 압축 실패: {c_resp.text}")
                                st.stop()

                            c_data = c_resp.json()
                            u_delta = c_data.get("token_usage", {})
                            total_usage["prompt_tokens"] += u_delta.get("prompt_tokens", 0)
                            total_usage["completion_tokens"] += u_delta.get("completion_tokens", 0)
                            total_usage["total_tokens"] += u_delta.get("total_tokens", 0)

                            comp_obj = c_data.get("compressed_chunk", raw_ch)
                            compressed_chunks_list.append(comp_obj)
                            ret_info = c_data.get("retention", {})
                            chunk_retentions.append({"chunk_index": c_idx, "repo_name": r_name, **ret_info})

                            sp_copy = dict(sp)
                            sp_copy["sub_llm_max_tokens"] = assigned_budget
                            sp_copy["sub_llm_output"] = comp_obj
                            sp_copy["retention"] = ret_info
                            sp_copy["final_input_tokens"] = c_data.get(
                                "compressed_tokens", _count_ui_tokens(json.dumps(comp_obj, ensure_ascii=False))
                            )
                            updated_sub_prompts.append(sp_copy)

                            msg = (
                                f"✅ 청크 #{c_idx} (`{r_name}`): "
                                f"{c_data.get('raw_tokens', 0)} → {c_data.get('compressed_tokens', 0)} tokens "
                                f"| 항목 보존율 **{ret_info.get('retention_rate_pct', 100)}%** "
                                f"({ret_info.get('retained_item_count', 0)}/{ret_info.get('raw_item_count', 0)}개) "
                                f"· {c_data.get('elapsed_sec', 0)}초"
                            )
                            live_logs.append(msg)
                            status_box.write(msg)
                            live_log_box.markdown("\n\n".join(live_logs))

                        progress_bar.progress(
                            total_chunks / max(1, total_chunks + 1),
                            text="서브 LLM 압축 완료! 저장소별 통합 후 종합 LLM 카드 생성 실행 중...",
                        )
                        status_box.write("⏳ 압축된 청크들을 저장소 단위로 모아 종합 프롬프트로 최종 카드 생성 중...")

                        syn_resp = requests.post(
                            f"{backend_url}/api/github/synthesize-chunks",
                            json={
                                "cleaned_data": cleaned_all,
                                "compressed_chunks": compressed_chunks_list,
                                "template_sections": normalized_tpl_sections,
                                "language": language,
                                "custom_final_base_prompt": custom_final_base_prompt,
                            },
                            timeout=180,
                        )
                        if syn_resp.status_code != 200:
                            st.error(f"종합 LLM 생성 실패: {syn_resp.text}")
                            st.stop()

                        syn_data = syn_resp.json()
                        s_usage = syn_data.get("token_usage", {})
                        total_usage["prompt_tokens"] += s_usage.get("prompt_tokens", 0)
                        total_usage["completion_tokens"] += s_usage.get("completion_tokens", 0)
                        total_usage["total_tokens"] += s_usage.get("total_tokens", 0)

                        progress_bar.progress(1.0, text="전체 청크 압축 및 종합 생성 완료!")
                        status_box.update(label="✅ 청크별 서브 LLM 압축 및 종합 생성 완료!", state="complete")

                    total_raw_items_sub = sum(r.get("raw_item_count", 0) for r in chunk_retentions)
                    total_ret_items_sub = sum(r.get("retained_item_count", 0) for r in chunk_retentions)
                    overall_sub_retention_pct = (
                        round((total_ret_items_sub / total_raw_items_sub) * 100, 2)
                        if total_raw_items_sub > 0
                        else 100.0
                    )
                    raw_sub_tok_sum = sum(int(sp.get("tokens", 0) or 0) for sp in updated_sub_prompts)
                    comp_sub_tok_sum = sum(int(sp.get("final_input_tokens", 0) or 0) for sp in updated_sub_prompts)
                    token_compression_pct = (
                        round((1 - (comp_sub_tok_sum / raw_sub_tok_sum)) * 100, 2)
                        if raw_sub_tok_sum > 0
                        else 0.0
                    )

                    run_summary = {
                        "apply_template_in_sub": apply_template_in_sub,
                        "allocation_mode": sub_alloc_mode,
                        "total_sub_pool": effective_total_sub_pool,
                        "raw_sub_tokens": raw_sub_tok_sum,
                        "compressed_sub_tokens": comp_sub_tok_sum,
                        "token_compression_pct": token_compression_pct,
                        "total_raw_items": total_raw_items_sub,
                        "retained_items": total_ret_items_sub,
                        "sub_item_retention_pct": overall_sub_retention_pct,
                        "final_mention_pct": (syn_data.get("comparison") or {}).get("direct_evidence_mention_pct", 0.0),
                        "chunk_retentions": chunk_retentions,
                    }

                    history_by_mode = dict(st.session_state.get("sub_tpl_mode_history") or {})
                    mode_key = "template_ON" if apply_template_in_sub else "template_OFF"
                    history_by_mode[mode_key] = run_summary
                    st.session_state["sub_tpl_mode_history"] = history_by_mode

                    new_pipe = {
                        **pipe,
                        "sub_prompts": updated_sub_prompts,
                        "combined_prompt": syn_data.get("combined_prompt", pipe.get("combined_prompt", "")),
                        "generated_sections": syn_data.get("generated_sections", []),
                        "token_usage": total_usage,
                        "model_used": syn_data.get("model_used", "-"),
                        "comparison": syn_data.get("comparison", {}),
                        "sub_run_summary": run_summary,
                    }
                    st.session_state["pipeline_res"] = new_pipe
                    st.session_state["pipeline_step_stage"] = 2
                    _save_trace({**_load_trace(), "pipeline_res": new_pipe, "pipeline_step_stage": 2})
                    st.rerun()

                except requests.exceptions.ConnectionError:
                    st.error(f"백엔드 서버({backend_url})에 연결할 수 없습니다. FastAPI 서버 상태를 확인해 주세요.")

            if has_ai_result:
                u = pipe.get("token_usage", {})
                run_sum = pipe.get("sub_run_summary") or {}
                u1, u2, u3, u4, u5 = st.columns(5)
                u1.metric("사용 모델", pipe.get("model_used", "-"))
                u2.metric("서브 템플릿 반영", "ON (반영)" if run_sum.get("apply_template_in_sub", True) else "OFF (미반영)")
                u3.metric("서브 단계 데이터 보존율", f"{run_sum.get('sub_item_retention_pct', 0)}%")
                u4.metric("서브 토큰 압축률", f"{run_sum.get('token_compression_pct', 0)}%")
                u5.metric("실측 총 API 토큰", f"{u.get('total_tokens', 0):,}")

                history_by_mode = st.session_state.get("sub_tpl_mode_history") or {}
                if history_by_mode:
                    st.markdown("**④ 서브 LLM 템플릿 반영 여부에 따른 데이터 보존율 비교**")
                    on_rec = history_by_mode.get("template_ON")
                    off_rec = history_by_mode.get("template_OFF")

                    comp_rows = []
                    for label, rec in [("템플릿 반영 (ON)", on_rec), ("템플릿 미반영 (OFF)", off_rec)]:
                        if rec:
                            comp_rows.append({
                                "모드": label,
                                "배분 방식": "비례 배분" if rec.get("allocation_mode") == "proportional" else "공통 배분",
                                "서브 압축 전→후 토큰": f"{rec.get('raw_sub_tokens', 0):,} → {rec.get('compressed_sub_tokens', 0):,}",
                                "토큰 압축률(%)": rec.get("token_compression_pct", 0.0),
                                "보존 항목 수": f"{rec.get('retained_items', 0)} / {rec.get('total_raw_items', 0)}개",
                                "서브 데이터 보존율(%)": rec.get("sub_item_retention_pct", 0.0),
                                "최종 종합 근거 반영률(%)": rec.get("final_mention_pct", 0.0),
                            })
                    st.dataframe(pd.DataFrame(comp_rows), width="stretch")

                    if on_rec and off_rec:
                        diff_ret = round(
                            on_rec.get("sub_item_retention_pct", 0.0) - off_rec.get("sub_item_retention_pct", 0.0), 2
                        )
                        diff_tok = on_rec.get("compressed_sub_tokens", 0) - off_rec.get("compressed_sub_tokens", 0)
                        st.info(
                            f"비교 결과: 템플릿 반영(ON)은 미반영(OFF) 대비 서브 단계 데이터 보존율이 "
                            f"**{diff_ret:+.2f}%p** 차이 나며, 압축 후 토큰은 **{diff_tok:+,} tokens** 차이가 발생했습니다."
                        )
                    else:
                        missing_label = "템플릿 미반영 (OFF)" if on_rec else "템플릿 반영 (ON)"
                        st.caption(f"💡 위 토글을 **{missing_label}**으로 바꾸어 한 번 더 실행하면 두 모드의 보존율 차이가 자동 비교됩니다.")

                with st.expander("청크별 서브 LLM 압축 전/후 결과 및 보존율 상세 보기"):
                    for idx, sp in enumerate(sub_prompts):
                        c_idx = sp.get("chunk_index", idx + 1)
                        r_name = sp.get("repo_name", "-")
                        raw_t = sp.get("tokens", 0)
                        fin_t = sp.get("final_input_tokens", raw_t)
                        b_lim = sp.get("sub_llm_max_tokens", "-")
                        ret = sp.get("retention") or {}
                        st.markdown(
                            f"**청크 #{c_idx} — {r_name}** "
                            f"(`{raw_t} tokens` → 한도 `{b_lim}` → 압축 후 `{fin_t} tokens` | "
                            f"보존율 `{ret.get('retention_rate_pct', '-')}%`)"
                        )
                        col_b, col_a = st.columns(2)
                        with col_b:
                            st.caption("압축 전 (청크 생 데이터)")
                            st.json(sp.get("raw_chunk_data"))
                        with col_a:
                            st.caption("압축 후 (서브 LLM 출력 -> 저장소별 통합 입력)")
                            st.json(sp.get("sub_llm_output"))

                with st.expander("종합 입력 프롬프트 전문 보기 (저장소별 통합 반영)"):
                    st.code(pipe.get("combined_prompt", ""), language="text")

                st.divider()
                st.subheader("3-4. 원본 데이터 vs 해석된 데이터 비교 (표 변환 및 CSV 저장)")
                comp = pipe.get("comparison", {})
                c_m1, c_m2, c_m3 = st.columns(3)
                c_m1.metric("원본 정제 항목 수", f"{comp.get('total_raw_items', 0)}개")
                c_m2.metric("원문 직접 매칭 항목 수", f"{comp.get('directly_matched_items', 0)}개")
                c_m3.metric("직접 근거 언급률", f"{comp.get('direct_evidence_mention_pct', 0)}%")

                st.markdown("#### ① 서브 LLM 청크별 압축 결과 표")
                sub_table_df = _flatten_sub_outputs_to_df(sub_prompts)
                if not sub_table_df.empty:
                    st.dataframe(sub_table_df, width="stretch")
                    st.download_button(
                        "⬇️ 서브 LLM 압축 결과 CSV 다운로드",
                        data=sub_table_df.to_csv(index=False).encode("utf-8-sig"),
                        file_name="sub_llm_compressed_table.csv",
                        mime="text/csv",
                        key="dl_sub_table_3_4",
                    )

                st.markdown("#### ② 종합 LLM 최종 해석 결과 (이력서 카드 뷰 & 전체 표)")
                gen_sections = pipe.get("generated_sections") or []
                gen_table_df = _flatten_generated_sections_to_df(gen_sections)

                if gen_sections:
                    for sec in gen_sections:
                        if not isinstance(sec, dict):
                            continue
                        sec_title = sec.get("section_title") or sec.get("title") or "섹션"
                        st.markdown(f"##### 📌 {sec_title}")
                        for item in sec.get("details") or []:
                            if not isinstance(item, dict):
                                continue
                            card_title = item.get("title", "항목")
                            col_vals = item.get("column_values") or {}
                            fields_html = ""
                            if isinstance(col_vals, dict):
                                for k, v in col_vals.items():
                                    val_html = _format_card_value_html(v)
                                    fields_html += (
                                        f'<div class="resume-field-label">▪ {k}</div>'
                                        f'<div class="resume-field-body">{val_html}</div>'
                                    )
                            st.markdown(
                                f'<div class="resume-card">'
                                f'<div class="resume-card-title">{card_title}</div>'
                                f'{fields_html}'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

                if not gen_table_df.empty:
                    with st.expander("전체 결과 표 형태로 펼쳐보기 (줄바꿈 적용)", expanded=False):
                        st.table(gen_table_df)

                    st.download_button(
                        "⬇️ 최종 생성 결과 표 CSV 다운로드",
                        data=gen_table_df.to_csv(index=False).encode("utf-8-sig"),
                        file_name="final_generated_table.csv",
                        mime="text/csv",
                        key="dl_gen_table_3_4",
                    )
                else:
                    st.info("표로 변환할 생성 결과가 없습니다.")

                with st.container(border=True):
                    st.markdown("**③ 실험 결과 CSV 세트 저장 (4단계 '해석 및 분석' 탭 연동)**")
                    sv_c1, sv_c2 = st.columns([2, 1])
                    exp_label = sv_c1.text_input(
                        "실험 라벨 메모",
                        placeholder="예: 비례배분_템플릿ON_한도2000",
                        key="ui_exp_label",
                    )
                    if sv_c2.button("📥 CSV 4종 일괄 저장 (4단계 탭 분석용)", type="primary", width="stretch", key="btn_save_csv_all"):
                        saved_id = _save_run_to_csvs(
                            pipe=pipe,
                            chunk_limit_mode=chunk_limit_mode,
                            run_label=exp_label,
                        )
                        st.success(f"저장 완료 (`run_id: {saved_id}`) — 상단 **4단계 · 해석 및 분석** 탭에서 확인하세요.")


# =========================================================
# TAB 3 (4단계): 저장된 CSV 기반 해석 및 분석
# =========================================================
with tab_stage4:
    st.header("4. 해석 및 분석 (저장된 CSV 표 기반 정량 검증)")

    if not SUMMARY_CSV_PATH.exists():
        st.info("아직 저장된 CSV 실험 기록이 없습니다. **3단계** 실행 후 **'CSV 4종 일괄 저장'**을 눌러 주세요.")
    else:
        summary_df = pd.read_csv(SUMMARY_CSV_PATH, dtype={"run_id": str})

        st.subheader("4-1. 누적 실험 비교표 (`runs_summary.csv`)")
        f1, f2, f3 = st.columns(3)
        tpl_opts = sorted(summary_df["서브템플릿반영"].dropna().unique().tolist())
        alloc_opts = sorted(summary_df["배분방식"].dropna().unique().tolist())

        sel_tpl = f1.multiselect("서브 템플릿 반영", options=tpl_opts, default=tpl_opts, key="s4_tpl")
        sel_alloc = f2.multiselect("배분 방식", options=alloc_opts, default=alloc_opts, key="s4_alloc")
        if f3.button("저장된 CSV 전체 초기화", width="stretch", key="s4_reset_csv"):
            for fp in EXPERIMENT_DIR.glob("*.csv"):
                fp.unlink(missing_ok=True)
            st.rerun()

        filtered_sum = summary_df[
            summary_df["서브템플릿반영"].isin(sel_tpl) & summary_df["배분방식"].isin(sel_alloc)
        ].copy()
        st.dataframe(filtered_sum, width="stretch")

        if not filtered_sum.empty:
            st.markdown("**조건별 평균 성능 비교 (템플릿 반영 ON/OFF × 비례/공통 배분)**")
            grp_df = (
                filtered_sum.groupby(["서브템플릿반영", "배분방식"], as_index=False)
                .agg(
                    실험수=("run_id", "count"),
                    평균_토큰압축률_pct=("토큰압축률(%)", "mean"),
                    평균_서브보존율_pct=("서브보존율(%)", "mean"),
                    평균_종합근거반영률_pct=("종합근거반영률(%)", "mean"),
                    평균_총API토큰=("실측총API토큰", "mean"),
                )
                .round(2)
            )
            st.dataframe(grp_df, width="stretch")

        st.download_button(
            "⬇️ 누적 실험 요약 CSV 다운로드 (runs_summary.csv)",
            data=filtered_sum.to_csv(index=False).encode("utf-8-sig"),
            file_name="runs_summary.csv",
            mime="text/csv",
            key="dl_s4_summary",
        )

        st.divider()
        st.subheader("4-2. 개별 실행(`run_id`) 상세 표 분석")
        run_ids = summary_df["run_id"].tolist()[::-1]
        label_map = {
            r["run_id"]: f"[{r['run_id']}] {r['실험라벨']} · 템플릿:{r['서브템플릿반영']} · {r['배분방식']} · 보존율:{r['서브보존율(%)']}%"
            for _, r in summary_df.iterrows()
        }
        chosen_run = st.selectbox("분석할 실행 선택", options=run_ids, format_func=lambda x: label_map.get(x, x), key="s4_run_pick")

        p_chunks = EXPERIMENT_DIR / f"chunks_{chosen_run}.csv"
        p_items = EXPERIMENT_DIR / f"items_{chosen_run}.csv"
        p_gen = EXPERIMENT_DIR / f"generated_{chosen_run}.csv"

        if p_gen.exists():
            st.markdown("#### ① 최종 생성 결과 표 (`generated_{run_id}.csv`)")
            df_g = pd.read_csv(p_gen, dtype={"run_id": str})
            st.dataframe(df_g, width="stretch")
            st.download_button(
                f"⬇️ 최종 생성 결과 CSV 다운로드 (generated_{chosen_run}.csv)",
                data=df_g.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"generated_{chosen_run}.csv",
                mime="text/csv",
                key="dl_s4_gen",
            )

        if p_chunks.exists():
            st.markdown("#### ② 청크별 압축 상세 표 (`chunks_{run_id}.csv`)")
            df_c = pd.read_csv(p_chunks, dtype={"run_id": str})
            st.dataframe(df_c, width="stretch")
            st.download_button(
                f"⬇️ 청크 상세 CSV 다운로드 (chunks_{chosen_run}.csv)",
                data=df_c.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"chunks_{chosen_run}.csv",
                mime="text/csv",
                key="dl_s4_chunks",
            )

        if p_items.exists():
            st.markdown("#### ③ 항목별 보존/유실 추적 표 (`items_{run_id}.csv`)")
            df_i = pd.read_csv(p_items, dtype={"run_id": str})
            item_mode = st.radio(
                "유실 추적 필터",
                options=["전체", "서브 유실(X)", "서브 보존(O) 후 종합 유실(X)"],
                horizontal=True,
                key="s4_item_mode",
            )
            if item_mode == "서브 유실(X)":
                df_i_view = df_i[df_i["서브보존여부"] == "X"]
            elif item_mode == "서브 보존(O) 후 종합 유실(X)":
                df_i_view = df_i[(df_i["서브보존여부"] == "O") & (df_i["종합반영여부"] == "X")]
            else:
                df_i_view = df_i

            st.dataframe(df_i_view, width="stretch", height=300)
            st.download_button(
                f"⬇️ 항목 추적 CSV 다운로드 (items_{chosen_run}.csv)",
                data=df_i.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"items_{chosen_run}.csv",
                mime="text/csv",
                key="dl_s4_items",
            )