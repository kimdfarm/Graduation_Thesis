import json
import os
from typing import Any

from dotenv import load_dotenv
import pandas as pd
import requests
import streamlit as st
import tiktoken

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


st.set_page_config(page_title="GitHub 활동 탐색 및 청크 가치 검증기", layout="wide")
st.title("GitHub 사용자 활동 탐색 & 청크 기반 데이터 가치 검증")

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

    if st.button("저장소 검색", type="primary", use_container_width=True, key="btn_search_repos"):
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
                st.session_state["repos"] = resp.json().get("repositories", [])
            else:
                st.error(f"저장소 조회 실패: {resp.text}")

# =========================================================
# 2. 저장소 선택(다중 가능), 탐색 항목 선택 및 활동 분석
# =========================================================
st.subheader("2. 저장소 선택(다중 가능) 및 사용자 활동 분석")

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

manual_repos_text = st.text_area(
    "추가/직접 입력 저장소 (owner/repo, 줄바꿈 또는 쉼표로 여러 개 가능)",
    placeholder="예:\nkimdfarm/News_quiz_by_slack\nkimdfarm/dmu_Graduation_project",
    height=75,
    key="ta_manual_repos",
)
if manual_repos_text.strip():
    raw_tokens = [
        item.strip()
        for line in manual_repos_text.splitlines()
        for item in line.split(",")
        if item.strip()
    ]
    existing_names = {r["full_name"] for r in selected_repos}
    for full in raw_tokens:
        if full not in existing_names:
            selected_repos.append({"full_name": full, "name": full.split("/")[-1]})
            existing_names.add(full)

for _k, _default_val in [
    ("chk_commits", True),
    ("chk_prs", True),
    ("chk_issues", True),
    ("chk_languages", True),
    ("chk_repo_meta", False),
]:
    if _k not in st.session_state:
        st.session_state[_k] = _default_val

st.markdown("**탐색 대상 항목 선택** (원하는 기능 체크박스를 선택하세요)")
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
            st.session_state["analysis_data"] = {
                "projects_data": collected_projects,
                "include_types": active_types,
                "target_author": target_author or github_id,
            }
            names_str = ", ".join(p.get("full_name") or p.get("repo_name", "") for p in collected_projects)
            types_str = ", ".join(EXPLORE_LABELS[t] for t in active_types)
            st.success(f"분석 완료 ({len(collected_projects)}개 저장소): {names_str} | 탐색 항목: {types_str}")

        for err_msg in errors:
            st.error(f"분석 실패 — {err_msg}")

# =========================================================
# 3. 청크 분할 가치 평가 및 백엔드 생성 기능 테스트
# =========================================================
analysis_data = st.session_state.get("analysis_data")
if analysis_data:
    st.divider()
    st.header("3. 청크 분할 가치 평가 및 백엔드 생성 기능 테스트")

    # [3-1] 청크 수 및 한도 설정 입력
    st.subheader("3-1. 청크 수·토큰 예산 및 한도 설정")

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
    base_prompt_tokens = _count_ui_tokens(raw_json_str[:8000]) + 450

    s1, s2, s3, s4 = st.columns(4)
    s1.metric(
        "원본 활동 항목 수",
        f"{raw_total_items:,}개",
        help=f"커밋 {raw_commits_n} / PR {raw_prs_n} / 이슈 {raw_issues_n}",
    )
    s2.metric("원본 생 데이터 토큰", f"{raw_tokens:,}", help=f"문자 수: {len(raw_json_str):,}자")
    s3.metric("기본 프롬프트 토큰", f"{base_prompt_tokens:,}")
    s4.metric(
        "선택 부가정보 요약",
        f"언어 {len(raw_langs_set)}종 · 메타 {meta_repos_n}건",
        help=", ".join(sorted(raw_langs_set)) if raw_langs_set else "선택된 언어 없음",
    )

    c1, c2, c3, c4 = st.columns(4)
    max_total_tokens = c1.number_input("전체 토큰 예산", min_value=500, max_value=50000, value=5000, step=500, key="ui_max_total_tokens")
    max_chunks = c2.number_input("제한 청크 수 (최대 개수)", min_value=1, max_value=50, value=6, step=1, key="ui_max_chunks")
    chunk_token_limit = c3.number_input("청크당 최대 토큰 한도", min_value=200, max_value=20000, value=1200, step=100, key="ui_chunk_token_limit")
    chunk_item_limit = c4.number_input(
        "청크당 최대 항목 수",
        min_value=5,
        max_value=500,
        value=40,
        step=5,
        help="한 청크에 포함될 수 있는 커밋+PR+이슈 항목의 최대 개수 한도",
        key="ui_chunk_item_limit",
    )

    col_opt1, col_opt2, col_opt3 = st.columns(3)
    force_chunking = col_opt1.checkbox("토큰 한도 미달 시에도 강제 청크 분할", value=True, key="ui_force_chunking")
    run_generation = col_opt2.checkbox("종합 프롬프트로 구조화 생성까지 실행", value=True, key="ui_run_generation")
    language = col_opt3.selectbox("출력 언어", options=["ko", "en"], index=0, key="ui_language")

    if st.button("단계별 청크 검증 및 생성 파이프라인 실행", type="primary", use_container_width=True, key="btn_run_pipeline"):
        req_payload = {
            "analysis_data": analysis_data,
            "target_author": target_author or github_id,
            "start_date": str(start_date) if start_date else None,
            "end_date": str(end_date) if end_date else None,
            "max_total_tokens": int(max_total_tokens),
            "max_chunks": int(max_chunks),
            "chunk_token_limit": int(chunk_token_limit),
            "chunk_item_limit": int(chunk_item_limit),
            "force_chunking": force_chunking,
            "language": language,
            "run_generation": run_generation,
        }
        resp = requests.post(f"{backend_url}/api/github/pipeline-inspect", json=req_payload, timeout=180)
        if resp.status_code == 200:
            st.session_state["pipeline_res"] = resp.json()
        else:
            st.error(f"파이프라인 실행 실패: {resp.text}")

    pipe = st.session_state.get("pipeline_res")
    if pipe:
        # [3-2] 종합 입력값(생 데이터 + 프롬프트) 및 생 데이터 표
        st.divider()
        st.subheader("3-2. 종합 입력값 (생 데이터 + 프롬프트) 및 생 데이터 표")

        st.markdown("**① 정제된 생 데이터 표**")
        raw_df = pd.DataFrame(pipe.get("raw_table_rows", []))
        if not raw_df.empty:
            st.dataframe(raw_df, use_container_width=True, height=280)
        else:
            st.info("표시할 정제 활동 항목이 없습니다.")

        with st.expander("② 종합 입력 프롬프트 + 생 데이터 전문 보기"):
            st.code(pipe.get("combined_prompt", ""), language="text")

        # [3-3] 청크 수 기준 한도 초과 여부 확인
        st.divider()
        st.subheader("3-3. 청크 수 및 각 청크 한도 초과 여부 확인")
        limits = pipe.get("chunk_limits", {})
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("실제 분할 청크 수", f"{limits.get('actual_chunk_count', 0)}개")
        m2.metric("제한 청크 수", f"{limits.get('max_chunks', 0)}개")
        m3.metric("청크 수 제한 초과 여부", "초과함" if limits.get("exceeded_chunk_count_limit") else "제한 이내")
        m4.metric("실측 API 총 토큰", f"{pipe.get('token_usage', {}).get('total_tokens', 0):,}")

        eval_df = pd.DataFrame(pipe.get("chunk_evaluations", []))
        if not eval_df.empty:
            st.dataframe(eval_df, use_container_width=True)

        # [3-4] 청크별 서브 프롬프트 + 청크 생 데이터
        st.divider()
        st.subheader("3-4. 청크별 서브 프롬프트 + 청크 생 데이터 (종합 프롬프트와 별도)")
        for sp in pipe.get("sub_prompts", []):
            status_tag = "⚠️ 한도 초과" if sp.get("exceeded") else "✅ 정상"
            with st.expander(f"청크 #{sp['chunk_index']} — {sp['repo_name']} ({sp['tokens']} tokens | {status_tag})"):
                st.markdown("**서브 프롬프트**")
                st.code(sp["sub_prompt"], language="text")
                st.markdown("**해당 청크 생 데이터**")
                st.json(sp["raw_chunk_data"])

        # [3-5] 원본 데이터 vs 해석된 데이터 비교
        st.divider()
        st.subheader("3-5. 원본 데이터 vs 해석된 데이터 비교")
        comp = pipe.get("comparison", {})
        c_m1, c_m2, c_m3 = st.columns(3)
        c_m1.metric("원본 정제 항목 수", f"{comp.get('total_raw_items', 0)}개")
        c_m2.metric("원문 직접 매칭 항목 수", f"{comp.get('directly_matched_items', 0)}개")
        c_m3.metric("직접 근거 언급률", f"{comp.get('direct_evidence_mention_pct', 0)}%")

        col_orig, col_interp = st.columns(2)
        with col_orig:
            st.markdown("**원본 정제 데이터 (Raw / Preprocessed)**")
            st.json(pipe.get("cleaned_data", []))
        with col_interp:
            st.markdown("**해석 및 구조화 데이터 (LLM Output)**")
            st.json(pipe.get("generated_sections", []))

        if comp.get("unmatched_sample"):
            with st.expander("해석 결과 문자열에 직접 포함되지 않은 원본 항목 샘플"):
                st.write(comp["unmatched_sample"])