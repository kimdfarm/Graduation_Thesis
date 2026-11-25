import datetime
import json
from typing import Any

import pandas as pd
import requests
import streamlit as st

from csv_exporter import (
    flatten_generated_sections_to_df,
    flatten_sub_outputs_to_df,
    format_card_value_html,
    save_run_to_csvs,
)
from ui_constants import (
    ALL_EXPLORE_TYPES,
    FINAL_BASE_PROMPT_TEMPLATE,
    SUB_BASE_PROMPT_TEMPLATE,
    calc_pure_prompt_tokens,
    compute_ui_sub_budgets,
    count_ui_tokens,
)
from ui_state import load_trace, save_trace


def render_stage3(backend_url: str, github_id: str, start_date: Any, end_date: Any) -> None:
    analysis_data = st.session_state.get("analysis_data")
    target_author_val = st.session_state.get("ti_target_author", github_id)

    if not analysis_data:
        st.info("먼저 **2단계** 탭에서 저장소를 선택하고 **선택 저장소 활동 분석 실행**을 완료해 주세요.")
        return

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
    raw_tokens = count_ui_tokens(raw_json_str)
    base_prompt_tokens = calc_pure_prompt_tokens(analysis_data.get("template_config"))

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("원본 활동 항목 수", f"{raw_total_items:,}개", help=f"커밋 {raw_commits_n} / PR {raw_prs_n} / 이슈 {raw_issues_n}")
    s2.metric("원본 생 데이터 토큰", f"{raw_tokens:,}", help=f"문자 수: {len(raw_json_str):,}자")
    s3.metric("기본 프롬프트 토큰", f"{base_prompt_tokens:,}", help="원본 데이터를 제외한 순수 프롬프트 지시문·템플릿 스키마 고정 토큰 수")
    s4.metric("선택 부가정보 요약", f"언어 {len(raw_langs_set)}종 · 메타 {meta_repos_n}건", help=", ".join(sorted(raw_langs_set)) if raw_langs_set else "선택된 언어 없음")

    chunk_limit_mode = st.radio(
        "청크 분할 기준 선택 (둘 중 하나만 적용)",
        options=["max_chunks", "chunk_tokens"],
        index=0,
        horizontal=True,
        format_func=lambda m: "제한 청크 수 (최대 개수 기준)" if m == "max_chunks" else "청크당 최대 토큰 한도 기준",
        key="ui_chunk_limit_mode",
    )

    max_chunks_val: int | None = None
    chunk_token_limit_val: int | None = None

    c1, c2 = st.columns(2)
    max_total_tokens = c1.number_input("전체 토큰 예산", min_value=500, max_value=50000, value=5000, step=500, key="ui_max_total_tokens")

    if chunk_limit_mode == "max_chunks":
        max_chunks_val = int(c2.number_input("제한 청크 수 (최대 개수)", min_value=1, max_value=50, value=6, step=1, key="ui_max_chunks"))
    else:
        chunk_token_limit_val = int(c2.number_input("청크당 최대 토큰 한도", min_value=200, max_value=20000, value=4000, step=100, key="ui_chunk_token_limit"))

    col_opt1, col_opt2, col_opt3 = st.columns(3)
    force_chunking = col_opt1.checkbox("단일 청크일 때도 분할 파이프라인 강제 적용", value=True, key="ui_force_chunking")
    col_opt2.checkbox("종합 프롬프트로 구조화 생성까지 실행", value=True, key="ui_run_generation")
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
                save_trace({**load_trace(), "pipeline_res": chunk_data, "pipeline_step_stage": 1})
                st.rerun()
            else:
                st.error(f"청크 분할 실행 실패: {resp.text}")
        except requests.exceptions.ConnectionError:
            st.error(f"백엔드 서버({backend_url})에 연결할 수 없습니다. FastAPI 서버가 실행 중인지 확인해 주세요.")

    pipe = st.session_state.get("pipeline_res")
    if not pipe:
        return

    sub_prompts = pipe.get("sub_prompts", [])
    limits = pipe.get("chunk_limits", {})
    eval_rows = pipe.get("chunk_evaluations", [])
    has_ai_result = bool(pipe.get("generated_sections")) or any(
        sp.get("sub_llm_output") is not None for sp in sub_prompts
    )

    # [3-2] AI 미사용 영역
    st.divider()
    st.subheader("3-2. 생 데이터 정보 · 전처리 되는 정보 · 청크 분해된 정보 (AI 미사용)")

    raw_src_projects = analysis_data.get("projects_data") or [analysis_data]
    raw_src_tokens = count_ui_tokens(json.dumps(analysis_data, ensure_ascii=False))
    cleaned_all = pipe.get("cleaned_data", [])
    cleaned_total_tokens = sum(count_ui_tokens(json.dumps(c, ensure_ascii=False)) for c in cleaned_all)
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
        hist_nodes = ((((p.get("defaultBranchRef") or {}).get("target") or {}).get("history") or {}).get("nodes")) or []
        pr_nodes = ((p.get("pullRequests") or {}).get("nodes")) or []
        issue_nodes = ((p.get("issues") or {}).get("nodes")) or []
        raw_info_rows.append({
            "저장소": p.get("full_name") or p.get("repo_name") or p.get("name") or "-",
            "대상 작성자": p.get("target_author") or target_author_val or "-",
            "원본 커밋 수": len(hist_nodes),
            "원본 PR 수": len(pr_nodes),
            "원본 이슈 수": len(issue_nodes),
            "생 데이터 토큰": count_ui_tokens(json.dumps(p, ensure_ascii=False)),
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

    # [3-3] AI 실행 영역
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
    sub_struct_tokens = count_ui_tokens(active_sub_struct_str)

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
    final_struct_tokens = count_ui_tokens(final_struct_str)

    sub_cmd_saved_tokens = count_ui_tokens(st.session_state["saved_sub_base_prompt"])
    final_cmd_saved_tokens = count_ui_tokens(st.session_state["saved_final_base_prompt"])
    sub_total_fixed_tokens = sub_cmd_saved_tokens + sub_struct_tokens
    final_total_fixed_tokens = final_cmd_saved_tokens + final_struct_tokens

    sub_is_custom = st.session_state["saved_sub_base_prompt"].strip() != SUB_BASE_PROMPT_TEMPLATE.strip()
    final_is_custom = st.session_state["saved_final_base_prompt"].strip() != FINAL_BASE_PROMPT_TEMPLATE.strip()

    expander_title = (
        f"② 프롬프트 설정 — 명령(커스텀) vs 구조(고정 · 서브 {sub_total_fixed_tokens:,}tok / 종합 {final_total_fixed_tokens:,}tok) "
        f"[{'저장됨' if (sub_is_custom or final_is_custom) else '기본값'} · {st.session_state['prompt_saved_at']}]"
    )

    with st.expander(expander_title, expanded=False):
        st.caption("**명령 프롬프트**는 커스텀 수정 후 저장할 수 있고, **구조 프롬프트**는 출력 파싱을 위해 고정으로 들어가는 기본 토큰입니다.")
        p_col1, p_col2 = st.columns(2)

        with p_col1:
            st.markdown("#### 서브 LLM 프롬프트 (컬럼 기준 데이터 압축 전용)")
            draft_sub_prompt = st.text_area(
                "서브 명령 프롬프트 (수정 가능)",
                value=st.session_state["saved_sub_base_prompt"],
                height=150,
                key="ui_draft_sub_base_prompt",
            )
            draft_sub_cmd_tok = count_ui_tokens(draft_sub_prompt)
            sub_has_unsaved = draft_sub_prompt.strip() != st.session_state["saved_sub_base_prompt"].strip()

            st.markdown("**🔒 서브 구조 프롬프트 (변경 불가 · 카드/섹션 미포함)**")
            st.code(active_sub_struct_str, language="json")
            st.markdown(
                f"- **명령 프롬프트 토큰(저장 기준)**: `{sub_cmd_saved_tokens:,} tokens` (입력창 `{draft_sub_cmd_tok:,} tokens`)\n"
                f"- **구조 프롬프트 토큰(고정 기본값)**: `{sub_struct_tokens:,} tokens`\n"
                f"- **서브 프롬프트 합계 토큰**: **`{sub_total_fixed_tokens:,} tokens`**\n"
                f"- **상태**: `{'⚠️ 미저장 변경 있음' if sub_has_unsaved else '✅ 저장 완료'}` · `{'커스텀' if sub_is_custom else '기본값'}`"
            )

        with p_col2:
            st.markdown("#### 종합 LLM 프롬프트 (저장소별 통합 & 카드 생성)")
            draft_final_prompt = st.text_area(
                "종합 명령 프롬프트 (수정 가능)",
                value=st.session_state["saved_final_base_prompt"],
                height=150,
                key="ui_draft_final_base_prompt",
            )
            draft_final_cmd_tok = count_ui_tokens(draft_final_prompt)
            final_has_unsaved = draft_final_prompt.strip() != st.session_state["saved_final_base_prompt"].strip()

            st.markdown("**🔒 종합 구조 프롬프트 (변경 불가 · 섹션/카드 스키마)**")
            st.code(final_struct_str, language="json")
            st.markdown(
                f"- **명령 프롬프트 토큰(저장 기준)**: `{final_cmd_saved_tokens:,} tokens` (입력창 `{draft_final_cmd_tok:,} tokens`)\n"
                f"- **구조 프롬프트 토큰(고정 기본값)**: `{final_struct_tokens:,} tokens` (JSON 스키마 + 템플릿 구조)\n"
                f"- **종합 프롬프트 합계 토큰**: **`{final_total_fixed_tokens:,} tokens`**\n"
                f"- **상태**: `{'⚠️ 미저장 변경 있음' if final_has_unsaved else '✅ 저장 완료'}` · `{'커스텀' if final_is_custom else '기본값'}`"
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
        per_chunk_budgets, effective_total_sub_pool = compute_ui_sub_budgets(
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
            st.caption(f"비례 배분 계산식: 전체 한도 **{effective_total_sub_pool:,} tokens** × 각 청크 데이터 토큰 비율(%)")

        budget_preview_rows = []
        sum_chunk_tok = sum(chunk_tok_list) or 1
        sum_budget_tok = sum(per_chunk_budgets) or 1
        for idx, sp in enumerate(sub_prompts):
            c_tok = chunk_tok_list[idx]
            b_tok = per_chunk_budgets[idx] if idx < len(per_chunk_budgets) else 50
            budget_preview_rows.append({
                "청크": f"#{sp.get('chunk_index', idx + 1)} {sp.get('repo_name', '-')}",
                "청크 데이터 토큰": c_tok,
                "데이터 비율(%)": round((c_tok / sum_chunk_tok) * 100, 1),
                "배분 비율(%)": round((b_tok / sum_budget_tok) * 100, 1),
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
                        "compressed_tokens", count_ui_tokens(json.dumps(comp_obj, ensure_ascii=False))
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
            save_trace({**load_trace(), "pipeline_res": new_pipe, "pipeline_step_stage": 2})
            st.rerun()

        except requests.exceptions.ConnectionError:
            st.error(f"백엔드 서버({backend_url})에 연결할 수 없습니다. FastAPI 서버 상태를 확인해 주세요.")

    if not has_ai_result:
        return

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
            diff_ret = round(on_rec.get("sub_item_retention_pct", 0.0) - off_rec.get("sub_item_retention_pct", 0.0), 2)
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
    sub_table_df = flatten_sub_outputs_to_df(sub_prompts)
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
    gen_table_df = flatten_generated_sections_to_df(gen_sections)

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
                        val_html = format_card_value_html(v)
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
        exp_label = sv_c1.text_input("실험 라벨 메모", placeholder="예: 비례배분_템플릿ON_한도2000", key="ui_exp_label")
        if sv_c2.button("📥 CSV 4종 일괄 저장 (4단계 탭 분석용)", type="primary", width="stretch", key="btn_save_csv_all"):
            saved_id = save_run_to_csvs(pipe=pipe, chunk_limit_mode=chunk_limit_mode, run_label=exp_label)
            st.success(f"저장 완료 (`run_id: {saved_id}`) — 상단 **4단계 · 해석 및 분석** 탭에서 확인하세요.")