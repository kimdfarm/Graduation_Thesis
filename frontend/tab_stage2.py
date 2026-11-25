from typing import Any

import requests
import streamlit as st

from ui_constants import EXPLORE_LABELS, GITHUB_TEMPLATE_PRESETS
from ui_state import load_trace, save_trace


def render_stage2(backend_url: str, github_token: str, github_id: str, start_date: Any, end_date: Any) -> None:
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
                save_trace({**load_trace(), "analysis_data": analysis_payload})
                names_str = ", ".join(p.get("full_name") or p.get("repo_name", "") for p in collected_projects)
                types_str = ", ".join(EXPLORE_LABELS[t] for t in active_types)
                cols_str = ", ".join(custom_template_payload["sections"][0]["columns"])
                st.success(
                    f"분석 완료 ({len(collected_projects)}개 저장소): {names_str} | "
                    f"탐색 항목: {types_str} | 컬럼: {cols_str} — 상단의 '3단계' 탭으로 이동해 청크 검증을 진행하세요."
                )

            for err_msg in errors:
                st.error(f"분석 실패 — {err_msg}")