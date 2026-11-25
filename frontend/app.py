import os
from typing import Any

from dotenv import load_dotenv
import requests
import streamlit as st

from frontend.tab_stage2 import render_stage2
from frontend.tab_stage3 import render_stage3
from frontend.tab_stage4 import render_stage4
from frontend.ui_constants import EXPERIMENT_DIR, TRACE_FILE
from frontend.ui_state import clear_trace, load_trace, restore_session_from_trace, save_trace

load_dotenv()
ENV_GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "").strip()

st.set_page_config(page_title="GitHub 활동 탐색 및 청크 가치 검증기", layout="wide")

st.markdown(
    """
    <style>
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { padding: 8px 18px; font-weight: 600; }
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

restore_session_from_trace()
st.title("GitHub 사용자 활동 탐색 & 청크 기반 데이터 가치 검증")

# 상단 캐시 초기화 바
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
    clear_trace("step3")
    st.rerun()
if top_btn2.button("캐시 초기화 (4단계 CSV)", width="stretch", key="top_btn_reset_step4"):
    clear_trace("step4")
    st.rerun()
if top_btn3.button("캐시 초기화 (2·3단계)", width="stretch", key="top_btn_reset_stage2"):
    clear_trace("stage2")
    st.rerun()
if top_btn4.button("전체 초기화 (1~4단계)", width="stretch", key="top_btn_reset_all"):
    clear_trace("all")
    st.rerun()

# 사이드바: 인증 및 대상 설정
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
                save_trace(
                    {
                        **load_trace(),
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
        clear_trace("step3")
        st.rerun()
    if rst_c2.button("4단계 CSV 초기화", width="stretch", key="btn_reset_step4"):
        clear_trace("step4")
        st.rerun()

    rst_c3, rst_c4 = st.columns(2)
    if rst_c3.button("2·3단계 초기화", width="stretch", key="btn_reset_stage2"):
        clear_trace("stage2")
        st.rerun()
    if rst_c4.button("전체 초기화 (1~4단계)", width="stretch", key="btn_reset_all"):
        clear_trace("all")
        st.rerun()

# 메인 탭 조립
tab_stage2, tab_stage3, tab_stage4 = st.tabs(
    [
        "2단계 · 저장소 선택 및 사용자 활동 분석",
        "3단계 · 청크 분할 가치 평가 및 생성 테스트",
        "4단계 · 해석 및 분석 (CSV 표 검증)",
    ]
)

with tab_stage2:
    render_stage2(backend_url, github_token, github_id, start_date, end_date)

with tab_stage3:
    render_stage3(backend_url, github_id, start_date, end_date)

with tab_stage4:
    render_stage4()