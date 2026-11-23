import requests
import streamlit as st

st.set_page_config(page_title="GitHub 활동 탐색 및 청크 가치 검증기", layout="wide")
st.title("GitHub 사용자 활동 탐색 & 청크 기반 데이터 가치 검증")

with st.sidebar:
    st.header("1. 인증 및 대상 설정")
    backend_url = st.text_input("백엔드 URL", value="http://localhost:8000")
    github_token = st.text_input("GitHub Personal Access Token", type="password")
    github_id = st.text_input("GitHub 사용자 ID", placeholder="예: kimdfarm")
    col_d1, col_d2 = st.columns(2)
    start_date = col_d1.date_input("시작일", value=None)
    end_date = col_d2.date_input("종료일", value=None)

    if st.button("저장소 검색", type="primary", use_container_width=True):
        if not github_id:
            st.error("GitHub 사용자 ID를 입력하세요.")
        else:
            params = {"github_id": github_id}
            if start_date:
                params["start_date"] = str(start_date)
            if end_date:
                params["end_date"] = str(end_date)
            headers = {"Authorization": f"Bearer {github_token}"} if github_token else {}
            r = requests.get(f"{backend_url}/api/develop/github/repositories", params=params, headers=headers, timeout=30)
            if r.status_code == 200:
                st.session_state["repos"] = r.json().get("repositories", [])
            else:
                st.error(f"저장소 조회 실패: {r.text}")

repos = st.session_state.get("repos", [])
selected_repo = None

st.subheader("2. 저장소 선택 및 사용자 활동 분석")
if repos:
    repo_options = {f"{r['full_name']} (★ {r['stargazers_count']} | {r.get('language') or '-'})": r for r in repos}
    picked_label = st.selectbox("검색된 저장소 중 하나를 선택하세요", list(repo_options.keys()))
    selected_repo = repo_options[picked_label]
else:
    manual_repo = st.text_input("또는 저장소 직접 입력 (owner/repo)", placeholder="예: kimdfarm/News_quiz_by_slack")
    if manual_repo:
        selected_repo = {"full_name": manual_repo, "name": manual_repo.split("/")[-1]}

target_author = st.text_input("분석할 사용자 활동 ID(author)", value=github_id)

if st.button("선택 저장소 활동 분석 실행"):
    if not github_token or not selected_repo:
        st.warning("GitHub 토큰과 저장소를 선택/입력하세요.")
    else:
        payload = {
            "github_id": target_author or github_id,
            "repo_name": selected_repo["full_name"],
            "start_date": str(start_date) if start_date else None,
            "end_date": str(end_date) if end_date else None,
        }
        r = requests.post(
            f"{backend_url}/api/github/analyze",
            json=payload,
            headers={"Authorization": f"Bearer {github_token}"},
            timeout=120,
        )
        if r.status_code == 200:
            st.session_state["analysis_data"] = r.json()
            st.success(f"[{selected_repo['full_name']}] 분석 완료!")
        else:
            st.error(f"분석 실패: {r.text}")

analysis_data = st.session_state.get("analysis_data")
if analysis_data:
    st.divider()
    st.subheader("3. 청크 분할 가치 평가 및 백엔드 생성 기능 테스트")

    c1, c2, c3 = st.columns(3)
    max_tokens = c1.number_input("청크/압축 토큰 예산", min_value=500, max_value=15000, value=2000, step=250)
    force_chunk = c2.checkbox("토큰 한도 미달 시에도 강제 청크 분할", value=True)
    run_judge = c3.checkbox("LLM 품질 비교 평가 포함", value=False)

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("청크 가치 평가 실행", use_container_width=True):
            eval_payload = {
                "analysis_data": analysis_data,
                "target_author": target_author or github_id,
                "start_date": str(start_date) if start_date else None,
                "end_date": str(end_date) if end_date else None,
                "max_total_tokens": int(max_tokens),
                "run_llm_judge": run_judge,
            }
            ev = requests.post(f"{backend_url}/api/github/evaluate-chunks", json=eval_payload, timeout=120)
            if ev.status_code == 200:
                st.session_state["eval_result"] = ev.json()
            else:
                st.error(ev.text)

    with col_btn2:
        if st.button("백엔드 구조화 생성 실행", use_container_width=True):
            gen_payload = {
                "title": f"{selected_repo['name'] if selected_repo else 'repo'} 활동 요약",
                "repo_name": selected_repo["full_name"] if selected_repo else "",
                "target_author": target_author or github_id,
                "analysis_data": analysis_data,
                "start_date": str(start_date) if start_date else None,
                "end_date": str(end_date) if end_date else None,
                "force_chunking": force_chunk,
                "max_total_tokens": int(max_tokens),
            }
            gn = requests.post(f"{backend_url}/api/resumes/github-generate", json=gen_payload, timeout=120)
            if gn.status_code == 200:
                st.session_state["gen_result"] = gn.json()
            else:
                st.error(gn.text)

    eval_res = st.session_state.get("eval_result")
    if eval_res:
        m = eval_res["metrics"]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("원본 추정 토큰", f"{m['raw_tokens']:,}")
        k2.metric("청크 합산 토큰", f"{m['chunked_tokens']:,}", delta=f"-{m['token_reduction_pct']}%")
        k3.metric("분할 청크 수", f"{m['chunk_count']}개")
        k4.metric("항목 보존율", f"{m['item_retention_rate_pct']}%")

        with st.expander("정제된 사용자 활동 & 청크 상세 보기"):
            st.json({"cleaned_data": eval_res["cleaned_data"], "chunks": eval_res["chunks"]})

        if eval_res.get("llm_judge"):
            st.info(f"LLM 평가: {eval_res['llm_judge']}")

    gen_res = st.session_state.get("gen_result")
    if gen_res:
        st.subheader("생성된 구조화 결과")
        st.json(gen_res)