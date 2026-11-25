import re
from pathlib import Path
from typing import Any
import altair as alt
import pandas as pd
import streamlit as st
from ui_constants import EXPERIMENT_DIR

ACADEMIC_REFERENCES = [
    {
        "ref_id": "[3]",
        "short_name": "Lost in the Middle (Liu et al., TACL 2024)",
        "full_citation": 'N. F. Liu, K. Lin, J. Hewitt, A. Paranjape, M. Bevilacqua, F. Petroni, and P. Liang, "Lost in the middle: How language models use long contexts," Transactions of the Association for Computational Linguistics (TACL), vol. 12, pp. 157–173, Feb. 2024.',
        "url": "https://aclanthology.org/2024.tacl-1.9/",
        "arxiv_url": "https://arxiv.org/abs/2307.03172",
        "applied_feature": "절차 2·4: 구조 보존형 청크 분할 및 저장소 단위 재통합",
        "interpretation": "긴 단일 문맥 중간부의 핵심 근거가 유실되는 현상을 막기 위해 청크 분할 및 저장소별 병합 구조를 쓰는 근거입니다.",
    },
    {
        "ref_id": "[4]",
        "short_name": "Chain of Density / CoD (Adams et al., 2023)",
        "full_citation": 'G. Adams, A. Fabbri, F. Ladhak, E. Lehman, and N. Elhadad, "From sparse to dense: GPT-4 summarization with chain of density prompting," in Proc. 4th New Frontiers in Summarization Workshop, pp. 68–74, Dec. 2023.',
        "url": "https://aclanthology.org/2023.newsum-1.7/",
        "arxiv_url": "https://arxiv.org/abs/2309.04269",
        "applied_feature": "절차 3: 토큰 압축률(TCR) vs. 서브 항목 보존율(ERR_sub) 트레이드오프",
        "interpretation": "제한된 토큰 상한 내에서 정보 밀도를 높일 때 발생하는 개체 누락과 압축률의 균형을 해석합니다.",
    },
    {
        "ref_id": "[8]",
        "short_name": "RAGAS (Es et al., EACL 2024)",
        "full_citation": 'S. Es, J. James, L. Espinosa-Anke, and S. Schockaert, "RAGAS: Automated evaluation of retrieval augmented generation," in Proc. 18th Conf. European Chapter of the ACL (EACL), pp. 150–158, Mar. 2024.',
        "url": "https://aclanthology.org/2024.eacl-demo.16/",
        "arxiv_url": "https://arxiv.org/abs/2309.15217",
        "applied_feature": "절차 3·4: 서브 보존율(ERR_sub) 및 종합 근거 반영률(EGR_final)",
        "interpretation": "생성 결과가 원천 데이터에 의해 얼마나 충실히 뒷받침되는지(Faithfulness)를 정량화합니다.",
    },
    {
        "ref_id": "[11]",
        "short_name": "LLMLingua (Jiang et al., EMNLP 2023)",
        "full_citation": 'H. Jiang, Q. Wu, C.-Y. Lin, Y. Yang, and L. Qiu, "LLMLingua: Compressing prompts for accelerated inference of large language models," in Proc. 2023 Conf. Empirical Methods in Natural Language Processing (EMNLP), pp. 13358–13376, Dec. 2023.',
        "url": "https://aclanthology.org/2023.emnlp-main.825/",
        "arxiv_url": "https://arxiv.org/abs/2310.05736",
        "applied_feature": "절차 2·3: 예산 상한 내 대용량 데이터 수용(TCR) 및 동적 예산 할당",
        "interpretation": "무료·저비용 LLM의 엄격한 토큰 한도 내에 대용량 컨텍스트를 수용하기 위해 압축률과 예산 배분을 설계하는 근거입니다.",
    },
    {
        "ref_id": "[12]",
        "short_name": "Selective Context (Li et al., EMNLP 2023)",
        "full_citation": 'Y. Li, B. Dong, F. Guerin, and C. Lin, "Compressing context to enhance inference efficiency of large language models," in Proc. 2023 Conf. Empirical Methods in Natural Language Processing (EMNLP), pp. 6342–6353, Dec. 2023.',
        "url": "https://aclanthology.org/2023.emnlp-main.391/",
        "arxiv_url": "https://arxiv.org/abs/2310.06201",
        "applied_feature": "절차 1·3: 중복 노이즈 제거 및 1차 압축 탈락(X→X) 해석",
        "interpretation": "정보량이 낮은 반복 로그 제거와 실제 핵심 근거 탈락을 구분해 해석하는 기준입니다.",
    },
    {
        "ref_id": "[13]",
        "short_name": "LongLLMLingua (Jiang et al., ACL 2024)",
        "full_citation": 'H. Jiang et al., "LongLLMLingua: Accelerating and enhancing LLMs in long context scenarios via prompt compression," in Proc. 62nd Annual Meeting of the ACL, pp. 1658–1677, Aug. 2024.',
        "url": "https://aclanthology.org/2024.acl-long.91/",
        "arxiv_url": "https://arxiv.org/abs/2310.06839",
        "applied_feature": "절차 3: 서브 단계 템플릿 컬럼 반영(Template ON vs. OFF) 비교",
        "interpretation": "목표 컬럼(target_columns)을 기준으로 압축할 때 불필요한 로그가 걸러지는 효과를 설명합니다.",
    },
    {
        "ref_id": "[14]",
        "short_name": "FActScore (Min et al., EMNLP 2023)",
        "full_citation": 'S. Min et al., "FActScore: Fine-grained atomic evaluation of factual precision in long form text generation," in Proc. 2023 Conf. Empirical Methods in Natural Language Processing (EMNLP), pp. 12076–12100, Dec. 2023.',
        "url": "https://aclanthology.org/2023.emnlp-main.741/",
        "arxiv_url": "https://arxiv.org/abs/2305.14251",
        "applied_feature": "절차 3·4: 원자적 항목(items_{run_id}.csv) 추적 및 완전일치(O→O)",
        "interpretation": "커밋·PR·이슈 단위의 원자적 사실(Atomic Facts)이 압축 후에도 살아남았는지 추적합니다.",
    },
    {
        "ref_id": "[15]",
        "short_name": "SummaC (Laban et al., TACL 2022)",
        "full_citation": 'P. Laban, T. Schnabel, P. N. Bennett, and M. A. Hearst, "SummaC: Re-visiting NLI-based models for inconsistency detection in summarization," Transactions of the Association for Computational Linguistics (TACL), vol. 10, pp. 163–177, Feb. 2022.',
        "url": "https://aclanthology.org/2022.tacl-1.10/",
        "arxiv_url": "https://arxiv.org/abs/2111.09525",
        "applied_feature": "절차 4: 2차 추상화 전환(O→X) 해석",
        "interpretation": "서브 단계에서 남았던 항목이 최종 카드에서 고수준 문장으로 합쳐지며 원문 문자열과 달라지는 현상을 해석합니다.",
    },
    {
        "ref_id": "[16]",
        "short_name": "FRANK Benchmark (Pagnoni et al., NAACL 2021)",
        "full_citation": 'A. Pagnoni, V. Balakrishnan, and Y. Tsvetkov, "Understanding factuality in abstractive summarization with FRANK: A benchmark for factuality metrics," in Proc. 2021 Conf. North American Chapter of the ACL (NAACL), pp. 4812–4829, Jun. 2021.',
        "url": "https://aclanthology.org/2021.naacl-main.383/",
        "arxiv_url": "https://arxiv.org/abs/2104.13346",
        "applied_feature": "절차 3·4: 오류 유형 분류 (1차 압축 누락 vs. 2차 추상화 검증)",
        "interpretation": "요약 단계에서 발생한 실제 정보 누락과 표현 변화(추상화)를 구분해 카드 품질을 진단합니다.",
    },
]


def _make_project_run_id(name: str) -> str:
    cleaned = re.sub(r"[^\w\-]+", "_", str(name).strip(), flags=re.UNICODE).strip("_")
    return cleaned or "project_run"


def save_project_as_single_id(
    folder_name: str,
    df_chunks: pd.DataFrame,
    df_items: pd.DataFrame,
    df_generated: pd.DataFrame,
    summary_extra: dict | None = None,
) -> str:
    proj_dir: Path = EXPERIMENT_DIR / folder_name
    proj_dir.mkdir(parents=True, exist_ok=True)
    proj_id = _make_project_run_id(folder_name)

    for df_target, prefix in [
        (df_chunks, "chunks"),
        (df_items, "items"),
        (df_generated, "generated"),
    ]:
        if df_target is not None and not df_target.empty:
            out_df = df_target.copy()
            out_df["run_id"] = proj_id
            out_df.to_csv(proj_dir / f"{prefix}_{proj_id}.csv", index=False, encoding="utf-8-sig")

    raw_tok = int(df_chunks["압축전토큰"].fillna(0).sum()) if (df_chunks is not None and not df_chunks.empty and "압축전토큰" in df_chunks.columns) else 0
    comp_tok = int(df_chunks["압축후토큰"].fillna(0).sum()) if (df_chunks is not None and not df_chunks.empty and "압축후토큰" in df_chunks.columns) else 0
    budget_sum = int(df_chunks["할당한도(tokens)"].fillna(0).sum()) if (df_chunks is not None and not df_chunks.empty and "할당한도(tokens)" in df_chunks.columns) else 0
    tcr_pct = round((1 - (comp_tok / raw_tok)) * 100, 2) if raw_tok > 0 else 0.0

    total_items = len(df_items) if df_items is not None else 0
    sub_o = int((df_items["서브보존여부"] == "O").sum()) if (df_items is not None and not df_items.empty and "서브보존여부" in df_items.columns) else 0
    sub_ret = round((sub_o / total_items) * 100, 2) if total_items > 0 else 0.0

    row_dict = {
        "run_id": proj_id,
        "대상저장소": folder_name,
        "프로젝트폴더": folder_name,
        "실험라벨": proj_id,
        "청크수": len(df_chunks) if df_chunks is not None else 0,
        "서브한도합계": budget_sum,
        "압축전토큰": raw_tok,
        "압축후토큰": comp_tok,
        "토큰압축률(%)": tcr_pct,
        "원본항목수": total_items,
        "서브보존항목수": sub_o,
        "서브보존율(%)": sub_ret,
        "최종생성카드수": len(df_generated) if df_generated is not None else 0,
    }
    if summary_extra:
        row_dict.update(summary_extra)
        row_dict["run_id"] = proj_id
        row_dict["실험라벨"] = proj_id

    s_path = proj_dir / "runs_summary.csv"
    if s_path.exists():
        s_df = pd.read_csv(s_path, dtype={"run_id": str})
        s_df = s_df[s_df["run_id"].astype(str) != proj_id]
        s_df = pd.concat([pd.DataFrame([row_dict]), s_df], ignore_index=True)
    else:
        s_df = pd.DataFrame([row_dict])
    s_df.to_csv(s_path, index=False, encoding="utf-8-sig")
    return proj_id


def _alt_grouped_bar(df_long: pd.DataFrame, x_col: str, y_col: str, group_col: str, title: str, unit: str = ""):
    """
    막대 폭을 그룹 수에 맞춰 넓게 유지하고, 축·범례·제목·막대 라벨이 이미지 저장 시
    잘리거나 맞닿지 않도록 넉넉한 여백(padding / titlePadding / labelPadding / offset)을 확보한 그룹 막대그래프입니다.
    """
    is_float = pd.api.types.is_float_dtype(df_long[y_col])
    fmt = ",.2f" if is_float else ","

    n_groups = max(1, df_long[group_col].nunique())
    n_x = max(1, df_long[x_col].nunique())
    bar_width = max(18, min(50, int(230 / (n_groups * max(1, n_x * 0.5)))))

    y_min = float(df_long[y_col].min())
    y_max = float(df_long[y_col].max())
    span = max(y_max - y_min, 1.0)
    top_pad = span * 0.20
    bottom_pad = span * 0.08
    y_scale = alt.Scale(
        domain=[min(0.0, y_min - bottom_pad), y_max + top_pad],
        nice=True,
    )

    base = alt.Chart(df_long).encode(
        x=alt.X(
            f"{x_col}:N",
            sort=None,
            title=None,
            axis=alt.Axis(
                labelAngle=-16,
                labelLimit=0,
                labelFontSize=11,
                labelPadding=12,
                titlePadding=16,
                tickSize=6,
            ),
        ),
        xOffset=alt.XOffset(f"{group_col}:N"),
        y=alt.Y(
            f"{y_col}:Q",
            scale=y_scale,
            title=unit if unit else None,
            axis=alt.Axis(
                labelPadding=10,
                titlePadding=14,
            ),
        ),
        color=alt.Color(
            f"{group_col}:N",
            title="구분",
            legend=alt.Legend(
                orient="bottom",
                columns=2,
                labelLimit=0,
                labelFontSize=11.5,
                titleFontSize=12.5,
                padding=14,
                offset=24,
                rowPadding=8,
                columnPadding=18,
            ),
        ),
        tooltip=[x_col, group_col, alt.Tooltip(f"{y_col}:Q", format=fmt)],
    )

    bars = base.mark_bar(
        width=bar_width,
        cornerRadiusTopLeft=4,
        cornerRadiusTopRight=4,
    )

    pos_labels = (
        base.transform_filter(alt.datum[y_col] >= 0)
        .mark_text(dy=-10, fontSize=11, fontWeight="bold", baseline="bottom")
        .encode(text=alt.Text(f"{y_col}:Q", format=fmt))
    )
    neg_labels = (
        base.transform_filter(alt.datum[y_col] < 0)
        .mark_text(dy=16, fontSize=11, fontWeight="bold", baseline="top")
        .encode(text=alt.Text(f"{y_col}:Q", format=fmt))
    )

    return (
        (bars + pos_labels + neg_labels)
        .properties(
            height=410,
            title=alt.TitleParams(
                text=title,
                fontSize=14.5,
                fontWeight="bold",
                anchor="start",
                offset=18,
            ),
            padding={"left": 24, "right": 28, "top": 22, "bottom": 28},
        )
        .configure_view(stroke=None)
    )


def _render_academic_reference_guide() -> None:
    with st.expander("📚 평가 지표 및 기능별 인용 논문 원작 정보 · 원본 링크 보기", expanded=False):
        st.caption("각 지표와 손실 분류는 아래 논문들의 평가 개념을 바탕으로 구성되었습니다.")
        for ref in ACADEMIC_REFERENCES:
            with st.container(border=True):
                c1, c2 = st.columns([3, 1])
                with c1:
                    st.markdown(f"**{ref['ref_id']} {ref['short_name']}**")
                    st.markdown(f"- **적용 절차/지표:** `{ref['applied_feature']}`")
                    st.markdown(f"- **해석 근거:** {ref['interpretation']}")
                    st.caption(ref["full_citation"])
                with c2:
                    st.link_button("🔗 학회 원문 (ACL)", ref["url"], use_container_width=True)
                    st.link_button("📄 arXiv 프리프린트", ref["arxiv_url"], use_container_width=True)


def _render_logical_methodology_blueprint() -> None:
    with st.container(border=True):
        st.markdown("#### 🧭 연구 증명 목표: 무료 API 토큰 한도 내 대용량 수용 vs. 원천 데이터 유지")
        p1, p2, p3, p4 = st.columns(4)
        with p1:
            st.markdown("**[절차 1] 원천 기준 집합 확보**")
            st.caption("정제된 커밋·PR·이슈(`items_{run_id}.csv`)를 검증 기준($E_{raw}$)으로 정의합니다 [12, 14].")
        with p2:
            st.markdown("**[절차 2] 한도 초과분 청크 축소**")
            st.caption("한도를 초과하는 원천 토큰을 청크 분할 및 예산 한도 내로 압축($TCR$)합니다 [3, 11].")
        with p3:
            st.markdown("**[절차 3] 1차 압축 후 데이터 유지**")
            st.caption("축소된 토큰 속에 원천 항목이 얼마나 남는지(`ERR_sub`, `X→X` 손실) 검증합니다 [4, 12, 13].")
        with p4:
            st.markdown("**[절차 4] 2차 통합 및 추상화 분리**")
            st.caption("서브 보존 항목(`O`)이 최종 카드에서 직접 일치(`O→O`) 또는 추상화(`O→X`)되는 비율을 분해합니다 [8, 15, 16].")


def _find_run_csv(proj_dir: Path, prefix: str, proj_id: str, raw_rid: str) -> Path | None:
    p_proj = proj_dir / f"{prefix}_{proj_id}.csv"
    if p_proj.exists():
        return p_proj
    p_raw = proj_dir / f"{prefix}_{raw_rid}.csv"
    if p_raw.exists():
        return p_raw
    return None


def _scan_all_experiment_folders() -> dict[str, dict]:
    catalog: dict[str, dict] = {}
    if not EXPERIMENT_DIR.exists():
        return catalog

    for folder_path in sorted(EXPERIMENT_DIR.iterdir()):
        if not folder_path.is_dir():
            continue

        folder_name = folder_path.name
        proj_id = _make_project_run_id(folder_name)
        s_path = folder_path / "runs_summary.csv"
        s_df = pd.read_csv(s_path, dtype={"run_id": str}) if s_path.exists() else pd.DataFrame()

        chosen_row = pd.Series({"run_id": proj_id, "대상저장소": folder_name, "실험라벨": proj_id})
        raw_rid = proj_id

        if not s_df.empty:
            matched = s_df[s_df["run_id"].astype(str).str.strip() == proj_id]
            if not matched.empty:
                chosen_row = matched.iloc[0]
                raw_rid = str(chosen_row.get("run_id", proj_id)).strip()
            else:
                chosen_row = s_df.iloc[0]
                raw_rid = str(chosen_row.get("run_id", proj_id)).strip() or proj_id

        repo_title = str(chosen_row.get("대상저장소") or chosen_row.get("프로젝트폴더") or folder_name)
        meta = {
            "folder": folder_name,
            "proj_id": proj_id,
            "raw_rid": raw_rid,
            "project_name": repo_title,
            "exp_label": proj_id,
            "alloc_mode": str(chosen_row.get("배분방식") or "-"),
            "tpl_mode": str(chosen_row.get("서브템플릿반영") or "-"),
            "chunk_cnt": chosen_row.get("청크수", "-"),
            "exec_time": str(chosen_row.get("실행시각") or "-"),
            "summary_row": chosen_row,
        }

        display_name = f"{proj_id} ({repo_title})"
        catalog[folder_name] = {
            "dir": folder_path,
            "proj_id": proj_id,
            "summary_df": s_df,
            "run_ids": [proj_id],
            "meta_by_rid": {proj_id: meta},
            "display_to_rid": {display_name: proj_id},
            "rid_to_display": {proj_id: display_name},
        }

    return catalog


def _safe_int(val: Any, default: int = 0) -> int:
    if val is None:
        return default
    try:
        if pd.isna(val):
            return default
    except (TypeError, ValueError):
        pass
    if isinstance(val, str):
        s = val.strip().replace(",", "")
        if not s or s.lower() in {"nan", "none", "null"}:
            return default
        try:
            return int(float(s))
        except ValueError:
            return default
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return default


def _calculate_from_run_csvs(proj_dir: Path, meta: dict, folder_name: str) -> dict:
    s_row = meta["summary_row"]
    proj_id = meta["proj_id"]
    raw_rid = meta["raw_rid"]

    p_chunks = _find_run_csv(proj_dir, "chunks", proj_id, raw_rid)
    p_items = _find_run_csv(proj_dir, "items", proj_id, raw_rid)
    p_gen = _find_run_csv(proj_dir, "generated", proj_id, raw_rid)

    df_c = pd.read_csv(p_chunks, dtype={"run_id": str}) if p_chunks else pd.DataFrame()
    df_i = pd.read_csv(p_items, dtype={"run_id": str}) if p_items else pd.DataFrame()
    df_g = pd.read_csv(p_gen, dtype={"run_id": str}) if p_gen else pd.DataFrame()

    if not df_c.empty and {"압축전토큰", "압축후토큰"}.issubset(df_c.columns):
        raw_tok = int(df_c["압축전토큰"].fillna(0).sum())
        comp_tok = int(df_c["압축후토큰"].fillna(0).sum())
        budget_sum = int(df_c["할당한도(tokens)"].fillna(0).sum()) if "할당한도(tokens)" in df_c.columns else int(s_row.get("서브한도합계", 0) or 0)
        chunk_cnt = len(df_c)
        tcr_pct = round((1 - (comp_tok / raw_tok)) * 100, 2) if raw_tok > 0 else 0.0
    else:
        raw_tok = int(s_row.get("압축전토큰", 0) or 0)
        comp_tok = int(s_row.get("압축후토큰", 0) or 0)
        budget_sum = int(s_row.get("서브한도합계", 0) or 0)
        chunk_cnt = int(s_row.get("청크수", 0) or 0)
        tcr_pct = float(s_row.get("토큰압축률(%)", 0.0) or 0.0)

    overflow_tok = max(0, raw_tok - budget_sum)
    saved_tok = max(0, raw_tok - comp_tok)
    overflow_resolved_pct = round(min(100.0, (saved_tok / overflow_tok) * 100), 2) if overflow_tok > 0 else 100.0

    total_items = len(df_i)
    if total_items > 0 and {"서브보존여부", "종합반영여부"}.issubset(df_i.columns):
        sub_o_cnt = int((df_i["서브보존여부"] == "O").sum())
        sub_x_cnt = int((df_i["서브보존여부"] == "X").sum())
        oo_cnt = int(((df_i["서브보존여부"] == "O") & (df_i["종합반영여부"] == "O")).sum())
        ox_cnt = int(((df_i["서브보존여부"] == "O") & (df_i["종합반영여부"] == "X")).sum())
        xx_cnt = int(((df_i["서브보존여부"] == "X") & (df_i["종합반영여부"] == "X")).sum())
        xo_cnt = int(((df_i["서브보존여부"] == "X") & (df_i["종합반영여부"] == "O")).sum())

        sub_ret_pct = round((sub_o_cnt / total_items) * 100, 2)
        final_ref_pct = round(((oo_cnt + xo_cnt) / total_items) * 100, 2)
        oo_pct = round((oo_cnt / total_items) * 100, 2)
        ox_pct = round((ox_cnt / total_items) * 100, 2)
        xx_pct = round((xx_cnt / total_items) * 100, 2)
    else:
        total_items = int(s_row.get("원본항목수", 0) or 0)
        sub_o_cnt = int(s_row.get("서브보존항목수", 0) or 0)
        sub_x_cnt = max(0, total_items - sub_o_cnt)
        oo_cnt = ox_cnt = xx_cnt = xo_cnt = 0
        sub_ret_pct = float(s_row.get("서브보존율(%)", 0.0) or 0.0)
        final_ref_pct = float(s_row.get("종합근거반영률(%)", 0.0) or 0.0)
        oo_pct = ox_pct = xx_pct = 0.0

    card_cnt = len(df_g) if not df_g.empty else int(s_row.get("최종생성카드수", 0) or 0)
    project_title = str(s_row.get("대상저장소") or s_row.get("프로젝트폴더") or folder_name)

    return {
        "folder": folder_name,
        "run_id": proj_id,
        "raw_rid": raw_rid,
        "project_title": project_title,
        "label": proj_id,
        "short_tag": proj_id,
        "time": str(s_row.get("실행시각") or "-"),
        "alloc_mode": str(s_row.get("배분방식") or "-"),
        "tpl_mode": str(s_row.get("서브템플릿반영") or "-"),
        "split_mode": str(s_row.get("분할기준") or "-"),
        "chunk_cnt": chunk_cnt,
        "card_cnt": card_cnt,
        "budget_sum": budget_sum,
        "raw_tok": raw_tok,
        "comp_tok": comp_tok,
        "overflow_tok": overflow_tok,
        "saved_tok": saved_tok,
        "overflow_resolved_pct": overflow_resolved_pct,
        "tcr_pct": tcr_pct,
        "total_items": total_items,
        "sub_o_cnt": sub_o_cnt,
        "sub_x_cnt": sub_x_cnt,
        "oo_cnt": oo_cnt,
        "ox_cnt": ox_cnt,
        "xx_cnt": xx_cnt,
        "xo_cnt": xo_cnt,
        "sub_ret_pct": sub_ret_pct,
        "final_ref_pct": final_ref_pct,
        "oo_pct": oo_pct,
        "ox_pct": ox_pct,
        "xx_pct": xx_pct,
        "api_tokens": _safe_int(s_row.get("실측총API토큰", 0), 0),
        "has_detail_csvs": (p_chunks is not None) and (p_items is not None),
        "df_chunks": df_c,
        "df_items": df_i,
        "df_generated": df_g,
    }


def _render_single_run_readable_card(m: dict) -> None:
    with st.container(border=True):
        st.markdown(f"### 📂 {m['project_title']} — ID: `{m['run_id']}`")
        st.caption(
            f"프로젝트 ID: **`{m['run_id']}`** | 폴더: `{m['folder']}` | 배분방식: **{m['alloc_mode']}** | "
            f"서브 템플릿: **{m['tpl_mode']}** | 청크 수: **{m['chunk_cnt']}개** → 최종 카드: **{m['card_cnt']}개**"
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("실제 절감 토큰량", f"-{m['saved_tok']:,} tok", delta=f"{m['raw_tok']:,} → {m['comp_tok']:,} tok ({m['tcr_pct']}%)")
        c2.metric("한도 초과분 해소율", f"{m['overflow_resolved_pct']}%", delta=f"초과분 {m['overflow_tok']:,} tok 대비")
        c3.metric("1차 데이터 유지율", f"{m['sub_ret_pct']}%", delta=f"유지 {m['sub_o_cnt']}개 / 탈락 {m['sub_x_cnt']}개")
        c4.metric("최종 직접 일치 (O→O)", f"{m['oo_pct']}% ({m['oo_cnt']}개)", delta=f"2차 추상화(O→X): {m['ox_cnt']}개")

        df_c = m["df_chunks"]
        if not df_c.empty and {"압축전토큰", "할당한도(tokens)", "압축후토큰"}.issubset(df_c.columns):
            idx_c = [f"청크 {int(r)}" for r in df_c.get("청크번호", range(1, len(df_c) + 1))]
            raw_vals = df_c["압축전토큰"].fillna(0).astype(int).values
            limit_vals = df_c["할당한도(tokens)"].fillna(0).astype(int).values
            comp_vals = df_c["압축후토큰"].fillna(0).astype(int).values
            saved_vals = (raw_vals - comp_vals).clip(min=0)

            g_left, g_right = st.columns(2)
            with g_left:
                bar_tok_long = pd.DataFrame({
                    "청크": idx_c * 3,
                    "구분": ["압축 전"] * len(idx_c) + ["할당 한도"] * len(idx_c) + ["압축 후"] * len(idx_c),
                    "토큰": list(raw_vals) + list(limit_vals) + list(comp_vals),
                })
                st.altair_chart(
                    _alt_grouped_bar(bar_tok_long, "청크", "토큰", "구분", f"[{m['run_id']}] 청크별 토큰 비교", "tokens"),
                    use_container_width=True,
                )

            with g_right:
                bar_saved_long = pd.DataFrame({
                    "청크": idx_c,
                    "구분": ["절감 토큰량"] * len(idx_c),
                    "절감량": list(saved_vals),
                })
                st.altair_chart(
                    _alt_grouped_bar(bar_saved_long, "청크", "절감량", "구분", f"[{m['run_id']}] 청크별 실제 절감 토큰량", "tokens"),
                    use_container_width=True,
                )


def _render_multi_run_thesis_figures(metrics_list: list[dict]) -> None:
    base = metrics_list[0]

    st.markdown("#### 📈 그림 1. 무료 LLM 토큰 한도 내 대용량 데이터 수용 및 절감 비교")
    c1_l, c1_r = st.columns(2)

    with c1_l:
        tok_long = []
        for m in metrics_list:
            tok_long.append({"단계": "1. 압축 전", "프로젝트ID": m["short_tag"], "토큰": m["raw_tok"]})
            tok_long.append({"단계": "2. 할당 한도", "프로젝트ID": m["short_tag"], "토큰": m["budget_sum"]})
            tok_long.append({"단계": "3. 압축 후", "프로젝트ID": m["short_tag"], "토큰": m["comp_tok"]})
        st.altair_chart(
            _alt_grouped_bar(pd.DataFrame(tok_long), "단계", "토큰", "프로젝트ID", "단계별 토큰 비교(압축 전·할당 한도·압축 후)", "tokens"),
            use_container_width=True,
        )

    with c1_r:
        saved_long = [
            {"프로젝트ID": m["short_tag"], "지표": "실제 절감량", "토큰": m["saved_tok"]}
            for m in metrics_list
        ] + [
            {"프로젝트ID": m["short_tag"], "지표": "압축 후 잔여", "토큰": m["comp_tok"]}
            for m in metrics_list
        ]
        st.altair_chart(
            _alt_grouped_bar(pd.DataFrame(saved_long), "프로젝트ID", "토큰", "지표", "프로젝트별 절감 토큰 vs 압축 후 잔여 토큰", "tokens"),
            use_container_width=True,
        )

    with st.container(border=True):
        st.markdown("**📚 그림 1 인용 논문 매핑 근거**")
        st.markdown(
            "- **[3] Lost in the Middle (TACL 2024):** 긴 문맥 중간부 근거 손실을 막기 위해 구조 보존형 청크 분할을 적용합니다.\n"
            "- **[11] LLMLingua (EMNLP 2023) & [12] Selective Context (EMNLP 2023):** 무료 Groq 토큰 한도 내에 대용량 데이터를 수용하기 위해 원천 토큰을 축소(`절감 토큰량`)해 한도 안으로 넣는 과정을 보여줍니다."
        )

    st.markdown("#### 📊 그림 2. 압축률(TCR) 대비 원천 데이터 유지율(ERR_sub) 및 기준 대비 차이(%p)")
    c2_l, c2_r = st.columns(2)

    with c2_l:
        rate_long = []
        for m in metrics_list:
            rate_long.append({"지표": "토큰 압축률(%)", "프로젝트ID": m["short_tag"], "비율": m["tcr_pct"]})
            rate_long.append({"지표": "1차 유지율(%)", "프로젝트ID": m["short_tag"], "비율": m["sub_ret_pct"]})
            rate_long.append({"지표": "최종 일치율(%)", "프로젝트ID": m["short_tag"], "비율": m["oo_pct"]})
        st.altair_chart(
            _alt_grouped_bar(pd.DataFrame(rate_long), "지표", "비율", "프로젝트ID", "압축률 vs 데이터 유지율(%)", "%"),
            use_container_width=True,
        )

    with c2_r:
        diff_long = []
        for m in metrics_list:
            diff_long.append({"프로젝트ID": m["short_tag"], "구분": "압축률 차이(%p)", "차이": round(m["tcr_pct"] - base["tcr_pct"], 2)})
            diff_long.append({"프로젝트ID": m["short_tag"], "구분": "1차 유지율 차이(%p)", "차이": round(m["sub_ret_pct"] - base["sub_ret_pct"], 2)})
        st.altair_chart(
            _alt_grouped_bar(pd.DataFrame(diff_long), "프로젝트ID", "차이", "구분", f"기준({base['short_tag']}) 대비 격차(%p)", "%p"),
            use_container_width=True,
        )

    with st.container(border=True):
        st.markdown("**📚 그림 2 인용 논문 매핑 근거**")
        st.markdown(
            "- **[4] Chain of Density (2023) & [13] LongLLMLingua (ACL 2024):** 압축률(`TCR`)을 높일 때 유지율(`ERR_sub`)이 어떻게 달라지는지 트레이드오프를 비교합니다.\n"
            "- **[8] RAGAS (EACL 2024):** 압축 결과가 원천 데이터에 충실한지 비율 격차(`%p`)로 검증합니다."
        )

    st.markdown("#### 🔍 그림 3. 원천 원자적 항목의 단계별 유지 · 1차 탈락 · 2차 추상화 건수")
    item_long = []
    for m in metrics_list:
        item_long.append({"상태": "1차 통과(서브 O)", "프로젝트ID": m["short_tag"], "건수": m["sub_o_cnt"]})
        item_long.append({"상태": "최종 일치(O→O)", "프로젝트ID": m["short_tag"], "건수": m["oo_cnt"]})
        item_long.append({"상태": "1차 탈락(X→X)", "프로젝트ID": m["short_tag"], "건수": m["xx_cnt"]})
        item_long.append({"상태": "2차 추상화(O→X)", "프로젝트ID": m["short_tag"], "건수": m["ox_cnt"]})

    st.altair_chart(
        _alt_grouped_bar(pd.DataFrame(item_long), "상태", "건수", "프로젝트ID", "원자적 항목 상태별 건수 비교", "개"),
        use_container_width=True,
    )

    with st.container(border=True):
        st.markdown("**📚 그림 3 인용 논문 매핑 근거**")
        st.markdown(
            "- **[14] FActScore (EMNLP 2023):** 원자적 사실(`Atomic Facts`) 단위 생존 건수(`서브 O`, `O→O`)를 집계합니다.\n"
            "- **[12] Selective Context & [16] FRANK (NAACL 2021):** 1차 압축 단계에서 걸러진 항목(`X→X`)과 2차 종합 요약에서 추상화된 항목(`O→X`)을 구분합니다.\n"
            "- **[15] SummaC (TACL 2022):** 1차 보존 항목이 최종 카드에서 고수준 문장으로 합쳐지며 원문 문자열과 달라지는 전환(`O→X`)을 해석합니다."
        )


def render_stage4() -> None:
    st.header("4단계: 프로젝트별 실험 데이터 손실·보존 비교 분석")

    _render_academic_reference_guide()
    _render_logical_methodology_blueprint()

    catalog = _scan_all_experiment_folders()
    if not catalog:
        st.warning(
            "저장된 프로젝트 실험 폴더(`experiments/{프로젝트명}/`)가 없습니다. "
            "3단계에서 실행 후 **CSV 4종 일괄 저장**을 먼저 진행해 주세요."
        )
        return

    folder_options = list(catalog.keys())
    sel_proj = st.selectbox(
        "📁 4-1 기준 프로젝트 선택 (프로젝트당 단일 ID 적용)",
        options=folder_options,
        index=0,
        format_func=lambda f: f"📂 {catalog[f]['proj_id']} (폴더: {f})",
        key="stage4_selected_project_folder",
    )

    proj_info = catalog[sel_proj]
    proj_dir: Path = proj_info["dir"]
    proj_id: str = proj_info["proj_id"]
    meta = proj_info["meta_by_rid"][proj_id]

    st.divider()
    st.subheader(f"4-1. 프로젝트 ID `{proj_id}` 수용량 · 절감량 · 유지율 분석")

    if st.button(f"💾 `{proj_id}` 이름으로 단일 ID CSV 저장/동기화", key=f"sync_proj_id_{sel_proj}"):
        cur_m = _calculate_from_run_csvs(proj_dir, meta, sel_proj)
        save_project_as_single_id(
            sel_proj,
            cur_m["df_chunks"],
            cur_m["df_items"],
            cur_m["df_generated"],
        )
        st.success(f"`{proj_id}` ID 기준으로 CSV 파일 저장이 완료되었습니다.")
        st.rerun()

    m_single = _calculate_from_run_csvs(proj_dir, meta, sel_proj)
    _render_single_run_readable_card(m_single)

    export_df = pd.DataFrame([{
        "프로젝트ID": m_single["run_id"],
        "폴더명": m_single["folder"],
        "프로젝트명": m_single["project_title"],
        "배분방식": m_single["alloc_mode"],
        "서브템플릿": m_single["tpl_mode"],
        "원천→한도→압축후(tok)": f"{m_single['raw_tok']} → {m_single['budget_sum']} → {m_single['comp_tok']}",
        "실제절감량(tok)": m_single["saved_tok"],
        "토큰압축률(%)": m_single["tcr_pct"],
        "한도초과해소율(%)": m_single["overflow_resolved_pct"],
        "1차데이터유지율(%)": m_single["sub_ret_pct"],
        "최종직접일치_O→O(%)": f"{m_single['oo_cnt']} ({m_single['oo_pct']}%)",
        "1차탈락_X→X(%)": f"{m_single['xx_cnt']} ({m_single['xx_pct']}%)",
        "2차추상화_O→X(%)": f"{m_single['ox_cnt']} ({m_single['ox_pct']}%)",
    }])

    st.markdown("#### 📊 논문 Table 2 수록용 요약표 (수용량 vs. 데이터 유지율)")
    st.dataframe(export_df, width="stretch", hide_index=True)

    st.download_button(
        f"⬇️ [{proj_id}] 논문 Table 2 비교표 CSV 다운로드",
        data=export_df.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"thesis_table2_{proj_id}.csv",
        mime="text/csv",
        key=f"dl_thesis_table2_{sel_proj}",
    )

    st.divider()
    st.subheader("4-2. 전체 프로젝트 교차 비교 그래프 (프로젝트명 단일 ID 기준) 및 논문 매핑")

    all_folders = list(catalog.keys())
    if len(all_folders) < 2:
        st.info("다른 프로젝트와 비교하려면 `experiments/` 아래에 프로젝트 폴더가 2개 이상 있어야 합니다.")
        return

    selected_folders = st.multiselect(
        "비교할 프로젝트 선택 (1번 선택 프로젝트가 격차 비교의 기준이 됩니다)",
        options=all_folders,
        default=all_folders[: min(3, len(all_folders))],
        format_func=lambda f: catalog[f]["proj_id"],
        key="stage4_global_multi_compare",
    )

    if len(selected_folders) < 2:
        st.warning("그래프 비교를 위해 **2개 이상**의 프로젝트를 선택해 주세요.")
        return

    compared_metrics = []
    for f_name in selected_folders:
        f_info = catalog[f_name]
        f_meta = f_info["meta_by_rid"][f_info["proj_id"]]
        compared_metrics.append(_calculate_from_run_csvs(f_info["dir"], f_meta, f_name))

    _render_multi_run_thesis_figures(compared_metrics)

    m_first, m_second = compared_metrics[0], compared_metrics[1]
    st.markdown(
        f"#### 🔎 기준(`{m_first['run_id']}`) vs 비교(`{m_second['run_id']}`) 원문 항목·청크·카드 상세 대조"
    )

    tab_items, tab_chunks, tab_cards = st.tabs([
        "🔍 절차 3·4: 원자적 항목 전이 비교 (FActScore / SummaC)",
        "📦 절차 2·3: 청크별 서브 토큰 압축 비교 (LLMLingua)",
        "🗂️ 절차 4: 최종 저장소 통합 카드 비교 (FRANK / RAGAS)",
    ])

    with tab_items:
        df_ia, df_ib = m_first["df_items"], m_second["df_items"]
        if not df_ia.empty and not df_ib.empty:
            join_keys = [k for k in ["저장소", "유형", "핵심문자열(core)"] if k in df_ia.columns and k in df_ib.columns]
            merged_items = pd.merge(
                df_ia[join_keys + ["서브보존여부", "종합반영여부"]],
                df_ib[join_keys + ["서브보존여부", "종합반영여부"]],
                on=join_keys,
                suffixes=(f"_{m_first['run_id']}", f"_{m_second['run_id']}"),
                how="outer",
            )
            only_diff = st.checkbox(
                "두 프로젝트 간 보존/반영 결과가 달라진 항목만 보기",
                value=False,
                key="filter_diff_items_global",
            )
            if only_diff:
                c1 = merged_items[f"서브보존여부_{m_first['run_id']}"] != merged_items[f"서브보존여부_{m_second['run_id']}"]
                c2 = merged_items[f"종합반영여부_{m_first['run_id']}"] != merged_items[f"종합반영여부_{m_second['run_id']}"]
                merged_items = merged_items[c1 | c2]
            st.dataframe(merged_items, width="stretch", height=320, hide_index=True)
        else:
            st.warning("항목 상세 CSV(`items_{프로젝트명}.csv`)가 없는 프로젝트가 포함되어 있습니다.")

    with tab_chunks:
        ca_col, cb_col = st.columns(2)
        with ca_col:
            st.markdown(f"**기준 프로젝트** — `{m_first['run_id']}`")
            if not m_first["df_chunks"].empty:
                st.dataframe(m_first["df_chunks"], width="stretch", height=300, hide_index=True)
            else:
                st.info("청크 CSV 없음")
        with cb_col:
            st.markdown(f"**비교 프로젝트** — `{m_second['run_id']}`")
            if not m_second["df_chunks"].empty:
                st.dataframe(m_second["df_chunks"], width="stretch", height=300, hide_index=True)
            else:
                st.info("청크 CSV 없음")

    with tab_cards:
        ga_col, gb_col = st.columns(2)
        with ga_col:
            st.markdown(f"**기준 프로젝트** — `{m_first['run_id']}`")
            if not m_first["df_generated"].empty:
                st.dataframe(m_first["df_generated"], width="stretch", height=300, hide_index=True)
            else:
                st.info("생성 카드 CSV 없음")
        with gb_col:
            st.markdown(f"**비교 프로젝트** — `{m_second['run_id']}`")
            if not m_second["df_generated"].empty:
                st.dataframe(m_second["df_generated"], width="stretch", height=300, hide_index=True)
            else:
                st.info("생성 카드 CSV 없음")