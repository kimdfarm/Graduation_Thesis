import pandas as pd
import streamlit as st

from frontend.ui_constants import EXPERIMENT_DIR, SUMMARY_CSV_PATH


def render_stage4() -> None:
    st.header("4. 해석 및 분석 (저장된 CSV 표 기반 정량 검증)")

    if not SUMMARY_CSV_PATH.exists():
        st.info("아직 저장된 CSV 실험 기록이 없습니다. **3단계** 실행 후 **'CSV 4종 일괄 저장'**을 눌러 주세요.")
        return

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