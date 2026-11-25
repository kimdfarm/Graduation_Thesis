import datetime
import json
from typing import Any

import pandas as pd

from ui_constants import EXPERIMENT_DIR, SUMMARY_CSV_PATH


def format_card_value_html(val: Any) -> str:
    if isinstance(val, list):
        lines = [str(x).strip() for x in val if str(x).strip()]
    else:
        raw = str(val or "").strip()
        normalized = raw.replace(" · ", "\n• ")
        lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    if not lines:
        return "-"
    formatted_lines = [f"• {line.lstrip('-• ').strip()}" for line in lines]
    return "<br>".join(formatted_lines)


def flatten_generated_sections_to_df(generated_sections: list[dict[str, Any]]) -> pd.DataFrame:
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


def flatten_sub_outputs_to_df(sub_prompts: list[dict[str, Any]]) -> pd.DataFrame:
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


def save_run_to_csvs(pipe: dict, chunk_limit_mode: str, run_label: str = "") -> str:
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

    gen_df = flatten_generated_sections_to_df(pipe.get("generated_sections") or [])
    if not gen_df.empty:
        gen_df.insert(0, "run_id", run_id)
    pd.DataFrame(chunk_rows).to_csv(EXPERIMENT_DIR / f"chunks_{run_id}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(item_rows).to_csv(EXPERIMENT_DIR / f"items_{run_id}.csv", index=False, encoding="utf-8-sig")
    gen_df.to_csv(EXPERIMENT_DIR / f"generated_{run_id}.csv", index=False, encoding="utf-8-sig")
    return run_id