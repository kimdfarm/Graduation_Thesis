import datetime
import json
from typing import Any

import streamlit as st

from frontend.ui_constants import EXPERIMENT_DIR, TRACE_FILE


def load_trace() -> dict[str, Any]:
    if not TRACE_FILE.exists():
        return {}
    try:
        return json.loads(TRACE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_trace(data: dict[str, Any]) -> None:
    try:
        TRACE_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def restore_session_from_trace() -> None:
    if "_trace_restored" in st.session_state:
        return
    saved = load_trace()
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


def clear_trace(scope: str) -> None:
    st.cache_data.clear()
    st.cache_resource.clear()
    trace = load_trace()

    if scope == "step3":
        trace.pop("pipeline_res", None)
        trace.pop("pipeline_step_stage", None)
        for k in ("pipeline_res", "pipeline_step_stage", "sub_tpl_mode_history"):
            st.session_state.pop(k, None)
        save_trace(trace)

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
        save_trace(trace)

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