import json
import random
import time as py_time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from chunker import (
    count_tokens,
    merge_compressed_chunks_by_repo,
    preprocess_github_data,
    split_repo_by_properties,
)
from github_client import (
    GITHUB_ENV_TOKEN,
    fetch_filtered_repositories,
    fetch_repository_analytics,
)
from llm_pipeline import (
    FALLBACK_MODEL_NAME,
    PRIMARY_MODEL_NAME,
    STRICT_ANALYSIS_SYSTEM_PROMPT,
    add_usage,
    ai_intelligent_preprocess,
    build_chunk_preview_data,
    calc_chunk_item_retention,
    compress_chunk_with_sub_llm_strict,
    compute_evidence_comparison,
    extract_usage_metrics,
    groq_client,
    invoke_final_synthesis,
    normalize_github_resume_data,
    resolve_template_sections,
    safe_json_parse,
)
from schemas import (
    ChunkEvaluateRequest,
    ChunkPreviewRequest,
    RepoAnalyzeRequest,
    ResumeGenerateRequest,
    SingleChunkCompressRequest,
    SynthesizeChunksRequest,
)

app = FastAPI(title="GitHub Activity & Chunk Value Evaluator")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GENERATED_DOCS: Dict[str, Any] = {}


@app.get("/api/develop/github/repositories")
def get_github_repositories(
    github_id: str,
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    authorization: Optional[str] = Header(None),
):
    repos = fetch_filtered_repositories(
        github_id=github_id,
        start_date=start_date,
        end_date=end_date,
        authorization=authorization,
    )
    return {"repositories": repos}


@app.get("/api/config/status")
def get_config_status():
    return {
        "has_env_github_token": bool(GITHUB_ENV_TOKEN),
        "has_groq_api_key": bool(GROQ_API_KEY),
    }


@app.post("/api/github/analyze")
async def analyze_repository_graphql(payload: RepoAnalyzeRequest, authorization: Optional[str] = Header(None)):
    return await fetch_repository_analytics(
        github_id=payload.github_id,
        repo_name_input=payload.repo_name,
        include_types=payload.include_types,
        authorization=authorization,
    )


@app.post("/api/github/evaluate-chunks")
def evaluate_chunk_value(req: ChunkEvaluateRequest):
    raw_projects = (req.analysis_data or {}).get("projects_data") or [req.analysis_data]
    cleaned_projects = [
        preprocess_github_data(
            repo,
            target_author=req.target_author or repo.get("target_author", ""),
            start_date=req.start_date,
            end_date=req.end_date,
        )
        for repo in raw_projects
        if isinstance(repo, dict)
    ]

    raw_tokens = sum(count_tokens(c) for c in cleaned_projects)
    raw_commits = sum(len(c.get("key_commits", [])) for c in cleaned_projects)
    raw_prs_issues = sum(len(c.get("key_prs_and_issues", [])) for c in cleaned_projects)

    chunks = []
    chunk_budget = max(400, req.max_total_tokens // 2)
    for repo in cleaned_projects:
        chunks.extend(split_repo_by_properties(repo, max_tokens=chunk_budget))

    chunk_tokens = sum(count_tokens(c) for c in chunks)
    token_delta = chunk_tokens - raw_tokens
    token_change_pct = round((token_delta / raw_tokens) * 100, 2) if raw_tokens > 0 else 0.0

    raw_items = set()
    chunk_items = set()
    for c in cleaned_projects:
        raw_items.update(c.get("key_commits", []))
        raw_items.update(c.get("key_prs_and_issues", []))
    for ch in chunks:
        chunk_items.update(ch.get("key_commits", []))
        chunk_items.update(ch.get("key_prs_and_issues", []))

    item_retention_rate = round((len(chunk_items & raw_items) / len(raw_items)) * 100, 2) if raw_items else 100.0

    judge_result = None
    judge_usage_tokens = 0
    if req.run_llm_judge and groq_client:
        judge_prompt = f"""Evaluate whether splitting RAW GitHub activity into CHUNKS preserves technical value without losing evidence or inventing facts.
Return ONLY valid JSON:
{{
  "faithfulness_score": 1-10,
  "coverage_score": 1-10,
  "signal_to_noise_score": 1-10,
  "verdict": "청크 분할의 가치에 대한 2~3문장 한국어 평가"
}}
RAW: {json.dumps(cleaned_projects, ensure_ascii=False)[:4500]}
CHUNKS: {json.dumps(chunks, ensure_ascii=False)[:4500]}
"""
        comp = groq_client.chat.completions.create(
            model=PRIMARY_MODEL_NAME,
            messages=[
                {"role": "system", "content": STRICT_ANALYSIS_SYSTEM_PROMPT},
                {"role": "user", "content": judge_prompt},
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        if getattr(comp, "usage", None):
            judge_usage_tokens = getattr(comp.usage, "total_tokens", 0) or 0
        judge_result = safe_json_parse(comp.choices[0].message.content or "{}")

    return {
        "tokenizer": "tiktoken (o200k_base)",
        "metrics": {
            "raw_tokens": raw_tokens,
            "chunked_tokens": chunk_tokens,
            "token_delta": token_delta,
            "token_change_pct": token_change_pct,
            "chunk_count": len(chunks),
            "filtered_commits": raw_commits,
            "filtered_prs_issues": raw_prs_issues,
            "item_retention_rate_pct": item_retention_rate,
            "judge_api_total_tokens": judge_usage_tokens,
        },
        "cleaned_data": cleaned_projects,
        "chunks": chunks,
        "llm_judge": judge_result,
    }


@app.post("/api/resumes/github-generate")
async def generate_github_resume(payload: ResumeGenerateRequest):
    if not groq_client:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY가 설정되지 않았습니다.")

    analysis_data = payload.analysis_data or {}
    raw_projects = analysis_data.get("projects_data") or ([analysis_data] if analysis_data.get("repo_name") else [])
    if not raw_projects:
        raise HTTPException(status_code=400, detail="분석된 저장소 데이터가 없습니다.")

    cleaned_projects = [
        preprocess_github_data(
            repo,
            target_author=payload.target_author or repo.get("target_author", ""),
            start_date=payload.start_date,
            end_date=payload.end_date,
        )
        for repo in raw_projects
        if isinstance(repo, dict)
    ]

    template_sections = (payload.template_structure or {}).get("sections") or [
        {
            "section_type": "PROJECTS",
            "section_title": "주요 프로젝트 및 기여 내역",
            "columns": payload.template_columns or ["역할 및 구현 내용", "기술 스택 및 성과"],
        }
    ]

    processed_projects, chunk_meta, compression_usage = ai_intelligent_preprocess(
        projects_data=cleaned_projects,
        template_sections=template_sections,
        max_total_tokens=payload.max_total_tokens,
        force_chunking=payload.force_chunking,
    )

    user_prompt = (
        f"[템플릿]\n{json.dumps(template_sections, ensure_ascii=False)}\n\n"
        f"[정제/압축 데이터]\n{json.dumps(processed_projects, ensure_ascii=False)}"
    )

    sections_data = []
    generation_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    call_logs: List[Dict[str, Any]] = []
    used_model = PRIMARY_MODEL_NAME

    for attempt in range(3):
        model_name = PRIMARY_MODEL_NAME if attempt == 0 else FALLBACK_MODEL_NAME
        used_model = model_name
        try:
            res = groq_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": STRICT_ANALYSIS_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            u = extract_usage_metrics(res)
            add_usage(generation_usage, u)
            call_logs.append({"attempt": attempt + 1, "model": model_name, "status": "ok", **u})

            parsed = safe_json_parse(res.choices[0].message.content or "{}")
            sections_data = normalize_github_resume_data(parsed).get("sections", [])
            if sections_data:
                break
        except Exception as err:
            call_logs.append({"attempt": attempt + 1, "model": model_name, "status": "error", "error": str(err)})
            if attempt < 2:
                py_time.sleep((2**attempt) + random.uniform(0.3, 0.8))

    total_api_usage = {
        "prompt_tokens": compression_usage["prompt_tokens"] + generation_usage["prompt_tokens"],
        "completion_tokens": compression_usage["completion_tokens"] + generation_usage["completion_tokens"],
        "total_tokens": compression_usage["total_tokens"] + generation_usage["total_tokens"],
    }

    doc_id = f"doc_{int(py_time.time())}"
    result_obj = {
        "id": doc_id,
        "title": payload.title,
        "category": payload.category,
        "model_used": used_model,
        "tokenizer_estimate": {
            "tokenizer": "tiktoken (o200k_base)",
            "cleaned_input_tokens": sum(count_tokens(p) for p in cleaned_projects),
            "processed_input_tokens": sum(count_tokens(p) for p in processed_projects),
            "prompt_tokens_est": count_tokens(STRICT_ANALYSIS_SYSTEM_PROMPT) + count_tokens(user_prompt),
        },
        "token_usage": {
            "compression_stage": compression_usage,
            "generation_stage": generation_usage,
            "grand_total": total_api_usage,
            "api_calls": call_logs,
        },
        "chunk_meta": chunk_meta,
        "sections": sections_data,
    }
    GENERATED_DOCS[doc_id] = result_obj
    return result_obj


@app.post("/api/github/compress-chunk")
def compress_single_chunk_endpoint(req: SingleChunkCompressRequest):
    t0 = py_time.time()
    usage_delta: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    raw_tokens = count_tokens(req.chunk)

    compressed_obj = compress_chunk_with_sub_llm_strict(
        chunk=req.chunk,
        template_sections=req.template_sections,
        sub_budget_tokens=max(100, int(req.sub_budget_tokens)),
        usage_summary=usage_delta,
        apply_template_in_sub=req.apply_template_in_sub,
        custom_sub_base_prompt=req.custom_sub_base_prompt,
    )
    compressed_tokens = count_tokens(compressed_obj)
    retention_info = calc_chunk_item_retention(req.chunk, compressed_obj)
    elapsed_sec = round(py_time.time() - t0, 2)

    return {
        "chunk_index": req.chunk_index,
        "repo_name": req.chunk.get("repo_name", "-"),
        "raw_tokens": raw_tokens,
        "sub_budget_tokens": req.sub_budget_tokens,
        "compressed_tokens": compressed_tokens,
        "elapsed_sec": elapsed_sec,
        "retention": retention_info,
        "token_usage": usage_delta,
        "compressed_chunk": compressed_obj,
    }


@app.post("/api/github/synthesize-chunks")
def synthesize_compressed_chunks_endpoint(req: SynthesizeChunksRequest):
    aggregated_by_repo = merge_compressed_chunks_by_repo(req.compressed_chunks)
    generated_sections, usage_delta, model_used, combined_raw_prompt = invoke_final_synthesis(
        template_sections=req.template_sections,
        processed_data=aggregated_by_repo,
        language=req.language,
        custom_base_prompt=req.custom_final_base_prompt,
    )
    comparison = compute_evidence_comparison(req.cleaned_data, generated_sections)

    return {
        "combined_prompt": combined_raw_prompt,
        "generated_sections": generated_sections,
        "token_usage": usage_delta,
        "model_used": model_used,
        "comparison": comparison,
    }


@app.post("/api/github/pipeline-inspect")
def run_pipeline_inspect(req: ChunkPreviewRequest):
    preview = build_chunk_preview_data(req)
    template_sections = preview.pop("template_sections")
    sub_prompts = preview["sub_prompts"]
    cleaned_projects = preview["cleaned_data"]
    sub_llm_budget = preview["chunk_limits"]["sub_llm_budget"]

    generated_sections: List[Dict[str, Any]] = []
    usage_summary: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    model_used = PRIMARY_MODEL_NAME

    compressed_chunks_for_final: List[Any] = []
    for sp in sub_prompts:
        sp["sub_llm_max_tokens"] = sub_llm_budget
        raw_ch = sp["raw_chunk_data"]
        if req.run_generation and groq_client:
            compressed_obj = compress_chunk_with_sub_llm_strict(
                chunk=raw_ch,
                template_sections=template_sections,
                sub_budget_tokens=sub_llm_budget,
                usage_summary=usage_summary,
            )
            sp["sub_llm_output"] = compressed_obj
            sp["final_input_tokens"] = count_tokens(compressed_obj)
            compressed_chunks_for_final.append(compressed_obj)
        else:
            sp["sub_llm_output"] = None
            compressed_chunks_for_final.append(raw_ch)

    aggregated_by_repo = merge_compressed_chunks_by_repo(compressed_chunks_for_final)
    if req.run_generation and groq_client:
        generated_sections, syn_usage, model_used, combined_prompt = invoke_final_synthesis(
            template_sections=template_sections,
            processed_data=aggregated_by_repo,
            language=req.language,
        )
        add_usage(usage_summary, syn_usage)
        preview["combined_prompt"] = combined_prompt
    else:
        _, _, _, combined_prompt = invoke_final_synthesis(
            template_sections=template_sections,
            processed_data=aggregated_by_repo,
            language=req.language,
        )
        preview["combined_prompt"] = combined_prompt

    comparison = compute_evidence_comparison(cleaned_projects, generated_sections)

    return {
        **preview,
        "generated_sections": generated_sections,
        "token_usage": usage_summary,
        "model_used": model_used,
        "comparison": comparison,
    }


@app.post("/api/github/chunk-preview")
def run_chunk_preview(req: ChunkPreviewRequest):
    data = build_chunk_preview_data(req)
    data.pop("template_sections", None)
    return data