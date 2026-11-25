import ast
import json
import os
import random
import re
import time as py_time
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from groq import Groq

from chunker import (
    count_tokens,
    merge_compressed_chunks_by_repo,
    preprocess_github_data,
    split_repo_by_properties,
    split_repo_into_max_chunks,
)
from schemas import ChunkPreviewRequest

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
PRIMARY_MODEL_NAME = os.getenv("PRIMARY_MODEL", "openai/gpt-oss-120b")
FALLBACK_MODEL_NAME = os.getenv("FALLBACK_MODEL", "openai/gpt-oss-20b")

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

STRICT_ANALYSIS_SYSTEM_PROMPT = """You are a strict technical evidence extractor and developer resume structuring engine.
Your job is to transform GitHub repository activity (commits, PRs, issues, languages) into structured JSON for engineering evaluation.

STRICT ANTI-HALLUCINATION & RIGOR RULES:
1. ZERO FABRICATION: Every technical claim, architecture decision, library/framework name, metric, or outcome MUST be directly grounded in the provided input data. If a detail is not present in the input, DO NOT invent or guess it.
2. HIGH-SIGNAL EXTRACTION ONLY: Prioritize concrete engineering actions (API/DB/auth/pipeline design, bug fixes, performance/concurrency/testing work, deployment/CI/CD, key PRs/issues). Strip out meaningless noise like "fix typo", "update readme", or empty labels unless tied to a concrete technical change.
3. EVIDENCE TRACEABILITY: Whenever possible, reference actual commit headlines, PR/issue titles, or line/repetition counts from the input data inside the descriptions so claims are verifiable.
4. SECTION & COLUMN CONSISTENCY: Every section must define a unified `ai_columns` array, and every detail card in that section must populate `column_values` matching those exact column names without omitting columns.
5. OUTPUT FORMAT: Respond ONLY with a valid JSON object matching the requested schema. No markdown fences outside JSON, no commentary.
"""

DEFAULT_SUB_BASE_PROMPT = """You are a strict technical data compression engine for developer activity analysis.
STRICT RULES:
1. ZERO FABRICATION: Every fact, commit message, PR/issue title, line count, repetition count, or tech stack MUST come strictly from the input chunk.
2. DO NOT CREATE RESUME SECTIONS OR CARDS: Only compress raw activity evidence into the flat JSON schema.
3. PRESERVE HIGH-SIGNAL EVIDENCE: Retain concrete technical actions, tech stacks (`tech_stacks`), key commits (`key_commits`), and PRs/issues (`key_prs_and_issues`).
4. DROP TRIVIAL NOISE: Remove meaningless commits/PRs/issues (e.g., typo, formatting) unless tied to concrete technical changes."""

DEFAULT_FINAL_BASE_PROMPT = """You are an expert developer resume formatting and structuring engine.

CRITICAL GOALS:
1. Analyze the aggregated GitHub repository data and organize them into the requested template sections (`template_sections`).
2. **Each Section must define its own unified `ai_columns` array at the section level** matching the provided template columns.
3. **DO NOT OMIT OR OVERSIMPLIFY DETAILS.** For each project or card, write rich, comprehensive resume descriptions covering architecture, technical challenges solved, specific features implemented, and concrete outcomes based *strictly* on the provided commit/PR/issue data. Avoid vague one-line summaries or raw log dumps.
4. All detail cards within that section **must share the exact same `ai_columns`**, acting like a uniform table where each card holds corresponding `column_values`.
5. Never invent or hallucinate missing details; stick strictly to provided facts.
6. Write clear, polished developer resume descriptions rather than copying raw commit headlines verbatim.
7. DETERMINE CARD COUNT BY REPOSITORY / DISTINCT PROJECT: Evaluate the aggregated repository data holistically and create ONE cohesive card per repository unless the data clearly contains multiple distinct standalone sub-projects. Do NOT split a single repository into multiple cards by chunk order.
8. CLEAN BULLET FORMATTING: Format multi-point fields (`핵심 구현 및 커밋 성과`, `트러블슈팅 및 개선` 등) as clean newline-separated bullet lines (`- ...`) rather than long run-on sentences."""


def extract_and_clean_json(text: str) -> str:
    cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", (text or "").strip())
    cleaned = re.sub(r"\n?```$", "", cleaned).strip()
    start_idx, end_idx = cleaned.find("{"), cleaned.rfind("}")
    if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
        return cleaned[start_idx : end_idx + 1]
    return cleaned


def safe_json_parse(json_str: str) -> dict:
    cleaned = extract_and_clean_json(json_str)
    for fn in (
        lambda s: json.loads(s, strict=False),
        lambda s: json.loads(re.sub(r",(\s*[}\]])", r"\1", re.sub(r"(?<!\\)[\r\n]+", r"\\n", s)), strict=False),
        lambda s: ast.literal_eval(s),
    ):
        try:
            res = fn(cleaned)
            if isinstance(res, dict):
                return res
        except Exception:
            continue
    raise ValueError("JSON 파싱 실패")


def normalize_github_resume_data(parsed_dict: dict) -> dict:
    raw_sections = parsed_dict.get("sections") or parsed_dict.get("data") or []
    if isinstance(raw_sections, dict):
        raw_sections = [raw_sections]

    normalized_sections = []
    for idx, sec in enumerate(raw_sections):
        if not isinstance(sec, dict):
            continue
        raw_details = sec.get("details") or sec.get("items") or []
        if isinstance(raw_details, dict):
            raw_details = [raw_details]

        cols = sec.get("ai_columns") or sec.get("columns") or ["항목", "내용"]
        details = []
        for d_idx, item in enumerate(raw_details):
            if not isinstance(item, dict):
                continue
            col_vals = item.get("column_values") or {}
            details.append({
                "id": str(item.get("id") or f"card_{idx+1}_{d_idx+1}"),
                "title": str(item.get("title") or item.get("name") or "항목"),
                "ai_columns": cols,
                "column_values": col_vals,
            })

        if details:
            normalized_sections.append({
                "section_type": str(sec.get("section_type") or "PROJECTS"),
                "section_title": str(sec.get("section_title") or "세부 정보"),
                "display_order": idx + 1,
                "columns": cols,
                "details": details,
            })

    return {"doc_type": parsed_dict.get("doc_type", "RESUME"), "sections": normalized_sections}


def extract_usage_metrics(completion_obj: Any) -> Dict[str, int]:
    usage = getattr(completion_obj, "usage", None)
    if not usage:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    prompt_t = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_t = int(getattr(usage, "completion_tokens", 0) or 0)
    total_t = int(getattr(usage, "total_tokens", 0) or (prompt_t + completion_t))
    return {
        "prompt_tokens": prompt_t,
        "completion_tokens": completion_t,
        "total_tokens": total_t,
    }


def add_usage(target: Dict[str, int], delta: Dict[str, int]) -> None:
    target["prompt_tokens"] += delta.get("prompt_tokens", 0)
    target["completion_tokens"] += delta.get("completion_tokens", 0)
    target["total_tokens"] += delta.get("total_tokens", 0)


def calc_chunk_item_retention(raw_chunk: dict, compressed_chunk: dict) -> Dict[str, Any]:
    raw_items = list(raw_chunk.get("key_commits", [])) + list(raw_chunk.get("key_prs_and_issues", []))
    comp_blob = json.dumps(compressed_chunk, ensure_ascii=False)
    retained = 0
    missing = []
    for line in raw_items:
        core = line.lstrip("- ").split(" (반복")[0].split(" (상태:")[0].strip()
        if core and core in comp_blob:
            retained += 1
        else:
            missing.append(line)
    total = len(raw_items)
    rate = round((retained / total) * 100, 2) if total > 0 else 100.0
    return {
        "raw_item_count": total,
        "retained_item_count": retained,
        "retention_rate_pct": rate,
        "dropped_sample": missing[:10],
    }


def resolve_template_sections(req: ChunkPreviewRequest) -> List[Dict[str, Any]]:
    tpl_struct = req.template_structure or {}
    raw_sections = tpl_struct.get("sections") or []
    normalized: List[Dict[str, Any]] = []

    if isinstance(raw_sections, list):
        for idx, sec in enumerate(raw_sections):
            if not isinstance(sec, dict):
                continue
            cols = [str(c).strip() for c in (sec.get("columns") or []) if str(c).strip()]
            if not cols and req.template_columns:
                cols = [str(c).strip() for c in req.template_columns if str(c).strip()]
            if not cols:
                cols = ["구현 및 문제 해결", "기술 스택 및 활동 근거"]
            normalized.append({
                "section_type": str(sec.get("section_type") or "CUSTOM"),
                "section_title": str(sec.get("title") or sec.get("section_title") or "사용자 기여 분석 및 구현 성과").strip(),
                "display_order": int(sec.get("display_order") or (idx + 1)),
                "columns": cols,
            })

    if normalized:
        return normalized

    fallback_cols = [str(c).strip() for c in (req.template_columns or []) if str(c).strip()] or [
        "구현 및 문제 해결",
        "기술 스택 및 활동 근거",
    ]
    return [
        {
            "section_type": "CUSTOM",
            "section_title": "사용자 기여 분석 및 구현 성과",
            "display_order": 1,
            "columns": fallback_cols,
        }
    ]


def build_sub_prompt(chunk: dict, template_sections: list, budget_tokens: int) -> str:
    target_cols = [col for sec in template_sections for col in (sec.get("columns") or [])]
    return (
        f"[Sub-Prompt | Token Budget: {budget_tokens}]\n"
        f"Extract & map evidence specifically for columns (DO NOT create cards/sections): {json.dumps(target_cols, ensure_ascii=False)}\n"
        f"Chunk Payload:\n{json.dumps(chunk, ensure_ascii=False, indent=2)}"
    )


def build_final_prompt(
    template_sections: list,
    processed_data: list,
    language: str = "ko",
    custom_base_prompt: Optional[str] = None,
) -> tuple[str, str]:
    if language == "ko":
        lang_instruction = (
            "CRITICAL LANGUAGE RULE: You MUST write ALL descriptions, project summaries, "
            "achievements, and text values in natural, professional **Korean (한국어)** suitable for a developer resume. "
            "Even if the original GitHub commit messages, PR titles, or repository descriptions are in English, "
            "translate and re-synthesize them into polished Korean resume sentences. "
            "Technical stack names (e.g., Python, TypeScript, React, FastAPI) and proper nouns can remain as they are."
        )
    else:
        lang_instruction = "Please write everything in professional **English** suitable for a developer resume."

    primary_sec = template_sections[0] if template_sections else {}
    example_cols = primary_sec.get("columns") or ["역할 및 구현", "기술 스택 및 근거"]
    example_col_values = {col: f"- {col} 관점의 구체적이고 상세한 이력서 설명" for col in example_cols}
    base_instruction = (custom_base_prompt or DEFAULT_FINAL_BASE_PROMPT).strip()

    sys_prompt = f"""{base_instruction}
{lang_instruction}
지정된 템플릿 섹션(`section_title`)과 컬럼(`columns` -> `ai_columns`, `column_values`) 구조에 맞춰 결과를 구성하세요.
반드시 아래 JSON 스키마로만 응답하세요:
{{
  "doc_type": "RESUME",
  "sections": [
    {{
      "section_type": "{primary_sec.get('section_type', 'CUSTOM')}",
      "section_title": "{primary_sec.get('section_title', '섹션 제목')}",
      "display_order": 1,
      "ai_columns": {json.dumps(example_cols, ensure_ascii=False)},
      "details": [
        {{
          "id": "card_1",
          "title": "저장소/프로젝트 대표 제목",
          "column_values": {json.dumps(example_col_values, ensure_ascii=False)}
        }}
      ]
    }}
  ]
}}"""
    usr_prompt = (
        f"[템플릿 구조]\n{json.dumps(template_sections, ensure_ascii=False, indent=2)}\n\n"
        f"[저장소별 통합 입력 데이터]\n{json.dumps(processed_data, ensure_ascii=False, indent=2)}"
    )
    return sys_prompt, usr_prompt


def resolve_sub_llm_budget(req: ChunkPreviewRequest, chunk_count: int, final_sys_prompt: str) -> int:
    final_overhead = count_tokens(final_sys_prompt) + 300
    available = max(200, int(req.max_total_tokens) - final_overhead)
    auto_budget = max(100, int((available * 0.9) // max(1, chunk_count)))
    if not req.sub_llm_auto and req.sub_llm_max_tokens and req.sub_llm_max_tokens > 0:
        return int(req.sub_llm_max_tokens)
    return auto_budget


def _build_sub_skeleton(chunk: dict, template_sections: list, apply_template_in_sub: bool) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "repo_name": str(chunk.get("repo_name") or ""),
        "target_author": str(chunk.get("target_author") or ""),
        "tech_stacks": [],
        "key_commits": [],
        "key_prs_and_issues": [],
    }
    if apply_template_in_sub:
        base["column_evidence"] = {
            col: []
            for sec in (template_sections or [])
            for col in (sec.get("columns") or [])
        }
    return base


def _shrink_chunk_to_budget(
    chunk: dict,
    template_sections: list,
    apply_template_in_sub: bool,
    content_budget_tokens: int,
) -> Dict[str, Any]:
    skeleton = _build_sub_skeleton(chunk, template_sections, apply_template_in_sub)
    skeleton_tokens = count_tokens(skeleton)
    target_total_tokens = max(skeleton_tokens + 20, skeleton_tokens + max(20, content_budget_tokens))

    out = json.loads(json.dumps(skeleton, ensure_ascii=False))
    for tech in (chunk.get("tech_stacks") or [])[:5]:
        cand = {**out, "tech_stacks": out["tech_stacks"] + [tech]}
        if count_tokens(cand) <= target_total_tokens:
            out["tech_stacks"].append(tech)

    commits = list(chunk.get("key_commits") or [])
    prs_issues = list(chunk.get("key_prs_and_issues") or [])
    for idx in range(max(len(commits), len(prs_issues))):
        progressed = False
        if idx < len(commits):
            cand = {**out, "key_commits": out["key_commits"] + [commits[idx]]}
            if count_tokens(cand) <= target_total_tokens:
                out["key_commits"].append(commits[idx])
                progressed = True
        if idx < len(prs_issues):
            cand = {**out, "key_prs_and_issues": out["key_prs_and_issues"] + [prs_issues[idx]]}
            if count_tokens(cand) <= target_total_tokens:
                out["key_prs_and_issues"].append(prs_issues[idx])
                progressed = True
        if not progressed and count_tokens(out) >= target_total_tokens:
            break

    return out


def compress_chunk_with_sub_llm_strict(
    chunk: dict,
    template_sections: list,
    sub_budget_tokens: int,
    usage_summary: Dict[str, int],
    apply_template_in_sub: bool = True,
    custom_sub_base_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    skeleton = _build_sub_skeleton(chunk, template_sections, apply_template_in_sub)
    skeleton_tokens = count_tokens(skeleton)
    content_budget = max(30, int(sub_budget_tokens) - skeleton_tokens)
    effective_max_tokens = max(128, skeleton_tokens + content_budget)

    if not groq_client:
        return _shrink_chunk_to_budget(chunk, template_sections, apply_template_in_sub, content_budget)

    base_instruction = (custom_sub_base_prompt or DEFAULT_SUB_BASE_PROMPT).strip()
    chunk_json = json.dumps(chunk, ensure_ascii=False)
    budget_rule = (
        f"\n[OUTPUT BUDGET RULE — EXCLUDING FIXED JSON SKELETON]\n"
        f"- Fixed JSON skeleton uses ~{skeleton_tokens} tokens.\n"
        f"- Keep your extracted content (commits, PRs/issues, tech_stacks, column_evidence) strictly within ~{content_budget} tokens "
        f"(total JSON around ~{effective_max_tokens} tokens).\n"
        f"- Drop lower-priority or noisy items first so the response stays concise and within budget."
    )

    target_columns = [
        col
        for sec in (template_sections or [])
        for col in (sec.get("columns") or [])
    ]

    if apply_template_in_sub and target_columns:
        sub_system_prompt = f"""{base_instruction}{budget_rule}

[Target Columns — EXTRACT EVIDENCE FOR THESE COLUMNS ONLY (DO NOT CREATE RESUME SECTIONS OR CARDS)]
{json.dumps(target_columns, ensure_ascii=False)}

5. STRICT JSON SCHEMA: Respond ONLY with valid JSON using this exact top-level structure:
{json.dumps(skeleton, ensure_ascii=False, indent=2)}"""
    else:
        sub_system_prompt = f"""{base_instruction}{budget_rule}

[Mode: Template-Agnostic Pure Fact Compression]
Do NOT filter or drop technical items for template alignment; pack maximum factual coverage of commits, PRs/issues, and tech stacks within the content budget.

5. STRICT JSON SCHEMA: Respond ONLY with valid JSON using this exact top-level structure:
{json.dumps(skeleton, ensure_ascii=False, indent=2)}"""

    for attempt in range(3):
        model_name = PRIMARY_MODEL_NAME if attempt == 0 else FALLBACK_MODEL_NAME
        try:
            chat_completion = groq_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": sub_system_prompt},
                    {"role": "user", "content": chunk_json},
                ],
                temperature=0.1,
                max_completion_tokens=effective_max_tokens,
                response_format={"type": "json_object"},
            )
            add_usage(usage_summary, extract_usage_metrics(chat_completion))
            content = chat_completion.choices[0].message.content or ""
            parsed = safe_json_parse(content)

            if isinstance(parsed, dict) and (
                "key_commits" in parsed or "key_prs_and_issues" in parsed or "column_evidence" in parsed
            ):
                res_obj: Dict[str, Any] = {
                    "repo_name": str(parsed.get("repo_name") or chunk.get("repo_name") or ""),
                    "target_author": str(parsed.get("target_author") or chunk.get("target_author") or ""),
                    "tech_stacks": parsed.get("tech_stacks") if isinstance(parsed.get("tech_stacks"), list) else [],
                    "key_commits": parsed.get("key_commits") if isinstance(parsed.get("key_commits"), list) else [],
                    "key_prs_and_issues": parsed.get("key_prs_and_issues") if isinstance(parsed.get("key_prs_and_issues"), list) else [],
                }
                if apply_template_in_sub:
                    res_obj["column_evidence"] = parsed.get("column_evidence") if isinstance(parsed.get("column_evidence"), dict) else {}

                if count_tokens(res_obj) > effective_max_tokens:
                    res_obj = _shrink_chunk_to_budget(res_obj, template_sections, apply_template_in_sub, content_budget)
                return res_obj
        except Exception:
            if attempt < 2:
                py_time.sleep((2 ** attempt) + random.uniform(0.3, 0.8))

    return _shrink_chunk_to_budget(chunk, template_sections, apply_template_in_sub, content_budget)


def ai_intelligent_preprocess(
    projects_data: list,
    template_sections: list,
    max_total_tokens: int = 5000,
    force_chunking: bool = False,
) -> tuple[list, dict, Dict[str, int]]:
    total_tokens = sum(count_tokens(p) for p in projects_data)
    compression_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    meta = {
        "tokenizer": "tiktoken (o200k_base)",
        "raw_tokens": total_tokens,
        "chunk_count": 0,
        "compression_applied": False,
        "chunks_detail": [],
    }

    if not force_chunking and total_tokens <= max_total_tokens:
        meta["processed_tokens"] = total_tokens
        return projects_data, meta, compression_usage

    all_chunks = []
    for repo in projects_data:
        if force_chunking or count_tokens(repo) > 1200:
            all_chunks.extend(split_repo_by_properties(repo, max_tokens=1200))
        else:
            all_chunks.append(repo)

    meta["chunk_count"] = len(all_chunks)
    meta["compression_applied"] = True
    meta["chunks_detail"] = [
        {
            "index": i + 1,
            "repo_name": c.get("repo_name"),
            "tokens": count_tokens(c),
            "commits_n": len(c.get("key_commits", [])),
            "prs_issues_n": len(c.get("key_prs_and_issues", [])),
        }
        for i, c in enumerate(all_chunks)
    ]

    if not groq_client:
        merged_fallback = merge_compressed_chunks_by_repo(all_chunks)
        meta["processed_tokens"] = sum(count_tokens(c) for c in merged_fallback)
        return merged_fallback, meta, compression_usage

    chunk_sizes = [count_tokens(c) for c in all_chunks]
    total_chunk_tokens = sum(chunk_sizes) or 1
    compressed_chunks = []

    for idx, (chunk, c_size) in enumerate(zip(all_chunks, chunk_sizes)):
        budget = max(350, int(max_total_tokens * (c_size / total_chunk_tokens)))
        try:
            compressed_obj = compress_chunk_with_sub_llm_strict(
                chunk=chunk,
                template_sections=template_sections,
                sub_budget_tokens=budget,
                usage_summary=compression_usage,
                apply_template_in_sub=True,
            )
            compressed_chunks.append(compressed_obj)
        except Exception:
            compressed_chunks.append(chunk)

    merged_projects = merge_compressed_chunks_by_repo(compressed_chunks)
    meta["processed_tokens"] = sum(count_tokens(p) for p in merged_projects)
    return merged_projects, meta, compression_usage


def build_chunk_preview_data(req: ChunkPreviewRequest) -> Dict[str, Any]:
    analysis_inc = (req.analysis_data or {}).get("include_types")
    raw_projects = (req.analysis_data or {}).get("projects_data") or [req.analysis_data]
    cleaned_projects = [
        preprocess_github_data(
            repo,
            target_author=req.target_author or repo.get("target_author", ""),
            start_date=req.start_date,
            end_date=req.end_date,
            include_types=repo.get("include_types") or analysis_inc,
        )
        for repo in raw_projects
        if isinstance(repo, dict)
    ]

    raw_rows: List[Dict[str, Any]] = []
    for proj in cleaned_projects:
        r_name = proj.get("repo_name", "")
        author = proj.get("target_author", "")
        techs = ", ".join(proj.get("tech_stacks", []))
        for c in proj.get("key_commits", []):
            raw_rows.append({
                "저장소": r_name,
                "작성자": author,
                "유형": "Commit",
                "기술스택": techs,
                "내용": c,
                "토큰수": count_tokens(c),
            })
        for item in proj.get("key_prs_and_issues", []):
            kind = "PR" if "[PR]" in item else "Issue"
            raw_rows.append({
                "저장소": r_name,
                "작성자": author,
                "유형": kind,
                "기술스택": techs,
                "내용": item,
                "토큰수": count_tokens(item),
            })

    mode = req.chunk_limit_mode
    token_limit = max(200, int(req.chunk_token_limit or 1200))
    chunk_cap = max(1, int(req.max_chunks or 6))

    chunks: List[dict] = []
    if mode == "max_chunks":
        per_repo_cap = max(1, chunk_cap // max(1, len(cleaned_projects)))
        for proj in cleaned_projects:
            chunks.extend(split_repo_into_max_chunks(proj, max_chunks=per_repo_cap))
        if len(chunks) > chunk_cap:
            chunks = chunks[:chunk_cap]
        exceeded_chunk_count_limit = len(chunks) > chunk_cap
    else:
        for proj in cleaned_projects:
            if req.force_chunking or count_tokens(proj) > token_limit:
                chunks.extend(split_repo_by_properties(proj, max_tokens=token_limit))
            else:
                chunks.append(proj)
        exceeded_chunk_count_limit = False

    template_sections = resolve_template_sections(req)
    sys_prompt_preview, _ = build_final_prompt(template_sections, [], language=req.language)
    sub_llm_budget = resolve_sub_llm_budget(req, len(chunks), sys_prompt_preview)

    chunk_evaluations: List[Dict[str, Any]] = []
    sub_prompts: List[Dict[str, Any]] = []

    for idx, ch in enumerate(chunks):
        c_tokens = count_tokens(ch)
        c_commits = len(ch.get("key_commits", []))
        c_prs_issues = len(ch.get("key_prs_and_issues", []))
        c_items = c_commits + c_prs_issues

        reasons = []
        if mode == "max_chunks":
            is_exceeded = (idx + 1) > chunk_cap
            if is_exceeded:
                reasons.append(f"제한 청크 수 초과({idx + 1}번째 > {chunk_cap}개)")
            eval_row = {
                "청크번호": idx + 1,
                "저장소/구분": ch.get("repo_name"),
                "커밋수": c_commits,
                "PR/이슈수": c_prs_issues,
                "총항목수": c_items,
                "청크_데이터_토큰수": c_tokens,
                "서브_LLM_출력상한": sub_llm_budget,
                "종합_프롬프트_반영_토큰수": c_tokens,
                "적용기준": f"최대 {chunk_cap}개",
                "한도초과여부": "초과" if is_exceeded else "정상",
                "초과사유": ", ".join(reasons) if reasons else "-",
            }
        else:
            is_exceeded = c_tokens > token_limit
            if is_exceeded:
                reasons.append(f"토큰 초과({c_tokens} > {token_limit})")
            eval_row = {
                "청크번호": idx + 1,
                "저장소/구분": ch.get("repo_name"),
                "커밋수": c_commits,
                "PR/이슈수": c_prs_issues,
                "총항목수": c_items,
                "청크_데이터_토큰수": c_tokens,
                "서브_LLM_출력상한": sub_llm_budget,
                "종합_프롬프트_반영_토큰수": c_tokens,
                "토큰한도": token_limit,
                "한도초과여부": "초과" if is_exceeded else "정상",
                "초과사유": ", ".join(reasons) if reasons else "-",
            }

        fixed_sub_instruction = (
            "[Fixed Sub-LLM Instruction]\n"
            "Compress ONLY the given GitHub chunk into flat JSON aligned with target columns. "
            "Do NOT create resume sections or cards. Zero fabrication, preserve concrete evidence."
        )
        fixed_sub_cmd_tokens = count_tokens(fixed_sub_instruction)
        sub_prompt_text = build_sub_prompt(ch, template_sections, sub_llm_budget)
        fixed_sub_llm_tokens = c_tokens + fixed_sub_cmd_tokens

        chunk_evaluations.append(eval_row)
        sub_prompts.append({
            "chunk_index": idx + 1,
            "repo_name": ch.get("repo_name"),
            "tokens": c_tokens,
            "fixed_sub_cmd_tokens": fixed_sub_cmd_tokens,
            "fixed_sub_llm_tokens": c_tokens,
            "sub_prompt_tokens": count_tokens(sub_prompt_text),
            "final_input_tokens": fixed_sub_llm_tokens,
            "exceeded": is_exceeded,
            "sub_prompt": sub_prompt_text,
            "raw_chunk_data": ch,
        })

    merged_preview = merge_compressed_chunks_by_repo(chunks)
    sys_prompt, final_user_prompt = build_final_prompt(template_sections, merged_preview, language=req.language)
    combined_raw_prompt = f"[SYSTEM PROMPT]\n{sys_prompt}\n\n[USER PROMPT]\n{final_user_prompt}"

    return {
        "tokenizer": "tiktoken (o200k_base)",
        "applied_template": {
            "section_count": len(template_sections),
            "sections": template_sections,
        },
        "cleaned_data": cleaned_projects,
        "raw_table_rows": raw_rows,
        "combined_prompt": combined_raw_prompt,
        "chunk_limits": {
            "max_total_tokens": req.max_total_tokens,
            "chunk_limit_mode": mode,
            "max_chunks": chunk_cap if mode == "max_chunks" else None,
            "chunk_token_limit": token_limit if mode == "chunk_tokens" else None,
            "actual_chunk_count": len(chunks),
            "exceeded_chunk_count_limit": exceeded_chunk_count_limit,
            "sub_llm_budget": sub_llm_budget,
        },
        "chunk_evaluations": chunk_evaluations,
        "sub_prompts": sub_prompts,
        "template_sections": template_sections,
    }


def invoke_final_synthesis(
    template_sections: List[Dict[str, Any]],
    processed_data: List[Dict[str, Any]],
    language: str = "ko",
    custom_base_prompt: Optional[str] = None,
) -> tuple[List[Dict[str, Any]], Dict[str, int], str, str]:
    sys_prompt, final_user_prompt = build_final_prompt(
        template_sections,
        processed_data,
        language=language,
        custom_base_prompt=custom_base_prompt,
    )
    combined_raw_prompt = f"[SYSTEM PROMPT]\n{sys_prompt}\n\n[USER PROMPT]\n{final_user_prompt}"

    generated_sections: List[Dict[str, Any]] = []
    usage_delta: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    model_used = PRIMARY_MODEL_NAME

    if groq_client:
        for attempt in range(3):
            model_used = PRIMARY_MODEL_NAME if attempt == 0 else FALLBACK_MODEL_NAME
            try:
                comp = groq_client.chat.completions.create(
                    model=model_used,
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": final_user_prompt},
                    ],
                    temperature=0.1,
                    response_format={"type": "json_object"},
                )
                add_usage(usage_delta, extract_usage_metrics(comp))
                parsed = safe_json_parse(comp.choices[0].message.content or "{}")
                generated_sections = normalize_github_resume_data(parsed).get("sections", [])
                if generated_sections:
                    break
            except Exception:
                if attempt < 2:
                    py_time.sleep((2 ** attempt) + random.uniform(0.3, 0.8))

    return generated_sections, usage_delta, model_used, combined_raw_prompt


def compute_evidence_comparison(cleaned_projects: List[Dict[str, Any]], generated_sections: List[Dict[str, Any]]) -> Dict[str, Any]:
    all_raw_items = []
    for proj in cleaned_projects:
        all_raw_items.extend(proj.get("key_commits", []))
        all_raw_items.extend(proj.get("key_prs_and_issues", []))

    generated_blob = json.dumps(generated_sections, ensure_ascii=False)
    matched_items = []
    missing_items = []
    for raw_line in all_raw_items:
        core = raw_line.lstrip("- ").split(" (반복")[0].split(" (상태:")[0].strip()
        if core and core in generated_blob:
            matched_items.append(raw_line)
        else:
            missing_items.append(raw_line)

    mention_rate = round((len(matched_items) / len(all_raw_items)) * 100, 2) if all_raw_items else 100.0
    return {
        "total_raw_items": len(all_raw_items),
        "directly_matched_items": len(matched_items),
        "unmatched_or_synthesized_items": len(missing_items),
        "direct_evidence_mention_pct": mention_rate,
        "unmatched_sample": missing_items[:20],
    }