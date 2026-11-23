import ast
import datetime
from datetime import timezone
import json
import os
import random
import re
import time as py_time
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
import httpx
from pydantic import BaseModel
import requests

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
PRIMARY_MODEL_NAME = os.getenv("PRIMARY_MODEL", "openai/gpt-oss-120b")
FALLBACK_MODEL_NAME = os.getenv("FALLBACK_MODEL", "openai/gpt-oss-20b")

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

app = FastAPI(title="GitHub Activity & Chunk Value Evaluator")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 메모리/파일 기반 임시 결과 저장소 (Supabase 대체)
GENERATED_DOCS: Dict[str, Any] = {}


class RepoAnalyzeRequest(BaseModel):
    github_id: str
    repo_name: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class ResumeGenerateRequest(BaseModel):
    title: str
    category: str = "BACKEND_AI"
    repo_name: str
    target_author: Optional[str] = None
    template_columns: Optional[List[str]] = None
    template_structure: Optional[Dict[str, Any]] = None
    language: Optional[str] = "ko"
    analysis_data: Optional[Dict[str, Any]] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    force_chunking: bool = False
    max_total_tokens: int = 5000


class ChunkEvaluateRequest(BaseModel):
    analysis_data: Dict[str, Any]
    target_author: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    max_total_tokens: int = 2000
    template_sections: Optional[List[Dict[str, Any]]] = None
    run_llm_judge: bool = False


GRAPHQL_REPO_QUERY = """
query GetRepoAnalytics($owner: String!, $name: String!, $afterCursor: String) {
  repository(owner: $owner, name: $name) {
    name
    description
    isFork
    stargazerCount
    forkCount

    languages(first: 10, orderBy: {field: SIZE, direction: DESC}) {
      totalSize
      edges {
        size
        node { name }
      }
    }

    defaultBranchRef {
      target {
        ... on Commit {
          history(first: 100, after: $afterCursor) {
            pageInfo {
              hasNextPage
              endCursor
            }
            nodes {
              messageHeadline
              committedDate
              additions
              deletions
              author {
                user {
                  login
                }
              }
            }
          }
        }
      }
    }

    issues(first: 100, orderBy: {field: CREATED_AT, direction: DESC}) {
      nodes {
        title
        state
        createdAt
        author {
          login
        }
        labels(first: 5) {
          nodes { name }
        }
      }
    }

    pullRequests(first: 100, orderBy: {field: CREATED_AT, direction: DESC}) {
      nodes {
        title
        state
        mergedAt
        createdAt
        additions
        deletions
        reviewDecision
        author {
          login
        }
        labels(first: 5) {
          nodes { name }
        }
      }
    }
  }
}
"""


def parse_iso_dt(date_str: Optional[str]) -> Optional[datetime.datetime]:
    if not date_str:
        return None
    try:
        dt = datetime.datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def check_user_has_commits_in_repo(repo_full_name: str, github_id: str, headers: dict) -> bool:
    if not repo_full_name or not github_id:
        return False
    try:
        url = f"https://api.github.com/repos/{repo_full_name}/commits?author={github_id}&per_page=1"
        res = requests.get(url, headers=headers, timeout=5)
        if res.status_code == 200 and isinstance(res.json(), list):
            return len(res.json()) > 0
        return False
    except Exception:
        return False


def preprocess_github_data(
    raw_graphql_data: dict,
    default_repo_name: str = "",
    target_author: str = "",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> dict:
    effective_author = (target_author or raw_graphql_data.get("target_author") or "").strip()
    repo_name = (
        raw_graphql_data.get("name")
        or raw_graphql_data.get("repo_name")
        or raw_graphql_data.get("full_name")
        or default_repo_name
        or "프로젝트"
    )

    languages_data = raw_graphql_data.get("languages") or {}
    tech_stacks: List[str] = []
    if isinstance(languages_data, dict):
        if "nodes" in languages_data:
            tech_stacks = [n.get("name") for n in (languages_data.get("nodes") or []) if n and n.get("name")]
        elif "edges" in languages_data:
            tech_stacks = [e.get("node", {}).get("name") for e in (languages_data.get("edges") or []) if e and e.get("node")]

    start_dt = None
    end_dt = None
    if start_date:
        start_dt = datetime.datetime.strptime(start_date[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
    if end_date:
        end_dt = datetime.datetime.strptime(end_date[:10], "%Y-%m-%d").replace(
            hour=23, minute=59, second=59, tzinfo=timezone.utc
        )

    # 1. 커밋 정제 (특정 사용자 author 필터링 + 날짜 필터링)
    commits_raw = (
        (((raw_graphql_data.get("defaultBranchRef") or {}).get("target") or {}).get("history") or {}).get("nodes")
        or []
    )
    merged_commits: Dict[str, Dict[str, int]] = {}
    for c in commits_raw:
        if not isinstance(c, dict):
            continue
        c_dt = parse_iso_dt(c.get("committedDate"))
        if c_dt and ((start_dt and c_dt < start_dt) or (end_dt and c_dt > end_dt)):
            continue

        author_login = (((c.get("author") or {}).get("user") or {}).get("login") or "").strip()
        if effective_author:
            if not author_login or author_login.lower() != effective_author.lower():
                continue

        msg = (c.get("messageHeadline") or c.get("message") or "").strip()
        if not msg:
            continue

        additions = c.get("additions", 0) or 0
        deletions = c.get("deletions", 0) or 0
        if additions > 5000:
            additions = 100
        if deletions > 5000:
            deletions = 100

        if msg not in merged_commits:
            merged_commits[msg] = {"additions": 0, "deletions": 0, "count": 0}
        merged_commits[msg]["additions"] += additions
        merged_commits[msg]["deletions"] += deletions
        merged_commits[msg]["count"] += 1

    clean_commits = [
        f"- {msg} (반복 {stat['count']}회, +{stat['additions']}/-{stat['deletions']}라인)"
        for msg, stat in merged_commits.items()
    ]

    # 2. PR & Issue 정제
    clean_prs: List[str] = []
    for pr in (raw_graphql_data.get("pullRequests") or {}).get("nodes") or []:
        if not isinstance(pr, dict):
            continue
        pr_dt = parse_iso_dt(pr.get("createdAt"))
        if pr_dt and ((start_dt and pr_dt < start_dt) or (end_dt and pr_dt > end_dt)):
            continue
        pr_author = ((pr.get("author") or {}).get("login") or "").strip()
        if effective_author and (not pr_author or pr_author.lower() != effective_author.lower()):
            continue
        title = (pr.get("title") or "").strip()
        if title:
            clean_prs.append(f"- [PR] {title} (상태: {pr.get('state', '')})")

    clean_issues: List[str] = []
    for issue in (raw_graphql_data.get("issues") or {}).get("nodes") or []:
        if not isinstance(issue, dict):
            continue
        issue_dt = parse_iso_dt(issue.get("createdAt"))
        if issue_dt and ((start_dt and issue_dt < start_dt) or (end_dt and issue_dt > end_dt)):
            continue
        issue_author = ((issue.get("author") or {}).get("login") or "").strip()
        if effective_author and (not issue_author or issue_author.lower() != effective_author.lower()):
            continue
        title = (issue.get("title") or "").strip()
        if title:
            clean_issues.append(f"- [Issue] {title} (상태: {issue.get('state', '')})")

    return {
        "repo_name": repo_name,
        "target_author": effective_author or "ALL",
        "tech_stacks": tech_stacks,
        "key_commits": clean_commits,
        "key_prs_and_issues": clean_prs + clean_issues,
    }


def estimate_tokens(obj: Any) -> int:
    text = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
    return max(1, len(text) // 2)


def split_repo_by_properties(repo_data: dict, max_tokens: int = 1200) -> list:
    chunks = []
    current_chunk = {
        "repo_name": repo_data.get("repo_name"),
        "target_author": repo_data.get("target_author"),
        "tech_stacks": repo_data.get("tech_stacks", []),
        "key_commits": [],
        "key_prs_and_issues": [],
    }

    for commit in repo_data.get("key_commits", []):
        candidate = dict(current_chunk)
        candidate["key_commits"] = current_chunk["key_commits"] + [commit]
        if estimate_tokens(candidate) > max_tokens and current_chunk["key_commits"]:
            chunks.append(current_chunk)
            current_chunk = {
                "repo_name": f"{repo_data.get('repo_name')} (commits part)",
                "target_author": repo_data.get("target_author"),
                "tech_stacks": [],
                "key_commits": [commit],
                "key_prs_and_issues": [],
            }
        else:
            current_chunk["key_commits"].append(commit)

    for item in repo_data.get("key_prs_and_issues", []):
        candidate = dict(current_chunk)
        candidate["key_prs_and_issues"] = current_chunk["key_prs_and_issues"] + [item]
        if estimate_tokens(candidate) > max_tokens and (current_chunk["key_commits"] or current_chunk["key_prs_and_issues"]):
            chunks.append(current_chunk)
            current_chunk = {
                "repo_name": f"{repo_data.get('repo_name')} (pr/issue part)",
                "target_author": repo_data.get("target_author"),
                "tech_stacks": [],
                "key_commits": [],
                "key_prs_and_issues": [item],
            }
        else:
            current_chunk["key_prs_and_issues"].append(item)

    if current_chunk["key_commits"] or current_chunk["key_prs_and_issues"] or current_chunk["tech_stacks"]:
        chunks.append(current_chunk)

    return chunks


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


def ai_intelligent_preprocess(
    projects_data: list,
    template_sections: list,
    max_total_tokens: int = 5000,
    force_chunking: bool = False,
) -> tuple[list, dict]:
    total_tokens = sum(estimate_tokens(p) for p in projects_data)
    meta = {
        "raw_tokens": total_tokens,
        "chunk_count": 0,
        "compression_applied": False,
        "chunks_detail": [],
    }

    if not force_chunking and total_tokens <= max_total_tokens:
        meta["processed_tokens"] = total_tokens
        return projects_data, meta

    all_chunks = []
    for repo in projects_data:
        if force_chunking or estimate_tokens(repo) > 1200:
            all_chunks.extend(split_repo_by_properties(repo, max_tokens=1200))
        else:
            all_chunks.append(repo)

    meta["chunk_count"] = len(all_chunks)
    meta["compression_applied"] = True
    meta["chunks_detail"] = [
        {
            "index": i + 1,
            "repo_name": c.get("repo_name"),
            "tokens": estimate_tokens(c),
            "commits_n": len(c.get("key_commits", [])),
            "prs_issues_n": len(c.get("key_prs_and_issues", [])),
        }
        for i, c in enumerate(all_chunks)
    ]

    if not groq_client:
        meta["processed_tokens"] = sum(estimate_tokens(c) for c in all_chunks)
        return all_chunks, meta

    chunk_sizes = [estimate_tokens(c) for c in all_chunks]
    total_chunk_tokens = sum(chunk_sizes) or 1
    template_str = json.dumps(template_sections, ensure_ascii=False)
    compressed_projects = []

    system_prompt = f"""You are a developer resume data compression engine.
Align the given GitHub activity chunk with the target template sections:
{template_str}

Rules:
1. Keep repository/author identity clear.
2. Preserve technical stack, key architecture/feature work, and concrete outcomes from commits/PRs/issues.
3. Drop trivial noise (typos, formatting) unless significant.
4. Respond ONLY with valid JSON.
"""

    for idx, (chunk, c_size) in enumerate(zip(all_chunks, chunk_sizes)):
        budget = max(350, int(max_total_tokens * (c_size / total_chunk_tokens)))
        try:
            completion = groq_client.chat.completions.create(
                model=PRIMARY_MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(chunk, ensure_ascii=False)},
                ],
                temperature=0.1,
                max_completion_tokens=budget,
                response_format={"type": "json_object"},
            )
            parsed = safe_json_parse(completion.choices[0].message.content or "")
            compressed_projects.append(parsed)
        except Exception:
            compressed_projects.append(chunk)

    meta["processed_tokens"] = sum(estimate_tokens(p) for p in compressed_projects)
    return compressed_projects, meta


@app.get("/api/develop/github/repositories")
def get_github_repositories(
    github_id: str,
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    authorization: Optional[str] = Header(None),
):
    if not start_date or not end_date:
        return {"repositories": []}

    token = authorization.replace("Bearer ", "").strip() if authorization else ""
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    raw_repos = []
    gh_res = requests.get(
        f"https://api.github.com/users/{github_id}/repos?sort=updated&per_page=100",
        headers=headers,
        timeout=10,
    )
    if gh_res.status_code == 200 and isinstance(gh_res.json(), list):
        raw_repos.extend(gh_res.json())

    search_res = requests.get(
        f"https://api.github.com/search/commits?q=author:{github_id}&sort=author-date&order=desc&per_page=30",
        headers=headers,
        timeout=10,
    )
    if search_res.status_code == 200:
        repo_names = {
            item["repository"]["full_name"]
            for item in (search_res.json().get("items") or [])
            if isinstance(item.get("repository"), dict) and item["repository"].get("full_name")
        }
        for full_name in repo_names:
            detail_res = requests.get(f"https://api.github.com/repos/{full_name}", headers=headers, timeout=5)
            if detail_res.status_code == 200:
                raw_repos.append(detail_res.json())

    start_dt = datetime.datetime.strptime(start_date[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.datetime.strptime(end_date[:10], "%Y-%m-%d").replace(
        hour=23, minute=59, second=59, tzinfo=timezone.utc
    )

    unique_repos: Dict[str, dict] = {}
    for repo in raw_repos:
        if not isinstance(repo, dict):
            continue
        full_name = repo.get("full_name")
        if not full_name:
            continue
        if repo.get("fork", False) and not check_user_has_commits_in_repo(full_name, github_id, headers):
            continue

        pushed_dt = parse_iso_dt(repo.get("pushed_at") or repo.get("updated_at"))
        if pushed_dt and not (start_dt <= pushed_dt <= end_dt):
            continue

        unique_repos[full_name] = {
            "id": repo.get("id"),
            "name": repo.get("name"),
            "full_name": full_name,
            "description": repo.get("description"),
            "is_private": repo.get("private", False),
            "is_fork": repo.get("fork", False),
            "stargazers_count": repo.get("stargazers_count", 0),
            "language": repo.get("language"),
            "updated_at": repo.get("updated_at"),
        }

    return {"repositories": list(unique_repos.values())}


@app.post("/api/github/analyze")
async def analyze_repository_graphql(payload: RepoAnalyzeRequest, authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="GitHub Access Token이 필요합니다.")
    token = authorization.split(" ", 1)[1].strip()

    if "/" in payload.repo_name:
        owner, repo_name = payload.repo_name.split("/", 1)
    else:
        owner, repo_name = payload.github_id, payload.repo_name

    all_commits = []
    after_cursor = None
    base_repo_data: Dict[str, Any] = {}

    async with httpx.AsyncClient(timeout=120.0) as client:
        for _ in range(3):
            resp = await client.post(
                "https://api.github.com/graphql",
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                json={
                    "query": GRAPHQL_REPO_QUERY,
                    "variables": {"owner": owner, "name": repo_name, "afterCursor": after_cursor},
                },
            )
            data = resp.json()
            if "errors" in data:
                raise HTTPException(status_code=400, detail=f"GraphQL 오류: {data['errors']}")

            repo_analytics = (data.get("data") or {}).get("repository") or {}
            if not repo_analytics:
                break
            if not base_repo_data:
                base_repo_data = repo_analytics

            history = (((repo_analytics.get("defaultBranchRef") or {}).get("target") or {}).get("history")) or {}
            all_commits.extend(history.get("nodes") or [])
            page_info = history.get("pageInfo") or {}
            if not page_info.get("hasNextPage"):
                break
            after_cursor = page_info.get("endCursor")

    if not base_repo_data:
        raise HTTPException(status_code=404, detail="저장소 데이터를 가져올 수 없습니다.")

    if base_repo_data.get("defaultBranchRef") and ((base_repo_data["defaultBranchRef"].get("target") or {}).get("history") is not None):
        base_repo_data["defaultBranchRef"]["target"]["history"]["nodes"] = all_commits

    base_repo_data["repo_name"] = base_repo_data.get("name") or repo_name
    base_repo_data["full_name"] = f"{owner}/{repo_name}"
    base_repo_data["target_author"] = payload.github_id
    return base_repo_data


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

    raw_tokens = sum(estimate_tokens(c) for c in cleaned_projects)
    raw_commits = sum(len(c.get("key_commits", [])) for c in cleaned_projects)
    raw_prs_issues = sum(len(c.get("key_prs_and_issues", [])) for c in cleaned_projects)

    chunks = []
    for repo in cleaned_projects:
        chunks.extend(split_repo_by_properties(repo, max_tokens=max(400, req.max_total_tokens // 2)))

    chunk_tokens = sum(estimate_tokens(c) for c in chunks)
    compression_ratio = round((1 - (chunk_tokens / raw_tokens)) * 100, 2) if raw_tokens > 0 else 0.0

    # 원본 항목 대비 청크에 보존된 고유 항목 비율
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
    if req.run_llm_judge and groq_client:
        prompt = f"""Compare RAW GitHub activity vs CHUNKED GitHub activity for resume/portfolio generation.
Return JSON with:
{{
  "faithfulness_score": 1-10,
  "coverage_score": 1-10,
  "conciseness_score": 1-10,
  "verdict": "청크 분할의 가치에 대한 2~3문장 한국어 평가"
}}
RAW: {json.dumps(cleaned_projects, ensure_ascii=False)[:4000]}
CHUNKS: {json.dumps(chunks, ensure_ascii=False)[:4000]}
"""
        comp = groq_client.chat.completions.create(
            model=PRIMARY_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        judge_result = safe_json_parse(comp.choices[0].message.content or "{}")

    return {
        "metrics": {
            "raw_tokens": raw_tokens,
            "chunked_tokens": chunk_tokens,
            "token_reduction_pct": compression_ratio,
            "chunk_count": len(chunks),
            "filtered_commits": raw_commits,
            "filtered_prs_issues": raw_prs_issues,
            "item_retention_rate_pct": item_retention_rate,
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

    processed_projects, chunk_meta = ai_intelligent_preprocess(
        projects_data=cleaned_projects,
        template_sections=template_sections,
        max_total_tokens=payload.max_total_tokens,
        force_chunking=payload.force_chunking,
    )

    system_prompt = """You are a developer resume structuring engine.
Generate structured JSON based strictly on the provided GitHub activity data and template sections.
All descriptions must be in natural Korean.
Return ONLY valid JSON:
{
  "doc_type": "RESUME",
  "sections": [
    {
      "section_type": "PROJECTS",
      "section_title": "섹션 제목",
      "display_order": 1,
      "ai_columns": ["컬럼1", "컬럼2"],
      "details": [
        {
          "id": "card_1",
          "title": "프로젝트명",
          "column_values": {"컬럼1": "내용", "컬럼2": "내용"}
        }
      ]
    }
  ]
}
"""
    user_prompt = (
        f"[템플릿]\n{json.dumps(template_sections, ensure_ascii=False)}\n\n"
        f"[정제 데이터]\n{json.dumps(processed_projects, ensure_ascii=False)}"
    )

    sections_data = []
    for attempt in range(3):
        model_name = PRIMARY_MODEL_NAME if attempt == 0 else FALLBACK_MODEL_NAME
        try:
            res = groq_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            parsed = safe_json_parse(res.choices[0].message.content or "{}")
            sections_data = normalize_github_resume_data(parsed).get("sections", [])
            if sections_data:
                break
        except Exception:
            if attempt < 2:
                py_time.sleep((2**attempt) + random.uniform(0.3, 0.8))

    doc_id = f"doc_{int(py_time.time())}"
    result_obj = {
        "id": doc_id,
        "title": payload.title,
        "category": payload.category,
        "chunk_meta": chunk_meta,
        "sections": sections_data,
    }
    GENERATED_DOCS[doc_id] = result_obj
    return result_obj