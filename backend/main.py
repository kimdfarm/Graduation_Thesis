import ast
import datetime
from datetime import timezone
import json
import os
import random
import re
import time as py_time
from typing import Any, Dict, List, Optional, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
import httpx
from pydantic import BaseModel
import requests
import tiktoken
# tiktoken 토크나이저 초기화 (o200k_base 기반, 실패 시 문자 길이 폴백)
try:
    TOKENIZER = tiktoken.get_encoding("o200k_base")
except Exception:
    TOKENIZER = None


def count_tokens(obj: Any) -> int:
    text = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
    if not text:
        return 0
    if TOKENIZER is not None:
        return len(TOKENIZER.encode(text))
    return max(1, len(text) // 2)

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


ALL_EXPLORE_TYPES = ["commits", "prs", "issues", "languages", "repo_meta"]


class RepoAnalyzeRequest(BaseModel):
    github_id: str
    repo_name: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    include_types: Optional[List[str]] = None  # 예: ["commits"], ["commits", "prs"] 등


def build_selective_graphql_query(include_types: List[str]) -> str:
    inc = set(include_types or ALL_EXPLORE_TYPES)

    meta_fields = """
    name
    description
    isFork
    stargazerCount
    forkCount
    """ if "repo_meta" in inc else """
    name
    isFork
    """

    lang_block = """
    languages(first: 10, orderBy: {field: SIZE, direction: DESC}) {
      totalSize
      edges {
        size
        node { name }
      }
    }
    """ if "languages" in inc else ""

    commit_block = """
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
    """ if "commits" in inc else ""

    issue_block = """
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
    """ if "issues" in inc else ""

    pr_block = """
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
    """ if "prs" in inc else ""

    return f"""
query GetSelectiveRepoAnalytics($owner: String!, $name: String!, $afterCursor: String) {{
  repository(owner: $owner, name: $name) {{
    {meta_fields}
    {lang_block}
    {commit_block}
    {issue_block}
    {pr_block}
  }}
}}
"""
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
    include_types: Optional[List[str]] = None,
) -> dict:
    inc = set(include_types or raw_graphql_data.get("include_types") or ALL_EXPLORE_TYPES)
    effective_author = (target_author or raw_graphql_data.get("target_author") or "").strip()
    repo_name = (
        raw_graphql_data.get("name")
        or raw_graphql_data.get("repo_name")
        or raw_graphql_data.get("full_name")
        or default_repo_name
        or "프로젝트"
    )

    tech_stacks: List[str] = []
    if "languages" in inc:
        languages_data = raw_graphql_data.get("languages") or {}
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

    clean_commits: List[str] = []
    if "commits" in inc:
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

    clean_prs: List[str] = []
    if "prs" in inc:
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
    if "issues" in inc:
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

    result = {
        "repo_name": repo_name,
        "target_author": effective_author or "ALL",
        "included_types": list(inc),
        "tech_stacks": tech_stacks,
        "key_commits": clean_commits,
        "key_prs_and_issues": clean_prs + clean_issues,
    }
    if "repo_meta" in inc:
        result["repo_meta"] = {
            "description": raw_graphql_data.get("description"),
            "isFork": raw_graphql_data.get("isFork", False),
            "stargazerCount": raw_graphql_data.get("stargazerCount", 0),
            "forkCount": raw_graphql_data.get("forkCount", 0),
        }
    return result


def estimate_tokens(obj: Any) -> int:
    text = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
    return max(1, len(text) // 2)


def split_repo_by_properties(repo_data: dict, max_tokens: int = 1200) -> list:
    chunks = []
    base_meta = {
        "repo_name": repo_data.get("repo_name"),
        "target_author": repo_data.get("target_author"),
        "tech_stacks": repo_data.get("tech_stacks", []),
        "key_commits": [],
        "key_prs_and_issues": [],
    }
    current_chunk = dict(base_meta)
    current_chunk["key_commits"] = []
    current_chunk["key_prs_and_issues"] = []

    for commit in repo_data.get("key_commits", []):
        candidate = {
            **current_chunk,
            "key_commits": current_chunk["key_commits"] + [commit],
        }
        has_existing_items = bool(current_chunk["key_commits"] or current_chunk["key_prs_and_issues"])
        if count_tokens(candidate) > max_tokens and has_existing_items:
            # 초과 직전 데이터까지만 하나의 청크로 확정
            chunks.append(current_chunk)
            # 초과를 유발한 현재 항목부터 새 청크 시작
            current_chunk = {
                "repo_name": f"{repo_data.get('repo_name')} (part {len(chunks) + 1})",
                "target_author": repo_data.get("target_author"),
                "tech_stacks": [],
                "key_commits": [commit],
                "key_prs_and_issues": [],
            }
        else:
            current_chunk["key_commits"].append(commit)

    for item in repo_data.get("key_prs_and_issues", []):
        candidate = {
            **current_chunk,
            "key_prs_and_issues": current_chunk["key_prs_and_issues"] + [item],
        }
        has_existing_items = bool(current_chunk["key_commits"] or current_chunk["key_prs_and_issues"])
        if count_tokens(candidate) > max_tokens and has_existing_items:
            chunks.append(current_chunk)
            current_chunk = {
                "repo_name": f"{repo_data.get('repo_name')} (part {len(chunks) + 1})",
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
        meta["processed_tokens"] = sum(count_tokens(c) for c in all_chunks)
        return all_chunks, meta, compression_usage

    chunk_sizes = [count_tokens(c) for c in all_chunks]
    total_chunk_tokens = sum(chunk_sizes) or 1
    template_str = json.dumps(template_sections, ensure_ascii=False)
    compressed_projects = []

    system_prompt = f"""You are a strict technical data compression engine for developer resume generation.
Align the given GitHub activity chunk with the target template sections:
{template_str}

Rules:
1. ZERO FABRICATION: Do not invent facts, metrics, or tools absent from the chunk.
2. Preserve technical stack, concrete implementation/bugfix/PR/issue details, and author identity.
3. Drop trivial noise (e.g., typo/formatting) unless tied to meaningful engineering work.
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
            add_usage(compression_usage, extract_usage_metrics(completion))
            parsed = safe_json_parse(completion.choices[0].message.content or "")
            compressed_projects.append(parsed)
        except Exception:
            compressed_projects.append(chunk)

    meta["processed_tokens"] = sum(count_tokens(p) for p in compressed_projects)
    return compressed_projects, meta, compression_usage


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

GITHUB_ENV_TOKEN = os.getenv("GITHUB_TOKEN", "").strip()


def resolve_github_token(authorization: Optional[str]) -> str:
    if authorization and authorization.startswith("Bearer "):
        header_token = authorization.split(" ", 1)[1].strip()
        if header_token:
            return header_token
    return GITHUB_ENV_TOKEN


@app.get("/api/config/status")
def get_config_status():
    return {
        "has_env_github_token": bool(GITHUB_ENV_TOKEN),
        "has_groq_api_key": bool(GROQ_API_KEY),
    }

@app.post("/api/github/analyze")
async def analyze_repository_graphql(payload: RepoAnalyzeRequest, authorization: Optional[str] = Header(None)):
    token = resolve_github_token(authorization)
    if not token:
        raise HTTPException(
            status_code=401,
            detail="GitHub Access Token이 필요합니다. (.env의 GITHUB_TOKEN 또는 화면 입력)",
        )

    selected_types = [t for t in (payload.include_types or ALL_EXPLORE_TYPES) if t in ALL_EXPLORE_TYPES]
    if not selected_types:
        selected_types = ["commits"]

    query_str = build_selective_graphql_query(selected_types)

    if "/" in payload.repo_name:
        owner, repo_name = payload.repo_name.split("/", 1)
    else:
        owner, repo_name = payload.github_id, payload.repo_name

    all_commits = []
    after_cursor = None
    base_repo_data: Dict[str, Any] = {}

    async with httpx.AsyncClient(timeout=120.0) as client:
        max_pages = 3 if "commits" in selected_types else 1
        for _ in range(max_pages):
            resp = await client.post(
                "https://api.github.com/graphql",
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                json={
                    "query": query_str,
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

            if "commits" in selected_types:
                history = (((repo_analytics.get("defaultBranchRef") or {}).get("target") or {}).get("history")) or {}
                all_commits.extend(history.get("nodes") or [])
                page_info = history.get("pageInfo") or {}
                if not page_info.get("hasNextPage"):
                    break
                after_cursor = page_info.get("endCursor")
            else:
                break

    if not base_repo_data:
        raise HTTPException(status_code=404, detail="저장소 데이터를 가져올 수 없습니다.")

    if "commits" in selected_types and base_repo_data.get("defaultBranchRef"):
        target_obj = base_repo_data["defaultBranchRef"].get("target")
        if isinstance(target_obj, dict) and isinstance(target_obj.get("history"), dict):
            target_obj["history"]["nodes"] = all_commits

    base_repo_data["repo_name"] = base_repo_data.get("name") or repo_name
    base_repo_data["full_name"] = f"{owner}/{repo_name}"
    base_repo_data["target_author"] = payload.github_id
    base_repo_data["include_types"] = selected_types
    return base_repo_data

STRICT_ANALYSIS_SYSTEM_PROMPT = """You are a strict technical evidence extractor and developer resume structuring engine.
Your job is to transform GitHub repository activity (commits, PRs, issues, languages) into structured JSON for engineering evaluation.

STRICT ANTI-HALLUCINATION & RIGOR RULES:
1. ZERO FABRICATION: Every technical claim, architecture decision, library/framework name, metric, or outcome MUST be directly grounded in the provided input data. If a detail is not present in the input, DO NOT invent or guess it.
2. HIGH-SIGNAL EXTRACTION ONLY: Prioritize concrete engineering actions (API/DB/auth/pipeline design, bug fixes, performance/concurrency/testing work, deployment/CI/CD, key PRs/issues). Strip out meaningless noise like "fix typo", "update readme", or empty labels unless tied to a concrete technical change.
3. EVIDENCE TRACEABILITY: Whenever possible, reference actual commit headlines, PR/issue titles, or line/repetition counts from the input data inside the descriptions so claims are verifiable.
4. SECTION & COLUMN CONSISTENCY: Every section must define a unified `ai_columns` array, and every detail card in that section must populate `column_values` matching those exact column names without omitting columns.
5. OUTPUT FORMAT: Respond ONLY with a valid JSON object matching the requested schema. No markdown fences outside JSON, no commentary.
"""


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


# 2) ChunkPreviewRequest 및 헬퍼 교체 (max_chunks 제거)
class ChunkPreviewRequest(BaseModel):
    analysis_data: Dict[str, Any]
    target_author: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    max_total_tokens: int = 5000
    chunk_limit_mode: Literal["max_chunks", "chunk_tokens"] = "max_chunks"
    max_chunks: Optional[int] = 6
    chunk_token_limit: Optional[int] = 1200
    sub_llm_max_tokens: Optional[int] = None
    sub_llm_auto: bool = True
    sub_llm_allocation_mode: Literal["equal", "proportional"] = "proportional"
    force_chunking: bool = True
    template_columns: Optional[List[str]] = None
    template_structure: Optional[Dict[str, Any]] = None
    language: str = "ko"
    run_generation: bool = False


DEFAULT_SUB_BASE_PROMPT = """You are a strict technical data compression engine for developer activity analysis.
STRICT RULES:
1. ZERO FABRICATION: Every fact, commit message, PR/issue title, line count, repetition count, or tech stack MUST come strictly from the input chunk.
2. DO NOT MERGE DIFFERENT REPOSITORIES: Keep the chunk's `repo_name` and `target_author` intact.
3. PRESERVE HIGH-SIGNAL EVIDENCE: Retain concrete technical actions, tech stacks (`tech_stacks`), key commits (`key_commits`), and PRs/issues (`key_prs_and_issues`).
4. DROP TRIVIAL NOISE: Remove meaningless commits/PRs/issues (e.g., typo, formatting) unless tied to concrete technical changes."""


class SingleChunkCompressRequest(BaseModel):
    chunk: Dict[str, Any]
    chunk_index: int = 1
    template_sections: List[Dict[str, Any]] = []
    sub_budget_tokens: int = 400
    apply_template_in_sub: bool = True
    custom_sub_base_prompt: Optional[str] = None


def _calc_chunk_item_retention(raw_chunk: dict, compressed_chunk: dict) -> Dict[str, Any]:
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


DEFAULT_FINAL_BASE_PROMPT = """You are a strict technical evidence extractor and developer resume structuring engine.
Your job is to transform GitHub repository activity (commits, PRs, issues, languages) into structured JSON for engineering evaluation.

STRICT ANTI-HALLUCINATION & RIGOR RULES:
1. ZERO FABRICATION: Every technical claim, architecture decision, library/framework name, metric, or outcome MUST be directly grounded in the provided input data. If a detail is not present in the input, DO NOT invent or guess it.
2. HIGH-SIGNAL EXTRACTION ONLY: Prioritize concrete engineering actions (API/DB/auth/pipeline design, bug fixes, performance/concurrency/testing work, deployment/CI/CD, key PRs/issues). Strip out meaningless noise like "fix typo", "update readme", or empty labels unless tied to a concrete technical change.
3. EVIDENCE TRACEABILITY: Whenever possible, reference actual commit headlines, PR/issue titles, or line/repetition counts from the input data inside the descriptions so claims are verifiable.
4. SECTION & COLUMN CONSISTENCY: Every section must define a unified `ai_columns` array, and every detail card in that section must populate `column_values` matching those exact column names without omitting columns.
5. OUTPUT FORMAT: Respond ONLY with a valid JSON object matching the requested schema. No markdown fences outside JSON, no commentary."""


class SynthesizeChunksRequest(BaseModel):
    cleaned_data: List[Dict[str, Any]]
    compressed_chunks: List[Dict[str, Any]]
    template_sections: List[Dict[str, Any]]
    language: str = "ko"
    custom_final_base_prompt: Optional[str] = None

def compute_sub_llm_budgets(
    chunk_tokens_list: List[int],
    max_total_tokens: int,
    final_sys_prompt: str,
    allocation_mode: str = "proportional",
    sub_llm_auto: bool = True,
    custom_sub_tokens: Optional[int] = None,
) -> List[int]:
    n = max(1, len(chunk_tokens_list))
    final_overhead = count_tokens(final_sys_prompt) + 300
    available = max(200, int(max_total_tokens) - final_overhead)
    total_pool = int(available * 0.9)

    if not sub_llm_auto and custom_sub_tokens and custom_sub_tokens > 0:
        if allocation_mode == "equal":
            return [int(custom_sub_tokens)] * n
        # 비례 배분 + 커스텀 입력 시: custom_sub_tokens * n 을 총 풀로 보고 청크 비중대로 배분
        total_pool = max(200, int(custom_sub_tokens) * n)

    if allocation_mode == "equal":
        per_chunk = max(100, total_pool // n)
        return [per_chunk] * n

    sum_tok = sum(chunk_tokens_list) or 1
    return [max(100, int(total_pool * (tok / sum_tok))) for tok in chunk_tokens_list]


@app.post("/api/github/compress-chunk")
def compress_single_chunk_endpoint(req: SingleChunkCompressRequest):
    """[3-3 실시간 진행용] 청크 1개씩 서브 LLM으로 압축하고 보존율·소요시간·토큰 사용량 반환"""
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
    retention_info = _calc_chunk_item_retention(req.chunk, compressed_obj)
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
    """[3-3 최종 종합 단계] 모든 서브 LLM 압축 결과를 종합 프롬프트로 구조화 생성"""
    sys_prompt, final_user_prompt = build_final_prompt(
        req.template_sections,
        req.compressed_chunks,
        language=req.language,
        custom_base_prompt=req.custom_final_base_prompt,
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

    all_raw_items = []
    for proj in req.cleaned_data:
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
        "combined_prompt": combined_raw_prompt,
        "generated_sections": generated_sections,
        "token_usage": usage_delta,
        "model_used": model_used,
        "comparison": {
            "total_raw_items": len(all_raw_items),
            "directly_matched_items": len(matched_items),
            "unmatched_or_synthesized_items": len(missing_items),
            "direct_evidence_mention_pct": mention_rate,
            "unmatched_sample": missing_items[:20],
        },
    }

def _resolve_sub_llm_budget(req: ChunkPreviewRequest, chunk_count: int, final_sys_prompt: str) -> int:
    final_overhead = count_tokens(final_sys_prompt) + 300
    available = max(200, int(req.max_total_tokens) - final_overhead)
    auto_budget = max(100, int((available * 0.9) // max(1, chunk_count)))
    if not req.sub_llm_auto and req.sub_llm_max_tokens and req.sub_llm_max_tokens > 0:
        return int(req.sub_llm_max_tokens)
    return auto_budget

def split_repo_into_max_chunks(repo_data: dict, max_chunks: int) -> list:
    """저장소의 커밋/PR/이슈 항목을 최대 max_chunks 개수로 나누어 분할"""
    safe_max = max(1, int(max_chunks))
    commits = list(repo_data.get("key_commits", []))
    prs_issues = list(repo_data.get("key_prs_and_issues", []))
    all_items = [("commit", c) for c in commits] + [("pr_issue", p) for p in prs_issues]

    if not all_items:
        return [
            {
                "repo_name": repo_data.get("repo_name"),
                "target_author": repo_data.get("target_author"),
                "tech_stacks": repo_data.get("tech_stacks", []),
                "key_commits": [],
                "key_prs_and_issues": [],
            }
        ]

    bucket_count = min(safe_max, len(all_items))
    buckets: List[List[tuple[str, str]]] = [[] for _ in range(bucket_count)]
    for idx, item in enumerate(all_items):
        buckets[idx % bucket_count].append(item)

    result_chunks = []
    for b_idx, bucket in enumerate(buckets):
        b_commits = [val for kind, val in bucket if kind == "commit"]
        b_prs_issues = [val for kind, val in bucket if kind == "pr_issue"]
        suffix = "" if bucket_count == 1 else f" (chunk {b_idx + 1}/{bucket_count})"
        result_chunks.append(
            {
                "repo_name": f"{repo_data.get('repo_name')}{suffix}",
                "target_author": repo_data.get("target_author"),
                "tech_stacks": repo_data.get("tech_stacks", []) if b_idx == 0 else [],
                "key_commits": b_commits,
                "key_prs_and_issues": b_prs_issues,
            }
        )
    return result_chunks


def _build_chunk_preview_data(req: ChunkPreviewRequest) -> Dict[str, Any]:
    """AI 호출 없이 순수 청크 분할 및 토큰/한도 계산만 수행하는 공통 함수 (3-2 전용)"""
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

    template_sections = _resolve_template_sections(req)
    sys_prompt_preview, _ = build_final_prompt(template_sections, [], language=req.language)
    sub_llm_budget = _resolve_sub_llm_budget(req, len(chunks), sys_prompt_preview)

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

        sub_prompt_text = build_sub_prompt(ch, template_sections, sub_llm_budget)
        sub_prompt_tokens = count_tokens(sub_prompt_text)
        fixed_sub_instruction = (
            "[Fixed Sub-LLM Instruction]\n"
            "Compress ONLY the given GitHub chunk into JSON aligned with target template sections. "
            "Zero fabrication, preserve concrete commits/PRs/issues/tech_stacks, drop trivial noise."
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

    sys_prompt, final_user_prompt = build_final_prompt(template_sections, chunks, language=req.language)
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


@app.post("/api/github/pipeline-inspect")
def run_pipeline_inspect(req: ChunkPreviewRequest):
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

    template_sections = _resolve_template_sections(req)

    sys_prompt_preview, _ = build_final_prompt(template_sections, [], language=req.language)
    sub_llm_budget = _resolve_sub_llm_budget(req, len(chunks), sys_prompt_preview)

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
                "토큰수": c_tokens,
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
                "토큰수": c_tokens,
                "토큰한도": token_limit,
                "한도초과여부": "초과" if is_exceeded else "정상",
                "초과사유": ", ".join(reasons) if reasons else "-",
            }

        sub_prompt_text = build_sub_prompt(ch, template_sections, sub_llm_budget)
        chunk_evaluations.append(eval_row)
        sub_prompts.append({
            "chunk_index": idx + 1,
            "repo_name": ch.get("repo_name"),
            "tokens": c_tokens,
            "exceeded": is_exceeded,
            "sub_prompt": sub_prompt_text,
            "raw_chunk_data": ch,
        })

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

    sys_prompt, final_user_prompt = build_final_prompt(
        template_sections, compressed_chunks_for_final, language=req.language
    )
    combined_raw_prompt = f"[SYSTEM PROMPT]\n{sys_prompt}\n\n[USER PROMPT]\n{final_user_prompt}"

    if req.run_generation and groq_client:
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
                u = extract_usage_metrics(comp)
                add_usage(usage_summary, u)
                parsed = safe_json_parse(comp.choices[0].message.content or "{}")
                generated_sections = normalize_github_resume_data(parsed).get("sections", [])
                if generated_sections:
                    break
            except Exception:
                if attempt < 2:
                    py_time.sleep((2**attempt) + random.uniform(0.3, 0.8))

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
        },
        "chunk_evaluations": chunk_evaluations,
        "sub_prompts": sub_prompts,
        "generated_sections": generated_sections,
        "token_usage": usage_summary,
        "model_used": model_used,
        "comparison": {
            "total_raw_items": len(all_raw_items),
            "directly_matched_items": len(matched_items),
            "unmatched_or_synthesized_items": len(missing_items),
            "direct_evidence_mention_pct": mention_rate,
            "unmatched_sample": missing_items[:20],
        },
    }


@app.post("/api/github/chunk-preview")
def run_chunk_preview(req: ChunkPreviewRequest):
    """[3-2 전용] AI를 전혀 호출하지 않고 순수 청크 분할 및 한도 판정만 즉시 반환"""
    data = _build_chunk_preview_data(req)
    data.pop("template_sections", None)
    return data




def _resolve_template_sections(req: ChunkPreviewRequest) -> List[Dict[str, Any]]:
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
        f"Target Template Sections: {json.dumps(template_sections, ensure_ascii=False)}\n"
        f"Extract & map evidence specifically for columns: {json.dumps(target_cols, ensure_ascii=False)}\n"
        f"Chunk Payload:\n{json.dumps(chunk, ensure_ascii=False, indent=2)}"
    )

def build_final_prompt(
    template_sections: list,
    processed_data: list,
    language: str = "ko",
    custom_base_prompt: Optional[str] = None,
) -> tuple[str, str]:
    lang_rule = "모든 설명은 자연스럽고 정확한 한국어로 작성하세요." if language == "ko" else "Write all descriptions in English."
    primary_sec = template_sections[0] if template_sections else {}
    example_cols = primary_sec.get("columns") or ["역할 및 구현", "기술 스택 및 근거"]
    example_col_values = {col: f"{col} 관점의 근거 기반 상세 내용" for col in example_cols}
    base_instruction = (custom_base_prompt or DEFAULT_FINAL_BASE_PROMPT).strip()

    sys_prompt = f"""{base_instruction}
{lang_rule}
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
          "title": "항목 제목",
          "column_values": {json.dumps(example_col_values, ensure_ascii=False)}
        }}
      ]
    }}
  ]
}}"""
    usr_prompt = (
        f"[템플릿 구조]\n{json.dumps(template_sections, ensure_ascii=False, indent=2)}\n\n"
        f"[청크 처리/종합 입력 데이터]\n{json.dumps(processed_data, ensure_ascii=False, indent=2)}"
    )
    return sys_prompt, usr_prompt



def _resolve_sub_llm_budget(req: ChunkPreviewRequest, chunk_count: int, final_sys_prompt: str) -> int:
    final_overhead = count_tokens(final_sys_prompt) + 300
    available = max(200, int(req.max_total_tokens) - final_overhead)
    auto_budget = max(100, int((available * 0.9) // max(1, chunk_count)))
    if not req.sub_llm_auto and req.sub_llm_max_tokens and req.sub_llm_max_tokens > 0:
        return int(req.sub_llm_max_tokens)
    return auto_budget

def compress_chunk_with_sub_llm_strict(
    chunk: dict,
    template_sections: list,
    sub_budget_tokens: int,
    usage_summary: Dict[str, int],
    apply_template_in_sub: bool = True,
    custom_sub_base_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    if not groq_client:
        return chunk

    base_instruction = (custom_sub_base_prompt or DEFAULT_SUB_BASE_PROMPT).strip()
    chunk_json = json.dumps(chunk, ensure_ascii=False)

    if apply_template_in_sub and template_sections:
        template_info_str = json.dumps(template_sections, ensure_ascii=False)
        sub_system_prompt = f"""{base_instruction}

[Target Template Structure & Columns — ALIGN EXTRACTION TO THESE]
{template_info_str}

5. STRICT JSON SCHEMA: Respond ONLY with valid JSON using this exact top-level structure:
{{
  "repo_name": "{chunk.get('repo_name', '')}",
  "target_author": "{chunk.get('target_author', '')}",
  "tech_stacks": [],
  "key_commits": [],
  "key_prs_and_issues": [],
  "column_evidence": {{}}
}}"""
    else:
        sub_system_prompt = f"""{base_instruction}

[Mode: Template-Agnostic Pure Fact Compression]
Do NOT filter or drop technical items for template alignment; preserve maximum factual coverage of commits, PRs/issues, and tech stacks within the token budget.

5. STRICT JSON SCHEMA: Respond ONLY with valid JSON using this exact top-level structure:
{{
  "repo_name": "{chunk.get('repo_name', '')}",
  "target_author": "{chunk.get('target_author', '')}",
  "tech_stacks": [],
  "key_commits": [],
  "key_prs_and_issues": []
}}"""

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
                max_completion_tokens=sub_budget_tokens,
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
                    "tech_stacks": parsed.get("tech_stacks") if isinstance(parsed.get("tech_stacks"), list) else chunk.get("tech_stacks", []),
                    "key_commits": parsed.get("key_commits") if isinstance(parsed.get("key_commits"), list) else chunk.get("key_commits", []),
                    "key_prs_and_issues": parsed.get("key_prs_and_issues") if isinstance(parsed.get("key_prs_and_issues"), list) else chunk.get("key_prs_and_issues", []),
                }
                if apply_template_in_sub:
                    res_obj["column_evidence"] = parsed.get("column_evidence") if isinstance(parsed.get("column_evidence"), dict) else {}
                return res_obj
        except Exception:
            if attempt < 2:
                py_time.sleep((2 ** attempt) + random.uniform(0.3, 0.8))

    return chunk





