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

class PipelineInspectRequest(BaseModel):
    analysis_data: Dict[str, Any]
    target_author: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    max_total_tokens: int = 5000
    max_chunks: int = 6
    chunk_token_limit: int = 1200
    chunk_item_limit: int = 40
    force_chunking: bool = True
    template_columns: Optional[List[str]] = None
    language: str = "ko"
    run_generation: bool = True


def build_sub_prompt(chunk: dict, template_sections: list, budget_tokens: int) -> str:
    return (
        f"[Sub-Prompt | Token Budget: {budget_tokens}]\n"
        f"Target Template: {json.dumps(template_sections, ensure_ascii=False)}\n"
        f"Chunk Payload:\n{json.dumps(chunk, ensure_ascii=False, indent=2)}"
    )


def build_final_prompt(template_sections: list, processed_data: list, language: str = "ko") -> tuple[str, str]:
    lang_rule = "모든 설명은 자연스럽고 정확한 한국어로 작성하세요." if language == "ko" else "Write all descriptions in English."
    sys_prompt = f"""{STRICT_ANALYSIS_SYSTEM_PROMPT}
{lang_rule}
반드시 아래 JSON 스키마로만 응답하세요:
{{
  "doc_type": "RESUME",
  "sections": [
    {{
      "section_type": "PROJECTS",
      "section_title": "섹션 제목",
      "display_order": 1,
      "ai_columns": ["역할 및 구현", "기술 스택 및 근거"],
      "details": [
        {{
          "id": "card_1",
          "title": "항목 제목",
          "column_values": {{
            "역할 및 구현": "근거 기반 상세 설명",
            "기술 스택 및 근거": "관련 커밋/PR/언어 근거"
          }}
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


@app.post("/api/github/pipeline-inspect")
def run_pipeline_inspect(req: PipelineInspectRequest):
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

    # 표 형태로 바로 변환하기 쉬운 row 목록 생성
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

    # 청크 분할
    chunks: List[dict] = []
    for proj in cleaned_projects:
        if req.force_chunking or count_tokens(proj) > req.chunk_token_limit:
            chunks.extend(split_repo_by_properties(proj, max_tokens=req.chunk_token_limit))
        else:
            chunks.append(proj)

    exceeded_chunk_count_limit = len(chunks) > req.max_chunks

    template_sections = [
        {
            "section_type": "PROJECTS",
            "section_title": "사용자 기여 분석 및 구현 성과",
            "columns": req.template_columns or ["구현 및 문제 해결", "기술 스택 및 활동 근거"],
        }
    ]

    # 각 청크별 한도 초과 여부 판정 및 서브 프롬프트 구성
    total_chunk_tokens = sum(count_tokens(c) for c in chunks) or 1
    chunk_evaluations: List[Dict[str, Any]] = []
    sub_prompts: List[Dict[str, Any]] = []

    for idx, ch in enumerate(chunks):
        c_tokens = count_tokens(ch)
        c_commits = len(ch.get("key_commits", []))
        c_prs_issues = len(ch.get("key_prs_and_issues", []))
        c_items = c_commits + c_prs_issues
        c_index_over = (idx + 1) > req.max_chunks

        over_tokens = c_tokens > req.chunk_token_limit
        over_items = c_items > req.chunk_item_limit
        is_exceeded = over_tokens or over_items or c_index_over

        reasons = []
        if over_tokens:
            reasons.append(f"토큰 초과({c_tokens} > {req.chunk_token_limit})")
        if over_items:
            reasons.append(f"항목 수 초과({c_items} > {req.chunk_item_limit})")
        if c_index_over:
            reasons.append(f"제한 청크 수 초과({idx + 1}번째 > {req.max_chunks}개)")

        budget = max(350, int(req.max_total_tokens * (c_tokens / total_chunk_tokens)))
        sub_prompt_text = build_sub_prompt(ch, template_sections, budget)

        chunk_evaluations.append({
            "청크번호": idx + 1,
            "저장소/구분": ch.get("repo_name"),
            "커밋수": c_commits,
            "PR/이슈수": c_prs_issues,
            "총항목수": c_items,
            "토큰수": c_tokens,
            "토큰한도": req.chunk_token_limit,
            "항목한도": req.chunk_item_limit,
            "한도초과여부": "초과" if is_exceeded else "정상",
            "초과사유": ", ".join(reasons) if reasons else "-",
        })
        sub_prompts.append({
            "chunk_index": idx + 1,
            "repo_name": ch.get("repo_name"),
            "tokens": c_tokens,
            "exceeded": is_exceeded,
            "sub_prompt": sub_prompt_text,
            "raw_chunk_data": ch,
        })

    sys_prompt, final_user_prompt = build_final_prompt(template_sections, chunks, language=req.language)
    combined_raw_prompt = f"[SYSTEM PROMPT]\n{sys_prompt}\n\n[USER PROMPT]\n{final_user_prompt}"

    generated_sections: List[Dict[str, Any]] = []
    usage_summary = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    model_used = PRIMARY_MODEL_NAME

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

    # 원본 항목 대비 해석 결과 텍스트 내 언급률 계산
    all_raw_items = []
    for proj in cleaned_projects:
        all_raw_items.extend(proj.get("key_commits", []))
        all_raw_items.extend(proj.get("key_prs_and_issues", []))

    generated_blob = json.dumps(generated_sections, ensure_ascii=False)
    matched_items = []
    missing_items = []
    for raw_line in all_raw_items:
        # "- 메시지 (반복 ...)" 형태에서 핵심 문구 추출
        core = raw_line.lstrip("- ").split(" (반복")[0].split(" (상태:")[0].strip()
        if core and core in generated_blob:
            matched_items.append(raw_line)
        else:
            missing_items.append(raw_line)

    mention_rate = round((len(matched_items) / len(all_raw_items)) * 100, 2) if all_raw_items else 100.0

    return {
        "tokenizer": "tiktoken (o200k_base)",
        "cleaned_data": cleaned_projects,
        "raw_table_rows": raw_rows,
        "combined_prompt": combined_raw_prompt,
        "chunk_limits": {
            "max_total_tokens": req.max_total_tokens,
            "max_chunks": req.max_chunks,
            "chunk_token_limit": req.chunk_token_limit,
            "chunk_item_limit": req.chunk_item_limit,
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