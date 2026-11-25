import datetime
from datetime import timezone
import os
from typing import Any, Dict, List, Optional

from fastapi import HTTPException
import httpx
import requests

from backend.chunker import ALL_EXPLORE_TYPES, parse_iso_dt

GITHUB_ENV_TOKEN = os.getenv("GITHUB_TOKEN", "").strip()


def resolve_github_token(authorization: Optional[str]) -> str:
    if authorization and authorization.startswith("Bearer "):
        header_token = authorization.split(" ", 1)[1].strip()
        if header_token:
            return header_token
    return GITHUB_ENV_TOKEN


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


def fetch_filtered_repositories(
    github_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    authorization: Optional[str] = None,
) -> List[Dict[str, Any]]:
    if not start_date or not end_date:
        return []

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

    return list(unique_repos.values())


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


async def fetch_repository_analytics(
    github_id: str,
    repo_name_input: str,
    include_types: Optional[List[str]] = None,
    authorization: Optional[str] = None,
) -> Dict[str, Any]:
    token = resolve_github_token(authorization)
    if not token:
        raise HTTPException(
            status_code=401,
            detail="GitHub Access Token이 필요합니다. (.env의 GITHUB_TOKEN 또는 화면 입력)",
        )

    selected_types = [t for t in (include_types or ALL_EXPLORE_TYPES) if t in ALL_EXPLORE_TYPES]
    if not selected_types:
        selected_types = ["commits"]

    query_str = build_selective_graphql_query(selected_types)

    if "/" in repo_name_input:
        owner, repo_name = repo_name_input.split("/", 1)
    else:
        owner, repo_name = github_id, repo_name_input

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
    base_repo_data["target_author"] = github_id
    base_repo_data["include_types"] = selected_types
    return base_repo_data