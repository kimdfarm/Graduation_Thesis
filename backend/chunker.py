import datetime
from datetime import timezone
import json
import re
from typing import Any, Dict, List, Optional

import tiktoken

ALL_EXPLORE_TYPES = ["commits", "prs", "issues", "languages", "repo_meta"]

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


def parse_iso_dt(date_str: Optional[str]) -> Optional[datetime.datetime]:
    if not date_str:
        return None
    try:
        dt = datetime.datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


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
            chunks.append(current_chunk)
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


def split_repo_into_max_chunks(repo_data: dict, max_chunks: int) -> list:
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


def extract_base_repo_name(name: str) -> str:
    cleaned = re.sub(
        r"\s*\((?:part|chunk|continuing)\s*\d*(?:\s*/\s*\d+)?\)\s*$",
        "",
        str(name or ""),
        flags=re.IGNORECASE,
    )
    return cleaned.strip() or "프로젝트"


def merge_compressed_chunks_by_repo(compressed_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}
    for ch in compressed_chunks or []:
        if not isinstance(ch, dict):
            continue
        base_repo = extract_base_repo_name(ch.get("repo_name") or "")
        if base_repo not in grouped:
            grouped[base_repo] = {
                "repo_name": base_repo,
                "target_author": ch.get("target_author") or "",
                "tech_stacks": [],
                "key_commits": [],
                "key_prs_and_issues": [],
                "column_evidence": {},
            }
        target = grouped[base_repo]
        if not target["target_author"] and ch.get("target_author"):
            target["target_author"] = ch.get("target_author")

        for tech in ch.get("tech_stacks") or []:
            if tech and tech not in target["tech_stacks"]:
                target["tech_stacks"].append(tech)

        for c in ch.get("key_commits") or []:
            if c and c not in target["key_commits"]:
                target["key_commits"].append(c)

        for p in ch.get("key_prs_and_issues") or []:
            if p and p not in target["key_prs_and_issues"]:
                target["key_prs_and_issues"].append(p)

        col_ev = ch.get("column_evidence")
        if isinstance(col_ev, dict):
            for col_name, ev_list in col_ev.items():
                existing = target["column_evidence"].setdefault(col_name, [])
                items = ev_list if isinstance(ev_list, list) else [ev_list]
                for item in items:
                    if item and item not in existing:
                        existing.append(item)

    merged_list = []
    for entry in grouped.values():
        if not entry["column_evidence"]:
            entry.pop("column_evidence", None)
        merged_list.append(entry)
    return merged_list