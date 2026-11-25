from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel


class RepoAnalyzeRequest(BaseModel):
    github_id: str
    repo_name: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    include_types: Optional[List[str]] = None


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


class SingleChunkCompressRequest(BaseModel):
    chunk: Dict[str, Any]
    chunk_index: int = 1
    template_sections: List[Dict[str, Any]] = []
    sub_budget_tokens: int = 400
    apply_template_in_sub: bool = True
    custom_sub_base_prompt: Optional[str] = None


class SynthesizeChunksRequest(BaseModel):
    cleaned_data: List[Dict[str, Any]]
    compressed_chunks: List[Dict[str, Any]]
    template_sections: List[Dict[str, Any]]
    language: str = "ko"
    custom_final_base_prompt: Optional[str] = None