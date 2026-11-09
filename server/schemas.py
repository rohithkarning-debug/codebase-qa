"""Request / response models for the web API."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    repo: str
    question: str = Field(min_length=3, max_length=2000)
    mode: Literal["agent", "rag"] = "agent"
    chunker: Literal["ast", "fixed"] = "ast"
    retriever: Literal["code", "general", "bm25", "hybrid"] = "code"
    provider: Optional[str] = None


class AddRepoRequest(BaseModel):
    url: str = Field(min_length=10, max_length=300)
    ref: str = ""
    name: str = ""


class JobStatus(BaseModel):
    id: str
    repo: str
    status: Literal["queued", "running", "done", "error"]
    stage: str = ""
    done: int = 0
    total: int = 0
    message: str = ""
