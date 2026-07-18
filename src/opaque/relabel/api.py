"""Relabeling HTTP API (spec §9). Single-user, local (§9.3)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from ..versioning.git import GitError
from .session import RelabelError

router = APIRouter(prefix='/api')


class GoldUpdate(BaseModel):
    gold: Any = None


class CommitBody(BaseModel):
    annotation: str | None = None


def _session(request: Request):
    return request.app.state.session


@router.get('/info')
def info(request: Request) -> dict:
    state = request.app.state
    session = state.session
    return {
        'project': state.project,
        'tool': state.tool.name,
        'task_type': state.tool.task_type,
        'eval_path': str(session.eval_path),
        'pending': session.pending,
        'sample_count': len(session.samples()),
        'predictions_available': bool(state.predictions),
    }


@router.get('/samples')
def samples(request: Request) -> dict:
    state = request.app.state
    return {
        'samples': state.session.samples(state.predictions),
        'predictions_available': bool(state.predictions),
    }


@router.patch('/samples/{sample_id}')
def update_sample(sample_id: str, body: GoldUpdate, request: Request) -> dict:
    state = request.app.state
    try:
        state.session.set_gold(sample_id, body.gold)
    except RelabelError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {
        'sample_id': sample_id,
        'gold': state.session.current_gold(sample_id),
        'edited': sample_id in state.session.pending,
        'pending': state.session.pending,
    }


@router.post('/session/commit')
def commit(body: CommitBody, request: Request) -> dict:
    try:
        return _session(request).commit(annotation=body.annotation)
    except (GitError, RelabelError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
