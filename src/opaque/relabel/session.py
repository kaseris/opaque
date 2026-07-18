"""Relabeling session state (spec §9.2).

Edits accumulate in an uncommitted in-memory working state as the user reviews samples —
nothing is committed per keystroke or per field. Saving the session applies every changed
gold value to the in-repo eval files and creates a *single* git commit (§3.2) with an
auto-generated message the user may annotate.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from ..config.models import ToolConfig
from ..versioning import git


class RelabelError(RuntimeError):
    pass


class RelabelSession:
    def __init__(self, repo: str | Path, tool: ToolConfig):
        self.repo = Path(repo)
        self.tool = tool
        self.eval_path = self.repo / tool.eval_data
        self._records: dict[str, dict] = {}     # sample_id -> {'path': Path, 'data': dict}
        self._order: list[str] = []             # sample_id order for stable listing
        self._edits: dict[str, Any] = {}        # sample_id -> pending new gold
        self._load()

    def _load(self) -> None:
        if self.eval_path.is_dir():
            files = sorted(self.eval_path.glob('*.json'))
            for f in files:
                data = json.loads(f.read_text())
                self._add(data, f)
        elif self.eval_path.exists():
            payload = json.loads(self.eval_path.read_text())
            for data in (payload if isinstance(payload, list) else [payload]):
                self._add(data, self.eval_path)
        else:
            raise RelabelError(f'Eval data not found at {self.eval_path}')

    def _add(self, data: dict, path: Path) -> None:
        sid = data['sample_id']
        self._records[sid] = {'path': path, 'data': data}
        self._order.append(sid)

    # --- reads ---------------------------------------------------------------------------

    def current_gold(self, sample_id: str) -> Any:
        """Gold with any pending edit applied (the working-state value)."""
        if sample_id in self._edits:
            return self._edits[sample_id]
        return self._records[sample_id]['data'].get('gold')

    def samples(self, predictions: dict[str, Any] | None = None) -> list[dict]:
        predictions = predictions or {}
        rows = []
        for sid in self._order:
            data = self._records[sid]['data']
            rows.append({
                'sample_id': sid,
                'raw_file_name': data.get('raw_file_name'),
                'input': data.get('input'),
                'gold': self.current_gold(sid),
                'prediction': predictions.get(sid),
                'edited': sid in self._edits,
            })
        return rows

    @property
    def pending(self) -> list[str]:
        return list(self._edits)

    # --- writes --------------------------------------------------------------------------

    def set_gold(self, sample_id: str, gold: Any) -> None:
        if sample_id not in self._records:
            raise RelabelError(f'Unknown sample_id: {sample_id}')
        # A pending edit that restores the committed value is a no-op — drop it.
        if gold == self._records[sample_id]['data'].get('gold'):
            self._edits.pop(sample_id, None)
        else:
            self._edits[sample_id] = gold

    def commit(self, annotation: str | None = None) -> dict:
        """Write pending edits into the eval files and create one git commit (§9.2)."""
        changed = list(self._edits)
        if not changed:
            return {'committed': False, 'reason': 'no changes', 'changed': []}

        touched_paths: set[Path] = set()
        for sid in changed:
            record = self._records[sid]
            record['data']['gold'] = self._edits[sid]
            touched_paths.add(record['path'])
        for path in touched_paths:
            self._write(path)

        message = self._message(len(changed), annotation)
        commit_hash = git.add_and_commit(self.repo, list(touched_paths), message)
        self._edits.clear()
        return {'committed': True, 'commit': commit_hash, 'message': message, 'changed': changed}

    def _write(self, path: Path) -> None:
        # For a single combined file, rewrite the whole list; for one-file-per-sample, the
        # record's own data.
        records_here = [r for r in self._records.values() if r['path'] == path]
        if path == self.eval_path and self.eval_path.is_file() and len(records_here) > 1:
            payload = [r['data'] for r in records_here]
        else:
            payload = records_here[0]['data']
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n')

    def _message(self, count: int, annotation: str | None) -> str:
        base = f"Relabel {count} sample(s) via Opaque — {date.today().isoformat()}"
        return f'{base}\n\n{annotation.strip()}' if annotation and annotation.strip() else base
