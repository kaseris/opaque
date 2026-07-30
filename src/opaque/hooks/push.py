"""Push-triggered local evaluation (the PR trigger, run on the developer's machine).

Why local rather than CI: an opaque run shells out to the target repo's real eval script,
which needs that project's model credentials and eval data, and it writes to an MLflow store
that has to persist across runs (§8). Neither survives an ephemeral CI runner. So the
*trigger* is PR-shaped while the *execution* stays on the machine that already has all of it.

``pre-push`` is the hook that makes that work: every PR creation and every PR update is
preceded by exactly one push from a developer's machine, so a pre-push hook fires exactly
once per PR-affecting event — no more (unlike ``post-commit``) and no less.

Experiment routing (important): ``prompt_impact.html`` diffs consecutive runs *within one
experiment* in wall-clock order and has no branch awareness, so concurrent branches logging
into the canonical experiment would produce deltas between unrelated prompt states. Branch
runs therefore go to ``{project}/{tool}@{branch}`` and only the integration branch writes the
canonical ``{project}/{tool}`` history.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from ..config.models import OpaqueConfig, ToolConfig
from ..versioning import git


@dataclass
class PushRef:
    """One line of a ``pre-push`` hook's stdin: ``<local ref> <local sha> <remote ref> <remote sha>``."""

    local_ref: str
    local_sha: str
    remote_ref: str
    remote_sha: str

    @property
    def is_delete(self) -> bool:
        return self.local_sha == git.NULL_SHA

    @property
    def is_new_branch(self) -> bool:
        """No counterpart on the remote yet — i.e. the push that opens the PR."""
        return self.remote_sha == git.NULL_SHA


def parse_push_refs(text: str) -> list[PushRef]:
    """Parse the ref lines git feeds a pre-push hook. Malformed lines are ignored."""
    refs = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 4:
            refs.append(PushRef(*parts))
    return refs


def push_ranges(
    repo: str | Path, refs: list[PushRef], base_branch: str | None = None
) -> list[tuple[str | None, str]]:
    """``(base, head)`` pairs describing what each pushed ref actually adds.

    For a branch that already exists on the remote, the base is the remote's current tip — so
    only the newly pushed commits count. For a brand-new branch (the PR-opening push) there is
    no remote tip, so the base is the merge point with the integration branch, which is exactly
    the diff the PR itself will show.
    """
    base_branch = base_branch or git.default_branch(repo)
    ranges: list[tuple[str | None, str]] = []
    for ref in refs:
        if ref.is_delete:
            continue
        if not ref.is_new_branch and git.rev_exists(repo, ref.remote_sha):
            base = ref.remote_sha
        else:
            base = git.merge_base(repo, ref.local_sha, base_branch)
        ranges.append((base, ref.local_sha))
    return ranges


def changed_in_ranges(repo: str | Path, ranges: list[tuple[str | None, str]]) -> list[str]:
    paths: set[str] = set()
    for base, head in ranges:
        paths.update(git.changed_paths(repo, base, head))
    return sorted(paths)


def _norm(path: str) -> str:
    return PurePosixPath(path).as_posix().strip('/')


def _within(path: str, root: str) -> bool:
    """True if ``path`` is ``root`` itself or sits underneath it (root may be a directory)."""
    path, root = _norm(path), _norm(root)
    return path == root or path.startswith(root + '/')


def tracked_paths(tool: ToolConfig) -> set[str]:
    """The paths whose change makes a run's headline metric attributable to *this* tool.

    Prompts and the field-matching schema only. Eval data is deliberately excluded: it is
    git-versioned in the same repo (§3.2) and the relabeling UI commits to it (§9.2), so
    including it would fire runs whose deltas reflect changed gold rather than a changed
    prompt. Eval-data changes are still detected and flagged — see ``Affected``.
    """
    paths = {_norm(p) for p in tool.prompts.values()}
    if tool.field_schema_path:
        paths.add(_norm(tool.field_schema_path))
    return paths


@dataclass
class Affected:
    tool: ToolConfig
    changed_prompts: list[str] = field(default_factory=list)
    eval_data_changed: bool = False


def affected_tools(config: OpaqueConfig, changed: list[str]) -> list[Affected]:
    """Tools whose tracked paths appear in ``changed``."""
    out = []
    for tool in config.tools:
        tracked = tracked_paths(tool)
        hits = sorted({t for t in tracked for c in changed if _within(c, t)})
        eval_hit = any(_within(c, tool.eval_data) for c in changed)
        if hits:
            out.append(Affected(tool=tool, changed_prompts=hits, eval_data_changed=eval_hit))
    return out


def experiment_for(project: str, tool_name: str, branch: str, base_branch: str) -> str:
    """Canonical §8.1 name on the integration branch, branch-scoped anywhere else."""
    canonical = f'{project}/{tool_name}'
    return canonical if branch == base_branch else f'{canonical}@{branch}'


@dataclass
class ToolCheck:
    """Result of evaluating one affected tool during a check."""

    tool_name: str
    experiment: str
    changed_prompts: list[str]
    eval_data_changed: bool
    metric_name: str | None = None
    metric: float | None = None
    baseline: float | None = None
    labeled: int = 0
    unlabeled: int = 0
    run_id: str | None = None
    report_path: str | None = None
    skipped: str | None = None

    @property
    def delta(self) -> float | None:
        if self.metric is None or self.baseline is None:
            return None
        return self.metric - self.baseline

    @property
    def regressed(self) -> bool:
        d = self.delta
        return d is not None and d < 0


@dataclass
class CheckResult:
    branch: str
    base_branch: str
    changed: list[str]
    checks: list[ToolCheck] = field(default_factory=list)
    note: str | None = None

    @property
    def ran(self) -> list[ToolCheck]:
        return [c for c in self.checks if c.skipped is None]

    def worst_delta(self) -> float | None:
        deltas = [c.delta for c in self.ran if c.delta is not None]
        return min(deltas) if deltas else None


def check(
    repo: str | Path,
    *,
    refs: list[PushRef] | None = None,
    base: str | None = None,
    head: str | None = None,
    tools: list[str] | None = None,
    tracking_uri: str = './mlruns',
    report: bool = True,
    log: bool = True,
) -> CheckResult:
    """Detect which tools a push touches and evaluate them locally.

    ``refs`` comes from a pre-push hook's stdin; ``base``/``head`` is the manual equivalent
    for running the same check by hand.
    """
    from .. import pipeline
    from ..config.loader import load_config
    from ..runner import RunnerError
    from .summary import latest_metric

    repo = Path(repo).resolve()
    config = load_config(repo)
    base_branch = git.default_branch(repo)
    branch = git.current_branch(repo)

    if refs is not None:
        ranges = push_ranges(repo, refs, base_branch)
    else:
        head = head or 'HEAD'
        ranges = [(base or git.merge_base(repo, head, base_branch), head)]

    result = CheckResult(branch=branch, base_branch=base_branch, changed=[])
    if not ranges:
        result.note = 'nothing to evaluate (no refs pushed, or a branch deletion)'
        return result

    result.changed = changed_in_ranges(repo, ranges)
    affected = affected_tools(config, result.changed)
    if tools:
        affected = [a for a in affected if a.tool.name in tools]
    if not affected:
        result.note = 'no tracked prompt or field-schema file changed'
        return result

    # The eval scores the *working tree*, so a push of some other branch's ref would produce a
    # run mislabelled as that branch's result. Refuse rather than log something untrue.
    head_sha = git.head_commit(repo)
    if refs is not None and head_sha is not None and all(h != head_sha for _, h in ranges):
        result.note = 'pushed ref is not the checked-out commit; opaque scores the working tree'
        return result

    for item in affected:
        experiment = experiment_for(config.project, item.tool.name, branch, base_branch)
        check_out = ToolCheck(
            tool_name=item.tool.name,
            experiment=experiment,
            changed_prompts=item.changed_prompts,
            eval_data_changed=item.eval_data_changed,
        )
        baseline = latest_metric(tracking_uri, f'{config.project}/{item.tool.name}')
        if baseline is not None:
            check_out.metric_name, check_out.baseline = baseline

        try:
            out = pipeline.evaluate(
                repo,
                item.tool.name,
                tracking_uri=tracking_uri,
                report=report,
                log=log,
                experiment=experiment,
            )
        except (RunnerError, FileNotFoundError, KeyError) as exc:
            check_out.skipped = str(exc)
            result.checks.append(check_out)
            continue

        r = out.result
        headline = 'field_accuracy' if r.task_type == 'extraction' else 'accuracy'
        check_out.metric_name = headline
        check_out.metric = r.metrics.get(headline)
        check_out.labeled = int(r.metrics.get('labeled_sample_count', 0))
        check_out.unlabeled = int(r.metrics.get('unlabeled_sample_count', 0))
        check_out.run_id = out.run_id
        check_out.report_path = str(out.report_path) if out.report_path else None
        result.checks.append(check_out)

    return result
