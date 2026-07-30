"""Push-triggered local evaluation (the PR trigger) — ref parsing, change detection,
experiment routing, and hook installation."""

from __future__ import annotations

import shutil
import stat
from pathlib import Path

import mlflow
import pytest
from typer.testing import CliRunner

from opaque.cli import app
from opaque.config.loader import load_config
from opaque.hooks import installer, push
from opaque.hooks.summary import render_markdown, render_terminal
from opaque.versioning.git import NULL_SHA

DEMO = Path(__file__).resolve().parents[1] / 'examples' / 'demo_project'
runner = CliRunner()


@pytest.fixture
def project(git_repo):
    """The demo project as a standalone git repo, committed clean."""
    shutil.copytree(DEMO, git_repo.path, dirs_exist_ok=True, ignore=shutil.ignore_patterns('.git'))
    git_repo.commit('initial')
    return git_repo


def _ref(sha: str, branch: str = 'feature', remote_sha: str = NULL_SHA) -> push.PushRef:
    return push.PushRef(f'refs/heads/{branch}', sha, f'refs/heads/{branch}', remote_sha)


def _tag_ref(sha: str, tag: str = 'v1.0') -> push.PushRef:
    return push.PushRef(f'refs/tags/{tag}', sha, f'refs/tags/{tag}', NULL_SHA)


# --- ref parsing -----------------------------------------------------------------------


def test_parse_push_refs_reads_git_stdin_format():
    refs = push.parse_push_refs(
        'refs/heads/f 1111111111111111111111111111111111111111 refs/heads/f ' + NULL_SHA + '\n'
        'garbage line\n'
    )
    assert len(refs) == 1
    assert refs[0].is_new_branch and not refs[0].is_delete


def test_deleted_branch_yields_no_range(project):
    refs = [_ref(NULL_SHA, remote_sha='2' * 40)]
    assert push.push_ranges(project.path, refs) == []


def test_tag_push_is_ignored(project):
    """`git push --tags` after landing a prompt change would otherwise re-score commits that
    were already evaluated when the branch itself was pushed."""
    project.git('checkout', '-q', '-b', 'feature')
    project.write('prompts/extraction.txt', 'v2\n')
    head = project.commit('tweak prompt')

    assert push.push_ranges(project.path, [_tag_ref(head)]) == []
    # The same commit pushed as a branch is still evaluated.
    assert push.push_ranges(project.path, [_ref(head)]) != []


def test_tag_only_push_reports_nothing_to_evaluate(project):
    project.git('checkout', '-q', '-b', 'feature')
    project.write('prompts/extraction.txt', 'v2\n')
    head = project.commit('tweak prompt')

    result = push.check(
        project.path, refs=[_tag_ref(head)], tracking_uri=str(project.path / 'store')
    )
    assert result.checks == []
    assert 'no branch refs' in result.note


def test_new_branch_bases_on_merge_point_with_default_branch(project):
    """The PR-opening push has no remote tip, so the diff must be against the merge base —
    which is exactly what the PR will show."""
    main_tip = project.git('rev-parse', 'HEAD').strip()
    project.git('checkout', '-q', '-b', 'feature')
    project.write('prompts/extraction.txt', 'v2 extraction prompt\n')
    head = project.commit('tweak prompt')

    ranges = push.push_ranges(project.path, [_ref(head)])
    assert ranges == [(main_tip, head)]
    assert push.changed_in_ranges(project.path, ranges) == ['prompts/extraction.txt']


def test_existing_remote_branch_bases_on_remote_tip(project):
    project.git('checkout', '-q', '-b', 'feature')
    project.write('prompts/extraction.txt', 'v2\n')
    first = project.commit('one')
    project.write('prompts/system.txt', 'v2 system\n')
    second = project.commit('two')

    # Remote already has `first`; only the second commit is being pushed.
    ranges = push.push_ranges(project.path, [_ref(second, remote_sha=first)])
    assert push.changed_in_ranges(project.path, ranges) == ['prompts/system.txt']


# --- change detection ------------------------------------------------------------------


def test_affected_tools_matches_only_tools_owning_the_changed_prompt(project):
    config = load_config(project.path)
    affected = push.affected_tools(config, ['prompts/extraction.txt'])
    assert [a.tool.name for a in affected] == ['invoices']


def test_shared_prompt_affects_every_tool_using_it(project):
    config = load_config(project.path)
    affected = push.affected_tools(config, ['prompts/system.txt'])
    assert sorted(a.tool.name for a in affected) == ['doc_type', 'invoices']


def test_field_schema_change_is_tracked(project):
    config = load_config(project.path)
    affected = push.affected_tools(config, ['field_schema.yaml'])
    assert [a.tool.name for a in affected] == ['invoices']


def test_eval_data_change_alone_does_not_trigger_a_run(project):
    """Relabeling commits (§9.2) touch eval data; their deltas are not prompt-attributable."""
    config = load_config(project.path)
    assert push.affected_tools(config, ['eval_data/invoices/sample_01.json']) == []


def test_eval_data_change_is_flagged_when_a_prompt_also_changed(project):
    config = load_config(project.path)
    affected = push.affected_tools(
        config, ['prompts/extraction.txt', 'eval_data/invoices/sample_01.json']
    )
    assert affected[0].eval_data_changed is True


def test_unrelated_change_affects_nothing(project):
    config = load_config(project.path)
    assert push.affected_tools(config, ['README.md', 'src/app.py']) == []


# --- experiment routing ----------------------------------------------------------------


def test_experiment_is_canonical_on_the_integration_branch():
    assert push.experiment_for('acme', 'invoices', 'main', 'main') == 'acme/invoices'


def test_experiment_is_branch_scoped_off_the_integration_branch():
    """prompt_impact.html diffs consecutive runs in one experiment with no branch awareness,
    so branch runs must not land in the canonical history."""
    assert push.experiment_for('acme', 'invoices', 'feature', 'main') == 'acme/invoices@feature'


# --- end to end ------------------------------------------------------------------------


def test_check_runs_affected_tool_and_logs_to_branch_experiment(project):
    store = project.path / 'store'
    project.git('checkout', '-q', '-b', 'feature')
    project.write('prompts/extraction.txt', 'v2 extraction prompt\n')
    head = project.commit('tweak prompt')

    result = push.check(project.path, refs=[_ref(head)], tracking_uri=str(store))

    assert [c.tool_name for c in result.checks] == ['invoices']
    check = result.checks[0]
    assert check.skipped is None
    assert check.experiment == 'acme_demo/invoices@feature'
    assert check.metric_name == 'field_accuracy'
    assert check.metric is not None
    assert check.baseline is None  # nothing on main yet
    assert check.changed_prompts == ['prompts/extraction.txt']

    client = mlflow.MlflowClient(tracking_uri=store.resolve().as_uri())
    assert client.get_experiment_by_name('acme_demo/invoices@feature') is not None
    assert client.get_experiment_by_name('acme_demo/invoices') is None


def test_check_reports_delta_against_the_canonical_baseline(project):
    """The delta a developer wants is 'versus what is on main today', so the baseline comes
    from the canonical experiment even though the run is logged to a branch-scoped one."""
    store = project.path / 'store'

    # A merge landing on main: the remote already has the previous tip, so the push range is
    # the merge itself. This is the run that writes the canonical history.
    tip0 = project.git('rev-parse', 'HEAD').strip()
    project.write('prompts/extraction.txt', 'v1 extraction prompt\n')
    tip1 = project.commit('prompt change on main')
    push.check(
        project.path,
        refs=[_ref(tip1, branch='main', remote_sha=tip0)],
        tracking_uri=str(store),
    )

    project.git('checkout', '-q', '-b', 'feature')
    project.write('prompts/extraction.txt', 'v2 extraction prompt\n')
    head = project.commit('tweak prompt')
    result = push.check(project.path, refs=[_ref(head)], tracking_uri=str(store))

    check = result.checks[0]
    assert check.baseline is not None
    assert check.delta == pytest.approx(check.metric - check.baseline)


def test_check_skips_when_no_tracked_file_changed(project):
    project.git('checkout', '-q', '-b', 'docs')
    project.write('README.md', 'docs only\n')
    head = project.commit('docs')

    result = push.check(project.path, refs=[_ref(head)], tracking_uri=str(project.path / 'store'))
    assert result.checks == []
    assert 'no tracked prompt' in result.note


def test_check_refuses_when_pushed_ref_is_not_the_working_tree(project):
    """The eval scores the checked-out tree, so scoring a ref that is not checked out would
    log a result under the wrong branch's name."""
    project.git('checkout', '-q', '-b', 'feature')
    project.write('prompts/extraction.txt', 'v2\n')
    head = project.commit('tweak')
    project.git('checkout', '-q', 'main')

    result = push.check(project.path, refs=[_ref(head)], tracking_uri=str(project.path / 'store'))
    assert result.checks == []
    assert 'not the checked-out commit' in result.note


def test_summaries_render_without_a_baseline(project):
    store = project.path / 'store'
    project.git('checkout', '-q', '-b', 'feature')
    project.write('prompts/extraction.txt', 'v2\n')
    head = project.commit('tweak')
    result = push.check(project.path, refs=[_ref(head)], tracking_uri=str(store))

    md = render_markdown(result)
    assert 'invoices' in md and 'field_accuracy' in md
    assert 'no baseline yet' in md.lower()
    assert 'invoices' in render_terminal(result)


def test_markdown_warns_when_eval_data_moved_too(project):
    store = project.path / 'store'
    project.git('checkout', '-q', '-b', 'feature')
    project.write('prompts/extraction.txt', 'v2\n')
    sample = next((project.path / 'eval_data' / 'invoices').glob('*.json'))
    project.write(f'eval_data/invoices/{sample.name}', sample.read_text())
    project.write('eval_data/invoices/extra.json', sample.read_text().replace('"sample', '"x_sample'))
    head = project.commit('prompt + relabel')

    result = push.check(project.path, refs=[_ref(head)], tracking_uri=str(store))
    assert result.checks[0].eval_data_changed is True
    assert 'not attributable to the prompt alone' in render_markdown(result)


# --- installer -------------------------------------------------------------------------


def test_install_writes_an_executable_hook(project):
    path = installer.install(project.path, python='/usr/bin/python3')
    assert path == project.path / '.git' / 'hooks' / 'pre-push'
    body = path.read_text()
    assert body.startswith('#!/bin/sh')
    assert '-m opaque check' in body
    assert str(project.path) in body
    assert path.stat().st_mode & stat.S_IXUSR


def test_installed_hook_honours_flags(project):
    body = installer.install(
        project.path, gate=True, comment=True, tracking_uri='/srv/mlruns'
    ).read_text()
    assert '--gate' in body and '--comment' in body and '/srv/mlruns' in body


def test_hook_filters_to_origin_by_default(project):
    body = installer.install(project.path).read_text()
    assert '"$1" != origin' in body


def test_hook_remote_filter_is_configurable(project):
    body = installer.install(project.path, remote='upstream').read_text()
    assert '"$1" != upstream' in body
    assert '"$1" != origin' not in body


def test_hook_remote_filter_can_be_disabled(project):
    body = installer.install(project.path, remote=None).read_text()
    assert '"$1" !=' not in body


def test_remote_guard_short_circuits_before_python(project):
    """The guard must sit above the exec line, or an irrelevant push still pays interpreter
    startup just to decide it is irrelevant."""
    body = installer.install(project.path).read_text()
    assert body.index('"$1" != origin') < body.index('exec ')


def test_reinstall_overwrites_our_own_hook(project):
    installer.install(project.path)
    path = installer.install(project.path, gate=True)
    assert '--gate' in path.read_text()


def test_install_refuses_to_clobber_a_foreign_hook(project):
    path = installer.hook_path(project.path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('#!/bin/sh\necho someone elses hook\n')

    with pytest.raises(installer.HookError, match='not written by opaque'):
        installer.install(project.path)

    installer.install(project.path, force=True)
    assert installer.is_ours(path)


def test_uninstall_leaves_a_foreign_hook_alone(project):
    path = installer.hook_path(project.path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('#!/bin/sh\necho mine\n')
    with pytest.raises(installer.HookError):
        installer.uninstall(project.path)
    assert path.exists()


def test_uninstall_removes_our_hook(project):
    installer.install(project.path)
    assert installer.uninstall(project.path) is not None
    assert installer.uninstall(project.path) is None


def test_hooks_dir_honours_core_hooks_path(project):
    custom = project.path / 'githooks'
    custom.mkdir()
    project.git('config', 'core.hooksPath', str(custom))
    assert installer.hooks_dir(project.path) == custom


# --- CLI -------------------------------------------------------------------------------


def test_hook_cli_remote_flag(project):
    runner.invoke(app, ['hook', 'install', str(project.path), '--remote', 'upstream'])
    assert '"$1" != upstream' in installer.hook_path(project.path).read_text()

    runner.invoke(app, ['hook', 'install', str(project.path), '--remote', ''])
    assert '"$1" !=' not in installer.hook_path(project.path).read_text()


def test_hook_cli_install_status_uninstall(project):
    assert 'Not installed' in runner.invoke(app, ['hook', 'status', str(project.path)]).output

    result = runner.invoke(app, ['hook', 'install', str(project.path)])
    assert result.exit_code == 0, result.output
    assert 'Installed' in runner.invoke(app, ['hook', 'status', str(project.path)]).output

    result = runner.invoke(app, ['hook', 'uninstall', str(project.path)])
    assert result.exit_code == 0, result.output
    assert 'Removed' in result.output


def test_check_cli_is_quiet_and_zero_exit_on_a_non_onboarded_repo(git_repo):
    git_repo.write('README.md', 'hi\n')
    git_repo.commit('init')
    result = runner.invoke(app, ['check', '--repo', str(git_repo.path)])
    assert result.exit_code == 0, result.output


def test_check_cli_writes_markdown_summary(project):
    store = project.path / 'store'
    out = project.path / 'summary.md'
    project.git('checkout', '-q', '-b', 'feature')
    project.write('prompts/extraction.txt', 'v2\n')
    project.commit('tweak')

    result = runner.invoke(
        app,
        ['check', '--repo', str(project.path), '--tracking-uri', str(store), '--out', str(out)],
    )
    assert result.exit_code == 0, result.output
    assert 'invoices' in out.read_text()


def test_check_cli_gate_blocks_on_regression(project, monkeypatch):
    """--gate turns the check into a push blocker; without it a regression still exits 0."""
    store = project.path / 'store'
    project.git('checkout', '-q', '-b', 'feature')
    project.write('prompts/extraction.txt', 'v2\n')
    project.commit('tweak')

    # Force a regression by reporting a baseline above whatever this run scores.
    monkeypatch.setattr('opaque.hooks.summary.latest_metric', lambda *a, **k: ('field_accuracy', 1.1))

    args = ['check', '--repo', str(project.path), '--tracking-uri', str(store)]
    assert runner.invoke(app, args).exit_code == 0
    gated = runner.invoke(app, [*args, '--gate'])
    assert gated.exit_code == 1
    assert 'blocking push' in gated.output + str(gated.stderr or '')
