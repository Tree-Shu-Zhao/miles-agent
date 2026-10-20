"""tools/r2s_decodes.py: a cell's mean +- std over its decodes, read from the job logs' eval lines."""

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("r2s_decodes", REPO / "tools" / "r2s_decodes.py")
rd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rd)

# miles/ray/rollout/metrics.py:55 `logger.info(f"eval {rollout_id}: {log_dict}")` as the job log carries it:
# the task launcher's task label, the driver logger's prefix (logging_utils.py:43 with arguments.py:2818's name; RolloutExecutor
# is not a ray actor), the logger's "file:line - " and the dict (floats; a value may print as np.float64(...)).
# The actors' lines reach the same task 0 through ray's forwarding, under ray's prefix.
DRIVER = "[2026-10-18 10:58:48.413 main] "
RAY = "\x1b[36m(MegatronTrainRayActor pid=4242, ip=10.0.0.7)\x1b[0m [2026-10-18 10:58:48.413 actor] "


def _eval_line(rollout_id: int, seen: float | str, unseen: float | str) -> str:
    return (
        f"0: {DRIVER}metrics.py:55 - eval {rollout_id}: {{'eval/alfworld_seen': {seen}, "
        f"'eval/alfworld_seen/env/won': {seen}, 'eval/alfworld_seen/env/step_count': np.float64(12.5), "
        f"'eval/alfworld_seen-none_reward_ratio': 0.0, 'eval/alfworld_unseen': {unseen}, "
        f"'eval/alfworld_unseen/env/won': {unseen}, 'eval/lag_steps': 0}}"
    )


def _train_line(rollout_id: int) -> str:
    """A training rollout's record (metrics.py:93, the rollout executor's perf dict): what an eval-only pass lacks."""
    return f"0: {RAY}metrics.py:93 - perf {rollout_id}: {{'rollout/response_length/mean': 812.5, 'perf/time': 51.0}}"


def _pass_line(tag: str) -> str:
    """The wrapper's JOB_PASS_miles_run_<RUN_TAG> echo after the entry returned 0 (Contract D): the wrapper
    writes it itself, so the job log carries it without a task label."""
    return f"JOB_PASS_miles_run_{tag}"


def test_r2s_decodes_mean_std(tmp_path, capsys):
    def log(name: str, lines: list[str]) -> None:
        (tmp_path / name).write_text("\n".join(lines) + "\n")

    # the arm: a first link that reached the eval at rollout 124, then the requeued link with the final eval at
    # rollout 149 (0-based: step 150 of 150). That line is one write of several KB, past the pipe's atomic size,
    # so the job log carries it in pieces with whole records of task 0 between them (ray's forwarded actor line
    # and the driver's own timer line) and another task's line
    final = _eval_line(149, 0.80, 0.70)
    cut = final.index("'eval/alfworld_seen/env/step_count'")
    log("arm_gym_101.log", ["gym up"])
    log(
        "run_101.log",
        [
            "0: + python3 opd.py",
            _train_line(123),
            _eval_line(124, 0.70, 0.60),
            "0: JOB_STOPPED_TO_RESUME_miles_run_arm",
        ],
    )
    log("arm_gym_102.log", ["gym up"])
    log(
        "run_102.log",
        [
            "0: + python3 opd.py",
            _train_line(148),
            final[:cut],
            f"0: {RAY}timer.py:31 - Timer actor_train start",
            f"0: {DRIVER}timer.py:31 - Timer eval_rollout end",
            "1: + sleep 5",
            "0: " + final[cut:],
            f"0: {RAY}timer.py:31 - Timer actor_train end",
            "0: JOB_PASS_miles_run_arm",
        ],
    )
    # decodes 2 and 3: eval_ckpt.py runs, one eval at rollout 0 each
    log("d1001_gym_103.log", ["gym up"])
    log("run_103.log", [_eval_line(0, 0.85, 0.70), _pass_line("d1001")])
    log("d1002_gym_104.log", ["gym up"])
    log("run_104.log", [_eval_line(0, 0.75, 0.76), _pass_line("d1002")])

    records, unparsed = rd.eval_records(str(tmp_path / "run_102.log"))
    assert [rollout_id for rollout_id, _ in records] == [149] and unparsed == 0
    assert records[0][1]["eval/alfworld_seen/env/step_count"] == 12.5
    assert rd.train_records(str(tmp_path / "run_102.log")) == 1
    assert rd.train_records(str(tmp_path / "run_103.log")) == 0
    assert rd.run_logs(str(tmp_path), "arm") == [str(tmp_path / "run_101.log"), str(tmp_path / "run_102.log")]

    assert rd.main(["--runs", "arm", "d1001", "d1002", "--runs-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[:3] == ["arm: eval 149  unparsed=0", "d1001: eval 0  unparsed=0", "d1002: eval 0  unparsed=0"]
    # seen 0.80 0.85 0.75 -> 0.8000 +- 0.0500; unseen 0.70 0.70 0.76 -> 0.7200 +- 0.0346 (sample std);
    # the paper's table entry, the seen/unseen mean per decode 0.75 0.775 0.755 -> 0.7600 +- 0.0132
    assert out[3].split() == ["alfworld_seen", "0.8000", "+-", "0.0500", "[0.8000", "0.8500", "0.7500]"]
    assert out[4].split() == ["alfworld_unseen", "0.7200", "+-", "0.0346", "[0.7000", "0.7000", "0.7600]"]
    assert out[5].split() == ["seen+unseen", "mean", "0.7600", "+-", "0.0132", "[0.7500", "0.7750", "0.7550]"]
    assert len(out) == 6
    # --logs: one job log per decode, the same numbers
    logs = [str(tmp_path / f"run_{job}.log") for job in (102, 103, 104)]
    assert rd.main(["--logs", *logs]) == 0
    assert capsys.readouterr().out.splitlines()[3:] == out[3:]


def test_r2s_decodes_counts_an_eval_record_that_never_closed(tmp_path, capsys):
    # a link killed in the middle of the final eval's write: the record never closes (the scheduler's unlabelled line is
    # glued on and does not close it either), the decode falls back to the eval before it and the header says so
    # instead of passing the earlier numbers off as the final ones
    final = _eval_line(149, 0.80, 0.70)
    log = tmp_path / "run_201.log"
    log.write_text(
        "\n".join(
            [
                _eval_line(124, 0.70, 0.60),
                final[: final.index("'eval/alfworld_unseen'")],
                "slurmstepd: error: *** JOB 201 ON node1 CANCELLED AT 2026-10-18T11:00:00 ***",
                _pass_line("arm"),
            ]
        )
        + "\n"
    )
    records, unparsed = rd.eval_records(str(log))
    assert [rollout_id for rollout_id, _ in records] == [124] and unparsed == 1
    assert rd.main(["--logs", str(log)]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "run_201.log: eval 124  unparsed=1"
    assert out[1].split()[:2] == ["alfworld_seen", "0.7000"]


def test_r2s_decodes_rejects_a_glue_that_closes_on_foreign_keys(tmp_path, capsys):
    """The tail piece of another wrapped line (a train dict's `, 'train/k': v}`) landing between the eval pieces
    would close the glue early with its keys mixed in: a joined dict is the eval only when every key is eval/...,
    so the glue goes on, cannot parse with the real tail and is counted as unparsed; the earlier eval stands,
    flagged by the header (and by the final-eval WARN, the run having trained)."""
    final = _eval_line(149, 0.80, 0.70)
    cut = final.index(" 'eval/alfworld_seen/env/step_count'")  # the first piece ends right after a value and its comma
    log = tmp_path / "run_301.log"
    log.write_text(
        "\n".join(
            [
                _train_line(148),
                _eval_line(124, 0.70, 0.60),
                final[:cut],
                "0:  'train/k21': 21.0}",
                "0: " + final[cut:],
                _pass_line("arm"),
            ]
        )
        + "\n"
    )
    records, unparsed = rd.eval_records(str(log))
    assert [rollout_id for rollout_id, _ in records] == [124] and unparsed == 1
    assert rd.main(["--logs", str(log)]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "run_301.log: eval 124  unparsed=1"
    assert out[1].startswith("WARN run_301.log: final eval is rollout 124, expected 149")


def test_r2s_decodes_nan_and_ddof(tmp_path, capsys):
    """A won miles printed as nan (no sample scored) is shown as nan for that decode and left out of the mean +- std
    with a WARN line, never a TypeError; --ddof picks the estimator the docstring states: 1 = sample std (the
    default, the paper tables' summary), 0 = population std."""
    for job, (seen, unseen) in ((401, (0.80, "nan")), (402, (0.85, 0.70)), (403, (0.75, 0.76))):
        (tmp_path / f"run_{job}.log").write_text(_eval_line(0, seen, unseen) + "\n" + _pass_line(f"d{job}") + "\n")
    logs = [str(tmp_path / f"run_{job}.log") for job in (401, 402, 403)]
    assert rd.main(["--logs", *logs]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[:3] == [f"run_{job}.log: eval 0  unparsed=0" for job in (401, 402, 403)]
    assert out[3] == "WARN run_401.log: eval/alfworld_unseen/env/won is nan (left out of the mean +- std)"
    assert out[4].split() == ["alfworld_seen", "0.8000", "+-", "0.0500", "[0.8000", "0.8500", "0.7500]"]
    # unseen over the two finite decodes 0.70 0.76 -> 0.7300 +- 0.0424 (sample std); the per-decode seen/unseen mean
    # is nan for the first decode too, 0.775 0.755 -> 0.7650 +- 0.0141
    assert out[5].split() == ["alfworld_unseen", "0.7300", "+-", "0.0424", "[nan", "0.7000", "0.7600]"]
    assert out[6].split() == ["seen+unseen", "mean", "0.7650", "+-", "0.0141", "[nan", "0.7750", "0.7550]"]
    assert len(out) == 7
    # population std: seen 0.0408, unseen 0.0300, the mean row 0.0100
    assert rd.main(["--logs", *logs, "--ddof", "0"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[4].split()[1:4] == ["0.8000", "+-", "0.0408"]
    assert out[5].split()[1:4] == ["0.7300", "+-", "0.0300"]
    assert out[6].split()[2:5] == ["0.7650", "+-", "0.0100"]
    assert "--ddof 1 (default) is the sample std" in rd.__doc__ and "--ddof 0 is the population std" in rd.__doc__


def test_r2s_decodes_docstring_names_the_decode_seeds():
    """The decodes are seeds 42 / 43 / 44 (docs/r2s.md section 2.4): the docstring and its --runs example name them,
    and the paper's own 1001 / 1002 appear in the one sentence that says whose seeds those were."""
    doc = " ".join(rd.__doc__.split())
    assert "(its training seed, 42)" in doc and "(seeds 43 and 44;" in doc
    assert doc.count("1001") == 1 and doc.count("1002") == 1 and "cells ran 42 / 1001 / 1002" in doc
    assert "--runs r2s_alf1 r2s_alf1_d43 r2s_alf1_d44" in doc


def test_r2s_decodes_warns_on_non_final_eval(tmp_path, capsys):
    """An arm's decode must be the eval of its final step (rollout --num-rollout - 1, default 149): a cut-short
    chain whose last eval is rollout 124 gets the WARN line under its header; the eval-only decodes (eval_ckpt.py:
    no training record, one eval at rollout 0, train.py:115-117) do not; --num-rollout moves the expectation."""
    arm = tmp_path / "run_501.log"
    lines = [_eval_line(0, 0.5, 0.4), _train_line(0), _train_line(124), _eval_line(124, 0.7, 0.6), _pass_line("arm")]
    arm.write_text("\n".join(lines) + "\n")
    d1001 = tmp_path / "run_502.log"
    d1001.write_text(_eval_line(0, 0.72, 0.61) + "\n" + _pass_line("d1001") + "\n")
    assert rd.train_records(str(arm)) == 2 and rd.train_records(str(d1001)) == 0
    assert rd.main(["--logs", str(arm), str(d1001)]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[:2] == ["run_501.log: eval 124  unparsed=0", "run_502.log: eval 0  unparsed=0"]
    assert out[2].startswith("WARN run_501.log: final eval is rollout 124, expected 149 (--num-rollout 150")
    assert not any(line.startswith("WARN run_502") for line in out) and len(out) == 6
    assert rd.main(["--logs", str(arm), str(d1001), "--num-rollout", "125"]) == 0
    assert not any(line.startswith("WARN") for line in capsys.readouterr().out.splitlines())


def test_r2s_decodes_warns_without_job_pass(tmp_path, capsys):
    """A run whose logs hold no JOB_PASS line did not finish: an arm that died inside its first rollout looks like an
    eval-only pass (its step-0 eval, no training record) and passed silently; it gets the WARN under the headers. The
    marker is matched as a prefix on the driver lines (the wrapper's unlabelled echo, or the task launcher's labelled form), a
    stopped link's JOB_STOPPED_TO_RESUME is not one, and a chain counts its links together (the requeued link
    carries it)."""
    died = tmp_path / "run_601.log"
    died.write_text(_eval_line(0, 0.5, 0.4) + "\nslurmstepd: error: *** JOB 601 ON node1 CANCELLED ***\n")
    passed = tmp_path / "run_602.log"
    passed.write_text(_eval_line(0, 0.72, 0.61) + "\n" + _pass_line("d43") + "\n")
    labelled = tmp_path / "run_603.log"
    labelled.write_text(_eval_line(0, 0.70, 0.60) + "\n0: " + _pass_line("d44") + "\n")
    assert [rd.job_pass_lines(str(p)) for p in (died, passed, labelled)] == [0, 1, 1]
    assert rd.main(["--logs", str(died), str(passed), str(labelled)]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[:3] == [f"run_{job}.log: eval 0  unparsed=0" for job in (601, 602, 603)]
    assert out[3] == "WARN run_601.log: no JOB_PASS in its logs (the run did not finish)"
    assert not any(line.startswith("WARN") for line in out[4:]) and len(out) == 7
    # a chain: the first link stopped to resume (no marker), the requeued link finished -> no WARN
    for name, lines in (
        ("arm_gym_701.log", ["gym up"]),
        ("run_701.log", [_train_line(123), _eval_line(124, 0.70, 0.60), "0: JOB_STOPPED_TO_RESUME_miles_run_arm"]),
        ("arm_gym_702.log", ["gym up"]),
        ("run_702.log", [_train_line(148), _eval_line(149, 0.80, 0.70), _pass_line("arm")]),
    ):
        (tmp_path / name).write_text("\n".join(lines) + "\n")
    assert rd.job_pass_lines(str(tmp_path / "run_701.log")) == 0
    assert rd.decode("arm", rd.run_logs(str(tmp_path), "arm")).job_pass == 1
    assert rd.main(["--runs", "arm", "--runs-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "arm: eval 149  unparsed=0" and not any(line.startswith("WARN") for line in out)


def test_run_logs_accepts_the_submit_output_name(tmp_path):
    """A submit that set its own -o writes miles_run_<jobid>.log in place of run_<jobid>.log (a wrapper that sets its own log name does);
    --runs resolves either, per link, and keeps the wrapper's name for a link whose log is not there yet."""
    for name in ("arm_gym_201.log", "miles_run_201.log", "arm_gym_202.log", "run_202.log", "arm_gym_203.log"):
        (tmp_path / name).write_text("x\n")
    assert rd.run_logs(str(tmp_path), "arm") == [
        str(tmp_path / "miles_run_201.log"),
        str(tmp_path / "run_202.log"),
        str(tmp_path / "run_203.log"),
    ]
