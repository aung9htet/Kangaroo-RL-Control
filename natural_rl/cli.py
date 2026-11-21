"""`python -m natural_rl ...`: the command-line contract of the benchmark (run inside the `rl` container).

Commands that are part of the contract but not implemented yet say so and exit with status 2; they never pretend.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

NOT_YET = {
    'rank': 'milestone M9', 'figures': 'milestone M10', 'benchmark': 'milestone M9',
    'plot': 'milestone M10',
}


def _unimplemented(name: str) -> int:
    print(f'natural_rl {name}: not implemented yet ({NOT_YET.get(name, "later milestone")}). '
          'No result is produced or invented.', file=sys.stderr)
    return 2


# ----------------------------------------------------------------------------- catalog
def cmd_catalog(args) -> int:
    from . import programme
    from .env.tasks import load_manifest
    from .models.registry import MODEL_IDS, implemented
    from .registry import SUITES, budgets_config, model_config, reward_config, suite_path

    what = args.what
    if what == 'models':
        print(f'{"id":12s} {"family":11s} {"implemented":12s} action space')
        for m in MODEL_IDS:
            try:
                cfg = model_config(m)
                family, action = cfg['family'], cfg['action_space']
            except FileNotFoundError:
                family, action = '?', '(config missing)'
            print(f'{m:12s} {family:11s} {"yes" if m in implemented() else "NO":12s} {action}')
    elif what == 'rewards':
        cfg = reward_config()
        print(f'register {cfg["version"]}  status={cfg["status"]}')
        for name, p in cfg['profiles'].items():
            print(f'  {name:16s} terms={"+".join(p["terms"]):34s} {p["question"]}')
        print('lambda:', cfg['calibration']['lambda'])
    elif what == 'suites':
        for s in SUITES:
            doc = load_manifest(suite_path(s))
            print(f'{s:9s} n={doc["n_tasks"]:4d}  tasks_sha256={doc["tasks_sha256"][:16]}  generator_seed={doc["generator_seed"]}')
        print('checkpoint_video_v1: task val_001, seed 20261121, camera canonical_v1, 640x480 @ 20 fps')
    elif what == 'budgets':
        cfg = budgets_config()
        for name, b in cfg['budgets'].items():
            print(f'{name:20s} limit={b["limit"]} expected unsafe proposal events per 100-decision episode')
    elif what == 'programme':
        policies = args.policies.split(',') if args.policies else None
        if args.select:
            from .selection import p3_selection
            policies, table = p3_selection()
            print('P3 selection (validation terminal-hold rate on r7_balanced, ties -> lower unsafe rate):')
            print(table.to_string(index=False))
            print('selected:', ','.join(policies))
        phases = [args.phase] if args.phase else ['P0', 'P1', 'P2', 'P3', 'P4']
        for ph in phases:
            if ph == 'P3' and not policies:
                print('P3: pass --policies decqn,A,B,C (selected on validation terminal-hold rate before any test result)')
                continue
            runs = programme.phase_runs(ph, policies if ph == 'P3' else None)
            t = programme.totals(runs)
            print(f'# {ph}: {t["runs"]} runs, {t["transitions"] / 1e6:.0f}M transitions, {t["videos"]} checkpoint videos')
            if args.commands:
                for r in runs:
                    print(programme.train_command(r))
        if not args.phase:
            tot = programme.programme_totals()['TOTAL']
            print(f'# TOTAL: {tot["runs"]} runs, {tot["transitions"] / 1e6:.0f}M transitions, {tot["videos"]} checkpoint videos')
    return 0


# ----------------------------------------------------------------------------- environment check
def cmd_verify_env(args) -> int:
    import os
    problems = []

    def check(name, fn):
        try:
            detail = fn()
            print(f'  ok    {name}' + (f': {detail}' if detail else ''))
        except Exception as error:
            problems.append(name)
            print(f'  FAIL  {name}: {type(error).__name__}: {error}')

    from .registry import repo_root, results_root

    def imports():
        import av, mujoco, numpy, pandas, pyarrow, torch  # noqa: E401
        import kangaroo_libero  # noqa: F401
        from sb3_contrib import CrossQ, TQC  # noqa: F401
        from stable_baselines3 import PPO, SAC, TD3  # noqa: F401
        return f'torch {torch.__version__}, mujoco {mujoco.__version__}'

    def submodule():
        sim = repo_root() / 'sim' / 'Kangaroo_Libero'
        if not (sim / 'src' / 'kangaroo_libero' / '__init__.py').exists():
            raise RuntimeError('simulator submodule missing: run `git submodule update --init` on the host')
        from .provenance import git_state
        return git_state(sim)['commit'][:12]

    def manifests():
        from .env.tasks import verify_manifests
        bad = verify_manifests(repo_root() / 'configs' / 'eval')
        if bad:
            raise RuntimeError('; '.join(bad[:3]))
        return 'train/val/test/ood intact, disjoint, rule-conforming'

    def scene_and_render():
        import numpy as np
        import yaml
        from kangaroo_rl.config import load_config
        from kangaroo_rl.scene import build_scene
        from .render import RawRenderer
        cfg = load_config(repo_root() / 'configs' / 'reach.yaml')['scene']
        scene = build_scene(suite=cfg['suite'], task_id=cfg['task_id'], seed=cfg['seed'], sim_overrides=cfg.get('sim_overrides'))
        doc = json.loads((repo_root() / 'configs' / 'eval' / f'{args.suite}.json').read_text())
        if doc['scene_signature'] != scene.signature():
            raise RuntimeError(f'scene signature {scene.signature()} != manifest {doc["scene_signature"]}: the manifests were '
                               'generated for a different scene')
        renderer = RawRenderer(scene, yaml.safe_load((repo_root() / 'configs/render/canonical_v1.yaml').read_text()))
        frame = renderer.render(scene.home)
        renderer.close()
        if frame.std() < 1:
            raise RuntimeError('rendered frame is blank (check MUJOCO_GL and the graphics stack)')
        return f'scene {scene.signature()}, frame {frame.shape} via MUJOCO_GL={os.environ.get("MUJOCO_GL")}'

    def lossless_video():
        import numpy as np
        import tempfile
        from .video import verify_ffv1, write_ffv1
        frames = np.random.default_rng(0).integers(0, 256, (5, 48, 64, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            rec = write_ffv1(frames, Path(tmp) / 'raw.mkv')
            out = verify_ffv1(Path(tmp) / 'raw.mkv', rec['frame_sha256'])
        if not out['ok']:
            raise RuntimeError('; '.join(out['problems']))
        return 'FFV1 write/readback bit-exact'

    def disk():
        from .provenance import disk_free_bytes
        root = results_root()
        probe = root / '.write_probe'
        root.mkdir(parents=True, exist_ok=True)
        probe.write_text('x')
        probe.unlink()
        return f'{disk_free_bytes(root) / 1e9:.0f} GB free under {root}, writable'

    print(f'natural_rl verify-env --suite {args.suite}')
    check('imports and pinned libraries', imports)
    check('simulator submodule', submodule)
    check('task manifests', manifests)
    check('scene, manifest match and raw rendering', scene_and_render)
    check('lossless video roundtrip', lossless_video)
    check('results directory', disk)
    print('FAILED: ' + ', '.join(problems) if problems else 'environment ok')
    return 1 if problems else 0


# ----------------------------------------------------------------------------- manifests
def cmd_manifests(args) -> int:
    from .env.tasks import build_manifests, verify_manifests
    from .registry import repo_root
    out = repo_root() / 'configs' / 'eval'
    if args.action == 'verify':
        problems = verify_manifests(out)
        print('manifests ok: counts, hashes, disjointness and rules verified' if not problems else '\n'.join(problems))
        return 1 if problems else 0
    from kangaroo_rl.config import load_config
    existing = [out / f'{s}_v1.json' for s in ('train', 'val', 'test', 'ood') if (out / f'{s}_v1.json').exists()]
    if existing and not args.force:
        print('manifest files already exist and are immutable; a rule change needs a NEW manifest id (pass --force only to '
              'regenerate identical content): ' + ', '.join(str(p) for p in existing), file=sys.stderr)
        return 1
    build_manifests(load_config(repo_root() / 'configs' / 'reach.yaml'), out)
    return 0


# ----------------------------------------------------------------------------- run helpers
def _run_context(run_id: str, keep_env: bool = False):
    """(run_dir, run manifest, scene, spec, profile) for replaying a run's checkpoints."""
    from kangaroo_rl.config import load_config
    from kangaroo_rl.experiment import make_experiment
    from .env.contract import ObsSpec
    from .registry import find_run, repo_root, reward_config
    from .rewards import load_profile
    out = find_run(run_id)
    manifest = json.loads((out / 'manifest.json').read_text())
    exp = make_experiment(load_config(repo_root() / 'configs' / 'reach.yaml'), keep_env=keep_env)
    spec = ObsSpec(manifest.get('obs_version', 'v2'), exp.scene.lower, exp.scene.upper, exp.pool.lower, exp.pool.upper)
    profile = load_profile(reward_config(), manifest['reward'])
    return out, manifest, exp.scene, spec, profile


def _recorder(run_id: str):
    from .checkpoint_video import VideoRecorder
    from .env.tasks import load_manifest, task_arrays
    from .registry import load_yaml, suite_path
    out, manifest, scene, spec, profile = _run_context(run_id)
    val_doc = load_manifest(suite_path('val_v1'))
    return out, manifest, VideoRecorder(out, manifest, scene, spec, profile, task_arrays(val_doc),
                                        load_yaml('configs/eval/checkpoint_video_v1.yaml'),
                                        load_yaml('configs/render/canonical_v1.yaml'), val_doc)


def cmd_train(args) -> int:
    from .models.registry import ModelNotImplemented
    from .train import run_training
    from .rewards import RewardNotLocked
    try:
        return run_training(args)
    except (ModelNotImplemented, RewardNotLocked) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        return 2


def cmd_checkpoints(args) -> int:
    from .checkpoint_video import read_video_index
    from .checkpoints import read_index, verify_checkpoint
    from .registry import find_run
    out = find_run(args.run_id)
    index = read_index(out)
    videos = read_video_index(out).set_index('checkpoint_step') if len(read_video_index(out)) else None
    print(f'run {args.run_id}  ({out})')
    print(f'{"step":>10s}  {"kind":9s}  {"sha256":14s}  {"files":6s}  {"video":8s}  saved_at')
    for _, row in index.iterrows():
        step = int(row['checkpoint_step'])
        intact = 'ok' if verify_checkpoint(out, step) else 'CHANGED'
        status = videos.loc[step, 'status'] if videos is not None and step in videos.index else row['video_status']
        print(f'{step:>10d}  {row["kind"]:9s}  {row["sha256"][:12]:14s}  {intact:6s}  {status:8s}  {row["saved_at"]}')
    print(f'{len(index)} checkpoints')
    return 0


def cmd_play(args) -> int:
    import numpy as np
    import pandas as pd
    from .checkpoints import checkpoint_dir, tree_sha256
    from .env.reach_env import ReachBench
    from .env.tasks import find_task, load_manifest, task_arrays
    from .models.registry import load_policy
    from .registry import load_yaml, suite_path
    from .rollout import rollout
    out, manifest, scene, spec, profile = _run_context(args.run_id)
    ckpt = checkpoint_dir(out, args.checkpoint_step)                          # exactly this saved step, never "latest"
    cm = json.loads((ckpt / 'manifest.json').read_text())
    if tree_sha256(ckpt / 'policy_state') != cm['sha256']:
        print(f'ERROR: checkpoint {args.checkpoint_step} files do not match their recorded SHA256', file=sys.stderr)
        return 1
    split = args.task_id.split('_')[0]
    doc = load_manifest(suite_path(f'{split}_v1'))
    tasks = task_arrays(doc)
    index = tasks['task_id'].index(args.task_id)
    policy = load_policy(manifest['model'], ckpt / 'policy_state', spec)
    env = ReachBench(scene, 1, profile, spec, tasks, horizon=int(find_task(doc, args.task_id)['horizon_decisions']))
    seed = int(load_yaml('configs/eval/checkpoint_video_v1.yaml')['evaluation_seed'])
    roll = rollout(policy, env, [index], seed=seed)
    ee, goal = roll['ee'][0], roll['goal'][0]
    d = np.linalg.norm(ee - goal, axis=1)
    print(f'play {args.run_id} checkpoint step {args.checkpoint_step} (sha256 {cm["sha256"][:12]}) task {args.task_id}  seed {seed}')
    print(f'{"k":>4s} {"t_s":>6s} {"dist_m":>8s}  collisions clip')
    for k in sorted(set(range(0, len(d), max(1, len(d) // 10))) | {len(d) - 1}):
        c = int(roll['collision_proposed'][0, k - 1]) if k else 0
        p = int(roll['clip_proposed'][0, k - 1]) if k else 0
        print(f'{k:>4d} {k * roll["dt"]:>6.2f} {d[k]:>8.4f}  {c:>10d} {p:>4d}')
    hold = bool((d[-5:] <= 0.02).all())
    print(f'final error {d[-1]:.4f} m | terminal 5-step hold: {hold} | collision proposals {int(roll["collision_proposed"].sum())} | '
          f'pre-clip limit proposals {int(roll["clip_proposed"].sum())}')
    trace = out / 'play' / f'step_{args.checkpoint_step:09d}' / f'{args.task_id}.parquet'
    trace.parent.mkdir(parents=True, exist_ok=True)
    k = np.arange(len(d))
    pd.DataFrame({'k': k, 't_s': k * roll['dt'], 'joint_angles_rad': list(roll['q'][0]), 'end_effector_xyz_m': list(ee),
                  'goal_xyz_m': [list(goal)] * len(d), 'distance_m': d, 'coordinate_frame': 'robot0_base_link',
                  'backend': 'kinematic'}).to_parquet(trace, index=False)
    print(f'trace written to {trace.resolve()}')
    return 0


def cmd_video(args) -> int:
    from .checkpoint_video import read_video_index, verify_run
    from .checkpoints import read_index
    from .registry import find_run
    if args.action == 'verify':
        out = find_run(args.run_id)
        problems = verify_run(out)
        n = len(read_index(out))
        if problems:
            print(f'VIDEO VERIFICATION FAILED for {args.run_id}: {len(problems)} problem(s)')
            for p in problems:
                print('  -', p)
            return 1
        print(f'video verify ok: {n} checkpoints, {n} raw.mkv files readable, lossless, frame-exact, hashes match ({out})')
        return 0
    out, manifest, recorder = _recorder(args.run_id)
    try:
        if args.action == 'record':
            existing = read_video_index(out)
            has = len(existing[(existing['checkpoint_step'] == args.checkpoint_step) & (existing['status'] == 'ok')]) > 0
            if has and not args.force:
                result = recorder.reproduce(args.checkpoint_step)
                print(f'step {args.checkpoint_step}: a verified raw video exists and is NOT overwritten; independent re-render '
                      f'{"is pixel-identical" if result["identical"] else "DIFFERS from it: " + str(result["differing_frames"])}')
                return 0 if result['identical'] else 1
            row = recorder.record(args.checkpoint_step, force=args.force)
            if row['status'] != 'ok':
                print(f'ERROR: video capture failed: {row["error"]}', file=sys.stderr)
                return 1
            print(f'recorded {row["frame_count"]} frames -> {Path(row["video_path"]).resolve()}')
            return 0
        from .checkpoints import read_index as ri
        failed = 0
        for step in ri(out)['checkpoint_step'].astype(int):
            v = read_video_index(out)
            if len(v[(v['checkpoint_step'] == step) & (v['status'] == 'ok')]) and (Path(v[v['checkpoint_step'] == step].iloc[0]['video_path'])).exists():
                continue                                              # only absent or failed archives are regenerated
            row = recorder.record(int(step))
            print(f'backfill step {step}: {row["status"]}')
            failed += row['status'] != 'ok'
        return 1 if failed else 0
    finally:
        recorder.close()


def cmd_results(args) -> int:
    from .checkpoints import read_index
    from .registry import find_run
    out = find_run(args.run_id)
    manifest = json.loads((out / 'manifest.json').read_text())
    print(f'run_id     {args.run_id}\nmodel      {manifest["model"]}\nreward     {manifest["reward"]}\nstatus     {manifest["status"]}'
          f'\nroot       {out.resolve()}')
    for name in ('manifest.json', 'config_resolved.yaml', 'train_history.parquet', 'train_episodes.parquet',
                 'checkpoint_index.csv', 'video_index.csv'):
        print(f'{name:26s} {(out / name).resolve() if (out / name).exists() else "(absent)"}')
    print(f'checkpoints/{"":14s} {len(read_index(out))} saved  ({(out / "checkpoints").resolve()})')
    print(f'videos/{"":19s} {(out / "videos").resolve()}')
    print(f'eval/{"":21s} {(out / "eval").resolve() if (out / "eval").exists() else "(none yet)"}')
    return 0


def cmd_eval(args) -> int:
    import numpy as np
    from .checkpoints import read_index
    from .eval import evaluate_checkpoint, select_checkpoint
    from .registry import repo_root, suite_path
    suite = args.suite
    if args.manifest and Path(args.manifest).resolve() != suite_path(suite).resolve():
        print(f'ERROR: --manifest {args.manifest} is not the registered manifest {suite_path(suite)} of suite {suite}', file=sys.stderr)
        return 2
    out, manifest, scene, spec, profile = _run_context(args.run_id, keep_env=args.backend == 'physics')
    index = read_index(out)
    saved = [int(s) for s in index['checkpoint_step']]
    if args.all_checkpoints:
        steps = saved
    elif args.checkpoint_step is not None:
        steps = [args.checkpoint_step]
    elif args.final:
        steps = [max(saved)]
    else:
        if suite in ('test_v1', 'ood_v1'):
            steps = [select_checkpoint(out, 'kinematic')]       # selection uses the training backend's validation results only
            print(f'validation-selected checkpoint: step {steps[0]} (highest kinematic val terminal-hold rate; tie -> fewer unsafe '
                  'events -> later step)')
        else:
            steps = [max(saved)]
    for step in steps:
        metrics = evaluate_checkpoint(out, manifest, scene, spec, profile, step, suite, args.backend)
        unsafe = metrics['unsafe_events'].mean()
        print(f'{args.run_id} step {step:>9d} {suite}: n={len(metrics)}  terminal hold {metrics["terminal_hold"].mean():.3f}  '
              f'first entry {metrics["first_entry"].mean():.3f}  final error median {metrics["final_error_m"].median():.4f} m  '
              f'unsafe events/episode {unsafe:.3f}  HKS {metrics["hks_status"].iloc[0]}', flush=True)
    return 0


def cmd_analyse(args) -> int:
    import pandas as pd
    from .registry import find_run
    out = find_run(args.run_id)
    frames = [pd.read_parquet(p) for p in sorted((out / 'eval').glob('*/*/step_*/episode_metrics.parquet'))]
    if not frames:
        print(f'no evaluation tables under {out}/eval; run `python -m natural_rl eval --run-id {args.run_id} --suite val_v1 --all-checkpoints` first',
              file=sys.stderr)
        return 1
    df = pd.concat(frames, ignore_index=True)
    summary = (df.groupby(['backend', 'split', 'checkpoint_step'])
                 .agg(n_tasks=('task_id', 'count'), terminal_hold_rate=('terminal_hold', 'mean'), first_entry_rate=('first_entry', 'mean'),
                      any_hold_rate=('any_hold', 'mean'), final_error_median_m=('final_error_m', 'median'),
                      path_median_m=('path_m', 'median'), unsafe_events_per_episode=('unsafe_events', 'mean'),
                      hks_scored=('hks_status', lambda x: bool((x == 'scored').all())),
                      conditional_HKS=('HKS', lambda x: x.dropna().mean() if len(x.dropna()) else float('nan')),
                      R_HKS=('R_HKS', 'mean'), quality_flagged=('quality_flag', lambda x: int((x != '').sum())))
                 .reset_index())
    path = out / 'analysis' / 'summary.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(path, index=False)
    with pd.option_context('display.width', 200, 'display.max_columns', 20, 'display.float_format', '{:.4f}'.format):
        print(summary.to_string(index=False))
    print(f'summary written to {path.resolve()}')
    return 0


def cmd_calibrate(args) -> int:
    import hashlib
    import pandas as pd
    import numpy as np
    from .registry import find_run, metrics_config, repo_root
    if args.what == 'rewards':
        from .rewards.calibrate import CALIBRATED_TERMS, collect_pilot, lambda_formula, lock_register, table_sha256
        runs = [r for r in args.pilot_runs.split(',') if r]
        table = collect_pilot(runs, 'train_v1', args.steps)
        lambdas = lambda_formula(table['sum_r0'], {t: table[f'sum_{t}'] for t in CALIBRATED_TERMS})
        print(f'pilot: {len(table)} episodes from {len(runs)} run(s), train_v1 tasks only, checkpoints={args.steps}')
        print(f'median |episode r0 return| = {np.median(np.abs(table["sum_r0"])):.4f}')
        for t in CALIBRATED_TERMS:
            print(f'  {t:6s} P75(|episodic raw term|) = {np.percentile(np.abs(table[f"sum_{t}"]), 75):.6f}   lambda = {lambdas[t]:.6g}')
        if args.dry_run:
            print('dry run: nothing written, register not locked')
            return 0
        out = repo_root() / 'results' / 'calibration'
        out.mkdir(parents=True, exist_ok=True)
        table.to_parquet(out / 'reward_v1_pilot.parquet', index=False)
        code_sha = hashlib.sha256((repo_root() / 'natural_rl' / 'rewards' / 'calibrate.py').read_bytes()).hexdigest()
        lock_register(repo_root() / 'configs' / 'rewards' / 'reward_v1.yaml', lambdas, len(table), table_sha256(table), code_sha,
                      f'deterministic rollouts of runs {runs} on train_v1 tasks (checkpoints: {args.steps})')
        print('reward_v1.yaml is now LOCKED (immutable); commit it before any r2-r11 training')
        return 0
    from .metrics.hks import ScalerError, SCALED_METRICS, fit_scalers
    import yaml
    runs = [r for r in args.runs.split(',') if r]
    frames = []
    for run_id in runs:
        run = find_run(run_id)
        paths = sorted((run / 'eval' / args.backend / 'val_v1').glob('step_*/episode_metrics.parquet'))
        if not paths:
            print(f'no val_v1 evaluation for {run_id}: run `eval --run-id {run_id} --suite val_v1 --final` first', file=sys.stderr)
            return 1
        frames.append(pd.read_parquet(paths[-1]))                          # the final saved step of the run
    pooled = pd.concat(frames, ignore_index=True)
    usable = pooled[pooled['terminal_hold'] & pooled['valid']]
    cfg = metrics_config()['hks']
    print(f'pooled {len(pooled)} validation episodes from {len(runs)} run(s); {len(usable)} successful and quality-valid ({args.backend})')
    try:
        scalers = fit_scalers({m: usable[m].to_numpy() for m in SCALED_METRICS}, int(cfg['scaler_min_episodes']),
                               upper_ladder=tuple(cfg['scaler_upper_ladder']))
    except ScalerError as error:
        print(f'BLOCKING metric-calibration error: {error}', file=sys.stderr)
        return 1
    path = repo_root() / 'configs' / 'metrics' / 'hks_scalers_v1.yaml'
    doc = yaml.safe_load(path.read_text()) if path.exists() else {'backends': {}}
    if doc.get('status') == 'locked' and args.backend in doc['backends']:
        print(f'{path} already holds locked scalers for {args.backend}; a new benchmark id is needed to change them', file=sys.stderr)
        return 1
    doc.update(version='hks_scalers_v1', status='locked')
    doc['backends'][args.backend] = {**scalers, '_provenance': {'source_runs': runs, 'n_episodes_pooled': int(len(usable)),
                                                                 'selection': 'terminal_hold and quality-valid validation episodes, final checkpoint of each run'}}
    path.write_text(yaml.safe_dump(doc, sort_keys=False))
    print(f'locked HKS scalers for {args.backend} written to {path}')
    return 0


def cmd_export(args) -> int:
    import numpy as np
    import pandas as pd
    from .registry import find_run
    if args.kind != 'trajectories':
        print(f'natural_rl export --kind {args.kind}: not implemented yet (milestone M10); only numerical trajectories are exported',
              file=sys.stderr)
        return 2
    out = find_run(args.run_id)
    paths = sorted((out / 'eval').glob(f'*/{args.suite}/step_*/trajectories.parquet'))
    if not paths:
        print(f'no evaluated trajectories for suite {args.suite} under {out}/eval; run `eval --suite {args.suite}` first', file=sys.stderr)
        return 1
    target = out / 'exports' / 'trajectories'
    target.mkdir(parents=True, exist_ok=True)
    for path in paths:
        backend, step = path.parents[2].name, int(path.parent.name.split('_')[1])
        df = pd.read_parquet(path)
        flat = df[['task_id', 'k', 't_s', 'coordinate_frame', 'backend']].copy()
        flat['checkpoint_step'] = step
        for name, column, labels in (('q', 'joint_angles_rad', [f'q{j + 1}_rad' for j in range(len(df['joint_angles_rad'].iloc[0]))]),
                                     ('ee', 'end_effector_xyz_m', ['ee_x_m', 'ee_y_m', 'ee_z_m']),
                                     ('goal', 'goal_xyz_m', ['goal_x_m', 'goal_y_m', 'goal_z_m']),
                                     ('v', 'velocity_m_s', ['vx_m_s', 'vy_m_s', 'vz_m_s'])):
            values = np.stack(df[column].to_numpy())
            for i, label in enumerate(labels):
                flat[label] = values[:, i]
        stem = f'{backend}_{args.suite}_step_{step:09d}'
        flat.to_parquet(target / f'{stem}.parquet', index=False)
        flat.to_csv(target / f'{stem}.csv', index=False)
        print(f'exported {len(flat)} frames ({flat["task_id"].nunique()} tasks) -> {(target / stem).resolve()}.{{csv,parquet}}')
    return 0


# ----------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog='python -m natural_rl', description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)

    c = sub.add_parser('catalog', help='list models, rewards, suites, budgets or the training programme')
    c.add_argument('what', choices=['models', 'rewards', 'suites', 'budgets', 'programme'])
    c.add_argument('--phase')
    c.add_argument('--policies', help='P3 only: decqn,A,B,C')
    c.add_argument('--commands', action='store_true', help='print every registered train command')
    c.add_argument('--select', action='store_true', help='P3: choose the three controllers from the P2 validation evaluations')
    c.set_defaults(fn=cmd_catalog)

    v = sub.add_parser('verify-env', help='check libraries, simulator, manifests, rendering, FFV1 and disk')
    v.add_argument('--suite', default='train_v1')
    v.set_defaults(fn=cmd_verify_env)

    m = sub.add_parser('manifests', help='verify or (immutably) build the task manifests')
    m.add_argument('action', choices=['verify', 'build'])
    m.add_argument('--force', action='store_true')
    m.set_defaults(fn=cmd_manifests)

    t = sub.add_parser('train', help='train one run with automatic checkpoint videos')
    t.add_argument('--model', required=True)
    t.add_argument('--reward', required=True)
    t.add_argument('--suite', default='train_v1')
    t.add_argument('--steps', type=int, required=True, help='environment transitions (single-environment steps)')
    t.add_argument('--seed', type=int, default=0)
    t.add_argument('--run-id', required=True)
    t.add_argument('--checkpoint-every', type=int, default=50000)
    t.add_argument('--record-every-checkpoint', action='store_true')
    t.add_argument('--cost-budget-id')
    t.add_argument('--contract', default='bench_v1', choices=['bench_v1', 'hist_v1'])
    t.add_argument('--strict-video', action='store_true', help='abort at the first failed video instead of at the end')
    t.add_argument('--smoke', action='store_true', help='disposable run under results/smoke (never part of main_v1)')
    t.add_argument('--allow-dirty', action='store_true')
    t.add_argument('--threads', type=int, default=4)
    t.add_argument('--device', default='auto')
    t.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE', help='override model config values, e.g. hyper.warmup=500')
    t.set_defaults(fn=cmd_train)

    k = sub.add_parser('checkpoints', help='list saved checkpoints')
    k.add_argument('action', choices=['list'])
    k.add_argument('--run-id', required=True)
    k.set_defaults(fn=cmd_checkpoints)

    pl = sub.add_parser('play', help='run one exact saved checkpoint on a task (headless)')
    pl.add_argument('--run-id', required=True)
    pl.add_argument('--checkpoint-step', type=int, required=True)
    pl.add_argument('--task-id', required=True)
    pl.set_defaults(fn=cmd_play)

    vi = sub.add_parser('video', help='record, backfill or verify raw checkpoint videos')
    vi.add_argument('action', choices=['record', 'backfill', 'verify'])
    vi.add_argument('--run-id', required=True)
    vi.add_argument('--checkpoint-step', type=int)
    vi.add_argument('--suite', default='checkpoint_video_v1')
    vi.add_argument('--all-checkpoints', action='store_true')
    vi.add_argument('--force', action='store_true', help='record anew even if a verified video exists (never default)')
    vi.set_defaults(fn=cmd_video)

    r = sub.add_parser('results', help='locate the results of a run')
    r.add_argument('action', choices=['locate'])
    r.add_argument('--run-id', required=True)
    r.set_defaults(fn=cmd_results)

    ex = sub.add_parser('export', help='export numerical trajectories of evaluated checkpoints (flat CSV and parquet, units in names)')
    ex.add_argument('--run-id', required=True)
    ex.add_argument('--kind', required=True, choices=['trajectories', 'geometry', 'pointcloud'])
    ex.add_argument('--suite', default='test_v1')
    ex.set_defaults(fn=cmd_export)

    cal = sub.add_parser('calibrate', help='lock reward coefficients (rewards) or HKS scalers (scalers) from real pilot data')
    cal.add_argument('what', choices=['rewards', 'scalers'])
    cal.add_argument('--pilot-runs', default='', help='rewards: comma-separated run ids to roll out on train_v1')
    cal.add_argument('--steps', default='final', choices=['final', 'all'], help='rewards: which checkpoints of the pilot runs')
    cal.add_argument('--runs', default='', help='scalers: comma-separated run ids with a val_v1 evaluation')
    cal.add_argument('--backend', default='kinematic')
    cal.add_argument('--dry-run', action='store_true')
    cal.set_defaults(fn=cmd_calibrate)

    for name in NOT_YET:
        s = sub.add_parser(name, help='(not implemented yet)')
        s.add_argument('rest', nargs=argparse.REMAINDER)
        s.set_defaults(fn=lambda a, n=name: _unimplemented(n))
    e = sub.add_parser('eval', help='evaluate saved checkpoints on a task suite (deterministic)')
    e.add_argument('--run-id', required=True)
    e.add_argument('--suite', required=True)
    e.add_argument('--manifest')
    e.add_argument('--checkpoint-step', type=int)
    e.add_argument('--all-checkpoints', action='store_true')
    e.add_argument('--final', action='store_true', help='evaluate the final checkpoint instead of the validation-selected one')
    e.add_argument('--backend', default='kinematic', choices=['kinematic', 'physics'])
    e.set_defaults(fn=cmd_eval)

    a = sub.add_parser('analyse', help='summarise the evaluation tables of a run')
    a.add_argument('--run-id', required=True)
    a.set_defaults(fn=cmd_analyse)
    return p


def main(argv=None) -> int:
    parser = build_parser()
    args, extra = parser.parse_known_args(argv)
    if extra and args.command not in NOT_YET:
        parser.error('unrecognized arguments: ' + ' '.join(extra))
    if args.command == 'video' and args.action == 'record' and args.checkpoint_step is None:
        print('video record needs --checkpoint-step', file=sys.stderr)
        return 2
    return args.fn(args)
