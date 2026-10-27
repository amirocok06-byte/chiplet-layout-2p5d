"""Case-held-out CNN training and finite-arm offline selection experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch

from .stage1_models import fit_cnn, LinUCB


def selection_report(rewards, selected):
    rewards = np.asarray(rewards)
    selected = np.asarray(selected, dtype=int)
    chosen = rewards[np.arange(len(rewards)), selected]
    oracle = rewards.max(axis=1)
    return {
        "mean_reward": float(chosen.mean()),
        "mean_oracle_reward": float(oracle.mean()),
        "mean_regret": float((oracle-chosen).mean()),
        "set_top1": float(np.isclose(chosen, oracle, rtol=1e-6, atol=1e-8).mean()),
    }


def _stack(dataset):
    return tuple(np.stack([getattr(s, name) for s in dataset.states])
                 for name in ("maps", "scalars", "rewards"))


def main(argv=None):
    from .stage1_data import load_stage1_dataset, split_by_case
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    for name in ("train", "validation", "test"):
        parser.add_argument(f"--{name}-cases", nargs="+", required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output directory must be new")
    dataset = load_stage1_dataset(args.labels_root, grid_size=64)
    train, validation, test = split_by_case(
        dataset, train_cases=args.train_cases,
        validation_cases=args.validation_cases, test_cases=args.test_cases)
    tm, ts, tr = _stack(train)
    vm, vs, vr = _stack(validation)
    qm, qs, qr = _stack(test)
    fitted = fit_cnn(tm, ts, tr, vm, vs, vr, device=args.device,
                     epochs=args.epochs, batch_size=args.batch_size, seed=args.seed)
    _, train_x = fitted.predict(tm, ts)
    _, validation_x = fitted.predict(vm, vs)
    if args.device == "cuda":
        torch.cuda.synchronize()
    prediction_start = time.perf_counter()
    test_pred, test_x = fitted.predict(qm, qs)
    if args.device == "cuda":
        torch.cuda.synchronize()
    prediction_seconds = time.perf_counter() - prediction_start
    # Feature scaling and reward scaling use train only, with a common reward
    # scale across arms so UCB scores remain directly comparable.
    xmean, xstd = train_x.mean(0), np.maximum(train_x.std(0), 1e-6)
    xmean[-1], xstd[-1] = 0., 1.
    train_x, validation_x, test_x = [(x-xmean)/xstd for x in (train_x, validation_x, test_x)]
    rmean, rstd = float(tr.mean()), max(float(tr.std()), 1e-6)

    def replay(xvalues, rewards, alpha):
        bandit = LinUCB(len(dataset.arm_ids), train_x.shape[1], alpha=alpha)
        for x, reward_vector in zip(train_x, tr):
            for arm, reward in enumerate(reward_vector):
                bandit.update(arm, x, (reward-rmean)/rstd)
        chosen = []
        for x, reward_vector in zip(xvalues, rewards):
            arm = bandit.select(x)
            chosen.append(arm)
            bandit.update(arm, x, (reward_vector[arm]-rmean)/rstd)
        return np.asarray(chosen)

    candidates = (0., 0.1, 0.5, 1.)
    validation_regrets = {
        alpha: selection_report(vr, replay(validation_x, vr, alpha))["mean_regret"]
        for alpha in candidates}
    alpha = min(candidates, key=validation_regrets.get)
    # Scalar Ridge baseline has its own train-only transform and validation
    # regularization choice, independent of the CNN embedding.
    scalar_mean, scalar_std = ts.mean(0), np.maximum(ts.std(0), 1e-6)
    def scalar_features(s):
        return np.column_stack(((s-scalar_mean)/scalar_std, np.ones(len(s))))
    tx, vx, qx = [scalar_features(s) for s in (ts, vs, qs)]
    ridge_choices = []
    for penalty in (0.1, 1., 10.):
        reg = penalty*np.eye(tx.shape[1])
        reg[-1, -1] = 0.
        weights = np.linalg.solve(tx.T@tx+reg, tx.T@tr)
        ridge_choices.append((float(np.mean((vx@weights-vr)**2)), penalty, weights))
    _, ridge_penalty, ridge_weights = min(ridge_choices, key=lambda row: row[0])
    selections = {
        "fixed": np.full(len(qr), int(np.argmax(tr.mean(0)))),
        "random": np.random.default_rng(args.seed).integers(len(dataset.arm_ids), size=len(qr)),
        "scalar_ridge": np.argmax(qx@ridge_weights, axis=1),
        "cnn_greedy": np.argmax(test_pred, axis=1),
        "cnn_linucb": replay(test_x, qr, alpha),
        "cnn_linucb_alpha0": replay(test_x, qr, 0.),
    }
    summary = {name: selection_report(qr, chosen) for name, chosen in selections.items()}
    for name, chosen in selections.items():
        summary[name]["mean_historical_action_seconds"] = float(np.mean([
            state.costs[arm] for state, arm in zip(test.states, chosen)]))
        summary[name]["by_case"] = {}
        for case in args.test_cases:
            indices = np.asarray([i for i, state in enumerate(test.states) if state.case_group == case])
            if len(indices):
                summary[name]["by_case"][case] = selection_report(qr[indices], chosen[indices])
    summary["cnn_prediction"] = {
        "per_arm_mae": np.mean(np.abs(test_pred-qr), axis=0).tolist(),
        "per_arm_mse": np.mean((test_pred-qr)**2, axis=0).tolist(),
    }
    config = vars(args).copy()
    config = {key: str(value) if isinstance(value, Path) else value for key, value in config.items()}
    label_hashes = {str(path.relative_to(args.labels_root)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in sorted(args.labels_root.glob("*/labels.jsonl"))}
    input_paths = sorted(args.labels_root.glob("*/problem.json")) + sorted(args.labels_root.glob("*/states/*.json"))
    input_hashes = {str(path.relative_to(args.labels_root)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in input_paths}
    source_paths = list(Path(__file__).parent.glob("stage1*.py")) + [
        Path(__file__).with_name(name) for name in ("state_projection.py", "trajectory_problem.py")]
    audit = dict(fitted.audit, alpha=alpha, validation_regrets=validation_regrets,
                 test_prediction_seconds=prediction_seconds,
                 ridge_penalty=ridge_penalty, python=sys.version, platform=platform.platform(),
                 split_case_disjoint=True, label_sha256=label_hashes, input_sha256=input_hashes,
                 split_states={name: [dict(case=state.case_group, state_id=state.state_id,
                                          geometry_version=state.geometry_version)
                                      for state in partition.states]
                               for name, partition in (("train", train), ("validation", validation), ("test", test))},
                 evaluation="offline full-information state sequence; selected reward updates only",
                 source_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in source_paths})
    args.output.mkdir(parents=True)
    for filename, value in (("config.json", config), ("summary.json", summary), ("audit.json", audit)):
        (args.output/filename).write_text(json.dumps(value, indent=2), encoding="utf-8")
    records = []
    for name, choices in selections.items():
        for i, (state, choice) in enumerate(zip(test.states, choices)):
            records.append(dict(method=name, state_id=state.state_id, case=state.case_group,
                                arm_id=dataset.arm_ids[choice], reward=float(qr[i, choice]),
                                oracle=float(qr[i].max()), evaluator_calls=state.budget,
                                historical_action_seconds=float(state.costs[choice])))
    (args.output/"decisions.jsonl").write_text(
        "".join(json.dumps(record)+"\n" for record in records), encoding="utf-8")
    report = ["# E2a 单切分实现验收", "",
              f"设备：{fitted.audit['device']}；train/validation/test 状态数："
              f"{len(tr)}/{len(vr)}/{len(qr)}；seed={args.seed}。", "",
              "结果为冻结状态序列上的离线一步选择；多 seed 与外层 Case 验证尚需单列。", "",
              "| 方法 | 平均收益 | 平均 regret | 集合 top-1 | 历史动作成本(s) |",
              "| --- | --- | --- | --- | --- |"]
    for name in selections:
        result = summary[name]
        report.append(f"| {name} | {result['mean_reward']:.6g} | "
                      f"{result['mean_regret']:.6g} | {result['set_top1']:.4f} | "
                      f"{result['mean_historical_action_seconds']:.6g} |")
    if args.device == "cuda":
        report += ["", f"PyTorch 峰值 allocated/reserved："
                   f"{audit['peak_allocated_bytes']/1024**2:.2f}/"
                   f"{audit['peak_reserved_bytes']/1024**2:.2f} MiB；"
                   f"allocator 上限 {audit['allocator_cap_bytes']/1024**2:.2f} MiB。"]
        report += [f"整卡已用显存的逐 batch 采样峰值："
                   f"{audit['sampled_device_peak_used_bytes']/1024**2:.2f} MiB"
                   "（包含其他进程，不等同连续监控峰值）。"]
    (args.output/"RESULTS.md").write_text("\n".join(report)+"\n", encoding="utf-8")
    # Save transforms and arm order alongside weights, sufficient for inference.
    torch.save({"state_dict": fitted.model.state_dict(), "arm_ids": dataset.arm_ids,
                "transforms": {key: torch.from_numpy(getattr(fitted, key)) for key in (
                    "map_mean", "map_std", "scalar_mean", "scalar_std", "reward_mean", "reward_std")}},
               args.output/"cnn.pt")
    print(json.dumps({"output": str(args.output), "audit": audit, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
