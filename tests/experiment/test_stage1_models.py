"""Also runnable with unittest in the existing PyTorch environment."""
import importlib.util
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional stage1 torch dependency")
class Stage1ModelsTest(unittest.TestCase):
    def test_cnn_outputs_six_rewards_and_keeps_spatial_embedding(self):
        import torch
        from chiplet_layout_experiment.stage1_models import SmallRewardCNN
        model = SmallRewardCNN()
        maps = torch.randn(3, 4, 64, 64)
        scalars = torch.randn(3, 4)
        self.assertEqual(tuple(model(maps, scalars).shape), (3, 6))
        self.assertEqual(tuple(model.encode(maps).shape), (3, 32))
        self.assertLess(sum(p.numel() for p in model.parameters()), 50000)
        model(maps, scalars).square().mean().backward()
        self.assertTrue(all(p.grad is not None for p in model.parameters()))

    def test_linucb_updates_only_selected_arm(self):
        from chiplet_layout_experiment.stage1_models import LinUCB
        bandit = LinUCB(arms=6, dimensions=3, alpha=0.2)
        x = np.array([1., 0., 1.])
        before_a, before_b = bandit.A.copy(), bandit.b.copy()
        bandit.update(2, x, 0.7)
        np.testing.assert_array_equal(bandit.A[[0, 1, 3, 4, 5]], before_a[[0, 1, 3, 4, 5]])
        np.testing.assert_array_equal(bandit.b[[0, 1, 3, 4, 5]], before_b[[0, 1, 3, 4, 5]])
        np.testing.assert_allclose(bandit.b[2], 0.7 * x)
        self.assertEqual(bandit.select(x), 2)

    def test_memory_budget_leaves_headroom_on_actual_8gb_class_card(self):
        from chiplet_layout_experiment.stage1_models import allocator_budget_bytes
        mib = 1024 ** 2
        self.assertEqual(allocator_budget_bytes(8188*mib, 8188*mib), 7164*mib)
        self.assertEqual(allocator_budget_bytes(8188*mib, 8000*mib), 6976*mib)
        with self.assertRaises(ValueError):
            allocator_budget_bytes(8188*mib, 512*mib)

    def test_cuda_required_does_not_silently_use_cpu(self):
        import torch
        from chiplet_layout_experiment.stage1_models import configure_device
        if not torch.cuda.is_available():
            with self.assertRaisesRegex(RuntimeError, "CUDA"):
                configure_device("cuda")

    def test_training_and_inference_return_finite_original_scale_rewards(self):
        from chiplet_layout_experiment.stage1_models import fit_cnn
        rng = np.random.default_rng(8)
        maps = rng.normal(size=(8, 4, 16, 16)).astype("float32")
        scalars = rng.normal(size=(8, 4)).astype("float32")
        rewards = rng.normal(size=(8, 6)).astype("float32")
        model = fit_cnn(maps[:6], scalars[:6], rewards[:6],
                        maps[6:], scalars[6:], rewards[6:],
                        device="cpu", epochs=3, batch_size=2, seed=11)
        pred, features = model.predict(maps[6:], scalars[6:])
        self.assertEqual(pred.shape, (2, 6))
        self.assertEqual(features.shape, (2, 37))
        self.assertTrue(np.isfinite(pred).all())
        self.assertEqual(model.audit["device"], "cpu")
        self.assertEqual(len(model.audit["validation_mse"]), 3)
        np.testing.assert_allclose(model.map_mean, maps[:6].mean((0, 2, 3), keepdims=True))
        np.testing.assert_allclose(model.reward_mean, rewards[:6].mean(0))

    def test_training_can_execute_under_the_real_cuda_budget(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest("real CUDA validation requires a CUDA-enabled environment")
        from chiplet_layout_experiment.stage1_models import fit_cnn
        rng = np.random.default_rng(31)
        maps = rng.normal(size=(6, 4, 64, 64)).astype("float32")
        scalars = rng.normal(size=(6, 4)).astype("float32")
        rewards = rng.normal(size=(6, 6)).astype("float32")
        result = fit_cnn(maps[:4], scalars[:4], rewards[:4],
                         maps[4:], scalars[4:], rewards[4:], epochs=2)
        self.assertGreater(result.audit["peak_allocated_bytes"], 0)
        self.assertLessEqual(result.audit["peak_reserved_bytes"], result.audit["allocator_cap_bytes"])
        self.assertLess(result.audit["allocator_cap_bytes"], 8 * 1024**3)

    def test_selection_report_counts_ties_and_original_scale_regret(self):
        from chiplet_layout_experiment.stage1_cnn_train import selection_report
        result = selection_report(np.array([[2., 2., 1.], [1., 4., 2.]]),
                                  np.array([1, 2]))
        self.assertEqual(result["set_top1"], 0.5)
        self.assertEqual(result["mean_regret"], 1.)

    def test_cpu_cli_saves_case_split_and_replay_audit(self):
        from chiplet_layout_experiment.stage1_cnn_train import main
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/"labels"
            for case in ("Case1", "Case2", "Case3"):
                directory = root/case
                (directory/"states").mkdir(parents=True)
                (directory/"problem.json").write_text(json.dumps({
                    "names": ["a", "b"], "board_mm": [10., 10.],
                    "sizes": [[.1, .1], [.1, .1]], "powers": [2., 3.],
                    "pin_indices": [[0, 1]], "pin_offsets": [[[0., 0.], [0., 0.]]],
                    "hpwl_scale": 8., "thermal_scale": 3.,
                }))
                (directory/"states/state.json").write_text(json.dumps({
                    "positions": [[.1, .1], [.7, .7]],
                }))
                rows = []
                for arm, reward in (("sa-local", .1), ("analytical-small", .2)):
                    for seed in (11, 12):
                        rows.append(dict(
                            state_id="s", phase="legal_optimization", arm_id=arm,
                            seed=seed, budget=65, candidate_budget=64, evaluator_calls=65,
                            state_metadata=dict(case=case, case_split_group=case,
                                                state_file="states/state.json"),
                            input_metrics=dict(hpwl_mm=8., thermal_proxy_peak=3.,
                                               overlap_fraction=0., valid=True),
                            bandit_reward=reward, total_action_seconds=.01))
                (directory/"labels.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows))
            output = Path(temp)/"run"
            with contextlib.redirect_stdout(io.StringIO()):
                main(["--labels-root", str(root), "--output", str(output),
                      "--train-cases", "Case1", "--validation-cases", "Case2",
                      "--test-cases", "Case3", "--device", "cpu", "--epochs", "1"])
            audit = json.loads((output/"audit.json").read_text())
            self.assertTrue(audit["split_case_disjoint"])
            self.assertEqual(audit["device"], "cpu")
            self.assertEqual(len((output/"decisions.jsonl").read_text().splitlines()), 6)
            self.assertTrue((output/"cnn.pt").is_file())
            self.assertTrue((output/"RESULTS.md").is_file())
            import torch
            saved = torch.load(output/"cnn.pt", map_location="cpu", weights_only=True)
            self.assertEqual(len(saved["arm_ids"]), 2)
            self.assertEqual(tuple(saved["transforms"]["map_mean"].shape), (1, 4, 1, 1))


if __name__ == "__main__":
    unittest.main()
