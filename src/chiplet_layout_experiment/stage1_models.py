"""Small reward CNN and disjoint LinUCB with explicit single-GPU accounting."""
from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import torch
from torch import nn


class SmallRewardCNN(nn.Module):
    """Preserve a 4x4 spatial grid before the 32-dimensional embedding."""

    def __init__(self, arms: int = 6):
        super().__init__()
        self.convolution = nn.Sequential(
            nn.Conv2d(4, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
        )
        self.embedding = nn.Sequential(nn.Linear(32 * 4 * 4, 32), nn.ReLU())
        self.head = nn.Linear(36, arms)

    def encode(self, maps):
        return self.embedding(self.convolution(maps))

    def forward(self, maps, scalars):
        return self.head(torch.cat((self.encode(maps), scalars), dim=1))


def allocator_budget_bytes(total_bytes: int, free_bytes: int) -> int:
    """Cap torch allocations below the smaller of physical/free/8 GiB budgets."""
    headroom = 1024 ** 3
    budget = min(8 * 1024 ** 3, total_bytes, free_bytes) - headroom
    if budget <= 0:
        raise ValueError("insufficient free GPU memory after 1 GiB headroom")
    return budget


def configure_device(device: str = "cuda") -> dict:
    """Apply allocator cap before model allocation; CUDA never falls back to CPU."""
    if device == "cpu":
        return {"device": "cpu", "gpu_training": False}
    if device != "cuda":
        raise ValueError("device must be 'cuda' or explicit test mode 'cpu'")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the formal 8GB GPU experiment")
    torch.cuda.set_device(0)
    free, total = torch.cuda.mem_get_info(0)
    cap = allocator_budget_bytes(total, free)
    torch.cuda.set_per_process_memory_fraction(cap / total, device=0)
    torch.cuda.reset_peak_memory_stats(0)
    return {
        "device": "cuda", "gpu_training": True,
        "gpu_name": torch.cuda.get_device_name(0),
        "total_bytes": total, "free_bytes_at_start": free,
        "allocator_cap_bytes": cap, "headroom_bytes": 1024 ** 3,
        "torch_version": torch.__version__, "torch_cuda": torch.version.cuda,
    }


@dataclass
class FittedCNN:
    model: SmallRewardCNN
    map_mean: np.ndarray
    map_std: np.ndarray
    scalar_mean: np.ndarray
    scalar_std: np.ndarray
    reward_mean: np.ndarray
    reward_std: np.ndarray
    audit: dict

    def predict(self, maps, scalars, batch_size=8):
        """Return original-scale reward and frozen [embedding, scalars, 1]."""
        if batch_size < 1 or len(maps) != len(scalars) or len(maps) == 0:
            raise ValueError("predict requires matching nonempty inputs and positive batch")
        device = next(self.model.parameters()).device
        predictions, features = [], []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(maps), batch_size):
                m = torch.as_tensor((maps[start:start+batch_size] - self.map_mean)
                                    / self.map_std, dtype=torch.float32, device=device)
                s = torch.as_tensor((scalars[start:start+batch_size] - self.scalar_mean)
                                    / self.scalar_std, dtype=torch.float32, device=device)
                z = self.model.encode(m)
                pred = self.model.head(torch.cat((z, s), dim=1)).cpu().numpy()
                predictions.append(pred * self.reward_std + self.reward_mean)
                feature = torch.cat((z, s, torch.ones((len(z), 1), device=device)), dim=1)
                features.append(feature.cpu().numpy())
        return np.concatenate(predictions), np.concatenate(features)


def fit_cnn(train_maps, train_scalars, train_rewards,
            validation_maps, validation_scalars, validation_rewards, *,
            device="cuda", epochs=100, batch_size=8, seed=11,
            learning_rate=1e-3, weight_decay=1e-4) -> FittedCNN:
    """Train only on train inputs; choose checkpoint using validation MSE."""
    start_time = time.perf_counter()
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    arrays = tuple(np.asarray(x, dtype=np.float32) for x in (
        train_maps, train_scalars, train_rewards,
        validation_maps, validation_scalars, validation_rewards))
    tm, ts, tr, vm, vs, vr = arrays
    for maps, scalars, rewards in ((tm, ts, tr), (vm, vs, vr)):
        if (maps.ndim != 4 or maps.shape[1] != 4 or scalars.ndim != 2
                or scalars.shape[1] != 4 or rewards.ndim != 2
                or len(maps) == 0 or len(maps) != len(scalars)
                or len(maps) != len(rewards)):
            raise ValueError("expected nonempty aligned maps[N,4,H,W], scalars[N,4], rewards[N,A]")
        if not all(np.isfinite(x).all() for x in (maps, scalars, rewards)):
            raise ValueError("training inputs must be finite")
    if tr.shape[1] != vr.shape[1]:
        raise ValueError("reward columns must share the same arm catalog")
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    audit = configure_device(device)
    sampled_device_used = (audit["total_bytes"]-audit["free_bytes_at_start"]
                           if device == "cuda" else 0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = SmallRewardCNN(tr.shape[1]).to(device)
    fitted = FittedCNN(
        model, tm.mean(axis=(0, 2, 3), keepdims=True),
        np.maximum(tm.std(axis=(0, 2, 3), keepdims=True), 1e-6),
        ts.mean(axis=0), np.maximum(ts.std(axis=0), 1e-6),
        tr.mean(axis=0), np.maximum(tr.std(axis=0), 1e-6), audit,
    )
    normalized_maps = (tm - fitted.map_mean) / fitted.map_std
    normalized_scalars = (ts - fitted.scalar_mean) / fitted.scalar_std
    normalized_rewards = (tr - fitted.reward_mean) / fitted.reward_std
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    rng = np.random.default_rng(seed)
    history, best, best_epoch, best_state = [], float("inf"), 0, None
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(tm))
        for start in range(0, len(tm), batch_size):
            indices = order[start:start+batch_size]
            m, s, r = (torch.as_tensor(x[indices], device=device) for x in (
                normalized_maps, normalized_scalars, normalized_rewards))
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(m, s), r)
            loss.backward()
            optimizer.step()
            if device == "cuda":
                free_bytes, total_bytes = torch.cuda.mem_get_info(0)
                sampled_device_used = max(sampled_device_used, total_bytes-free_bytes)
        prediction, _ = fitted.predict(vm, vs, batch_size)
        error = float(np.mean(((prediction - vr) / fitted.reward_std) ** 2))
        if not np.isfinite(error):
            raise RuntimeError("nonfinite validation loss")
        history.append(error)
        if error < best:
            best, best_epoch = error, epoch + 1
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if device == "cuda":
        torch.cuda.synchronize()
        audit.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
                     peak_reserved_bytes=torch.cuda.max_memory_reserved(0),
                     sampled_device_peak_used_bytes=sampled_device_used,
                     device_memory_sample_scope="whole device after optimizer steps; sampled, not continuous")
    audit.update(parameter_count=sum(p.numel() for p in model.parameters()),
                 batch_size=batch_size, epochs=epochs, seed=seed,
                 best_epoch=best_epoch, validation_mse=history,
                 train_states=len(tm), validation_states=len(vm),
                 elapsed_seconds=time.perf_counter()-start_time,
                 learning_rate=learning_rate, weight_decay=weight_decay)
    return fitted


class LinUCB:
    """Disjoint linear UCB; update accepts only one chosen action's reward."""

    def __init__(self, arms: int, dimensions: int, alpha: float = 0.1,
                 regularization: float = 1.0):
        if arms < 1 or dimensions < 1 or alpha < 0 or regularization <= 0:
            raise ValueError("invalid LinUCB dimensions or hyperparameters")
        self.A = np.repeat((regularization*np.eye(dimensions))[None], arms, axis=0)
        self.b = np.zeros((arms, dimensions))
        self.alpha = alpha

    def _context(self, context):
        x = np.asarray(context, dtype=float)
        if x.shape != self.b.shape[1:] or not np.isfinite(x).all():
            raise ValueError("invalid context shape or nonfinite context")
        return x

    def scores(self, context):
        x = self._context(context)
        scores = []
        for matrix, vector in zip(self.A, self.b):
            theta = np.linalg.solve(matrix, vector)
            uncertainty = np.sqrt(max(float(x @ np.linalg.solve(matrix, x)), 0.))
            scores.append(float(x @ theta) + self.alpha*uncertainty)
        return np.asarray(scores)

    def select(self, context):
        return int(np.argmax(self.scores(context)))

    def update(self, arm: int, context, reward: float):
        x = self._context(context)
        if not 0 <= arm < len(self.A) or not np.isfinite(reward):
            raise ValueError("invalid chosen arm or reward")
        self.A[arm] += np.outer(x, x)
        self.b[arm] += float(reward)*x
