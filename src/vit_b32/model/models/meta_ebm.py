import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MetaEBM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        task_embed_dim: int,
        pool_length: int = 1024,
        hidden_dim: int = 256,
        task_hidden_dim: int = 128,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.task_embed_dim = int(task_embed_dim)
        self.pool_length = int(pool_length)

        self.vector_proj = nn.Sequential(
            nn.Linear(self.pool_length, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.task_proj = nn.Sequential(
            nn.Linear(self.task_embed_dim, task_hidden_dim),
            nn.LayerNorm(task_hidden_dim),
            nn.GELU(),
        )
        fusion_dim = hidden_dim + task_hidden_dim
        self.energy_head = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, max(64, hidden_dim // 2)),
            nn.GELU(),
            nn.Linear(max(64, hidden_dim // 2), 1),
        )

    def _pool_vector(self, v: torch.Tensor) -> torch.Tensor:
        if v.dim() == 3:
            v = v[:, 0, :]
        if v.dim() != 2:
            raise ValueError(f"Expected v to have shape [B, D], got {tuple(v.shape)}")

        if v.size(1) % self.pool_length != 0:
            pad = self.pool_length - (v.size(1) % self.pool_length)
            v = F.pad(v, (0, pad), mode="constant", value=0.0)
        chunk_size = v.size(1) // self.pool_length
        return v.view(v.size(0), self.pool_length, chunk_size).mean(dim=-1)

    def energy(self, v: torch.Tensor, task_embed: torch.Tensor) -> torch.Tensor:
        pooled = self._pool_vector(v)
        h_v = self.vector_proj(pooled)
        h_t = self.task_proj(task_embed)
        return self.energy_head(torch.cat([h_v, h_t], dim=-1)).view(-1)

    def forward(self, v: torch.Tensor, task_embed: torch.Tensor) -> torch.Tensor:
        return self.energy(v, task_embed)

    def grad_v(self, v: torch.Tensor, task_embed: torch.Tensor, create_graph: bool = False) -> torch.Tensor:
        x = v.detach().requires_grad_(True)
        energy = self.energy(x, task_embed)
        return torch.autograd.grad(energy.sum(), x, create_graph=create_graph)[0]

    def refine(
        self,
        v_init: torch.Tensor,
        task_embed: torch.Tensor,
        steps: int = 5,
        step_size: float = 1e-3,
        grad_clip: float = 0.0,
        beta: float = 1.0,
        langevin_noise_std: float = 0.0,
    ) -> torch.Tensor:
        with torch.enable_grad():
            v = v_init.to(task_embed.device)
            for _ in range(max(0, int(steps))):
                v = v.detach().requires_grad_(True)
                energy = float(beta) * self.energy(v, task_embed)
                grad = torch.autograd.grad(energy.sum(), v, create_graph=False)[0]
                if grad_clip > 0:
                    grad = grad.clamp(-grad_clip, grad_clip)
                v = (v - step_size * grad).detach()
                if langevin_noise_std > 0.0:
                    noise_scale = float(langevin_noise_std) * math.sqrt(max(2.0 * step_size / max(float(beta), 1e-12), 0.0))
                    v = (v + torch.randn_like(v) * noise_scale).detach()
        return v
