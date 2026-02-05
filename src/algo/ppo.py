# PPO loss + update step (clip obj, value loss, entropy, KL, logging)

from __future__ import annotations

from typing import Mapping
import torch
from src.models.base import ActorCritic
from src.types import MiniBatch, PPOCfg


class PPOLearner:
    def __init__(
        self, model: ActorCritic, optim: torch.optim.Optimizer, config: PPOCfg
    ) -> None:
        self.model = model
        self.optim = optim
        self.clip_coef = config.clip_coef
        self.ent_coef = config.ent_coef
        self.vf_coef = config.vf_coef
        self.max_grad_norm = config.max_grad_norm

    def update(self, mini_batch: MiniBatch) -> Mapping[str, float]:
        # Forward pass on the minibatch
        logp, entropy, value = self.model.evaluate_actions(
            mini_batch.obs, mini_batch.actions
        )
        # PPO policy objective (clipped surrogate)
        ratio = torch.exp(logp - mini_batch.logp_old)
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
        objective = torch.min(
            ratio * mini_batch.advantages_norm,
            clipped_ratio * mini_batch.advantages_norm,
        )
        policy_loss = -objective.mean()
        # Value loss (+ optional value clipping)
        value_clipped = mini_batch.values_old + torch.clamp(
            value - mini_batch.values_old, -self.clip_coef, self.clip_coef
        )
        value_loss_unclipped = (value - mini_batch.returns).pow(2)
        value_loss_clipped = (value_clipped - mini_batch.returns).pow(2)
        value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
        # Entropy bonus
        entropy_mean = entropy.mean()
        loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy_mean
        # Backward + optimizer step + grad clipping
        self.optim.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.max_grad_norm
        )
        if not torch.isfinite(grad_norm):  # (optional) NaN/Inf guard
            raise RuntimeError(f"Non-finite grad norm: {grad_norm}")
        self.optim.step()
        # Metrics to compute and return (log “dashboard”)
        with torch.no_grad():

            def to_scalar(x):
                if isinstance(x, torch.Tensor):
                    return x.detach().item()
                return float(x)

            metrics = {}
            # losses
            metrics["policy_loss"] = to_scalar(policy_loss)
            metrics["value_loss"] = to_scalar(value_loss)
            metrics["entropy"] = to_scalar(entropy_mean)
            metrics["total_loss"] = to_scalar(loss)
            # PPO health
            metrics["approx_kl"] = to_scalar((mini_batch.logp_old - logp).mean())
            metrics["clipfrac"] = to_scalar(
                ((ratio - 1.0).abs() > self.clip_coef).float().mean()
            )
            metrics["ratio_mean"] = to_scalar(ratio.mean())
            # value fit
            metrics["explained_variance"] = to_scalar(
                (
                    1.0
                    - (value - mini_batch.values_old).var()
                    / mini_batch.values_old.var()
                )
            )
            # stability
            metrics["grad_norm"] = to_scalar(grad_norm)
            metrics["lr"] = to_scalar(self.optim.param_groups[0]["lr"])
            return metrics

    def state_dict(self) -> dict:
        raise NotImplementedError

    def load_state_dict(self, state: dict) -> None:
        raise NotImplementedError
