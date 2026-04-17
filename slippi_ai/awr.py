"""Advantage-Weighted Regression for IL.

Reference: Peng et al. 2019, "Advantage-Weighted Regression: Simple and
Scalable Off-Policy Reinforcement Learning".

This module computes per-timestep weights from advantages, which are then
applied to the per-timestep negative log-likelihood in Policy.imitation_loss.
When AWR is disabled, the loss reduces exactly to plain BC.

Burn-in: the default burnin_steps=100000 targets a ~500k-step full IL run
(roughly 20% of total steps). For smoke tests, override to a small number
(e.g. 100) via flag to verify AWR turns on, or disable AWR entirely.
"""
import dataclasses

import tensorflow as tf


@dataclasses.dataclass
class AWRConfig:
  """Config for advantage-weighted regression during IL.

  When enabled=False (the default), behaves as a no-op and training is
  bit-identical to plain BC.
  """
  enabled: bool = False
  mode: str = 'exp'            # 'exp', 'linear', or 'filter'
  temperature: float = 1.0     # beta in exp(A / beta); lower = sharper
  clip: float = 20.0           # max weight after exp, to prevent blow-up
  normalize: bool = True       # subtract batch mean, divide by batch std
  burnin_steps: int = 100000   # steps before AWR weights kick in; see note below
  # For 'filter' mode: keep top-quantile transitions, drop the rest.
  filter_quantile: float = 0.5


def compute_weights(
    advantages: tf.Tensor,     # [T, B]
    config: AWRConfig,
    global_step: tf.Tensor,    # scalar int
) -> tf.Tensor:
  """Compute per-timestep AWR weights from advantages.

  Returns a tensor of shape [T, B] where each element weights that timestep's
  negative log-likelihood in the policy loss. When config.enabled is False or
  global_step < burnin_steps, returns all-ones (equivalent to plain BC).
  """
  # Always stop-gradient: AWR weights must not train the value head.
  advantages = tf.stop_gradient(advantages)

  ones = tf.ones_like(advantages)
  if not config.enabled:
    return ones

  # Burn-in: use plain BC weights until value head has trained enough.
  in_burnin = global_step < config.burnin_steps

  if config.normalize:
    # Batch-level normalization. Compute over all [T, B] elements.
    mean = tf.reduce_mean(advantages)
    std = tf.math.reduce_std(advantages) + 1e-6
    advantages = (advantages - mean) / std

  if config.mode == 'exp':
    weights = tf.exp(advantages / config.temperature)
    weights = tf.minimum(weights, config.clip)
  elif config.mode == 'linear':
    # Shift so min weight is 0, preserving relative ordering.
    weights = tf.maximum(advantages, 0.0)
  elif config.mode == 'filter':
    # Keep top (1 - filter_quantile) fraction. Threshold at the quantile.
    # Note: tf.stats.percentile is not available; use sort-based approx.
    flat = tf.reshape(advantages, [-1])
    k = tf.cast(
        tf.cast(tf.shape(flat)[0], tf.float32) * config.filter_quantile,
        tf.int32)
    threshold = tf.sort(flat)[k]
    weights = tf.cast(advantages > threshold, tf.float32)
  else:
    raise ValueError(f'Unknown AWR mode: {config.mode}')

  # Apply burn-in switch.
  weights = tf.where(in_burnin, ones, weights)
  return weights


def awr_metrics(weights: tf.Tensor, advantages: tf.Tensor) -> dict:
  """Metrics for logging AWR behavior during training."""
  advantages = tf.stop_gradient(advantages)
  return {
      'weight_mean': tf.reduce_mean(weights),
      'weight_max': tf.reduce_max(weights),
      'weight_min': tf.reduce_min(weights),
      'weight_std': tf.math.reduce_std(weights),
      'advantage_mean': tf.reduce_mean(advantages),
      'advantage_std': tf.math.reduce_std(advantages),
  }
