from frozendict import frozendict
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from force.env.util import space_dim, is_standard_box
from force.nn import ConfigurableModule, Optimizer
from force.nn.models.transformer import DecoderOnlyTransformer
from force.nn.util import get_device, freepeat, torchify


BATCH_KEYS = {'rtgs', 'observations', 'actions', 'timesteps', 'masks'}
NUM_COMPONENTS = 3  # (rtg, obs, act) for each timestep


def _pad(x, H):
    # Check input
    if x.ndim == 1:
        x = x.unsqueeze(1)
        needs_squeeze = True
    elif x.ndim == 2:
        needs_squeeze = False
    else:
        raise ValueError('Too many dimensions')

    l, d = x.shape
    assert l <= H, f'Cannot pad tensor of sequence length {l} when history length is {H}'

    # Pad if necessary
    if l < H:
        kwargs = {'device': x.device, 'dtype': x.dtype}
        x = torch.cat([torch.zeros(H - l, d, **kwargs), x], dim=0)

    # Squeeze if necessary
    if needs_squeeze:
        x = x.squeeze(1)

    return x

def _padding_mask(l, H, device=None):
    return torch.cat([
        torch.ones(H - l, dtype=bool, device=device),
        torch.zeros(l, dtype=bool, device=device)
    ])


class DecisionTransformer(ConfigurableModule):
    class Config(ConfigurableModule.Config):
        transformer = DecoderOnlyTransformer.Config
        history_len = 20
        discount = 1.0
        max_timestep = 1000
        optimizer = Optimizer.Config
        grad_norm_clip = 0.25

    def __init__(self, cfg, obs_space, act_space, device=None):
        assert cfg.transformer.layer.num_heads == 1, 'Currently only single head supported'
        super().__init__(cfg)
        obs_dim = space_dim(obs_space)
        act_dim = space_dim(act_space)
        device = get_device(device)
        self.embed_dim = embed_dim = cfg.transformer.dim_model
        self.zero_action = torch.zeros_like(torchify(act_space.sample(), device=device))
        self.squash_actions = is_standard_box(act_space)
        self.H = H = cfg.history_len
        self.total_seq_len = NUM_COMPONENTS * H

        # Specify input and output shapes
        self._input_shape = frozendict({
            'rtgs': torch.Size([H]),
            'observations': torch.Size([H, obs_dim]),
            'actions': torch.Size([H, act_dim]),
            'timesteps': torch.Size([H]),
            'masks': torch.Size([H])
        })
        self._output_shape = torch.Size([H, act_dim])

        # Instantiate model components
        self.rtg_encoder = nn.Linear(1, embed_dim)
        self.obs_encoder = nn.Linear(obs_dim, embed_dim)
        self.action_encoder = nn.Linear(act_dim, embed_dim)
        self.position_embedding = nn.Embedding(cfg.max_timestep, embed_dim)
        self.embed_layer_norm = nn.LayerNorm(embed_dim)
        self.action_decoder = nn.Linear(embed_dim, act_dim)
        self.transformer = DecoderOnlyTransformer(cfg.transformer)

        self.attn_mask = torch.triu(
            torch.full((self.total_seq_len, self.total_seq_len), True,
                       dtype=bool, device=device),
            diagonal=1,
        )

        # Construct optimizer
        self.parameter_tuple = tuple(self.parameters())
        self.parameter_count = self.count_parameters()
        self.optimizer = Optimizer(cfg.optimizer, self.parameter_tuple)

    def forward(self, batch: dict):
        batch_size = len(batch['observations'])

        # Add position embedding
        position_encoding = self.position_embedding(batch['timesteps'])
        rtg = self.rtg_encoder(batch['rtgs'].unsqueeze(2)) + position_encoding
        obs = self.obs_encoder(batch['observations']) + position_encoding
        act = self.action_encoder(batch['actions']) + position_encoding

        # Create full sequence by interleaving the 3 component sequences
        components = [rtg, obs, act]
        stacked_seq = torch.stack(components, dim=1)
        stacked_seq = stacked_seq \
            .permute(0, 2, 1, 3) \
            .reshape(batch_size, self.total_seq_len, self.embed_dim)
        stacked_seq = self.embed_layer_norm(stacked_seq)

        # To make the attention mask fit the stacked inputs, have to stack it as well
        padding_mask = batch['masks']
        stacked_padding_mask = torch.stack([padding_mask for _ in components], dim=1) \
            .permute(0, 2, 1) \
            .reshape(batch_size, self.total_seq_len)

        # Run full sequence through the transformer
        output = self.transformer(
            stacked_seq,
            attn_mask=self.attn_mask,
            key_padding_mask=stacked_padding_mask
        )

        # Predict actions using indices 1, 4, 7, ..., 3i+1, ...
        # These correspond to encodings of the trajectory
        # (R_0, o_0, a_0, ..., R_i-1, o_i-1, a_i-1, R_i, o_i)
        # which notably excludes a_i
        actions = self.action_decoder(output[:,1::3])

        # Squash to [-1,1] if desired
        if self.squash_actions:
            actions = torch.tanh(actions)

        return actions

    def _prep(self, list_of_lists):
        """Truncates the history to history_len and pads if necessary"""
        ret = []
        for tensor_list in list_of_lists:
            if len(tensor_list) > self.H:
                tensor_list = tensor_list[-self.H:]
            tensor = torch.stack(tensor_list)
            ret.append(_pad(tensor, self.H))
        return torch.stack(ret)

    def act(self, rtgs, obs, acts):
        """Action selection is batched across envs. The arguments to act()
        are lists of lists of tensors, with the first list indexing over envs
        and the second indexing over time.
        (They are stored as lists instead of tensors for quick append.)
        """
        batch_size = len(rtgs)
        assert len(obs) == batch_size and len(acts) == batch_size

        # Append zero action. Transformer expects an action there,
        # but it doesn't exist yet at eval time. Note: this action padding
        # won't affect next action prediction due to causal mask.
        acts = [act_seq + [self.zero_action] for act_seq in acts]

        # Check sequence lengths match up
        history_len = len(rtgs[0])
        for list_of_lists in [rtgs, obs, acts]:
            for tensor_list in list_of_lists:
                assert len(tensor_list) == history_len

        # Determine timesteps and masks (same for all envs)
        seq_len = min(history_len, self.H)
        timesteps = _pad(torch.arange(history_len)[-self.H:], self.H)
        mask = _padding_mask(seq_len, self.H, device=self.device)

        # Prepare inputs
        batch = {
            'rtgs': self._prep(rtgs),
            'observations': self._prep(obs),
            'actions': self._prep(acts),
            'timesteps': freepeat(timesteps, batch_size, 0),
            'masks': freepeat(mask, batch_size, 0)
        }
        batch = {k: v.to(self.device) for k, v in batch.items()}

        # Forward pass
        action_predictions = self(batch)

        # We have predictions for all actions in the sequence (history),
        # but only need the most recent
        return action_predictions[:,-1,:]

    def update(self, batch):
        batch = {k: v.to(self.device) for k, v in batch.items()}
        action_predictions = self(batch)
        action_targets = batch['actions']
        unmasked = ~batch['masks']
        relevant_predictions = action_predictions[unmasked]
        relevant_targets = action_targets[unmasked]
        loss = F.mse_loss(relevant_predictions, relevant_targets)
        assert not loss.isnan()
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameter_tuple, self.cfg.grad_norm_clip,
                                 error_if_nonfinite=True)
        self.optimizer.step()
        return loss.item()


class DTSampler:
    def __init__(self, trajectories, seq_len, device):
        self.trajectories = trajectories
        self.seq_len = seq_len
        self.device = device

    def sample(self):
        traj = random.choice(self.trajectories)
        T = len(traj)
        start_t = random.randrange(T)
        end_t = min(start_t + self.seq_len, T)
        slice = traj[start_t:end_t]
        slice_len = end_t - start_t
        assert 1 <= slice_len <= self.seq_len

        # Sample before (possibly) padding front with zeros
        sample = {
            'rtgs': slice['rtgs'],
            'observations': slice['observations'],
            'actions': slice['actions'],
            'timesteps': torch.arange(start_t, end_t, device=traj.device)
        }

        # Pad
        for k in sample.keys():
            v = sample[k]
            needs_squeeze = False
            if v.ndim == 1:
                v = v.unsqueeze(1)
                needs_squeeze = True
            v = _pad(v, self.seq_len)
            if needs_squeeze:
                v = v.squeeze(1)
            sample[k] = v

        # Mask
        sample['masks'] = _padding_mask(slice_len, self.seq_len, device=traj.device)

        assert set(sample.keys()) == BATCH_KEYS
        return sample


def evaluate_dt(eval_envs, dt, target_return):
    dt.eval()
    n_envs = len(eval_envs)

    # Keeping track
    all_rtgs = [[] for _ in range(n_envs)]
    all_obs = [[] for _ in range(n_envs)]
    all_acts = [[] for _ in range(n_envs)]
    active_indices = set(range(n_envs))
    total_rewards = torch.zeros(n_envs)

    for i, env in enumerate(eval_envs):
        obs, info = env.reset()
        all_obs[i].append(obs)
        all_rtgs[i].append(torch.tensor(target_return))

    while len(active_indices) > 0:
        # Get actions for all active envs
        with torch.no_grad():
            actions = dt.act(
                [all_rtgs[i] for i in active_indices],
                [all_obs[i] for i in active_indices],
                [all_acts[i] for i in active_indices]
            )

        # Step each env individually
        done_indices = set()
        for act_i, env_i in enumerate(tuple(active_indices)):
            action = actions[act_i]
            all_acts[env_i].append(action)
            next_obs, reward, terminated, truncated, info = eval_envs[env_i].step(action)
            total_rewards[env_i] += reward
            if terminated or truncated:
                done_indices.add(env_i)
            else:
                all_rtgs[env_i].append(all_rtgs[env_i][-1] - reward)
                all_obs[env_i].append(next_obs)
        active_indices -= done_indices

    dt.train()
    return total_rewards