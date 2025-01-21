import math
import random

import numpy as np
import torch
from einops import rearrange
from torch.nn import functional as F


def add_gumbel_noise(t, temperature, device):
    return t + torch.Tensor(temperature * np.random.gumbel(size=t.shape)).to(device)


class NATSchedule(object):
    def __init__(
        self,
        codebook_size,
        device,
        ignore_ind=-1,
        smoothing=0.0,
        beta_alpha_beta=(12, 3),
    ):
        self.mask_ind = codebook_size  # for input masking
        self.ignore_ind = ignore_ind  # for ce loss, excluding visible
        self.device = device
        self.smoothing = smoothing
        self.beta_a, self.beta_b = beta_alpha_beta

    @staticmethod
    def cosine_schedule(t):
        return torch.cos(t * math.pi * 0.5)

    def sample(self, x0):
        N, L, device = *x0.shape, self.device
        beta_dist = torch.distributions.Beta(self.beta_a, self.beta_b)
        rand_mask_probs = beta_dist.sample((N,)).to(device).float()
        num_token_masked = (L * rand_mask_probs).round().clamp(min=1)
        batch_randperm = torch.rand(N, L, device=device).argsort(dim=-1)
        mask = batch_randperm < rearrange(num_token_masked, "b -> b 1")
        masked_ids = torch.where(mask, self.mask_ind, x0)
        labels = torch.where(mask, x0, self.ignore_ind)
        return None, labels, masked_ids  # timestep is not needed for nnet

    def loss(self, pred, label):  # pred: N, L, C
        return F.cross_entropy(
            pred.transpose(1, 2),
            label.long(),
            ignore_index=self.ignore_ind,
            label_smoothing=self.smoothing,
        )

    def create_random_index_groups(self, seq_len, n_groups):
        """
        Creates randomly distributed groups of indices.

        Args:
            seq_len: The total number of indices (e.g., length of a sequence).
            n_groups: The desired number of groups.

        Returns:
            A dictionary where keys are group indices (0 to n_groups-1) and values
            are lists of randomly assigned indices.
            Returns None if n_groups is greater than seq_len
        """

        if n_groups > seq_len:
            return None
        all_indices = list(range(seq_len))
        random.shuffle(all_indices)  # Shuffle the indices randomly

        group2indices = {}
        indices_per_group = seq_len // n_groups
        remainder = seq_len % n_groups

        start_index = 0
        for g in range(n_groups):
            num_indices_in_group = indices_per_group + (1 if g < remainder else 0)
            group2indices[g] = all_indices[
                start_index : start_index + num_indices_in_group
            ]
            start_index += num_indices_in_group

        return group2indices

    @torch.no_grad()
    def generate(
        self,
        gen_steps,
        _n_samples,
        nnet,
        decode_fn,
        manual_ratios,
        manual_temp,
        manual_samp_temp,
        manual_cfg,
        **kwargs
    ):
        device = self.device

        fmap_size = 16
        seq_len = fmap_size * fmap_size

        mysampler = False
        if mysampler:
            samp_temp = 1.0
            vocab_size = 1024
            cfg_scale = manual_cfg[0]

            # Sample the initial set of ids randomly
            # ids = torch.randint(0, vocab_size, (_n_samples, seq_len), device=device)

            # Sample the initial set of ids by taking the argmax of an empty image.
            ids = torch.full(
                (_n_samples, seq_len), self.mask_ind, dtype=torch.long, device=device
            )
            logits = nnet(ids, **kwargs, scale=cfg_scale)
            ids = torch.distributions.Categorical(
                logits=logits / max(samp_temp, 1e-4)
            ).sample()
            # Because this initialization requires inference, decrement the number of remaining steps
            gen_steps = gen_steps - 1

            n_groups = 2
            spacing = 2  # 1

            scan_order = "grid"
            if scan_order == "random":
                # Assign groups randomly
                group2indices = self.create_random_index_groups(seq_len, n_groups)
            else:
                # Assign groups in a grid
                group2indices = {}
                start_ind = 0
                for g in range(n_groups):
                    group2indices[g] = list(
                        range(start_ind, (g + 1) * (seq_len // n_groups), spacing)
                    )
                    start_ind = (g + 1) * (seq_len // n_groups)

        else:
            ids = torch.full(
                (_n_samples, seq_len), self.mask_ind, dtype=torch.long, device=device
            )

        for step in range(gen_steps):
            if mysampler:
                i_group = step % n_groups
                if scan_order == "random" and i_group == 0 and step > 1:
                    group2indices = self.create_random_index_groups(seq_len, n_groups)
                indices_to_update = group2indices[i_group]

                cfg_scale = manual_cfg[step]
                ids[:, indices_to_update] = self.mask_ind
                logits = nnet(ids, **kwargs, scale=cfg_scale)

                sampled_ids = torch.distributions.Categorical(
                    logits=logits / max(samp_temp, 1e-4)
                ).sample()
                logits = torch.log_softmax(logits, dim=-1)
                sampled_logits = torch.squeeze(
                    torch.gather(
                        logits, dim=-1, index=torch.unsqueeze(sampled_ids, -1)
                    ),
                    -1,
                )
                ids[:, indices_to_update] = sampled_ids[:, indices_to_update]

            else:
                mask_ratio = manual_ratios[step] if step < gen_steps - 1 else 0
                # scaling temp
                annealed_temp = manual_temp[step] if step < gen_steps - 1 else 0
                # scaling cfg
                cfg_scale = manual_cfg[step]
                samp_temp = manual_samp_temp[step]
                # sampling & scoring
                is_mask = ids == self.mask_ind
                logits = nnet(ids, **kwargs, scale=cfg_scale)
                sampled_ids = torch.distributions.Categorical(
                    logits=logits / max(samp_temp, 1e-4)
                ).sample()
                logits = torch.log_softmax(logits, dim=-1)
                sampled_logits = torch.squeeze(
                    torch.gather(
                        logits, dim=-1, index=torch.unsqueeze(sampled_ids, -1)
                    ),
                    -1,
                )
                sampled_ids = torch.where(is_mask, sampled_ids, ids)
                sampled_logits = torch.where(is_mask, sampled_logits, +np.inf).float()
                # masking
                mask_len = torch.Tensor([np.floor(seq_len * mask_ratio)]).to(device)
                mask_len = torch.maximum(
                    torch.Tensor([1]).to(device),
                    torch.minimum(
                        torch.sum(is_mask, dim=-1, keepdims=True) - 1, mask_len
                    ),
                )[0].squeeze()
                confidence = add_gumbel_noise(sampled_logits, annealed_temp, device)
                sorted_confidence, _ = torch.sort(confidence, axis=-1)
                cut_off = sorted_confidence[:, mask_len.long() - 1 : mask_len.long()]
                masking = confidence <= cut_off
                ids = torch.where(masking, self.mask_ind, sampled_ids)
        if mysampler:
            _z = rearrange(ids, "b (i j) -> b i j", i=fmap_size, j=fmap_size)
            out = decode_fn(_z)
        else:
            _z = rearrange(sampled_ids, "b (i j) -> b i j", i=fmap_size, j=fmap_size)
            out = decode_fn(_z)
        return out
