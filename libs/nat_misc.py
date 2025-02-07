import math
import random

import einops
import numpy as np
import torch
from einops import rearrange
from torch.nn import functional as F

from .nat_model import UViT


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

    def sample(self, x0, sampling_type):
        if sampling_type == "improvednat":
            N, L, device = *x0.shape, self.device
            beta_dist = torch.distributions.Beta(self.beta_a, self.beta_b)
            rand_mask_probs = beta_dist.sample((N,)).to(device).float()
            num_token_masked = (L * rand_mask_probs).round().clamp(min=1)
            batch_randperm = torch.rand(N, L, device=device).argsort(dim=-1)
            mask = batch_randperm < rearrange(num_token_masked, "b -> b 1")
            masked_ids = torch.where(mask, self.mask_ind, x0)
            labels = torch.where(mask, x0, self.ignore_ind)
            return None, labels, masked_ids  # timestep is not needed for nnet
        elif sampling_type == "grid":
            n_grid = 2
            N, L, device = *x0.shape, self.device
            grid_size = L // n_grid
            random_dist = torch.distributions.Uniform(0, n_grid)
            rand_mask_indices = random_dist.sample((N,)).to(device).float()
            rand_mask_indices = torch.floor(rand_mask_indices).int()
            rand_mask_indices = rand_mask_indices * grid_size
            mask = torch.zeros((N, L), device=device)
            all_indices = (
                torch.arange(L, device=device).unsqueeze(0).expand(N, -1)
            )  # Shape: (N, L)
            mask = (all_indices >= rand_mask_indices.unsqueeze(1)) & (
                all_indices < (rand_mask_indices.unsqueeze(1) + grid_size)
            )
            masked_ids = torch.where(mask, self.mask_ind, x0)
            labels = torch.where(mask, x0, self.ignore_ind)
            return None, labels, masked_ids  # timestep is not needed for nnet
        elif sampling_type == "checkerboard":
            N, L, device = *x0.shape, self.device
            random_dist = torch.distributions.Uniform(0, 2)
            rand_mask_indices = random_dist.sample((N,)).to(device).float()
            rand_mask_indices = torch.floor(rand_mask_indices).int()
            mask = torch.zeros((N, L), device=device)
            all_indices = (
                torch.arange(L, device=device).unsqueeze(0).expand(N, -1)
            )  # Shape: (N, L)
            mask = (all_indices % 2) == rand_mask_indices.unsqueeze(-1)
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

    # @torch.no_grad()
    # def generate(
    #     self,
    #     gen_steps,
    #     _n_samples,
    #     nnet,
    #     decode_fn,
    #     manual_ratios,
    #     manual_temp,
    #     manual_samp_temp,
    #     manual_cfg,
    #     **kwargs
    # ):
    #     device = self.device

    #     fmap_size = 16
    #     seq_len = fmap_size * fmap_size

    #     ids = torch.full(
    #         (_n_samples, seq_len), self.mask_ind, dtype=torch.long, device=device
    #     )

    #     for step in range(gen_steps):
    #         mask_ratio = manual_ratios[step] if step < gen_steps - 1 else 0
    #         # scaling temp
    #         annealed_temp = manual_temp[step] if step < gen_steps - 1 else 0
    #         # scaling cfg
    #         cfg_scale = manual_cfg[step]
    #         samp_temp = manual_samp_temp[step]
    #         # sampling & scoring
    #         is_mask = ids == self.mask_ind
    #         logits = nnet(ids, **kwargs, scale=cfg_scale)
    #         sampled_ids = torch.distributions.Categorical(
    #             logits=logits / max(samp_temp, 1e-4)
    #         ).sample()
    #         logits = torch.log_softmax(logits, dim=-1)
    #         sampled_logits = torch.squeeze(
    #             torch.gather(logits, dim=-1, index=torch.unsqueeze(sampled_ids, -1)), -1
    #         )
    #         sampled_ids = torch.where(is_mask, sampled_ids, ids)
    #         sampled_logits = torch.where(is_mask, sampled_logits, +np.inf).float()
    #         # masking
    #         mask_len = torch.Tensor([np.floor(seq_len * mask_ratio)]).to(device)
    #         mask_len = torch.maximum(
    #             torch.Tensor([1]).to(device),
    #             torch.minimum(torch.sum(is_mask, dim=-1, keepdims=True) - 1, mask_len),
    #         )[0].squeeze()
    #         confidence = add_gumbel_noise(sampled_logits, annealed_temp, device)
    #         sorted_confidence, _ = torch.sort(confidence, axis=-1)
    #         cut_off = sorted_confidence[:, mask_len.long() - 1 : mask_len.long()]
    #         masking = confidence <= cut_off
    #         ids = torch.where(masking, self.mask_ind, sampled_ids)

    #     _z = rearrange(sampled_ids, "b (i j) -> b i j", i=fmap_size, j=fmap_size)
    #     out = decode_fn(_z)

    #     return out

    @torch.no_grad()
    def generate(
        self,
        gen_steps,
        _n_samples,
        nnet,
        decode_fn,
        sampling_type,
        manual_ratios,
        manual_temp,
        manual_samp_temp,
        manual_cfg,
        **kwargs
    ):
        device = self.device

        fmap_size = 16
        seq_len = fmap_size * fmap_size

        if sampling_type == "grid":
            # Grid sampler
            vocab_size = 1024
            ids = torch.randint(0, vocab_size, (_n_samples, seq_len), device=device)
            n_groups = 2
            group2indices = {}
            start_ind = 0
            for g in range(n_groups):
                group2indices[g] = list(
                    range(start_ind, (g + 1) * (seq_len // n_groups))
                )
                start_ind = (g + 1) * (seq_len // n_groups)
            samp_temp = 1.0
        elif sampling_type == "checkerboard":
            # checkerboard sampler
            vocab_size = 1024
            # ids = torch.randint(0, vocab_size, (_n_samples, seq_len), device=device)
            ids = torch.full(
                (_n_samples, seq_len), self.mask_ind, dtype=torch.long, device=device
            )
            n_groups = 2
            group2indices = {}
            start_ind = 0
            for g in range(n_groups):
                group2indices[g] = list(range(g, seq_len, n_groups))
            samp_temp = 1.0
        else:
            ids = torch.full(
                (_n_samples, seq_len), self.mask_ind, dtype=torch.long, device=device
            )

        for step in range(gen_steps):
            if sampling_type == "grid" or sampling_type == "checkerboard":
                i_group = step % n_groups
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
        if sampling_type == "grid" or sampling_type == "checkerboard":
            _z = rearrange(ids, "b (i j) -> b i j", i=fmap_size, j=fmap_size)
            out = decode_fn(_z)
        else:
            _z = rearrange(sampled_ids, "b (i j) -> b i j", i=fmap_size, j=fmap_size)
            out = decode_fn(_z)
        return out


class NAT_Schedule_MRFSampler(NATSchedule):
    def __init__(
        self,
        codebook_size,
        device,
        ignore_ind=-1,
        smoothing=0.0,
        beta_alpha_beta=(12, 3),
    ):
        super().__init__(codebook_size, device, ignore_ind, smoothing, beta_alpha_beta)
        self.mask_ind = 1024  # for input masking
        self.pretrained_nnet = UViT(
            **{
                "img_size": 16,
                "codebook_size": 1024,
                "embed_dim": 768,
                "depth": 24,
                "num_heads": 8,
                "mlp_ratio": 4,
                "qkv_bias": False,
                "num_classes": 1001,
                "use_checkpoint": False,
                "skip": True,
            }
        )
        ckpt = torch.load("assets/nnet_ema.pth", map_location="cpu")
        self.pretrained_nnet.load_state_dict(ckpt)
        self.unfolder = torch.nn.Unfold((3, 3), stride=1)
        self.empty_ctx = torch.from_numpy(np.array([[1000]], dtype=np.longlong)).to(
            device
        )
        self.cutoffs = [5, 6, 7]

    @torch.no_grad()
    def generate_pretrained(
        self,
        nnet,
        cutoff,
        manual_ratios=[
            0.9531645174140464,
            0.8795409806037042,
            0.7940230737075623,
            0.745995860948045,
            0.6249785820461957,
            0.407321956652571,
            0.260658917067054,
        ],
        manual_temp=[
            2.404912789740805,
            2.477616274962803,
            1.191696577294597,
            0.7685444738649722,
            1.0673489052089495,
            0.9353720330515827,
            0.7843839537838292,
        ],
        manual_samp_temp=[
            2.8999037830310685,
            1.867261830931786,
            1.1443729673819831,
            1.3796334226014182,
            1.4699896959810144,
            1.3760684112198358,
            1.7082165405200722,
            0.7133060703601533,
        ],
        manual_cfg=[
            0.0001,
            0.0001,
            0.0001,
            0.9151274967591447,
            1.6160182878189675,
            1.1945509026838659,
            1.38322370140345,
            1.22678520936251,
        ],
        gen_steps=8,
        _n_samples=1,
    ):
        kwargs = {}

        device = self.device

        fmap_size = 16
        seq_len = fmap_size * fmap_size
        ids = torch.full(
            (_n_samples, seq_len), self.mask_ind, dtype=torch.long, device=device
        )

        for step in range(gen_steps):
            if step == cutoff:
                break
            mask_ratio = manual_ratios[step] if step < gen_steps - 1 else 0
            # scaling temp
            annealed_temp = manual_temp[step] if step < gen_steps - 1 else 0
            # scaling cfg
            cfg_scale = manual_cfg[step]
            samp_temp = manual_samp_temp[step]
            # sampling & scoring
            is_mask = ids == self.mask_ind
            logits = self.cfg_nnet(self.pretrained_nnet, ids, **kwargs, scale=cfg_scale)
            sampled_ids = torch.distributions.Categorical(
                logits=logits / max(samp_temp, 1e-4)
            ).sample()
            logits = torch.log_softmax(logits, dim=-1)
            sampled_logits = torch.squeeze(
                torch.gather(logits, dim=-1, index=torch.unsqueeze(sampled_ids, -1)),
                -1,
            )
            sampled_ids = torch.where(is_mask, sampled_ids, ids)
            sampled_logits = torch.where(is_mask, sampled_logits, +np.inf).float()
            # masking
            mask_len = torch.Tensor([np.floor(seq_len * mask_ratio)]).to(device)
            mask_len = torch.maximum(
                torch.Tensor([1]).to(device),
                torch.minimum(torch.sum(is_mask, dim=-1, keepdims=True) - 1, mask_len),
            )[0].squeeze()
            confidence = add_gumbel_noise(sampled_logits, annealed_temp, device)
            sorted_confidence, _ = torch.sort(confidence, axis=-1)
            cut_off = sorted_confidence[:, mask_len.long() - 1 : mask_len.long()]
            masking = confidence <= cut_off
            ids = torch.where(masking, self.mask_ind, sampled_ids)
        return ids
        _z = rearrange(sampled_ids, "b (i j) -> b i j", i=fmap_size, j=fmap_size)
        out = decode_fn(_z)
        return out

    @torch.no_grad()
    def generate(
        self,
        gen_steps,
        _n_samples,
        nnet,
        decode_fn,
        sampling_type,
        manual_ratios,
        manual_temp,
        manual_samp_temp,
        manual_cfg,
        **kwargs
    ):
        device = self.device
        self.pretrained_nnet = self.pretrained_nnet.to(device)

        fmap_size = 16
        seq_len = fmap_size * fmap_size

        ids = torch.full(
            (_n_samples, seq_len), self.mask_ind, dtype=torch.long, device=device
        )

        for step in range(gen_steps):
            mask_ratio = manual_ratios[step] if step < gen_steps - 1 else 0
            # scaling temp
            annealed_temp = manual_temp[step] if step < gen_steps - 1 else 0
            # scaling cfg
            cfg_scale = manual_cfg[step]
            samp_temp = manual_samp_temp[step]
            # sampling & scoring
            is_mask = ids == self.mask_ind
            with torch.no_grad():
                logits = self.cfg_nnet(
                    self.pretrained_nnet, ids, scale=cfg_scale, **kwargs
                )
            # logits = self.pretrained_nnet(ids, **kwargs)
            sampled_ids = torch.distributions.Categorical(
                logits=logits / max(samp_temp, 1e-4)
            ).sample()
            logits = torch.log_softmax(logits, dim=-1)
            sampled_logits = torch.squeeze(
                torch.gather(logits, dim=-1, index=torch.unsqueeze(sampled_ids, -1)),
                -1,
            )
            sampled_ids = torch.where(is_mask, sampled_ids, ids)
            sampled_logits = torch.where(is_mask, sampled_logits, +np.inf).float()
            # masking
            mask_len = torch.Tensor([np.floor(seq_len * mask_ratio)]).to(device)
            mask_len = torch.maximum(
                torch.Tensor([1]).to(device),
                torch.minimum(torch.sum(is_mask, dim=-1, keepdims=True) - 1, mask_len),
            )[0].squeeze()
            confidence = add_gumbel_noise(sampled_logits, annealed_temp, device)
            sorted_confidence, _ = torch.sort(confidence, axis=-1)
            confidence_thresh = sorted_confidence[
                :, mask_len.long() - 1 : mask_len.long()
            ]

            # , **kwargs)
            masking = confidence <= confidence_thresh
            ids = torch.where(masking, self.mask_ind, sampled_ids)

            if step in self.cutoffs:
                tokens_to_unmask = torch.logical_and(
                    confidence >= confidence_thresh, is_mask
                ).squeeze(0)
                windows_preds = self.unfolder(
                    self.pad_tensor(
                        rearrange(
                            sampled_ids, "b (l w) -> b 1 l w", l=fmap_size, w=fmap_size
                        ).to(torch.float32)
                    )
                ).to(torch.int32)
                windows_preds = rearrange(
                    windows_preds, "b (l w) n -> (n b) (l w)", l=3, w=3
                )
                with torch.no_grad():
                    refined_logits = nnet(
                        windows_preds,
                        timesteps=None,
                        scale=cfg_scale,
                    )[:, :1024]
                refined_tokens = torch.distributions.Categorical(
                    logits=refined_logits / max(samp_temp, 1e-4)
                ).sample()
                refined_tokens = rearrange(
                    refined_tokens,
                    "(b l w) -> b (l w)",
                    l=fmap_size,
                    w=fmap_size,
                )

                tokens_to_refine = torch.logical_and(
                    torch.rand(is_mask.shape, device=device) < 0.25,
                    torch.logical_not(is_mask),
                )
                sampled_ids[tokens_to_refine] = refined_tokens[tokens_to_refine]

                # , **kwargs)
                masking = confidence <= confidence_thresh
                ids = torch.where(masking, self.mask_ind, sampled_ids)

        _z = rearrange(sampled_ids, "b (i j) -> b i j", i=fmap_size, j=fmap_size)
        out = decode_fn(_z)
        return out

    def pad_tensor(self, tensor, pad_size=1, pad_value=1025):
        return F.pad(
            tensor,
            (pad_size, pad_size, pad_size, pad_size),
            mode="constant",
            value=pad_value,
        )

    def sample(self, x0, **kwargs):
        self.pretrained_nnet = self.pretrained_nnet.to(x0.device)
        N, L, device = *x0.shape, self.device
        manual_ratios = [
            0.9531645174140464,
            0.8795409806037042,
            0.7940230737075623,
            0.745995860948045,
            0.6249785820461957,
            0.407321956652571,
            0.260658917067054,
        ]
        manual_cfg = [
            0.0001,
            0.0001,
            0.0001,
            0.9151274967591447,
            1.6160182878189675,
            1.1945509026838659,
            1.38322370140345,
            1.22678520936251,
        ]
        manual_samp_temp = [
            2.8999037830310685,
            1.867261830931786,
            1.1443729673819831,
            1.3796334226014182,
            1.4699896959810144,
            1.3760684112198358,
            1.7082165405200722,
            0.7133060703601533,
        ]
        manual_temp = [
            2.404912789740805,
            2.477616274962803,
            1.191696577294597,
            0.7685444738649722,
            1.0673489052089495,
            0.9353720330515827,
            0.7843839537838292,
        ]
        fmap_size = 16
        inputs = []
        gt = []
        idxs = []
        for i in range(N):
            for c in range(len(self.cutoffs)):
                cutoff = self.cutoffs[c]
                batch_randperm = torch.rand(1, L, device=device).argsort(dim=-1)
                is_mask = batch_randperm < manual_ratios[cutoff - 1] * L
                masked_ids = torch.where(is_mask, self.mask_ind, x0[i])
                labels = torch.where(is_mask, x0[i], self.ignore_ind)

                with torch.no_grad():
                    logits = self.cfg_nnet(
                        self.pretrained_nnet,
                        masked_ids,
                        context=kwargs["context"][i].unsqueeze(0),
                        scale=manual_cfg[cutoff - 1],
                    )[:, :, :1024]
                    sampled_ids = torch.distributions.Categorical(
                        logits=logits / max(1.0, 1e-4)
                    ).sample()

                    sampled_ids = torch.where(is_mask, sampled_ids, x0[i])

                    logits = torch.log_softmax(logits, dim=-1)
                    sampled_logits = torch.squeeze(
                        torch.gather(
                            logits, dim=-1, index=torch.unsqueeze(sampled_ids, -1)
                        ),
                        -1,
                    )
                    sampled_logits = torch.where(
                        is_mask, sampled_logits, +np.inf
                    ).float()
                    mask_ratio = manual_ratios[cutoff] if cutoff < 8 - 1 else 0
                    # scaling temp
                    annealed_temp = manual_temp[cutoff] if cutoff < 8 - 1 else 0
                    # scaling cfg
                    # masking
                    mask_len = torch.Tensor([np.floor(L * mask_ratio)]).to(device)
                    mask_len = torch.maximum(
                        torch.Tensor([1]).to(device),
                        torch.minimum(
                            torch.sum(is_mask, dim=-1, keepdims=True) - 1, mask_len
                        ),
                    )[0].squeeze()
                    confidence = add_gumbel_noise(sampled_logits, annealed_temp, device)
                    sorted_confidence, _ = torch.sort(confidence, axis=-1)
                    confidence_thresh = sorted_confidence[
                        :, mask_len.long() - 1 : mask_len.long()
                    ]
                    windows_to_keep = torch.logical_and(
                        confidence >= confidence_thresh, is_mask
                    ).squeeze(0)
                    masking = confidence <= confidence_thresh
                    ids = torch.where(masking, self.mask_ind, sampled_ids)

                ids = rearrange(ids, "b (i j) -> b 1 i j", i=fmap_size, j=fmap_size)
                windows_pretrained_preds = self.unfolder(
                    self.pad_tensor(ids.to(torch.float32))
                ).to(torch.int32)
                windows_pretrained_preds = rearrange(
                    windows_pretrained_preds, "b (l w) n -> n b l w", l=3, w=3
                )
                # windows_to_keep = [w[0][1][1] == self.mask_ind for w in windows_mask]

                # labels = rearrange(labels, "b (i j) -> b 1 i j", i=fmap_size, j=fmap_size)
                # windows_labels = self.unfolder(self.pad_tensor(labels.to(torch.float32))).to(
                #     torch.int32
                # )
                # windows_labels = rearrange(windows_labels, "b (l w) n -> n b l w", l=3, w=3)
                windows_labels = labels[:, windows_to_keep]

                # sampled_ids = rearrange(
                #     sampled_ids, "b (i j) -> b 1 i j", i=fmap_size, j=fmap_size
                # )
                # windows_pretrained_preds = self.unfolder(
                #     self.pad_tensor(sampled_ids.to(torch.float32))
                # ).to(torch.int32)
                # windows_pretrained_preds = rearrange(
                #     windows_pretrained_preds, "b (l w) n -> n b l w", l=3, w=3
                # )

                windows_pretrained_preds = windows_pretrained_preds[windows_to_keep]
                inputs += windows_pretrained_preds
                gt += windows_labels

        return (
            None,
            torch.concat(gt, -1),
            torch.stack(inputs).reshape(len(inputs), -1),
        )  # timestep is not needed

    def loss(self, pred, label):  # pred: N, L, C
        return F.cross_entropy(
            pred,
            label.long(),
            ignore_index=self.ignore_ind,
            label_smoothing=self.smoothing,
        )

    def cfg_nnet(self, nnet_ema, x, scale, **kwargs):
        _cond = nnet_ema(x, **kwargs)
        kwargs["context"] = einops.repeat(self.empty_ctx, "1 ... -> B ...", B=x.size(0))
        _uncond = nnet_ema(x, **kwargs)
        res = _cond + scale * (_cond - _uncond)
        return res
