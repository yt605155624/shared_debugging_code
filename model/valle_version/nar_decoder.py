import torch
from torch import Tensor, nn
from torch.nn import functional as F
from model.embedding import TokenEmbedding, SinePositionalEmbedding
import random
from random import choice
from model.utils import sequence_mask, make_pad_mask
from modules.transformer import (
    AdaptiveLayerNorm,
    TransformerEncoderLayer,
    TransformerEncoder
)
from torchmetrics.classification import MulticlassAccuracy
NUM_AUDIO_TOKENS = 1024
class NARDecoder(nn.Module):

    def __init__(self, config, norm_first=False):
        super(NARDecoder, self).__init__()
        self.model_dim = config["hidden_dim"]
        self.embedding_dim = config["embedding_dim"]
        self.num_head = config["num_head"]
        self.num_layers = config["num_layers"]
        self.norm_first = norm_first
        self.vocab_size = config["vocab_size"]
        self.p_dropout = config["p_dropout"]
        self.EOS = config["EOS"]
        self.norm_first = norm_first
        self.stages = [1, 2, 3, 4, 5, 6, 7, 8]
        self.rng = random.Random(0)
        self.prefix_mode = 1
        self.num_quantizers = 9
        assert self.EOS == 1024

        self.rng = random.Random(0)
        # phoneme embeddings
        self.nar_semantic_embedding = TokenEmbedding(self.embedding_dim, self.vocab_size, self.p_dropout)

        # audio codebook token embedding
        self.nar_audio_embeddings = nn.ModuleList([
                TokenEmbedding(self.embedding_dim, self.vocab_size-1, self.p_dropout) for _ in range(len(self.stages))
        ])
        self.nar_audio_position = SinePositionalEmbedding(self.embedding_dim, dropout=0.1, scale=False, alpha=False)

        self.h = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=self.model_dim,
                nhead=self.num_head,
                dim_feedforward=self.model_dim * 4,
                dropout=0.1,
                batch_first=True,
                norm_first=norm_first,
                adaptive_layer_norm=True
            ),
            num_layers=self.num_layers,
            norm=AdaptiveLayerNorm(self.model_dim, norm=nn.LayerNorm(self.model_dim)) if self.norm_first else None,
        )
        self.nar_predict_layers = nn.ModuleList([
            nn.Linear(self.model_dim, NUM_AUDIO_TOKENS, bias=False) for _ in self.stages
        ])
        self.nar_stage_embeddings = nn.ModuleList([
                TokenEmbedding(self.model_dim, 1) for _ in self.stages
        ])
        self.loss_fct = nn.CrossEntropyLoss(reduction='sum')
        self.nar_accuracy_metric = MulticlassAccuracy(
            NUM_AUDIO_TOKENS + 1,
            top_k=10,
            average="micro",
            multidim_average="global",
            ignore_index=NUM_AUDIO_TOKENS,
        )

        for j in range(0, self.num_quantizers - 2):
            self.nar_predict_layers[
                j
            ].weight = self.nar_audio_embeddings[j + 1].weight


    def forward(self, x, x_lens, y, y_lens):

        x = self.nar_semantic_embedding(x)
        semantic_len = x.size(1)
        acoustic_len = y.size(1)

        fetch_idx = torch.arange(0, acoustic_len).to(x.device) * 2.0 / 3
        fetch_idx_int = fetch_idx.to(torch.int64).clamp(0, semantic_len - 1)
        fetch_idx_res = fetch_idx - fetch_idx_int
        # print(sem_cond.size(), fetch_idx_int.size(), fetch_idx_res.size())
        x = x[:, fetch_idx_int] * (1 - fetch_idx_res).unsqueeze(0).unsqueeze(2) \
                + x[:, (fetch_idx_int + 1).clamp(0, semantic_len - 1)] * fetch_idx_res.unsqueeze(0).unsqueeze(2)

        y_mask = make_pad_mask(y_lens).to(y.device)
        y_mask_int = y_mask.type(torch.int64)
        codes = y.type(torch.int64) * (1 - y_mask_int.unsqueeze(dim=-1))
        y = codes[:, :, 0] + self.EOS * y_mask_int

        num_nar_layers = self.num_quantizers - 1
        nar_stage = self.rng.choices(
            [_k for _k in range(1, self.num_quantizers)],
            weights=[1.0 / num_nar_layers] * num_nar_layers,
            k=1,
        )[0]


        y_prompts_codes = None
        y_emb, prefix_len = self._prepare_prompts(
            x, y, y_lens, codes, nar_stage, y_prompts_codes
        )

        y_len = y_lens.max()
        targets = codes[..., nar_stage - 1] + self.EOS * y_mask_int
        targets = targets[:, prefix_len:]

        y_pos = self.nar_audio_position(y_emb)

        xy_padding_mask = y_mask
        xy_pos = y_pos
        xy_dec, _ = self.h(
            (xy_pos, self.nar_stage_embeddings[nar_stage - 1].weight),
            src_key_padding_mask=xy_padding_mask,
            # is_causal=False,
        )

        xy_dec = xy_dec[:, prefix_len:]

        logits = self.nar_predict_layers[nar_stage - 1](xy_dec).permute(0, 2, 1)

        # loss
        total_length = (y_lens).sum().type(torch.float32)
        loss = F.cross_entropy(
                    logits,
                    targets,
                    ignore_index=1024,
                    reduction='sum',
                ) * (total_length / (total_length - prefix_len * x.shape[0]))

        acc = (
                self.nar_accuracy_metric(
                    F.pad(
                        logits.detach(),
                        (0, 0, 0, 1, 0, 0),
                        value=logits.min().cpu().item(),
                    ),
                    targets,
                ).item()
                * total_length
        )


        n_frames = y_lens.sum().type(torch.float32)
        acc = acc / n_frames
        #topk = torch.topk(logits.permute(0, 2, 1), k=10)
        #topk_hit = torch.sum(torch.eq(topk.indices, targets.unsqueeze(-1)), dim=-1)
        return loss, acc, nar_stage

    def infer(self, x, x_len, y):

        x = self.nar_semantic_embedding(x)
        semantic_len = x.size(1)
        acoustic_len = y.size(1)
        n_prompt = min(225, acoustic_len // 2)
        prompts = y[:, :n_prompt]

        fetch_idx = torch.arange(0, acoustic_len).to(x.device) * 2.0 / 3
        fetch_idx_int = fetch_idx.to(torch.int64).clamp(0, semantic_len - 1)
        fetch_idx_res = fetch_idx - fetch_idx_int
        # print(sem_cond.size(), fetch_idx_int.size(), fetch_idx_res.size())
        x = x[:, fetch_idx_int] * (1 - fetch_idx_res).unsqueeze(0).unsqueeze(2) \
            + x[:, (fetch_idx_int + 1).clamp(0, semantic_len - 1)] * fetch_idx_res.unsqueeze(0).unsqueeze(2)

        y_emb = x
        prefix_len = prompts.shape[1]
        codes = []

        if self.prefix_mode != 0:
            for j in range(1, self.num_quantizers):

                y_emb[:, :prefix_len] += self.nar_audio_embeddings[j-1](
                    prompts[..., j-1]
                )

        for i, (predict_layer, embedding_layer) in enumerate(
                zip(
                    self.nar_predict_layers,
                    self.nar_audio_embeddings,
                )
        ):
            y_pos = self.nar_audio_position(y_emb)
            xy_pos = y_pos

            xy_dec, _ = self.h(
                (xy_pos, self.nar_stage_embeddings[i].weight)
            )

            logits = predict_layer(xy_dec[:, prefix_len:])
            samples = torch.argmax(logits, dim=-1)
            codes.append(samples)
            # Formula (4) (5)
            if i < self.num_quantizers - 2:
                y_emb[:, prefix_len:] += embedding_layer(samples)

        return torch.concat([prompts, torch.stack(codes, dim=-1)], dim=1)

    def _prepare_prompts(self, x, y, y_lens, codes, nar_stage, y_prompts_codes):
        # 5.1 For the NAR acoustic prompt tokens, we select a random segment waveform of 3 seconds
        # from the same utterance.
        # We implement this differently.
        if self.prefix_mode == 0:
            # no prefix
            prefix_len = 0
            y_emb = self.nar_audio_embeddings[0](y)
            for j in range(1, nar_stage):
                # Formula (4) (5)
                y_emb = y_emb + self.nar_audio_embeddings[j](codes[..., j])
        elif self.prefix_mode == 1:
            # prefix at begining
            int_low = (0.25 * y_lens.min()).type(torch.int64).item()
            prefix_len = torch.randint(int_low, int_low * 2, size=()).item()
            prefix_len = min(prefix_len, 225)  # 24000/320 * 3s = 225 frames

            y_prompts = x[:, :prefix_len]
            y_emb = x[:, prefix_len:]

            for j in range(1, self.num_quantizers):
                y_prompts += self.nar_audio_embeddings[j-1](
                    codes[:, :prefix_len, j-1]
                )
                if j < nar_stage:
                    y_emb += self.nar_audio_embeddings[j-1](
                        codes[:, prefix_len:, j-1]
                    )
            y_emb = torch.concat([y_prompts, y_emb], axis=1)
        elif self.prefix_mode in [2, 4]:
            if self.prefix_mode == 2:
                # random prefix
                prefix_len = min(225, int(0.25 * y_lens.min().item()))

                y_prompts_codes = []
                for b in range(codes.shape[0]):
                    start = self.rng.randint(0, y_lens[b].item() - prefix_len)
                    y_prompts_codes.append(
                        torch.clone(codes[b, start : start + prefix_len])
                    )
                    codes[
                        b, start : start + prefix_len, nar_stage
                    ] = NUM_AUDIO_TOKENS
                y_prompts_codes = torch.stack(y_prompts_codes, dim=0)
            else:
                prefix_len = y_prompts_codes.shape[1]

            y_prompts = self.nar_audio_embeddings[0](y_prompts_codes[..., 0])
            y_emb = self.nar_audio_embeddings[0](y)
            for j in range(1, self.num_quantizers):
                y_prompts += self.nar_audio_embeddings[j](
                    y_prompts_codes[..., j]
                )
                if j < nar_stage:
                    y_emb += self.nar_audio_embeddings[j](codes[..., j])
            y_emb = torch.concat([y_prompts, y_emb], axis=1)
        else:
            raise ValueError

        return y_emb, prefix_len

