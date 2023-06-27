import math
import random
import joblib
import numpy as np
import torch
from einops import rearrange, reduce
from einops.layers.torch import Rearrange, EinMix
from torch import nn
import torch.nn.functional as F
from model.conformer2.conformer import Conformer  # renamed core to conformer2 in rishikkish20 repo
from torchmetrics.classification import MulticlassAccuracy
from encodec import EncodecModel
_CONFIDENCE_OF_KNOWN_TOKENS = torch.Tensor([torch.inf]).to("cuda")

def uniform(shape, min = 0, max = 1, device = None):
    return torch.zeros(shape, device = device).float().uniform_(0, 1)

def cosine_schedule(t):
    return torch.cos(t * math.pi * 0.5)

def gamma_func(t):
    return np.cos(t * np.pi / 2)

def top_k(logits, thres = 0.9):
    k = math.ceil((1 - thres) * logits.shape[-1])
    val, ind = logits.topk(k, dim = -1)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(2, ind, val)
    return probs


def weights_init(m):
    classname = m.__class__.__name__
    if "Linear" in classname or "Embedding" == classname:
        #print(f"Initializing Module {classname}.")
        nn.init.trunc_normal_(m.weight.data, 0.0, 0.02)
    # elif "Parameter" in classname:
    #     return nn.init.trunc_normal_(m, 0.0, 0.02)

def log(t, eps = 1e-10):
    return torch.log(t + eps)

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(t, temperature = 1., dim=-1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim=dim)


class SoundStorm(nn.Module):

    def __init__(self, dim=768, heads=12, linear_units=1024, num_blocks=12, semantic_codebook_size=1025,
                semantic_num_quantizers=1, acoustic_codebook_size=1025, acoustic_num_quantizers=8,
                positionwise_conv_kernel_size=5, encodec=None, hubert_kmean_path=None):

        super().__init__()
        num_codes_with_mask = acoustic_codebook_size + 1
        #sos_token = 0
        self.steps = 8
        self.ignore_index = 1025
        self.n_q = acoustic_num_quantizers

        self.semantic_embeds = nn.Embedding((semantic_codebook_size + 1) * semantic_num_quantizers, 768)

        self.code_embeds = nn.ModuleList(
            [
                nn.Embedding(num_codes_with_mask + 2, dim)
                for _ in range(acoustic_num_quantizers)
            ]
        )


        self.mask_token_id = 1025
        self.mask_upper_level = 1025

        #self.sos_tokens = sos_token

        self.lm = Conformer(
            attention_dim=dim,
            attention_heads=heads,
            linear_units=linear_units,
            num_blocks=num_blocks,
            positionwise_conv_kernel_size=positionwise_conv_kernel_size
        )

        self.heads = nn.Sequential(
            nn.Linear(dim, dim * acoustic_num_quantizers),
            Rearrange('b n (h d) -> b (n h) d', h=acoustic_num_quantizers),
            nn.GELU(),
            nn.LayerNorm(dim, eps=1e-12),
            Rearrange('b (n q) d -> b n q d', q=acoustic_num_quantizers)
        )


        self.bias = nn.ParameterList([
                nn.Parameter(torch.zeros(num_codes_with_mask + 2))
                for _ in range(acoustic_num_quantizers)

            ]
        )

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            #Rearrange('b (n q) d -> b n q d', q=acoustic_num_quantizers),
            EinMix(
                'b n q d -> b n q l',
                weight_shape='q d l',
                bias_shape='q l',
                q=acoustic_num_quantizers,
                l=acoustic_codebook_size,
                d=dim
            )
        )

        self.loss = nn.CrossEntropyLoss(reduction='mean', ignore_index=self.ignore_index)
        self.accuracy_metric = MulticlassAccuracy(
            num_classes=1025,
            top_k=10,
            average="micro",
            multidim_average="global",
            ignore_index=1024,
        )

        self.sem_cond_proj = nn.Linear(768, dim)
        self.apply(weights_init)

        # initialize conformer token embedding as encodec/hubert vectors
        if encodec is not None:
            self._read_embedding_from_encodec(encodec)

        if hubert_kmean_path is not None:
            self._read_embedding_from_hubert_kmeans(hubert_kmean_path)

    def _read_embedding_from_encodec(self, encodec: EncodecModel):
        for i, layer in enumerate(encodec.quantizer.vq.layers[:self.n_q]):
            layer_weight = layer.codebook
            layer_dim = layer_weight.size(1)
            self.code_embeds[i].weight.data[:1024, :layer_dim] = layer_weight.clone().data

    def _read_embedding_from_hubert_kmeans(self, km_path: str):
        km_model = joblib.load(km_path)
        centers = km_model.cluster_centers_.transpose()
        centers = torch.tensor(centers, dtype=torch.float32).transpose(0, 1)
        self.semantic_embeds.weight.data[:centers.size(0), :] = centers.clone()

    def level_mask(self, code, seq_len, b, t, device):

        rand_times = torch.empty(b, device=device).uniform_(0, 1)
        batched_randperm = torch.rand((b, seq_len - t), device=device).argsort(dim=-1).float()

        rand_probs = cosine_schedule(rand_times)

        num_tokens_mask = (rand_probs * (seq_len - t)).clamp(min=1.)

        mask = batched_randperm < rearrange(num_tokens_mask, 'b -> b 1')

        prompt_mask = torch.ones((b, t), device=device).eq(0)
        mask = torch.cat([prompt_mask, mask], dim=1)

        labels = torch.where(mask, code, self.ignore_index)

        code = torch.where(mask, self.mask_token_id, code)

        return code, labels

    def fine_mask(self, code, t):
        code[:, t:] = self.mask_upper_level
        return code


    def masking(self, codes, q=None, t=None):
        seq_len = codes.shape[1]
        batch = codes.shape[0]
        codes = rearrange(codes, 'b n q -> q b n')

        masked_codes = []

        for i, code in enumerate(codes):
            if q == i:
                c, label = self.level_mask(code, seq_len, batch, t, codes.device)
                masked_codes.append(c)
            elif i > q:
                masked_codes.append(self.fine_mask(code, t))
            else:
                masked_codes.append(code)

        return masked_codes, label





    def forward(self, cond, codes):
        """
        cond: [B, Len]
        codes: [B, Len, n_q]
        """

        #b, q, n = codes.shape

        #codes = rearrange(codes, 'b q n -> b n q', q=q)

        q = random.randint(0, self.n_q - 1)
        t = random.randint(0, codes.shape[1] - 1)

        masked_codes, labels = self.masking(codes, q, t)

        masked_codes = torch.stack(masked_codes, dim=0)
        masked_codes = rearrange(masked_codes, 'q b n -> b n q')


        emb = None


        for i, layer in enumerate(self.code_embeds):
            if emb is None:
                emb = layer(masked_codes[:, :, i].unsqueeze(-1)).squeeze(-2)
            else:
                emb =  emb + layer(masked_codes[:, :, i].unsqueeze(-1)).squeeze(-2)

        # upsample the semantic tokens
        acoustic_len = codes.size(1)
        semantic_len = cond.size(1)
        semb = self.semantic_embeds(cond)               # [B, n, d]
        fetch_idx = torch.arange(0, acoustic_len).to(semb.device) * 2 / 3
        fetch_idx_int = fetch_idx.to(torch.int64).clamp(0, semantic_len - 1)
        fetch_idx_res = fetch_idx - fetch_idx_int
        # print(sem_cond.size(), fetch_idx_int.size(), fetch_idx_res.size())
        sem_cond_upscale = semb[:, fetch_idx_int] * (1 - fetch_idx_res).unsqueeze(0).unsqueeze(2) \
                           + semb[:, (fetch_idx_int + 1).clamp(0, semantic_len - 1)] * fetch_idx_res.unsqueeze(
            0).unsqueeze(2)
        semb = self.sem_cond_proj(sem_cond_upscale)

        # emb = reduce(emb, 'b n q d -> b n d', 'sum')                  # [B, n, d]

        emb = emb + semb

        out, _ = self.lm(emb, None)                            # [B, n, d]

        out = self.heads(out)                         # [B, q*n, d]

        logits = self.to_logits(out)                  # [B, n, q, d]

        #logits = torch.matmul(out[:, :, q], self.code_embeds[q].weight.T) + self.bias[q]
        logits = logits[:, :, q]      # [B, n, d]

        loss = F.cross_entropy(
            rearrange(logits, 'b n c -> b c n'),
            labels,
            ignore_index=self.ignore_index
        )

        acc_mask = rearrange(~labels.eq(1025), 'b n -> (b n)')

        # top 10 accuracy of acoustic tokens
        acc = self.accuracy_metric(rearrange(logits, 'b n c -> (b n) c')[acc_mask], rearrange(labels, 'b n -> (b n)')[acc_mask]).item()

        return loss, acc, 0





    def tokens_to_logits(self, semb, input_codes):
        # [B, n, q]
        emb = semb
        for i, layer in enumerate(self.code_embeds):
            emb = emb + layer(input_codes[:, :, i])
        # emb = self.code_embeds(emb.long())  # [B, n, q, d]


        # emb = reduce(emb, 'b n q d -> b n d', 'sum')  # [B, n, d]
        out, _ = self.lm(emb, None)  # [B, n, d]

        out = self.heads(out)  # [B, q*n, d]

        logits = self.to_logits(out)  # [B, n, q, d]

        return logits

    def mask_by_random_topk(self, mask_len, probs, temperature=1.0):
        confidence = torch.log(probs) + temperature * torch.distributions.gumbel.Gumbel(0, 1).sample(probs.shape).to("cuda")
        sorted_confidence, _ = torch.sort(confidence, dim=-1)
        # Obtains cut off threshold given the mask lengths.
        cut_off = torch.take_along_dim(sorted_confidence, mask_len.to(torch.long), dim=-1)
        # Masks tokens with lower confidence.
        masking = (confidence < cut_off)
        return masking

    @torch.no_grad()
    def generate(self, conds, codes):

        # clip the first 3 sec of ground truth as prompt, remove rest
        # if sample too short, use first half
        if codes is not None:
            num_latents_input = codes.size(1)
        else:
            num_latents_input = int(conds.size(1) * 1.5)  # Scale by 1.5 because HuBERT is 50Hz, Encodec is 75Hz
        num_prompt = min(int(num_latents_input * 0.5), 225)  # Default is 3 seconds (3*75Hz = 225 frames)
        prompt = codes[:, :num_prompt, :]

        device = next(self.lm.parameters()).device
        num_latents_to_generate = num_latents_input - num_prompt
        batch_size = 1

        acoustic_len = codes.size(1)
        semantic_len = conds.size(1)

        codes = prompt

        # upsample sem tokens
        semb = self.semantic_embeds(conds)  # [B, n, d]
        fetch_idx = torch.arange(0, acoustic_len).to(semb.device) * 2 / 3
        fetch_idx_int = fetch_idx.to(torch.int64).clamp(0, semantic_len - 1)
        fetch_idx_res = fetch_idx - fetch_idx_int
        # print(sem_cond.size(), fetch_idx_int.size(), fetch_idx_res.size())
        sem_cond_upscale = semb[:, fetch_idx_int] * (1 - fetch_idx_res).unsqueeze(0).unsqueeze(2) \
                           + semb[:, (fetch_idx_int + 1).clamp(0, semantic_len - 1)] * fetch_idx_res.unsqueeze(
            0).unsqueeze(2)
        semb = self.sem_cond_proj(sem_cond_upscale)

        # masked = []
        # for i in range(q):
        #     masked.append(torch.zeros((b, len - n, 1), device="cuda", dtype=torch.int).fill_(self.mask_token_id))
        #
        # masked = torch.cat(masked, dim=1)
        #
        # inputs = torch.cat((prompt, masked), dim=1)

        seq_len = num_latents_to_generate

        # sequence starts off as all masked
        shape = (batch_size, seq_len, 8)
        seq = torch.full(shape, self.mask_token_id, device=device)
        mask = torch.full(shape, True, device=device)

        # Calculate number of tokens to have masked at each time step
        times = torch.linspace(0., 1., self.steps + 1)
        all_mask_num_tokens = (cosine_schedule(times[1:]) * seq_len).long()

        # from lucidrain's inference code
        filter_thres = 0.7
        start_temperature = 1.0
        for rvq_layer in range(8):
            for mask_num_tokens, steps_until_x0 in zip(all_mask_num_tokens.tolist(), reversed(range(self.steps))):

                logits = self.tokens_to_logits(semb, torch.cat([prompt, seq], dim=1))
                logits = logits.view(batch_size, num_latents_to_generate + num_prompt, 8, 1025)
                logits = logits[:, num_prompt:, rvq_layer, :]  # Get the logits we want to consider (post-prompt and on given RVQ layer)


                logits = top_k(logits, filter_thres)  # Remove logits below a certain threshold (convert to -inf)

                annealing_scale = steps_until_x0 / self.steps
                temperature = start_temperature * annealing_scale
                # print(temperature)
                temperature = 1.0
                # probs = (logits / max(temperature, 1e-3)).softmax(dim = -1)

                # Top codebook vector index for each of the timestamps
                sampled_ids = gumbel_sample(logits, temperature=max(temperature, 1e-3))

                # Temporarily replace all tokens where mask is still True with sample tokens, will be undone below after mask is recomputed
                # Only tokens that are unmasked in the update will be kept
                seq[:, :, rvq_layer] = torch.where(mask[:, :, rvq_layer], sampled_ids, seq[:, :, rvq_layer])


                scores = 1 - logits.softmax(dim=-1)
                scores = scores.gather(2, rearrange(sampled_ids, 'b n -> b n 1'))  # gather the logits that it sampled
                scores = rearrange(scores, 'b n 1 -> b n')

                # No more tokens left to unmask, move to next RVQ layer
                if mask_num_tokens == 0:
                    continue

                # Remove scores corresponding to positions that have already been unmasked
                scores = scores.masked_fill(~mask[:, :, rvq_layer], -torch.finfo(scores.dtype).max)

                # High score = low probability logit value so select the highest `mask_num_tokens` to remain masked after this step
                mask_indices = scores.topk(mask_num_tokens, dim=-1).indices
                mask[:, :, rvq_layer] = torch.zeros_like(scores, dtype=torch.bool).scatter(1, mask_indices, True)
                # Update seq with the newly calculated mask
                seq[:, :, rvq_layer] = seq[:, :, rvq_layer].masked_fill(mask[:, :, rvq_layer], self.mask_token_id)

        out = torch.cat([prompt, seq], dim=1)
        return out

        # rishkkish's old code for reference
        # i = 0
        #
        #
        # cur_ids = inputs  # [b, q, n]
        #
        # for _ in range(q):
        #     unknown_number_in_the_beginning = torch.sum(inputs[:, i] == self.mask_token_id, dim=-1)
        #     if i == 0:
        #         # Confidence based sampling:
        #         for t in range(T[i]-1):
        #             logits = self.tokens_to_logits(semb, cur_ids)          # [B, n, q, d]
        #             #logits = rearrange(logits, 'b n q d -> q b n d')
        #
        #             target_logits = logits[:, :, i]                               # [B, n, d]
        #             cur_ids = rearrange(cur_ids, 'b q n -> q b n')
        #             target_ids = cur_ids[i]    # [B, n]
        #
        #             sampled_ids = torch.distributions.categorical.Categorical(logits=target_logits).sample()
        #
        #             unknown_map = (target_ids == self.mask_token_id)  # which tokens need to be sampled -> bool [8, 257]
        #             sampled_ids = torch.where(unknown_map, sampled_ids, target_ids)  # replace all -1 with their samples and leave the others untouched [8, 257]
        #
        #             ratio = 1. * (t + 1) / T[i]  # just a percentage e.g. 1 / 12
        #             ratio = torch.zeros((b,), device="cuda", dtype=torch.int).fill_(ratio)
        #             mask_ratio = cosine_schedule(ratio)
        #
        #             probs = F.softmax(target_logits, dim=-1)  # convert logits into probs [8, 257, 1024]
        #             selected_probs = torch.squeeze(torch.take_along_dim(probs, torch.unsqueeze(sampled_ids, -1), -1),
        #                                            -1)  # get probability for selected tokens in categorical call, also for already sampled ones [8, 257]
        #
        #             selected_probs = torch.where(unknown_map, selected_probs, _CONFIDENCE_OF_KNOWN_TOKENS)
        #
        #             mask_len = torch.unsqueeze(torch.floor(unknown_number_in_the_beginning * mask_ratio),
        #                                        1)  # floor(256 * 0.99) = 254 --> [254, 254, 254, 254, ....]
        #             mask_len = torch.maximum(torch.zeros_like(mask_len),
        #                                      torch.minimum(torch.sum(unknown_map, dim=-1, keepdim=True) - 1,
        #                                                    mask_len))  # add -1 later when conditioning and also ones_like. Zeroes just because we have no cond token
        #             # max(1, min(how many unknown tokens, how many tokens we want to sample))
        #
        #             masking = self.mask_by_random_topk(mask_len, selected_probs,
        #                                                temperature=choice_temperature * (1. - ratio))
        #
        #             target_ids = torch.where(masking, self.mask_token_id, sampled_ids)
        #
        #             cur_ids[i] = target_ids
        #
        #             cur_ids = rearrange(cur_ids, 'q b n -> b q n')
        #
        #     # Greedy Sampling:
        #     logits = self.tokens_to_logits(conds, cur_ids)  # [B, n, q, d]
        #
        #     logits = rearrange(logits, 'b n q d -> q b n d')
        #
        #     cur_ids = rearrange(cur_ids, 'b q n -> q b n')
        #     target_ids = cur_ids[i]  # [B, n]
        #     sampled_ids = torch.argmax(logits[i], dim=-1)
        #     unknown_map = (target_ids == self.mask_token_id)
        #     target_ids = torch.where(unknown_map, sampled_ids, target_ids)
        #
        #     cur_ids[i] = target_ids
        #
        #     cur_ids = rearrange(cur_ids, 'q b n -> b q n')
        #
        #     i = i + 1
        #
        # return cur_ids      #[B, q, n]






def num_params(model, print_out=True):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    if print_out:
        print("Trainable Parameters: %.3fM" % parameters)


if __name__ == '__main__':

    cond = torch.randint(1, 1024, (2, 20)).long()
    codes = torch.randint(1, 1024, (2, 8, 20)).long()

    model = SoundStorm()

    num_params(model)


    logits, out, mask = model(cond, codes)

