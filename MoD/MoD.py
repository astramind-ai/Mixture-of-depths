from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

class TokenRouter(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.weight_predictor = nn.Linear(embed_dim, 1)

    def forward(self, x):
        weights = self.weight_predictor(x).squeeze(
            -1
        )  # [batch_size, seq_len]
        return weights

class MoD(nn.Module):
    def __init__(self, capacity, block):
        super().__init__()
        self.router = TokenRouter(block.hidden_size)
        self.block = block
        self.capacity = capacity
        self.training_step = 0

    def forward(self,
                x: Tensor,
                causal_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
                cache_position,
                **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        b, s, d = x.shape
        weights = self.router(x)
        if self.router.training:
            # this is done to avoid shocking the model with a sudden change in capacity
            self.training_step += 1 if self.training_step < 1000 else 999
            self.capacity = 0.125 + ((1-0.125) * (1. / (self.training_step)))
        # Compute B-th percentile for router weights to determine the capacity threshold
        k = int(self.capacity * s)
        top_k_values, _ = torch.topk(weights, k, dim=1, sorted=True)
        threshold = top_k_values[:, -1]

        # Determine which tokens exceed the threshold
        selected_mask = weights > threshold.unsqueeze(-1)

        # Process only selected tokens through the block
        processed_tokens = torch.zeros_like(x)
        for i in range(b):
            # Process tokens for each block
            selected_tokens = x[i][selected_mask[i]]
            position_ids = position_ids[i][selected_mask[i]].unsqueeze(0)
            cache_position = cache_position[selected_mask[i]]
            if selected_tokens.size(0) > 0:
                processed_tokens[i][selected_mask[i]] = self.block(
                    selected_tokens.unsqueeze(0),
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    **kwargs
                ) * weights[i][selected_mask[i]]  # TODO: Check if this works

        # Combine processed tokens with unprocessed ones
        output = processed_tokens + (
                x * (~selected_mask).unsqueeze(-1).to(x.dtype)  # TODO: Warn?
        )
        return output

def apply_to_hf(model, enabled=True):
    if not enabled:
        return model

    new_layers = nn.ModuleList()
    for i, layer in enumerate(model.model.layers):
        if i %2 != 0:
            new_layer = MoD(0.125, layer)
        else:
            new_layer = layer
        new_layers.append(new_layer)

    model.model.layers = new_layers
    return model