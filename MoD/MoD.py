import os

import torch
import torch.nn as nn
from typing import Optional, Tuple, Any

from transformers import PreTrainedModel



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
                x: torch.Tensor,
                causal_mask: torch.Tensor,
                position_ids: torch.Tensor,
                past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]],
                output_attentions: bool,
                use_cache: bool,
                cache_position: Optional[torch.Tensor] = None,
                **kwargs: Any
                ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        b, s, d = x.shape
        weights = self.router(x)
        if self.router.training:
            self.training_step += 1 if self.training_step < 1000 else 999
            self.capacity = 0.125 + ((1 - 0.125) * (1. / self.training_step))

        k = int(self.capacity * s)
        top_k_values, _ = torch.topk(weights, k, dim=1, sorted=True)
        threshold = top_k_values[:, -1]
        selected_mask = weights > threshold.unsqueeze(-1)

        processed_tokens = torch.zeros_like(x)
        for i in range(b):
            selected_tokens = x[i][selected_mask[i]]
            selected_position_ids = position_ids[i][selected_mask[i]].unsqueeze(0)

            if selected_tokens.size(0) > 0:
                # Gestione dinamica del cache_position
                if cache_position is not None:
                    selected_cache_position = cache_position[selected_mask[i]]
                    processed_tokens[i][selected_mask[i]] = self.block(
                        selected_tokens.unsqueeze(0),
                        attention_mask=causal_mask,
                        position_ids=selected_position_ids,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=selected_cache_position,
                        **kwargs
                    )[0] * weights[i][selected_mask[i]].unsqueeze(-1)
                else:
                    processed_tokens[i][selected_mask[i]] = self.block(
                        selected_tokens.unsqueeze(0),
                        attention_mask=causal_mask,
                        position_ids=selected_position_ids,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        **kwargs
                    )[0] * weights[i][selected_mask[i]].unsqueeze(-1)

        output = processed_tokens + (x * (~selected_mask).unsqueeze(-1).to(x.dtype))
        return (output,) if len(output.shape) == 3 else (output.unsqueeze(0),)


def apply_mod_to_hf(model: PreTrainedModel, enabled: bool = True) -> PreTrainedModel:
    if not enabled:
        return model

    new_layers = nn.ModuleList()
    for i, layer in enumerate(model.model.layers):
        if i % 2 != 0:
            new_layer = MoD(0.125, layer)
        else:
            new_layer = layer
        new_layers.append(new_layer)

    model.model.layers = new_layers
    # Prendi il nome della classe corrente
    class_name = model.__class__.__name__

    # Inserisci 'MoD' prima di 'For'
    if 'For' in class_name:
        parts = class_name.split('For', 1)
        modified_class_name = parts[0] + 'MoDFor' + parts[1]
    else:
        modified_class_name = 'MoD' + class_name  # Se non trova 'For', aggiunge 'MoD' all'inizio

    # Ora puoi impostare l'attributo __name__ della classe dell'istanza
    model.__class__.__name__ = modified_class_name

    return model
