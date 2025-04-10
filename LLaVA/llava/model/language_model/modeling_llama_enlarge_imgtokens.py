import inspect
import math
import warnings
from functools import partial
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

import transformers
from transformers.models.llama.modeling_llama import *
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter,
    _prepare_4d_attention_mask,
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)


def model_forward(
    self,
    input_ids: torch.LongTensor = None,
    modality_indicators: torch.Tensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    img_start_end_idx: Optional[List[List[int]]] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:

    layer_id = 30
    resize_len = 576
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape[:2]
    elif inputs_embeds is not None:
        batch_size, seq_length = inputs_embeds.shape[:2]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False
    past_key_values_length = 0
    if use_cache:
        use_legacy_cache = not isinstance(past_key_values, Cache)
        if use_legacy_cache:
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        past_key_values_length = past_key_values.get_usable_length(seq_length)

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
        )
        position_ids = position_ids.unsqueeze(0)

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if self._use_flash_attention_2:
        # 2d mask is passed through the layers
        attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        if (img_start_end_idx is not None) and (attention_mask is not None):
            delta_len = resize_len - (img_start_end_idx[0][1]-img_start_end_idx[0][0])
            delta_mask = torch.ones([batch_size, delta_len]).bool()
            new_attention_mask = torch.cat([delta_mask, attention_mask.cpu()], dim=1)
        
    elif self._use_sdpa and not output_attentions:
        # output_attentions=True can not be supported when using SDPA, and we fall back on
        # the manual implementation that requires a 4D causal mask in all cases.
        
        # enlarge the attention_mask, which can be replaced in the later part
        # if attention_mask is not None:
        #     delta_len = resize_len - (img_start_end_idx[0][1]-img_start_end_idx[0][0])
        #     delta_mask = torch.ones([batch_size, delta_len]).bool()
            # new_attention_mask = torch.cat([delta_mask, attention_mask.cpu()], dim=1)
            # new_mask_len = new_attention_mask.shape[1]
            # new_attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
            #         new_attention_mask,
            #         (batch_size, new_mask_len),
            #         inputs_embeds,
            #         past_key_values_length,
            #     )
            
        attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
        )
    else:
        # 4d mask is passed through the layers
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

    # embed positions
    hidden_states = inputs_embeds

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None
    # import ipdb; ipdb.set_trace()
    for idx, decoder_layer in enumerate(self.layers):
        # Upsampling the img tokens
        if (idx==layer_id) and (img_start_end_idx is not None) and (seq_length > 100):
            new_hidden_states = []
            new_position_ids = []
            # position_ids: (L')
            delta_len = resize_len - (img_start_end_idx[0][1]-img_start_end_idx[0][0])
            L_ = delta_len + seq_length
            new_position_ids = torch.arange(L_).unsqueeze(0).to(position_ids.device)
            if attention_mask is not None:
                attention_mask = new_attention_mask.to(attention_mask.device)

            for bs_idx, hidden_state_ in enumerate(hidden_states):
                start_end = img_start_end_idx[bs_idx]
                img_tokens = hidden_state_[start_end[0]: start_end[1], :] 
                img_tokens = img_tokens.t().unsqueeze(0)
                img_tokens = F.interpolate(img_tokens, size=(resize_len), mode='linear', align_corners=True)
                img_tokens = img_tokens.squeeze(0).t()
                # hidden_state_: (L', D)
                hidden_state_ = torch.cat((hidden_state_[0:start_end[0], :], img_tokens, hidden_state_[start_end[1]:, :]), dim=0)
                del img_tokens
                new_hidden_states.append(hidden_state_)
            
            del hidden_states
            hidden_states = torch.stack(new_hidden_states, dim=0) # B, L', D
            position_ids = new_position_ids
            
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                attention_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        hidden_states = layer_outputs[0]
        # TODO try to increase the image tokens in the later layers
        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = None
    if use_cache:
        next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )

def model_forward_interpolate_channels(
    self,
    input_ids: torch.LongTensor = None,
    modality_indicators: torch.Tensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    img_start_end_idx: Optional[List[List[int]]] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:

    layer_id = 30
    resize_len = 576
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape[:2]
    elif inputs_embeds is not None:
        batch_size, seq_length = inputs_embeds.shape[:2]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    past_key_values_length = 0
    if use_cache:
        use_legacy_cache = not isinstance(past_key_values, Cache)
        if use_legacy_cache:
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        past_key_values_length = past_key_values.get_usable_length(seq_length)

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
        )
        position_ids = position_ids.unsqueeze(0)

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if self._use_flash_attention_2:
        # 2d mask is passed through the layers
        attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        if (img_start_end_idx is not None) and (attention_mask is not None):
            # judge whether img_end whether exceed the maximum length (context length=2048)
            output_len = img_start_end_idx[0][1]-img_start_end_idx[0][0]
            k = int(resize_len/output_len)
            new_resize_len = k * output_len
            delta_len = new_resize_len - output_len
            delta_mask = torch.ones([batch_size, delta_len]).bool()
            new_attention_mask = torch.cat([delta_mask, attention_mask.cpu()], dim=1)
        
    elif self._use_sdpa and not output_attentions:
        # output_attentions=True can not be supported when using SDPA, and we fall back on
        # the manual implementation that requires a 4D causal mask in all cases.
        
        # enlarge the attention_mask, which can be replaced in the later part
        # if attention_mask is not None:
        #     delta_len = resize_len - (img_start_end_idx[0][1]-img_start_end_idx[0][0])
        #     delta_mask = torch.ones([batch_size, delta_len]).bool()
            # new_attention_mask = torch.cat([delta_mask, attention_mask.cpu()], dim=1)
            # new_mask_len = new_attention_mask.shape[1]
            # new_attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
            #         new_attention_mask,
            #         (batch_size, new_mask_len),
            #         inputs_embeds,
            #         past_key_values_length,
            #     )
            
        attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
        )
    else:
        # 4d mask is passed through the layers
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

    # embed positions
    hidden_states = inputs_embeds

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None
    for idx, decoder_layer in enumerate(self.layers):
        # Upsampling the img tokens
        if (idx==layer_id) and (img_start_end_idx is not None) and (seq_length > 100):
            new_hidden_states = []
            new_position_ids = []
            D = hidden_states.shape[-1]
            output_len = img_start_end_idx[0][1] - img_start_end_idx[0][0]
            k = int(resize_len/(img_start_end_idx[0][1] - img_start_end_idx[0][0]))
            new_resize_len = k * output_len
            delta_len = new_resize_len - output_len
            L_ = delta_len + seq_length
            new_position_ids = torch.arange(L_).unsqueeze(0).to(position_ids.device)
            if attention_mask is not None:
                attention_mask = new_attention_mask.to(attention_mask.device)

            for bs_idx, hidden_state_ in enumerate(hidden_states):
                start_end = img_start_end_idx[bs_idx]
                img_tokens = hidden_state_[start_end[0]: start_end[1], :] 
                img_tokens = img_tokens.unsqueeze(0)
                img_tokens = F.interpolate(img_tokens, size=(k*D), mode='linear', align_corners=True)
                img_tokens = img_tokens.view(1, -1, D)
                img_tokens = img_tokens.squeeze(0)
                # hidden_state_: (L', D)
                hidden_state_ = torch.cat((hidden_state_[0:start_end[0], :], img_tokens, hidden_state_[start_end[1]:, :]), dim=0)
                del img_tokens
                new_hidden_states.append(hidden_state_)
            
            del hidden_states
            hidden_states = torch.stack(new_hidden_states, dim=0) # B, L', D
            position_ids = new_position_ids
            
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        try:
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
        except:
            print(f"hidden_states.shape:{hidden_states.shape}, attention_mask.shape:{attention_mask.shape}, position_ids.shape:{position_ids.shape}")
            
        hidden_states = layer_outputs[0]
        # TODO try to increase the image tokens in the later layers
        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = None
    if use_cache:
        next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


def hack_llama_model_increase_patches():
    transformers.models.llama.modeling_llama.LlamaModel.forward = model_forward

def hack_llama_model_increase_patches_interpolate_channels():
    transformers.models.llama.modeling_llama.LlamaModel.forward = model_forward_interpolate_channels


if __name__ == '__main__':
    # from llava_llama2 import hack_llava
    # hack_llava()
    # hack_llama_model_increase_patches()
    hack_llama_model_increase_patches_interpolate_channels()
    
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path
    from llava.eval.run_llava import eval_model
    # from run_llava2 import eval_model

    model_path = "/home/liyifa11/MyCodes/PronePatches/LLaVA/checkpoints/llava-v1.5-7b-SAWI_enlarge_imgtokens_L28"

    # tokenizer, model, image_processor, context_len = load_pretrained_model(
    #     model_path=model_path,
    #     model_base=None,
    #     model_name=get_model_name_from_path(model_path)
    # )    
    prompt = "What are the things I should be cautious about when I visit here?"
    image_file = "https://llava-vl.github.io/static/images/view.jpg"

    args = type('Args', (), {
        "model_path": model_path,
        "model_base": None,
        "model_name": get_model_name_from_path(model_path),
        "query": prompt,
        "conv_mode": None,
        "image_file": image_file,
        "sep": ",",
        "temperature": 0,
        "top_p": None,
        "num_beams": 1,
        "max_new_tokens": 512
    })()

    eval_model(args)