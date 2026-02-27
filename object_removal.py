import os
import torch
from diffusers import LTXLatentUpsamplePipeline, AutoencoderKLLTXVideo
from .wrapper.transformer_ltx import LTXVideoTransformer3DModel
from diffusers.pipelines.ltx.modeling_latent_upsampler import LTXLatentUpsamplerModel
from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
from transformers import T5TokenizerFast
from safetensors.torch import load_file
from .OmnimatteZero import OmnimatteZero
from contextlib import contextmanager
import sys
from diffusers import GGUFQuantizationConfig
from accelerate import init_empty_weights

@contextmanager
def temp_patch_module_attr(module_name: str, attr_name: str, new_obj):
    mod = sys.modules.get(module_name)
    if mod is None:
        yield
        return
    had = hasattr(mod, attr_name)
    orig = getattr(mod, attr_name, None)
    setattr(mod, attr_name, new_obj)
    try:
        yield
    finally:
        if had:
            setattr(mod, attr_name, orig)
        else:
            try:
                delattr(mod, attr_name)
            except Exception:
                pass

def round_to_nearest_resolution_acceptable_by_vae(height, width):
    height = height - (height % 32)
    width = width - (width % 32)
    return height, width

def load_model(model_path, gguf_path, vae_path, cur_dir, compose_mode=False, device=torch.device("cuda")):
    if isinstance(device, str):
        device = torch.device(device)

    # Critical: float32 on CPU for stability & no dtype mismatch
    dtype = torch.bfloat16 if device.type == "cpu" else torch.bfloat16

    # VAE loading
    if compose_mode:
        from .foreground_composition import MyAutoencoderKLLTXVideo
        with temp_patch_module_attr("diffusers", "AutoencoderKLLTXVideo", MyAutoencoderKLLTXVideo):
            try:
                vae_config = MyAutoencoderKLLTXVideo.load_config(os.path.join(cur_dir, "LTX-Video-0.9.7-diffusers/vae/config.json"))
                vae = MyAutoencoderKLLTXVideo.from_config(vae_config, torch_dtype=dtype)
                vae.load_state_dict(load_file(vae_path), strict=False)
            except Exception:
                print("load vae error, using normal load mode")
                vae = MyAutoencoderKLLTXVideo.from_single_file(
                    vae_path,
                    config=os.path.join(cur_dir, "LTX-Video-0.9.7-diffusers/vae/config.json"),
                    torch_dtype=dtype
                )
    else:
        try:
            vae_config = AutoencoderKLLTXVideo.load_config(os.path.join(cur_dir, "LTX-Video-0.9.7-diffusers/vae/config.json"))
            vae = AutoencoderKLLTXVideo.from_config(vae_config, torch_dtype=dtype)
            vae.load_state_dict(load_file(vae_path), strict=False)
        except Exception:
            print("load vae error, using normal load mode")
            vae = AutoencoderKLLTXVideo.from_single_file(
                vae_path,
                config=os.path.join(cur_dir, "LTX-Video-0.9.7-diffusers/vae/config.json"),
                torch_dtype=dtype
            )

    vae = vae.to(device, dtype)

    # Transformer loading
    with temp_patch_module_attr("diffusers", "LTXVideoTransformer3DModel", LTXVideoTransformer3DModel):
        if gguf_path is not None:
            transformer = LTXVideoTransformer3DModel.from_single_file(
                gguf_path,
                config=os.path.join(cur_dir, "LTX-Video-0.9.7-diffusers/transformer"),
                quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
                torch_dtype=dtype
            )
        else:
            try:
                transformer = LTXVideoTransformer3DModel.from_single_file(
                    model_path,
                    config=os.path.join(cur_dir, "LTX-Video-0.9.7-diffusers/transformer/config.json"),
                    torch_dtype=dtype
                )
            except Exception:
                print("load model error, using normal load mode")
                transformer_config = LTXVideoTransformer3DModel.load_config(
                    os.path.join(cur_dir, "LTX-Video-0.9.7-diffusers/transformer/config.json")
                )
                with init_empty_weights():
                    transformer = LTXVideoTransformer3DModel.from_config(transformer_config, torch_dtype=dtype)
                transformer.load_state_dict(load_file(model_path), strict=False, assign=True)

    transformer = transformer.to(device)

    # Pipeline — no extra torch_dtype
    pipe = OmnimatteZero.from_pretrained(
        os.path.join(cur_dir, "LTX-Video-0.9.7-diffusers"),
        text_encoder=None,
        vae=vae,
        transformer=transformer,
    )

    if pipe.device != device:
        pipe = pipe.to(device)

    pipe.vae.enable_tiling()

    return pipe


def get_diffusers_mask(tokenizer, prompt, device):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=128,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    prompt_attention_mask = text_inputs.attention_mask.bool().to(device)
    prompt_attention_mask = prompt_attention_mask.view(1, -1).repeat(1, 1)
    return prompt_attention_mask


def inference(pipe, prompt_embeds, negative_prompt_embeds, video, mask, num_frames, expected_height, expected_width,
              seed, guidance_scale, num_inference_steps, cur_dir, device):
    condition1 = LTXVideoCondition(video=video, frame_index=0)
    condition2 = LTXVideoCondition(video=mask, frame_index=0)

    prompt = "Empty"
    negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"

    tokenizer = T5TokenizerFast.from_pretrained(os.path.join(cur_dir, "LTX-Video-0.9.7-diffusers/tokenizer"))
    prompt_attention_mask = get_diffusers_mask(tokenizer, prompt, device)
    negative_prompt_attention_mask = get_diffusers_mask(tokenizer, negative_prompt, device)

    downscaled_height, downscaled_width = round_to_nearest_resolution_acceptable_by_vae(expected_height, expected_width)

    video = pipe.my_call(
        conditions=[condition1, condition2],
        prompt=None,
        negative_prompt=None,
        width=downscaled_width,
        height=downscaled_height,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=torch.Generator().manual_seed(seed),
        prompt_embeds=prompt_embeds,
        prompt_attention_mask=prompt_attention_mask,
        negative_prompt_embeds=negative_prompt_embeds,
        negative_prompt_attention_mask=negative_prompt_attention_mask,
        output_type="pil",
    )
    video = video.frames[0]
    return video
