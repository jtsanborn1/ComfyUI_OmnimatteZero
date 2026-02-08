 # !/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import torch
import os
from diffusers.hooks import apply_group_offloading
import folder_paths
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io
import nodes
from .model_loader_utils import clear_comfyui_cache,load_images_list,tensor2pillist_upscale
from .object_removal import load_model,inference
MAX_SEED = np.iinfo(np.int32).max
node_cr_path = os.path.dirname(os.path.abspath(__file__))
device = torch.device(
    "cuda:0") if torch.cuda.is_available() else torch.device(
    "mps") if torch.backends.mps.is_available() else torch.device(
    "cpu")

weigths_gguf_current_path = os.path.join(folder_paths.models_dir, "gguf")
if not os.path.exists(weigths_gguf_current_path):
    os.makedirs(weigths_gguf_current_path)

folder_paths.add_model_folder_path("gguf", weigths_gguf_current_path) #  gguf dir


class OmnimatteZero_SM_Model(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        
        return io.Schema(
            node_id="OmnimatteZero_SM_Model",
            display_name="OmnimatteZero_SM_Model",
            category="OmnimatteZero",
            inputs=[
                io.Combo.Input("dit",options= ["none"] + folder_paths.get_filename_list("diffusion_models") ),
                io.Combo.Input("gguf",options= ["none"] + folder_paths.get_filename_list("gguf")),  
                io.Combo.Input("vae",options= ["none"] + folder_paths.get_filename_list("vae")),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                ],
            )
    @classmethod
    def execute(cls, dit,gguf,vae) -> io.NodeOutput:
        clear_comfyui_cache()
        dit_path=folder_paths.get_full_path("diffusion_models", dit) if dit != "none" else None
        gguf_path=folder_paths.get_full_path("gguf", gguf) if gguf != "none" else None
        vae_path=folder_paths.get_full_path("vae", vae) if vae != "none" else None
        pipeline = load_model(dit_path,gguf_path,vae_path,node_cr_path)
        return io.NodeOutput(pipeline)
    

class OmnimatteZero_SM_KSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="OmnimatteZero_SM_KSampler",
            display_name="OmnimatteZero_SM_KSampler",
            category="OmnimatteZero",
            inputs=[
                io.Model.Input("model"),
                io.Image.Input("images"),
                io.Image.Input("mask"),
                io.Int.Input("width", default=768, min=256, max=nodes.MAX_RESOLUTION,step=32,display_mode=io.NumberDisplay.number),
                io.Int.Input("height", default=512, min=256, max=nodes.MAX_RESOLUTION,step=32,display_mode=io.NumberDisplay.number),
                io.Int.Input("num_frames", default=121, min=8, max=nodes.MAX_RESOLUTION,step=1,display_mode=io.NumberDisplay.number),
                io.Float.Input("guidance_scale", default=3.0, min=1.0, max=50.0,step=0.1,display_mode=io.NumberDisplay.number),           
                io.Int.Input("steps", default=25, min=1, max=1024,step=1,display_mode=io.NumberDisplay.number),
                io.Int.Input("seed", default=0, min=0, max=MAX_SEED,display_mode=io.NumberDisplay.number),
                io.Int.Input("block_num", default=5, min=0, max=MAX_SEED,display_mode=io.NumberDisplay.number),
                io.Conditioning.Input("positive",optional=True),
                io.Conditioning.Input("negative",optional=True),
            ], # io.Float.Input("noise", default=0.0, min=0.0, max=1.0,step=0.01,display_mode=io.NumberDisplay.number),
            outputs=[
                io.Image.Output(display_name="images"),
            ],
        )
    @classmethod
    def execute(cls, model,images,mask,width,height,num_frames,guidance_scale,steps,seed,block_num,positive=None,negative=None,) -> io.NodeOutput:
        if mask.shape[0]==1:
            mask=mask.repeat(num_frames,1,1,1)
        if images.shape[0]==1:
            images=images.repeat(num_frames,1,1,1)
        num_frames=min(images.shape[0],num_frames,mask.shape[0])
        images=tensor2pillist_upscale(images, width, height)[:num_frames]
        mask=tensor2pillist_upscale(mask, width, height)[:num_frames]

        clear_comfyui_cache()
        # apply offloading
        if block_num>0:
            apply_group_offloading(model.transformer, onload_device=torch.device("cuda"), offload_type="block_level", num_blocks_per_group=block_num)
        else:
            model.enable_model_cpu_offload() 
        # infer
        if positive is None or negative is None: 
            positive=torch.load(os.path.join(node_cr_path, "wrapper/positive.pt")).to(device,torch.bfloat16)
            negative=torch.load(os.path.join(node_cr_path, "wrapper/negative.pt")).to(device,torch.bfloat16)
        else:
            positive=positive[0][0].to(device,torch.bfloat16)
            negative=negative[0][0].to(device,torch.bfloat16)
        #print("OmnimatteZero_SM_KSampler:",positive.shape,negative.shape)#torch.Size([1, 128, 4096])
        images=inference(model, positive, negative, images, mask, num_frames,height, width,seed,guidance_scale,
              steps, node_cr_path,device)
        return io.NodeOutput(load_images_list(images))



class OmnimatteZero_SM_Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            OmnimatteZero_SM_Model,
            OmnimatteZero_SM_KSampler,
        ]
async def comfy_entrypoint() -> OmnimatteZero_SM_Extension:  # ComfyUI calls this to load your extension and its nodes.
    return OmnimatteZero_SM_Extension()
