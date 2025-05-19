import torch
import sys
import os
import importlib

class AQ_multiface_ApplyPulidFlux:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "pulid_flux": ("PULIDFLUX", ),
                "eva_clip": ("EVA_CLIP", ),
                "face_analysis": ("FACEANALYSIS", ),
                "images": ("IMAGE_LIST", ),
                "weight": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 5.0, "step": 0.05 }),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
            },
            "optional": {
                "attn_masks": ("IMAGE_LIST", ),
                "options": ("OPTIONS",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID"
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_multi_pulid_flux"
    CATEGORY = "AQ/Pulid"

    def apply_multi_pulid_flux(self, model, pulid_flux, eva_clip, face_analysis, images, weight, start_at, end_at, attn_masks=None, options={}, unique_id=None):
        # Rather than importing the class, we'll call the original node directly
        # We know the original node is successfully registered in ComfyUI
        from nodes import NODE_CLASS_MAPPINGS
        
        if "ApplyPulidFlux" not in NODE_CLASS_MAPPINGS:
            # This fallback will help with debugging if the node isn't found
            print("Available nodes:", list(NODE_CLASS_MAPPINGS.keys()))
            
            # Try an alternative approach - import from the PuLID_Flux module directly
            try:
                # Add the parent directory to sys.path to make absolute imports work
                parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                if parent_dir not in sys.path:
                    sys.path.insert(0, parent_dir)
                
                # Import the module (not relatively)
                print(f"Trying to import from {parent_dir}")
                pulidflux_module = importlib.import_module("ComfyUI_PuLID_Flux_ll.pulidflux")
                ApplyPulidFlux = pulidflux_module.ApplyPulidFlux
                apply_pulid = ApplyPulidFlux()
            except Exception as e:
                print(f"Error importing directly from module: {e}")
                raise ValueError(f"Could not find or import ApplyPulidFlux. Is ComfyUI_PuLID_Flux_ll installed? Error: {e}")
        else:
            # Use the registered node class
            ApplyPulidFlux = NODE_CLASS_MAPPINGS["ApplyPulidFlux"]
            apply_pulid = ApplyPulidFlux()
        
        # Work directly on the original model
        current_model = model
        
        # Get number of images in the list
        num_images = len(images)
        
        # Process each image-mask pair
        for i in range(num_images):
            # Get single image from the list
            single_image = images[i].unsqueeze(0) if len(images[i].shape) == 3 else images[i]
            
            # Extract corresponding mask if provided
            attn_mask = None
            if attn_masks is not None:
                if i < len(attn_masks):
                    attn_mask = attn_masks[i].unsqueeze(0) if len(attn_masks[i].shape) == 3 else attn_masks[i]
            
            # Apply PuLID Flux to this image-mask pair
            print(f"Processing image {i+1}/{num_images}")
            current_model, = apply_pulid.apply_pulid_flux(
                model=current_model,
                pulid_flux=pulid_flux,
                eva_clip=eva_clip,
                face_analysis=face_analysis,
                image=single_image,
                weight=weight,
                start_at=start_at,
                end_at=end_at,
                attn_mask=attn_mask,
                options=options,
                unique_id=unique_id
            )
        
        return (current_model,) 