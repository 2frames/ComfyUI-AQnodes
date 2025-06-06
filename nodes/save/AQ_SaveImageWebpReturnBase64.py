import os
import folder_paths as comfy_paths
import torch
from PIL import Image, ImageFilter, ImageEnhance, ImageOps, ImageDraw, ImageChops, ImageFont
import numpy as np
import torch.nn.functional as F
from server import PromptServer
import piexif
import piexif.helper
import json
import requests
import io
import base64


MODELS_DIR =  comfy_paths.models_dir
AQ_IMAGES_OUTPUT_DIR =  comfy_paths.get_output_directory()


class AQ_SaveImageWebpReturnBase64:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "name": ("STRING", {"default": "image"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "prompt_text": ("STRING", {"default": ""}),
                "cfg": ("FLOAT", { "default": 0, "min": 0.0, "max": 50.0 }),
                "steps": ("INT", {"default": 0}),   
                "scheduler": ("STRING", {"default": ""}),
                "sampler": ("STRING", {"default": ""}),
            },
            "hidden": {"unique_id": "UNIQUE_ID", 
                       "extra_pnginfo": "EXTRA_PNGINFO",
                       "prompt": "PROMPT"
                       },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "AQ/images"

    def save_images(self, images, name, seed, prompt_text, cfg, steps, scheduler, sampler, unique_id, extra_pnginfo, prompt):
        import random

        is_dev = os.environ.get("IS_DEV", False)
        resultsDev = []

        results = []
        index = 0
        server = PromptServer.instance 
        #subfolder = os.path.normpath(os.path.join('user_', server.client_id, server.last_prompt_id))
        subfolder = os.path.normpath(os.path.join('user_', server.client_id))
        if not os.path.exists(AQ_IMAGES_OUTPUT_DIR):
            os.makedirs(AQ_IMAGES_OUTPUT_DIR)

        out_dir = os.path.join(AQ_IMAGES_OUTPUT_DIR, subfolder)

        for tensor in images:
            index += 1
            name_with_index = f"{name}_{index}_{seed}_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5)) + '.webp'
            array = 255.0 * tensor.cpu().numpy()
            image = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))

            import io
            import base64
            buffer = io.BytesIO()

            userdata = {
                'prompt': prompt_text,
                'cfg': cfg,
                'steps': steps,
                'scheduler': scheduler,
                'sampler': sampler,
                'seed': seed,
            }

            exifData = piexif.helper.UserComment.dump(
                json.dumps(userdata),
                encoding="unicode"
            )

            # Create EXIF dictionary
            exif_dict = {"Exif": {piexif.ExifIFD.UserComment: exifData}}

            # Convert the EXIF dictionary to bytes
            exif_bytes = piexif.dump(exif_dict)


            image.save(buffer, format="WEBP")
            webp_image_data = buffer.getvalue()
            webp_base64 = base64.b64encode(webp_image_data).decode('utf-8')

            full_output_path = f"{out_dir}/{name_with_index}"
            if os.path.commonpath((AQ_IMAGES_OUTPUT_DIR, os.path.abspath(full_output_path))) != AQ_IMAGES_OUTPUT_DIR:
                print("Saving image outside the output folder is not allowed.")
                return {}
            
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            image.save(full_output_path, format="WEBP", exif=exif_bytes)


            full_output_path_txt = full_output_path.replace(".webp", ".txt")
            with open(full_output_path_txt, "w") as f:
                f.write(prompt_text)

            if is_dev and extra_pnginfo is not None:
                resultsDev.append({
                    "filename": name_with_index + " [output]",
                    "subfolder": out_dir,
                    "type": "base64Image",
                    "name": name_with_index
                })

                return { "ui": { "images": resultsDev } }
        
            results.append(
                {
                    "type": "base64Image",
                    "name": name_with_index,
                    "data": webp_base64,
                    "seed": seed,
                }
            )


        return ({"ui": {"images": results}}) 