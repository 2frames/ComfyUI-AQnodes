import torch
import os
import folder_paths
import numpy as np
from PIL import Image
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    BitsAndBytesConfig,
    AutoProcessor,
)
from pathlib import Path
import json

model_directory = os.path.join(folder_paths.models_dir, "VLM")
os.makedirs(model_directory, exist_ok=True)

class AQ_QwenLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (
                    [
                        "unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit",
                        "unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit",
                        "unsloth/Qwen2.5-VL-32B-Instruct-bnb-4bit",
                    ],
                    {"default": "unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit"},
                ),
                "quantization": (
                    ["none", "4bit"],
                    {"default": "4bit"},
                ),
                "attention": (
                    ["flash_attention_2", "sdpa", "eager"],
                    {"default": "sdpa"},
                ),
                "device": (
                    ["auto", "cuda:0", "cuda:1"],
                    {"default": "auto"},
                ),
            }
        }
    
    RETURN_TYPES = ("QWEN_MODEL",)
    RETURN_NAMES = ("qwen_model",)
    FUNCTION = "load_model"
    CATEGORY = "AQ/LLM"

    def load_model(self, model, quantization, attention, device):
        qwen_model = {"model": None, "processor": None, "model_path": None}
        
        # Setup model path
        model_name = model.rsplit("/", 1)[-1]
        model_path = os.path.join(model_directory, model_name)
        
        # Download if needed
        if not os.path.exists(model_path):
            print(f"Downloading Qwen model to: {model_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=model,
                local_dir=model_path,
                local_dir_use_symlinks=False
            )

        # Setup quantization
        if quantization == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif quantization == "8bit":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            quantization_config = None

        # Load model and processor
        qwen_model["model"] = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=device,
            attn_implementation=attention,
            quantization_config=quantization_config,
            trust_remote_code=True,
        )
        qwen_model["processor"] = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=True
        )
        qwen_model["model_path"] = model_path
        
        return (qwen_model,)

class AQ_Qwen:
    @classmethod
    def INPUT_TYPES(s):
        default_format = '''{
    "type": "object",
    "properties": {
      "user_idea": {
        "type": "string"
      },
      "new_scene_description": {
        "type": "string"
      },
      "new_image_type":{
        "type": "string"
      },
      "new_style":{
        "type": "string"
      }
    },
    "required": [
      "user_idea",
      "new_scene_description",
      "new_image_type",
      "new_style"
    ]
  }'''

        return {
            "required": {
                "qwen_model": ("QWEN_MODEL",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "system_message": ("STRING", {"default": "You are a helpful assistant.", "multiline": True}),
                "temperature": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 2.0, "step": 0.05}),
                "max_tokens": ("INT", {"default": 256, "min": 1, "max": 2048}),
                "top_p": ("FLOAT", {"default": 0.001, "min": 0.0, "max": 1.0, "step": 0.001}),
                "repetition_penalty": ("FLOAT", {"default": 1.05, "min": 1.0, "max": 2.0, "step": 0.01}),
                "min_pixels": ("INT", {"default": 224, "min": 16, "max": 1280}),
                "max_pixels": ("INT", {"default": 1280, "min": 224, "max": 2048}),
                "enable_json": ("BOOLEAN", {"default": False}),
                "json_format": ("STRING", {"default": default_format, "multiline": True}),
                "result_template": ("STRING", {
                    "default": "{json[new_scene_description]} in style {json[new_style]}, {json[new_image_type]}", 
                    "multiline": True
                }),
            },
            "optional": {
                "image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("response", "formatted_response")
    FUNCTION = "generate"
    CATEGORY = "AQ/LLM"

    def format_response(self, response, enable_json, json_format, result_template):
        if not enable_json:
            return response, response
            
        try:
            # First try to parse the response as JSON
            try:
                json_response = json.loads(response)
            except json.JSONDecodeError:
                # If response is not JSON, it means the model didn't follow the format
                print("Response is not valid JSON, returning as is")
                return response, response

            # If result_template is provided, format using it
            if result_template:
                try:
                    formatted_response = result_template.format(json=json_response)
                    return response, formatted_response
                except KeyError as e:
                    error_msg = f"Error: Missing required field in JSON response: {str(e)}"
                    print(error_msg)
                    return response, error_msg
                except Exception as e:
                    error_msg = f"Error formatting template: {str(e)}"
                    print(error_msg)
                    return response, error_msg

            return response, response

        except Exception as e:
            print(f"Error in JSON formatting: {str(e)}")
            return response, response

    def generate(self, qwen_model, prompt, system_message, temperature=0.1, 
                max_tokens=256, top_p=0.001, repetition_penalty=1.05,
                min_pixels=224, max_pixels=1280, enable_json=False,
                json_format="", result_template="", image=None):
        try:
            min_pixels = min_pixels * min_pixels
            max_pixels = max_pixels * max_pixels
            
            content = []
            
            # Handle system message
            if system_message:
                messages = [{"role": "system", "content": system_message}]
            else:
                messages = []
                
            # Handle image if provided
            user_content = []
            if image is not None:
                # Convert tensor to numpy array
                if isinstance(image, torch.Tensor):
                    image = image.cpu().numpy()
                
                # Handle ComfyUI image format (B, H, W, C)
                if len(image.shape) == 4:
                    image = image[0]  # Take first image from batch
                
                # Ensure proper type and range for PIL
                if image.dtype != np.uint8:
                    if image.max() <= 1:
                        image = (image * 255).astype(np.uint8)
                    else:
                        image = image.astype(np.uint8)
                
                # Convert numpy array to PIL Image and save temporarily
                img = Image.fromarray(image)
                image_path = Path(folder_paths.temp_directory) / "temp_qwen_image.png"
                img.save(str(image_path))
                
                # Add image to content
                user_content.append({
                    "type": "image",
                    "image": f"file://{image_path.resolve().as_posix()}",
                    "min_pixels": min_pixels,
                    "max_pixels": max_pixels,
                })

            # Modify prompt if JSON is enabled
            if enable_json and json_format:
                try:
                    schema = json.loads(json_format)
                    prompt = f"""Please provide your response in the following JSON format:
{json.dumps(schema, indent=2)}

{prompt}"""
                except json.JSONDecodeError:
                    print("Invalid JSON format provided, using prompt as is")
            
            # Add prompt
            user_content.append({"type": "text", "text": prompt})
            messages.append({"role": "user", "content": user_content})
            
            # Process with Qwen's template
            model_text = qwen_model["processor"].apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Process vision info
            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                messages,
                return_video_kwargs=True
            )
            
            # Prepare inputs
            inputs = qwen_model["processor"](
                text=[model_text],
                images=image_inputs,
                padding=True,
                return_tensors="pt",
                **video_kwargs,
            )
            
            # Move to correct device
            inputs = inputs.to(qwen_model["model"].device)
            
            # Generate response
            generated_ids = qwen_model["model"].generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                pad_token_id=qwen_model["processor"].tokenizer.pad_token_id,
                eos_token_id=qwen_model["processor"].tokenizer.eos_token_id,
            )
            
            # Process output
            generated_ids_trimmed = [
                out_ids[len(in_ids):] 
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            response = qwen_model["processor"].batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            
            # Format response if needed
            return self.format_response(response, enable_json, json_format, result_template)
            
        except Exception as e:
            print(f"Error in Qwen generation: {str(e)}")
            return ("")
            
        finally:
            # Clean up temp files
            if image is not None and 'image_path' in locals() and image_path.exists():
                image_path.unlink() 