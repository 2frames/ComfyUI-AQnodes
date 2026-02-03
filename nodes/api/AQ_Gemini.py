import requests
import base64
from PIL import Image
import io
import json
import numpy as np
import torch
from google import genai
from google.genai import types

class AQ_Gemini:
    @classmethod
    def INPUT_TYPES(s):
        default_schema = '''{
    "type": "object",
    "required": ["description", "style"],
    "properties": {
        "description": {
            "type": "string"
        },
        "style": {
            "type": "string"
        },
        "tags": {
            "type": "array",
            "items": {
                "type": "string"
            }
        }
    }
}'''
        
        models = [
            "gemma-3-12b-it",  # default
            "gemma-3-1b-it",
            "gemma-3-4b-it",
            "gemma-3-27b-it",
            "gemini-2.5-flash-preview-04-17",
            "gemini-2.5-pro-preview-05-06",
            "gemini-2.0-flash",
            "gemini-2.0-flash-preview-image-generation",
            "gemini-2.0-flash-lite",
            "gemini-1.5-flash",
            "gemini-1.5-flash-8b",
            "gemini-1.5-pro",
            "gemini-embedding-exp",
            "imagen-3.0-generate-002",
            "veo-2.0-generate-001",
            "gemini-2.0-flash-live-001",
            "custom"  # Option for custom model
        ]
        
        return {
            "required": {
                "gemini_api_key": ("STRING", {"default": ""}),
                "model_selection": (models, {"default": "gemma-3-12b-it"}),
                "custom_model": ("STRING", {"default": "", "multiline": False}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "system_message": ("STRING", {"default": "You are a helpful assistant.", "multiline": True}),
                "temperature": ("FLOAT", {"default": 1, "min": 0.0, "max": 2.0, "step": 0.05}),
                "top_k": ("INT", {"default": 64, "min": 0, "max": 100}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "enable_json": ("BOOLEAN", {"default": False}),
                "json_schema": ("STRING", {"default": default_schema, "multiline": True}),
                "result_template": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING",)
    RETURN_NAMES = ("response", "formatted_response", "instruction")
    FUNCTION = "generate"
    CATEGORY = "Aquasite/LLM"

    def generate(self, gemini_api_key, model_selection, custom_model, prompt, system_message, temperature=0.8, 
                top_k=64, top_p=0.95, enable_json=False, json_schema="", result_template="", image=None):
        
        instruction = """
How to use AQ_Gemini node:
1. Provide your Gemini API key in the 'gemini_api_key' field
2. Select a model from the dropdown or choose 'custom' and enter your model name
3. Enter your prompt in the 'prompt' field
4. Customize the system message to control the assistant's behavior
5. Optionally provide an image for image-based prompts
6. For JSON output:
   - Enable 'enable_json'
   - Provide JSON schema in 'json_schema'
   - Use 'result_template' to format the response

Available Models:
- Gemma Models: gemma-3-{1b,4b,12b,27b}-it
- Gemini 2.5: flash-preview, pro-preview
- Gemini 2.0: flash, flash-preview-image-generation, flash-lite, flash-live
- Gemini 1.5: flash, flash-8b, pro
- Special Models: gemini-embedding-exp, imagen-3.0, veo-2.0

Note: If any error occurs, empty strings will be returned instead of raising an error.
"""

        try:
            if not gemini_api_key:
                return ("", "", instruction)

            # Initialize Gemini client
            client = genai.Client(api_key=gemini_api_key)
            
            # Determine which model to use
            model_name = custom_model if model_selection == "custom" and custom_model else model_selection

            # Prepare contents list
            contents = []

            # Add system message if provided
            if system_message:
                contents.append(
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=system_message)]
                    )
                )

            # Handle image if provided
            if image is not None:
                # Convert tensor to numpy array
                if isinstance(image, torch.Tensor):
                    image = image.cpu().numpy()
                
                # Handle ComfyUI image format (B, H, W, C)
                if len(image.shape) == 4:
                    image = image[0]
                
                # Ensure proper type and range for PIL
                if image.dtype != np.uint8:
                    if image.max() <= 1:
                        image = (image * 255).astype(np.uint8)
                    else:
                        image = image.astype(np.uint8)
                
                # Convert numpy array to PIL Image and then to base64
                img = Image.fromarray(image)
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format="PNG")
                img_byte_arr.seek(0)
                img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
                
                # Create content with image and prompt
                contents.append(
                    types.Content(
                        role="user",
                        parts=[{
                            "inline_data": {
                                "mime_type": "image/png",
                                "data": img_base64
                            }
                        },
                        {
                            "text": prompt
                        }]
                    )
                )
            else:
                # Text-only content
                contents.append(
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=prompt)]
                    )
                )

            # Configure generation settings
            generate_config = types.GenerateContentConfig(
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )

            # Add JSON schema if enabled and not a Gemma-3 model
            is_gemma_model = model_name.startswith("gemma-3")
            if enable_json and json_schema and not is_gemma_model:
                try:
                    schema_dict = json.loads(json_schema)
                    generate_config.response_mime_type = "application/json"
                    generate_config.response_schema = self._convert_schema_dict_to_genai(schema_dict)
                except json.JSONDecodeError:
                    print("Invalid JSON schema provided")

            # Generate content
            response = ""
            for chunk in client.models.generate_content_stream(
                model=model_name,
                contents=contents,
                config=generate_config,
            ):
                if chunk.text:
                    response += chunk.text

            formatted_response = response

            # Handle template formatting if result_template is provided and response might be JSON
            if result_template and enable_json:
                success, json_response = self.try_parse_json(response)
                if success:
                    try:
                        formatted_response = result_template.format(json=json_response)
                    except KeyError as e:
                        print(f"Template formatting error: {str(e)}")
                        formatted_response = response
                else:
                    print("Response is not valid JSON")
                    formatted_response = response

            return (response, formatted_response, instruction)
            
        except Exception as e:
            print(f"Error in Gemini API: {str(e)}")
            return ("", "", instruction)

    def _convert_schema_dict_to_genai(self, schema_dict):
        """Convert a JSON schema dictionary to genai Schema object."""
        schema_type = schema_dict.get("type", "object")
        
        if schema_type == "object":
            properties = {}
            for prop_name, prop_schema in schema_dict.get("properties", {}).items():
                properties[prop_name] = self._convert_schema_dict_to_genai(prop_schema)
            
            return genai.types.Schema(
                type=genai.types.Type.OBJECT,
                required=schema_dict.get("required", []),
                properties=properties
            )
        
        elif schema_type == "array":
            items_schema = self._convert_schema_dict_to_genai(schema_dict.get("items", {}))
            return genai.types.Schema(
                type=genai.types.Type.ARRAY,
                items=items_schema
            )
        
        elif schema_type == "string":
            return genai.types.Schema(type=genai.types.Type.STRING)
        
        elif schema_type == "number":
            return genai.types.Schema(type=genai.types.Type.NUMBER)
        
        elif schema_type == "integer":
            return genai.types.Schema(type=genai.types.Type.INTEGER)
        
        elif schema_type == "boolean":
            return genai.types.Schema(type=genai.types.Type.BOOLEAN)
        
        else:
            return genai.types.Schema(type=genai.types.Type.STRING)

    def prepare_json(self, text):
        """Clean up text that might contain markdown JSON code blocks."""
        # Remove markdown JSON code block markers
        text = text.replace("```json", "").replace("```", "").strip()
        return text

    def try_parse_json(self, text):
        """Try to parse JSON from text, handling markdown code blocks.
        Returns tuple (success, result) where result is either parsed JSON or original text."""
        try:
            # First clean up any markdown formatting
            cleaned_text = self.prepare_json(text)
            # Try to parse as JSON
            return True, json.loads(cleaned_text)
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {str(e)}")
            return False, text
        except Exception as e:
            print(f"Unexpected error while parsing JSON: {str(e)}")
            return False, text


class AQ_Gemini_acstep15:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "gemini_api_key": ("STRING", {"default": ""}),
                "model_selection": ([
                    "gemma-3-12b-it",  # default
                    "gemma-3-1b-it",
                    "gemma-3-4b-it",
                    "gemma-3-27b-it",
                    "gemini-flash-latest",
                    "gemini-flash-lite-latest",
                    "gemini-2.5-flash",
                    "gemini-2.5-flash-lite",
                    "gemini-2.5-pro",
                    "gemini-3-flash-preview",
                    "gemini-3-pro-preview",
                    "custom"  # Option for custom model
                ], {"default": "gemma-3-12b-it"}),
                "custom_model": ("STRING", {"default": "", "multiline": False}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "system_message": ("STRING", {"default": "You are a helpful assistant.", "multiline": True}),
                "temperature": ("FLOAT", {"default": 1, "min": 0.0, "max": 2.0, "step": 0.05}),
                "top_k": ("INT", {"default": 64, "min": 0, "max": 100}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = (
        "STRING",
        "STRING",
        "STRING",
        "INT",
        "FLOAT",
        "STRING",
        "STRING",
        "STRING",
    )
    RETURN_NAMES = (
        "response_json",
        "tags",
        "lyrics",
        "bpm",
        "duration",
        "timesignature",
        "language",
        "keyscale",
    )
    FUNCTION = "generate"
    CATEGORY = "Aquasite/LLM"

    def generate(self, gemini_api_key, model_selection, custom_model, prompt, system_message, temperature=0.8,
                top_k=64, top_p=0.95, image=None):
        if not gemini_api_key:
            return ("", "", "", 120, 120.0, "4", "en", "C major")

        json_schema = {
            "type": "object",
            "required": [
                "description",
                "tags",
                "lyrics",
                "bpm",
                "keyscale",
                "durationSeconds",
                "timesignature",
                "language"
            ],
            "properties": {
                "description": {"type": "string"},
                "tags": {"type": "string"},
                "lyrics": {"type": "string"},
                "bpm": {"type": "integer"},
                "keyscale": {"type": "string"},
                "durationSeconds": {"type": "number"},
                "timesignature": {"type": "string"},
                "language": {"type": "string"}
            }
        }

        json_instruction = (
            "Return ONLY valid JSON matching this schema (no markdown, no comments): "
            f"{json.dumps(json_schema)}"
        )

        try:
            client = genai.Client(api_key=gemini_api_key)
            model_name = custom_model if model_selection == "custom" and custom_model else model_selection

            contents = []
            if system_message:
                contents.append(
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=system_message)]
                    )
                )

            combined_prompt = f"{prompt}\n\n{json_instruction}".strip()

            if image is not None:
                if isinstance(image, torch.Tensor):
                    image = image.cpu().numpy()

                if len(image.shape) == 4:
                    image = image[0]

                if image.dtype != np.uint8:
                    if image.max() <= 1:
                        image = (image * 255).astype(np.uint8)
                    else:
                        image = image.astype(np.uint8)

                img = Image.fromarray(image)
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format="PNG")
                img_byte_arr.seek(0)
                img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

                contents.append(
                    types.Content(
                        role="user",
                        parts=[{
                            "inline_data": {
                                "mime_type": "image/png",
                                "data": img_base64
                            }
                        },
                        {
                            "text": combined_prompt
                        }]
                    )
                )
            else:
                contents.append(
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=combined_prompt)]
                    )
                )

            generate_config = types.GenerateContentConfig(
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )

            # Use schema-based JSON when supported; otherwise rely on the instruction.
            is_gemma_model = model_name.startswith("gemma-3")
            if not is_gemma_model:
                generate_config.response_mime_type = "application/json"
                generate_config.response_schema = self._convert_schema_dict_to_genai(json_schema)

            response = ""
            for chunk in client.models.generate_content_stream(
                model=model_name,
                contents=contents,
                config=generate_config,
            ):
                if chunk.text:
                    response += chunk.text

            success, json_response = self.try_parse_json(response)
            if not success or not isinstance(json_response, dict):
                json_response = {}

            tags = json_response.get("tags") or json_response.get("description") or ""
            lyrics = json_response.get("lyrics", "")
            bpm = self._coerce_int(json_response.get("bpm"), default=120)
            duration = self._coerce_float(json_response.get("durationSeconds"), default=120.0)
            timesignature = str(json_response.get("timesignature", "4"))
            language = str(json_response.get("language", "en"))
            keyscale = str(json_response.get("keyscale", "C major"))

            response_json = json.dumps({
                "description": json_response.get("description", ""),
                "tags": tags,
                "lyrics": lyrics,
                "bpm": bpm,
                "keyscale": keyscale,
                "durationSeconds": duration,
                "timesignature": timesignature,
                "language": language
            })

            return (response_json, tags, lyrics, bpm, duration, timesignature, language, keyscale)
        except Exception as e:
            print(f"Error in Gemini API: {str(e)}")
            return ("", "", "", 0, 120, 120.0, "4", "en", "C major")

    def _convert_schema_dict_to_genai(self, schema_dict):
        schema_type = schema_dict.get("type", "object")

        if schema_type == "object":
            properties = {}
            for prop_name, prop_schema in schema_dict.get("properties", {}).items():
                properties[prop_name] = self._convert_schema_dict_to_genai(prop_schema)

            return genai.types.Schema(
                type=genai.types.Type.OBJECT,
                required=schema_dict.get("required", []),
                properties=properties
            )
        if schema_type == "array":
            items_schema = self._convert_schema_dict_to_genai(schema_dict.get("items", {}))
            return genai.types.Schema(
                type=genai.types.Type.ARRAY,
                items=items_schema
            )
        if schema_type == "string":
            return genai.types.Schema(type=genai.types.Type.STRING)
        if schema_type == "number":
            return genai.types.Schema(type=genai.types.Type.NUMBER)
        if schema_type == "integer":
            return genai.types.Schema(type=genai.types.Type.INTEGER)
        if schema_type == "boolean":
            return genai.types.Schema(type=genai.types.Type.BOOLEAN)
        return genai.types.Schema(type=genai.types.Type.STRING)

    def prepare_json(self, text):
        text = text.replace("```json", "").replace("```", "").strip()
        return text

    def try_parse_json(self, text):
        try:
            cleaned_text = self.prepare_json(text)
            return True, json.loads(cleaned_text)
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {str(e)}")
            return False, text
        except Exception as e:
            print(f"Unexpected error while parsing JSON: {str(e)}")
            return False, text

    def _coerce_int(self, value, default=0):
        try:
            return int(value)
        except Exception:
            return default

    def _coerce_float(self, value, default=0.0):
        try:
            return float(value)
        except Exception:
            return default
