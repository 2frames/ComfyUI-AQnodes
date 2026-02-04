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
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
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
        "INT",
        "FLOAT",
        "COMBO",
        "COMBO",
        "COMBO",
    )
    RETURN_NAMES = (
        "response_json",
        "tags",
        "lyrics",
        "seed",
        "bpm",
        "duration",
        "timesignature",
        "language",
        "keyscale",
    )
    FUNCTION = "generate"
    CATEGORY = "Aquasite/LLM"

    def generate(self, gemini_api_key, model_selection, custom_model, prompt, system_message, temperature=0.8, top_p=0.95, seed=0, image=None):
        if not gemini_api_key:
            return ("", "", "", seed, 120, 120.0, "4", "en", "C major")

        json_schema = {
            "type": "object",
            "required": [
                "description",
                "tags",
                "lyrics",
                "seed",
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
                "seed": {"type": "integer"},
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
       

            seed_instruction = f"Seed: {seed}. Use it to vary output across runs."
            combined_prompt = f"{prompt}\n\n{seed_instruction}\n\n{json_instruction}".strip()

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
                top_p=top_p,
            )

            if system_message:
                generate_config.system_instruction = [types.Part.from_text(text=system_message)]
               

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
            if isinstance(lyrics, str):
                lyrics = lyrics.replace("\\n\\n", "\n").replace("\n\n", "\n")
            seed_value = self._coerce_int(json_response.get("seed"), default=seed)
            bpm = self._coerce_int(json_response.get("bpm"), default=120)
            duration = self._coerce_float(json_response.get("durationSeconds"), default=120.0)
            timesignature = str(json_response.get("timesignature", "4"))
            language = str(json_response.get("language", "en"))
            keyscale = str(json_response.get("keyscale", "C major"))

            response_json = json.dumps({
                "description": json_response.get("description", ""),
                "tags": tags,
                "lyrics": lyrics,
                "seed": seed_value,
                "bpm": bpm,
                "keyscale": keyscale,
                "durationSeconds": duration,
                "timesignature": timesignature,
                "language": language
            })

            return (response_json, tags, lyrics, seed_value, bpm, duration, timesignature, language, keyscale)
        except Exception as e:
            print(f"Error in Gemini API: {str(e)}")
            return ("", "", "", seed, 120, 120.0, "4", "en", "C major")

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


class AQ_OpenAI_acstep15:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "openai_api_key": ("STRING", {"default": ""}),
                "model": ([
                    "gpt-5.2",
                    "gpt-5-nano",
                    "gpt-5-mini",
                    "gpt-5",
                    "gpt-4.1-mini",
                    "gpt-4.1",
                    "gpt-4o-mini",
                    "gpt-4o",
                    "custom"
                ], {"default": "gpt-5-nano"}),
                "custom_model": ("STRING", {"default": "", "multiline": False}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "system_message": ("STRING", {"default": "You are a helpful assistant.", "multiline": True}),
                "temperature": ("FLOAT", {"default": 1, "min": 0.0, "max": 2.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "verbosity": (["none", "low", "medium", "high"], {"default": "medium"}),
                "reasoning_effort": (["none", "minimal", "low", "medium", "high"], {"default": "medium"}),
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
        "INT",
        "FLOAT",
        "COMBO",
        "COMBO",
        "COMBO",
    )
    RETURN_NAMES = (
        "response_json",
        "tags",
        "lyrics",
        "seed",
        "bpm",
        "duration",
        "timesignature",
        "language",
        "keyscale",
    )
    FUNCTION = "generate"
    CATEGORY = "Aquasite/LLM"

    def generate(self, openai_api_key, model, custom_model, prompt, system_message, temperature=0.8, top_p=0.95, seed=0, verbosity="medium", reasoning_effort="medium", image=None):
        if not openai_api_key:
            return ("", "", "", seed, 120, 120.0, "4", "en", "C major")

        schema = {
            "type": "object",
            "properties": {
                "tags": {
                    "type": "string",
                    "description": "Style prompt describing the song's musical characteristics. This is the primary control for musical style, instrumentation, and mood. Can be formatted as: (1) Comma-separated keywords: \"rock, pop, energetic, electric guitar, male vocals, 120 bpm\", (2) Descriptive phrases: \"A smooth jazzy lo-fi hip-hop track with gentle piano melody\", (3) Mixed format combining both. Include any of: genre/subgenre (rock, jazz, EDM, folk, metal, hip-hop, R&B, country, classical, electronic), instruments (guitar, piano, drums, bass, synthesizer, violin, saxophone, trumpet), vocal style (male/female vocals, raspy, smooth, powerful, whispered, choir, a cappella), mood/energy (energetic, melancholic, uplifting, dark, romantic, aggressive, chill, epic), tempo descriptor (slow, mid-tempo, fast, 120 bpm), production style (lo-fi, polished, raw, orchestral, acoustic, electronic), era/influence (1980s, vintage, modern, retro). For instrumental-only tracks, include \"instrumental\" in tags. For specific vocal techniques: \"a cappella\" for vocals only, \"b-box\" for beatboxing. The model supports all mainstream music styles and responds well to detailed, specific descriptions."
                },
                "lyrics": {
                    "type": "string",
                    "description": "Complete song lyrics with structure tags and optional language codes. STRUCTURE TAGS (in square brackets): [intro], [verse], [verse 1], [verse 2], [pre-chorus], [chorus], [hook], [bridge], [breakdown], [drop], [outro], [instrumental], [inst]. Use [inst] or [instrumental] for instrumental-only sections or entire instrumental tracks. LANGUAGE CODES: For non-English lyrics, prefix each line or section with language code in brackets: [zh] for Chinese, [ko] for Korean, [ja] for Japanese, [es] for Spanish, [de] for German, [fr] for French, [pt] for Portuguese, [it] for Italian, [ru] for Russian, [pl] for Polish, [ar] for Arabic, [th] for Thai, [vi] for Vietnamese. Example: \"[verse]\\n[zh]wo de xin li zhi you ni\\n[zh]mei yi tian dou xiang nian ni\". For romanized input of non-Latin scripts, include tone numbers for Chinese (e.g., \"wo3 ai4 ni3\"). PERFORMANCE DIRECTIONS: Can include [whispered], [spoken], [shouted], or descriptive brackets like [Intro - Acoustic Guitar], [Guitar Solo], [Drop - Heavy Bass]. FORMATTING: Use \\n for line breaks between lyrics lines. Separate sections with blank lines. Keep lyrics concise and rhythmic to match the musical style. For vocal harmony or duets, indicate with [Male Vocal], [Female Vocal], or [Duet]."
                },
                "seed": {
                    "type": "integer",
                    "description": "Random seed for reproducible generation. Use -1 for random seed (recommended for exploring variations). Use a specific positive integer (e.g., 42, 12345) to reproduce the exact same output with identical parameters. When batch generating, different seeds produce different variations of the same prompt. Useful for: iterating on a good result, sharing reproducible outputs, A/B testing parameter changes."
                },
                "bpm": {
                    "type": "integer",
                    "description": "Tempo in beats per minute. Typical ranges by genre: Ballad/Ambient (60-80), R&B/Soul (70-100), Hip-Hop (80-115), Pop (100-130), House/Disco (115-130), Rock (110-140), Techno/Trance (125-150), Drum and Bass (160-180), Hardcore/Speedcore (180+). Common tempos: 80 (slow groove), 100 (moderate), 120 (standard dance/pop), 140 (energetic), 170 (DnB). The tempo should match the energy described in tags. Can also include BPM directly in tags field for emphasis (e.g., \"120 bpm\").",
                    "minimum": 1
                },
                "keyscale": {
                    "type": "string",
                    "description": "Musical key and scale defining the harmonic foundation. Format: \"Root Scale\" where Root is the note (A, B, C, D, E, F, G with optional # or b for sharp/flat) and Scale is \"major\" or \"minor\". Examples: \"C major\" (bright, happy), \"A minor\" (emotional, versatile), \"G major\" (warm, popular for folk/pop), \"E minor\" (powerful, common in rock), \"D minor\" (dramatic, melancholic), \"F# minor\" (dark, atmospheric), \"Bb major\" (smooth, jazzy). Major keys generally sound happier/brighter; minor keys sound more emotional/darker. Choose based on mood: uplifting songs → major keys; emotional/dark songs → minor keys. Common pairings: Pop/Dance → C/G major; Rock/Metal → E/A minor; Jazz → Bb/Eb major; Sad ballads → D/A minor."
                },
                "durationSeconds": {
                    "type": "integer",
                    "description": "Target duration of generated audio in seconds. Supported range: 30-600 seconds (ACE-Step 1.5 supports up to 10 minutes). Recommended ranges: Short/loops (30-60), Standard song (90-180), Extended (180-300), Long-form (300-600). IMPORTANT: Start with 90-120 seconds for most consistent results. Longer durations (180+ seconds) may require multiple generation attempts to maintain musical coherence throughout. Very short durations (30-60s) work well for loops, samples, or song sections. Match duration to lyric length - longer lyrics need more time.",
                    "minimum": 1
                },
                "timesignature": {
                    "type": "string",
                    "description": "Time signature as beats per measure, affecting the rhythmic feel and groove. Valid values: \"2\" (cut time/alla breve - march-like, polka, fast classical), \"3\" (waltz time/3/4 - waltzes, some ballads, folk), \"4\" (common time/4/4 - most popular music: rock, pop, hip-hop, electronic, jazz), \"6\" (compound time/6/8 - shuffle feel, triplet-based grooves, some ballads, Irish jigs). Most music uses \"4\" (4/4 time). Use \"3\" for waltzes or swaying ballads. Use \"6\" for shuffle/swing feels or compound grooves. Use \"2\" for marches or fast-paced classical styles.",
                    "enum": ["2", "3", "4", "6"]
                },
                "language": {
                    "type": "string",
                    "description": "ISO 639-1 two-character code for the primary language of the lyrics. Must be exactly 2 lowercase letters. Top 10 best-supported languages: \"en\" (English), \"zh\" (Chinese Mandarin), \"ja\" (Japanese), \"ko\" (Korean), \"es\" (Spanish), \"de\" (German), \"fr\" (French), \"pt\" (Portuguese), \"it\" (Italian), \"ru\" (Russian). Additional supported languages include: \"pl\" (Polish), \"ar\" (Arabic), \"th\" (Thai), \"vi\" (Vietnamese), \"nl\" (Dutch), \"sv\" (Swedish), \"tr\" (Turkish), \"id\" (Indonesian), \"hi\" (Hindi), \"uk\" (Ukrainian), and 40+ more. For multilingual songs, set to the dominant language and use language codes in lyrics field for mixed-language sections. Performance may vary for less common languages due to training data distribution.",
                    "pattern": "^[a-z]{2}$",
                    "minLength": 2,
                    "maxLength": 2
                }
            },
            "required": [
                "tags",
                "lyrics",
                "seed",
                "bpm",
                "keyscale",
                "durationSeconds",
                "timesignature",
                "language"
            ],
            "additionalProperties": False
        }

        model_name = custom_model if model == "custom" and custom_model else model
        seed_instruction = f"Seed: {seed}. Use it to vary output across runs."

        user_content = [{"type": "input_text", "text": f"{prompt}\n\n{seed_instruction}"}]

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
            img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")
            user_content.append({
                "type": "input_image",
                "image_url": f"data:image/png;base64,{img_base64}"
            })

        payload = {
            "model": model_name,
            "input": [
                {
                    "role": "developer",
                    "content": [{"type": "input_text", "text": system_message}] if system_message else []
                },
                {
                    "role": "user",
                    "content": user_content
                }
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "song_generation_response",
                    "strict": True,
                    "schema": schema
                }
            },
            "temperature": temperature,
            "top_p": top_p,
            "tools": [],
            "store": False,
            "include": []
        }
        if verbosity != "none":
            payload["text"]["verbosity"] = verbosity
        if reasoning_effort != "none":
            payload["reasoning"] = {"effort": reasoning_effort}

        try:
            response = requests.post(
                "https://api.openai.com/v1/responses",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {openai_api_key}"
                },
                json=payload,
                timeout=120
            )
            if not response.ok:
                print(f"OpenAI API error status: {response.status_code}")
                print(f"OpenAI API error body: {response.text}")
                error_json = json.dumps({
                    "error": {
                        "status_code": response.status_code,
                        "body": response.text
                    }
                })
                return (error_json, "", "", seed, 120, 120.0, "4", "en", "C major")
            response.raise_for_status()
            data = response.json()

            content = data.get("output_text", "")
            if not content:
                output = data.get("output", [])
                parts = []
                for item in output:
                    for part in item.get("content", []) or []:
                        if part.get("type") in ("output_text", "text"):
                            parts.append(part.get("text", ""))
                content = "".join(parts)

            success, json_response = self.try_parse_json(content)
            if not success or not isinstance(json_response, dict):
                json_response = {}

            tags = str(json_response.get("tags", "") or "")

            lyrics = json_response.get("lyrics", "")
            if isinstance(lyrics, str):
                lyrics = lyrics.replace("\\n\\n", "\n").replace("\n\n", "\n")

            seed_value = self._coerce_int(json_response.get("seed"), default=seed)
            bpm = self._coerce_int(json_response.get("bpm"), default=120)
            duration = self._coerce_float(json_response.get("durationSeconds"), default=120.0)
            timesignature = str(json_response.get("timesignature", "4"))
            language = str(json_response.get("language", "en"))
            keyscale = str(json_response.get("keyscale", "C major"))

            response_json = json.dumps({
                "tags": tags,
                "lyrics": lyrics,
                "seed": seed_value,
                "bpm": bpm,
                "keyscale": keyscale,
                "durationSeconds": duration,
                "timesignature": timesignature,
                "language": language
            })

            return (response_json, tags, lyrics, seed_value, bpm, duration, timesignature, language, keyscale)
        except Exception as e:
            print(f"Error in OpenAI API: {str(e)}")
            error_json = json.dumps({"error": str(e)})
            return (error_json, "", "", seed, 120, 120.0, "4", "en", "C major")

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
