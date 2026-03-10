import os


class AQ_CoverPrompt:
    ASPECT_RATIO_NOTES = {
        "16:9 YouTube": "optimized for 16:9 widescreen YouTube thumbnail format (1280x720)",
        "1:1 Square":   "optimized for 1:1 square format (1080x1080)",
        "4:5 Portrait": "optimized for 4:5 portrait format (1080x1350)",
    }

    NEGATIVE_PROMPT = (
        "blur, blurry, out of focus, low resolution, low quality, pixelated, "
        "watermark, logo, signature, text errors, misspelled text, distorted letters, "
        "distorted anatomy, disfigured, bad proportions, extra limbs, missing limbs, "
        "bad lighting, flat lighting, overexposed, underexposed, washed out, "
        "generic stock photo, clip art, cartoon (unless style selected), "
        "oversaturated, noise, grain, compression artifacts"
    )

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "cover_prompt":  ("STRING", {"multiline": True, "default": ""}),
                "title":         ("STRING", {"multiline": False, "default": ""}),
                "tagline":       ("STRING", {"multiline": False, "default": ""}),
                "headline":      ("STRING", {"multiline": False, "default": ""}),
                "tags":          ("STRING", {"multiline": False, "default": ""}),
                "keyscale":      ("STRING", {"multiline": False, "default": "C major"}),
                "style":         (["cinematic photo", "digital illustration", "oil painting", "neon art", "minimalist", "watercolor", "3D render"], {"default": "cinematic photo"}),
                "aspect_ratio":  (["16:9 YouTube", "1:1 Square", "4:5 Portrait"], {"default": "16:9 YouTube"}),
            },
            "optional": {
                "template_override": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "negative_prompt")
    FUNCTION = "compose"
    CATEGORY = "Aquasite/LLM"

    def compose(self, cover_prompt, title, tagline, headline, tags, keyscale, style, aspect_ratio, template_override=""):
        if template_override and template_override.strip():
            template = template_override
        else:
            template_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "prompts", "cover.md"
            )
            with open(template_path, "r", encoding="utf-8") as f:
                template = f.read()

        aspect_ratio_note = self.ASPECT_RATIO_NOTES.get(aspect_ratio, aspect_ratio)

        prompt = template.format(
            style=style,
            aspect_ratio_note=aspect_ratio_note,
            coverImage=cover_prompt,
            title=title,
            tagline=tagline,
            headline=headline,
            tags=tags,
            keyscale=keyscale,
        )

        return (prompt, self.NEGATIVE_PROMPT)
