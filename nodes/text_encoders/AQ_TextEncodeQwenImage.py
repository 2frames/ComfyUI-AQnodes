import node_helpers
import comfy.utils
import math


class AQ_TextEncodeQwenImageEdit:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "use_image": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "vae": ("VAE",),
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"

    CATEGORY = "advanced/conditioning"

    def encode(self, clip, prompt, use_image=True, vae=None, image=None):
        ref_latent = None
        if image is None or not use_image:
            images = []
        else:
            samples = image.movedim(-1, 1)
            total = int(1024 * 1024)

            scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
            width = round(samples.shape[3] * scale_by)
            height = round(samples.shape[2] * scale_by)

            s = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
            image = s.movedim(1, -1)
            images = [image[:, :, :, :3]]
            if vae is not None:
                ref_latent = vae.encode(image[:, :, :, :3])

        tokens = clip.tokenize(prompt, images=images)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        if ref_latent is not None:
            conditioning = node_helpers.conditioning_set_values(
                conditioning, {"reference_latents": [ref_latent]}, append=True
            )
        return (conditioning,)


class AQ_TextEncodeQwenImageEditPlus:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "use_image1": ("BOOLEAN", {"default": True}),
                "use_image2": ("BOOLEAN", {"default": True}),
                "use_image3": ("BOOLEAN", {"default": True}),
                "prompt_template": (["original", "detailed", "transformation_focused", "preserving", "creative", "technical", "artistic_generation", "style_transfer", "scene_reimagining", "playful", "descriptive", "precise_surgical", "multi_image_blend", "view_angle_control"], {"default": "original"}),
            },
            "optional": {
                "vae": ("VAE",),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"

    CATEGORY = "advanced/conditioning"

    def encode(self, clip, prompt, use_image1=True, use_image2=True, use_image3=True, prompt_template="original", vae=None, image1=None, image2=None, image3=None):
        ref_latents = []
        images = [image1, image2, image3]
        use_images = [use_image1, use_image2, use_image3]
        images_vl = []
        
        # Define available prompt templates
        templates = {
            "original": "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            "detailed": "<|im_start|>system\nAnalyze the input image systematically: identify dominant colors and color palette, describe shapes and compositional elements, note textures and material properties, catalog all visible objects and their spatial relationships, and characterize the background and lighting. Then, interpret the user's modification request and generate a new image that implements these changes while preserving the original aesthetic and maintaining visual coherence.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            "transformation_focused": "<|im_start|>system\nExamine the input image and understand its current state. Carefully analyze the user's instruction to determine what specific changes are requested. Apply the transformation precisely as described, modifying only the elements mentioned while keeping all other aspects unchanged. Ensure the modified image maintains photorealistic quality and natural appearance.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            "preserving": "<|im_start|>system\nStudy the input image and identify its core visual elements, style, composition, and atmosphere. Understand the user's modification request and implement it in a way that preserves the fundamental character and identity of the original image. Maintain consistency in lighting, perspective, and artistic style while carefully integrating the requested changes.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            "creative": "<|im_start|>system\nObserve the input image and capture its visual essence, mood, and artistic qualities. Interpret the user's creative direction and generate a new image that fulfills their vision while enhancing the overall aesthetic. Feel free to refine and improve the image artistically while staying true to the user's intent and maintaining recognizable elements from the original.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            "technical": "<|im_start|>system\nProcess the input image data: extract color values (RGB/HSV), identify geometric shapes and dimensions, analyze texture patterns and surface properties, detect and classify objects with bounding regions, and evaluate background composition and depth. Parse the user's modification parameters and generate output image that satisfies the specified constraints while maintaining structural integrity and visual fidelity of unmodified regions.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            "artistic_generation": "<|im_start|>system\nLook at the reference image and identify the main subject, character, or object shown. Use this as inspiration to create a new artistic image based on the user's description. The subject from the reference image should be recognizable in the new generation, but feel free to interpret it artistically, change the style, setting, and context according to the user's creative vision. Focus on producing beautiful, imaginative artwork.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            "style_transfer": "<|im_start|>system\nIdentify the main subject or character in the reference image. Read the user's description to understand what artistic style, medium, or visual treatment they want applied. Generate a new image that depicts the same subject but rendered in the requested style - whether it's a painting technique (watercolor, oil, pastel), art movement (impressionist, cartoon, anime), or other visual style. Keep the subject recognizable while fully embracing the new artistic style.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            "scene_reimagining": "<|im_start|>system\nExamine the reference image to understand the main subject, character, or object. The user will describe a new scene, setting, environment, or situation for this subject. Generate a new image that places the subject from the reference into the described context. The subject should maintain its essential characteristics and identity, but be integrated naturally into the new scene with appropriate lighting, perspective, and atmosphere.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            "playful": "<|im_start|>system\nLook at the picture and see what's in it - maybe it's a toy, a pet, a drawing, or something else fun! Now read what the user wants to create. Make a new, fun picture that brings their idea to life! Be creative and imaginative. If they mention colors, styles, places, or activities, include all of those. Make something cheerful and exciting that captures their imagination. The new image should be vibrant, engaging, and full of personality.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            "descriptive": "<|im_start|>system\nObserve the reference image to understand what subject is being shown. Read the user's text description carefully - it will tell you exactly what to generate. Follow their description literally and precisely: if they mention specific colors, use those colors; if they describe a style, apply that style; if they mention objects, settings, or actions, include all of them. Generate an image that matches their written description as closely as possible while keeping the main subject from the reference recognizable.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            "precise_surgical": "<|im_start|>system\nYou are performing precise, surgical edits. The user will tell you exactly what to ADD, REPLACE, or REMOVE using clear, direct language. Make ONLY the specific change requested - nothing more. Keep everything else completely unchanged: preserve all colors, textures, lighting, shadows, reflections, proportions, perspective, and composition. Do not add distortion, do not warp text, do not create duplicate elements, do not alter the background. Match original font, size, and alignment if editing text. Maintain natural appearance with no artifacts. The edit should be seamless and invisible except for the exact element being modified.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            "multi_image_blend": "<|im_start|>system\nYou are blending and combining multiple reference images into one cohesive result. Analyze each input image to identify the key subjects, elements, or features. The user's description will explain how to merge these images together. Integrate all specified elements naturally - match lighting across sources, blend shadows and highlights realistically, ensure consistent perspective and scale, harmonize color palettes. Create seamless transitions between elements from different sources. The final image should look unified and natural, not like a collage or photo-bash. Preserve the identity and key features from each reference while creating visual coherence.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            "view_angle_control": "<|im_start|>system\nYou are re-rendering the scene from a different camera angle or perspective. Study the reference image to understand the 3D space, object positions, lighting setup, and scene composition. The user will specify a new camera angle, viewpoint, or perspective. Re-project the entire scene from this new angle while maintaining spatial consistency: keep all objects in their correct 3D positions, preserve relative sizes and distances, adjust lighting and shadows to match the new angle, maintain the same materials and textures. The result should look like the same scene photographed from a different position. Do not change object identities, add new elements, or alter the fundamental scene composition - only change the viewing angle.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        }
        
        llama_template = templates.get(prompt_template, templates["original"])
        image_prompt = ""

        for i, (image, use_image) in enumerate(zip(images, use_images)):
            if image is not None and use_image:
                samples = image.movedim(-1, 1)
                total = int(384 * 384)

                scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
                width = round(samples.shape[3] * scale_by)
                height = round(samples.shape[2] * scale_by)

                s = comfy.utils.common_upscale(
                    samples, width, height, "area", "disabled"
                )
                images_vl.append(s.movedim(1, -1))
                if vae is not None:
                    total = int(1024 * 1024)
                    scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
                    width = round(samples.shape[3] * scale_by / 8.0) * 8
                    height = round(samples.shape[2] * scale_by / 8.0) * 8

                    s = comfy.utils.common_upscale(
                        samples, width, height, "area", "disabled"
                    )
                    ref_latents.append(vae.encode(s.movedim(1, -1)[:, :, :, :3]))

                image_prompt += (
                    "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(
                        i + 1
                    )
                )

        tokens = clip.tokenize(
            image_prompt + prompt, images=images_vl, llama_template=llama_template
        )
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        if len(ref_latents) > 0:
            conditioning = node_helpers.conditioning_set_values(
                conditioning, {"reference_latents": ref_latents}, append=True
            )
        return (conditioning,)
