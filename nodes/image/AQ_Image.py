import torch
import comfy.utils
import comfy.model_management
from torchvision.transforms import Resize, CenterCrop, GaussianBlur
from PIL import Image
import numpy as np
import base64
from io import BytesIO

class AQ_LoadImageBase64:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("STRING", {"multiline": False})}}

    RETURN_TYPES = ("IMAGE", "MASK")
    CATEGORY = "AQ/Image"
    FUNCTION = "load_image"

    def load_image(self, image):
        imgdata = base64.b64decode(image)
        img = Image.open(BytesIO(imgdata))

        if "A" in img.getbands():
            mask = np.array(img.getchannel("A")).astype(np.float32) / 255.0
            mask = 1.0 - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")

        img = img.convert("RGB")
        img = np.array(img).astype(np.float32) / 255.0
        img = torch.from_numpy(img)[None,]

        return (img, mask)

class AQ_Image_Pad:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "left": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                "right": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                "top": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                "bottom": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "pad_image"

    CATEGORY = "AQ/Image"

    def pad_image(self, image, left, right, top, bottom):
        # Get the current shape of the image
        batch, height, width, channels = image.shape

        # Create a new tensor with the padded dimensions
        new_height = height + top + bottom
        new_width = width + left + right
        padded_image = torch.zeros((batch, new_height, new_width, channels), dtype=image.dtype, device=image.device)

        # Set the alpha channel to 0 (fully transparent) for the entire padded image
        padded_image[:, :, :, 3] = 0

        # Copy the original image into the padded image
        padded_image[:, top:top+height, left:left+width, :] = image

        return (padded_image,)

class AQ_Image_DetailTransfer:
  @classmethod
  def INPUT_TYPES(s):
    return {
      "required": {
        "target": ("IMAGE",),
        "source": ("IMAGE",),
        "mode": (["add", "multiply", "screen", "overlay", "soft_light", "hard_light", "color_dodge", "color_burn", "difference", "exclusion", "divide",],{"default": "add"}),
        "blur_sigma": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 100.0, "step": 0.01}),
        "blend_factor": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.001, "round": 0.001}),
        "image_output": (["Hide", "Preview", "Save", "Hide/Save"], {"default": "Preview"}),
        "save_prefix": ("STRING", {"default": "ComfyUI"}),
      },
      "optional": {
        "mask": ("MASK",),
      },
      "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
    }

  RETURN_TYPES = ("IMAGE",)
  RETURN_NAMES = ("image",)
  OUTPUT_NODE = False
  FUNCTION = "transfer"
  CATEGORY = "AQ/Image"



  def transfer(self, target, source, mode, blur_sigma, blend_factor, image_output, save_prefix, mask=None, prompt=None, extra_pnginfo=None):
    batch_size, height, width, _ = target.shape
    device = comfy.model_management.get_torch_device()
    target_tensor = target.permute(0, 3, 1, 2).clone().to(device)
    source_tensor = source.permute(0, 3, 1, 2).clone().to(device)

    if target.shape[1:] != source.shape[1:]:
      source_tensor = comfy.utils.common_upscale(source_tensor, width, height, "bilinear", "disabled")

    if source.shape[0] < batch_size:
      source = source[0].unsqueeze(0).repeat(batch_size, 1, 1, 1)

    kernel_size = int(6 * int(blur_sigma) + 1)

    gaussian_blur = GaussianBlur(kernel_size=(kernel_size, kernel_size), sigma=(blur_sigma, blur_sigma))

    blurred_target = gaussian_blur(target_tensor)
    blurred_source = gaussian_blur(source_tensor)

    if mode == "add":
      new_image = (source_tensor - blurred_source) + blurred_target
    elif mode == "multiply":
      new_image = source_tensor * blurred_target
    elif mode == "screen":
      new_image = 1 - (1 - source_tensor) * (1 - blurred_target)
    elif mode == "overlay":
      new_image = torch.where(blurred_target < 0.5, 2 * source_tensor * blurred_target,
                               1 - 2 * (1 - source_tensor) * (1 - blurred_target))
    elif mode == "soft_light":
      new_image = (1 - 2 * blurred_target) * source_tensor ** 2 + 2 * blurred_target * source_tensor
    elif mode == "hard_light":
      new_image = torch.where(source_tensor < 0.5, 2 * source_tensor * blurred_target,
                               1 - 2 * (1 - source_tensor) * (1 - blurred_target))
    elif mode == "difference":
      new_image = torch.abs(blurred_target - source_tensor)
    elif mode == "exclusion":
      new_image = 0.5 - 2 * (blurred_target - 0.5) * (source_tensor - 0.5)
    elif mode == "color_dodge":
      new_image = blurred_target / (1 - source_tensor)
    elif mode == "color_burn":
      new_image = 1 - (1 - blurred_target) / source_tensor
    elif mode == "divide":
      new_image = (source_tensor / blurred_source) * blurred_target
    else:
      new_image = source_tensor

    new_image = torch.lerp(target_tensor, new_image, blend_factor)
    if mask is not None:
      mask = mask.to(device)
      new_image = torch.lerp(target_tensor, new_image, mask)
    new_image = torch.clamp(new_image, 0, 1)
    new_image = new_image.permute(0, 2, 3, 1).cpu().float()

    return {"ui": {},
            "result": (new_image,)}
  
class AQ_ImageMaskSwitch:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "select": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
            "images1": ("IMAGE",),
        },

            "optional": {
                "mask1_opt": ("MASK",),
                "images2_opt": ("IMAGE",),
                "mask2_opt": ("MASK",),
                "images3_opt": ("IMAGE",),
                "mask3_opt": ("MASK",),
                "images4_opt": ("IMAGE",),
                "mask4_opt": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK",)

    OUTPUT_NODE = False

    FUNCTION = "doit"

    CATEGORY = "AQ/Image"

    def doit(self, select, images1, mask1_opt=None, images2_opt=None, mask2_opt=None, images3_opt=None, mask3_opt=None,
             images4_opt=None, mask4_opt=None):
        if select == 1:
            return images1, mask1_opt,
        elif select == 2:
            return images2_opt, mask2_opt,
        elif select == 3:
            return images3_opt, mask3_opt,
        else:
            return images4_opt, mask4_opt, 