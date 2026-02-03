from .nodes.pulid.AQ_MasksAndImagesAsList import AQ_MasksAndImagesAsList
from .nodes.transformers.AQ_Qwen import AQ_Qwen
from .nodes.transformers.AQ_Qwen import AQ_QwenLoader
from .nodes.pulid.AQ_multiface_ApplyPulidFlux import AQ_multiface_ApplyPulidFlux
from .nodes.api.AQ_Gemini import AQ_Gemini, AQ_Gemini_acstep15
from .nodes.api.AQ_SendImageToAPI import AQ_SendImageToAPI

from .nodes.clip.AQ_CLIPSetLastLayer import AQ_CLIPSetLastLayer
from .nodes.numbers.AQ_Numbers import AQ_Increment, AQ_Random
from .nodes.blend.AQ_BlendImages import AQ_BlendImages
from .nodes.image.AQ_Image import AQ_LoadImageBase64, AQ_Image_Pad, AQ_Image_DetailTransfer, AQ_ImageMaskSwitch
from .nodes.save.AQ_SaveImageWebpReturnBase64 import AQ_SaveImageWebpReturnBase64
from .nodes.filters.AQ_filters import AQ_BatchAverageImage, AQ_ColorMatchImage
from .nodes.text_encoders.AQ_TextEncodeQwenImage import AQ_TextEncodeQwenImageEdit
from .nodes.text_encoders.AQ_TextEncodeQwenImage import AQ_TextEncodeQwenImageEditPlus

NODE_CLASS_MAPPINGS = {
    "AQ_MasksAndImagesAsList": AQ_MasksAndImagesAsList,
    "AQ_multiface_ApplyPulidFlux": AQ_multiface_ApplyPulidFlux,
    "AQ_Qwen": AQ_Qwen,
    "AQ_QwenLoader": AQ_QwenLoader,
    "AQ_Gemini": AQ_Gemini,
    "AQ_Gemini_acstep15": AQ_Gemini_acstep15,
    "AQ_SendImageToAPI": AQ_SendImageToAPI,
    "AQ_CLIPSetLastLayer": AQ_CLIPSetLastLayer,
    "AQ_Increment": AQ_Increment,
    "AQ_Random": AQ_Random,
    "AQ_BlendImages": AQ_BlendImages,
    "AQ_LoadImageBase64": AQ_LoadImageBase64,
    "AQ_Image_Pad": AQ_Image_Pad,
    "AQ_Image_DetailTransfer": AQ_Image_DetailTransfer,
    "AQ_ImageMaskSwitch": AQ_ImageMaskSwitch,
    "AQ_SaveImageWebpReturnBase64": AQ_SaveImageWebpReturnBase64,
    "AQ_BatchAverageImage": AQ_BatchAverageImage,
    "AQ_ColorMatchImage": AQ_ColorMatchImage,
    "AQ_TextEncodeQwenImageEdit": AQ_TextEncodeQwenImageEdit,
    "AQ_TextEncodeQwenImageEditPlus": AQ_TextEncodeQwenImageEditPlus
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AQ_MasksAndImagesAsList": "AQ_MasksAndImagesAsList",
    "AQ_multiface_ApplyPulidFlux": "AQ_multiface_ApplyPulidFlux",
    "AQ_Qwen": "AQ_Qwen",
    "AQ_QwenLoader": "AQ_QwenLoader",
    "AQ_Gemini": "AQ_Gemini",
    "AQ_Gemini_acstep15": "AQ_Gemini_acstep15",
    "AQ_SendImageToAPI": "AQ_SendImageToAPI",
    "AQ_CLIPSetLastLayer": "AQ_CLIPSetLastLayer",
    "AQ_Increment": "AQ_Increment",
    "AQ_Random": "AQ_Random",
    "AQ_BlendImages": "AQ_BlendImages",
    "AQ_LoadImageBase64": "AQ_LoadImageBase64",
    "AQ_Image_Pad": "AQ_Image_Pad",
    "AQ_Image_DetailTransfer": "AQ_Image_DetailTransfer",
    "AQ_ImageMaskSwitch": "AQ_ImageMaskSwitch",
    "AQ_SaveImageWebpReturnBase64": "AQ_SaveImageWebpReturnBase64",
    "AQ_BatchAverageImage": "AQ_BatchAverageImage",
    "AQ_ColorMatchImage": "AQ_ColorMatchImage",
    "AQ_TextEncodeQwenImageEdit": "AQ_TextEncodeQwenImageEdit",
    "AQ_TextEncodeQwenImageEditPlus": "AQ_TextEncodeQwenImageEditPlus"

}


__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
