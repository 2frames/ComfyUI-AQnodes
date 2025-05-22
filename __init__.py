from .nodes.pulid.AQ_MasksAndImagesAsList import AQ_MasksAndImagesAsList
from .nodes.transformers.AQ_Qwen import AQ_Qwen
from .nodes.transformers.AQ_Qwen import AQ_QwenLoader
from .nodes.pulid.AQ_multiface_ApplyPulidFlux import AQ_multiface_ApplyPulidFlux
from .nodes.api.AQ_Gemini import AQ_Gemini
from .nodes.api.AQ_SendImageToAPI import AQ_SendImageToAPI

NODE_CLASS_MAPPINGS = {
    "AQ_MasksAndImagesAsList": AQ_MasksAndImagesAsList,
    "AQ_multiface_ApplyPulidFlux": AQ_multiface_ApplyPulidFlux,
    "AQ_Qwen": AQ_Qwen,
    "AQ_QwenLoader": AQ_QwenLoader,
    "AQ_Gemini": AQ_Gemini,
    "AQ_SendImageToAPI": AQ_SendImageToAPI
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AQ_MasksAndImagesAsList": "AQ_MasksAndImagesAsList",
    "AQ_multiface_ApplyPulidFlux": "AQ_multiface_ApplyPulidFlux",
    "AQ_Qwen": "AQ_Qwen",
    "AQ_QwenLoader": "AQ_QwenLoader",
    "AQ_Gemini": "AQ_Gemini",
    "AQ_SendImageToAPI": "AQ_SendImageToAPI"
}


__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
