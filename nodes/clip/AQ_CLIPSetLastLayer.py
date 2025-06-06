class AQ_CLIPSetLastLayer:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "clip": ("CLIP", ),
                              "stop_at_clip_layer": ("INT", {"default": 0, "min": -24, "max": 0, "step": 1}),
                              }}
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "set_last_layer"

    CATEGORY = "AQ/conditioning"


    def set_last_layer(self, clip, stop_at_clip_layer):
        if(stop_at_clip_layer == 0):
            return (clip,)

        clip = clip.clone()
        clip.clip_layer(stop_at_clip_layer)
        return (clip,) 