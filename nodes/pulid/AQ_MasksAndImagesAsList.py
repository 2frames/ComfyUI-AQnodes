import torch

class AQ_MasksAndImagesAsList:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "masks": ("MASK", {"forceInput": True}),  # From SEGSToMaskList
                "images": ("IMAGE", {"forceInput": True}), # From SEGSToImageList
            }
        }

    RETURN_TYPES = ("IMAGE_LIST", "IMAGE_LIST", "STRING")
    RETURN_NAMES = ("images", "masks", "count")
    FUNCTION = "convert_to_lists"
    CATEGORY = "AQ/Pulid"
    INPUT_IS_LIST = True
    
    def convert_to_lists(self, masks, images):
        # Ensure we have the same number of masks and images
        if len(masks) != len(images):
            print(f"Warning: Number of masks ({len(masks)}) does not match number of images ({len(images)})")
            # Use the shorter length to avoid index errors
            length = min(len(masks), len(images))
            masks = masks[:length]
            images = images[:length]
        
        # Process masks and images together to maintain order
        processed_images = []
        processed_masks = []
        
        for mask, img in zip(masks, images):
            # Process mask
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(-1)
            processed_masks.append(mask)
            
            # Process image
            processed_images.append(img)
        
        # Create count string
        count_str = f"{len(processed_images)}"
            
        return (processed_images, processed_masks, count_str) 