import torch


class AQ_BatchAverageImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "operation": (["mean", "median"],),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"

    CATEGORY = "AQ/filters"

    def apply(self, images, operation):
        t = images.detach().clone()
        if operation == "mean":
            return (torch.mean(t, dim=0, keepdim=True),)
        elif operation == "median":
            return (torch.median(t, dim=0, keepdim=True)[0],)
        return (t,)


class AQ_ColorMatchImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "reference": ("IMAGE",),
                "blur_type": (["blur", "guidedFilter"],),
                "blur_size": ("INT", {"default": 0, "min": 0, "max": 1023}),
                "factor": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -10.0,
                        "max": 10.0,
                        "step": 0.01,
                        "round": 0.01,
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "batch_normalize"

    CATEGORY = "AQ/filters"

    def batch_normalize(self, images, reference, blur_type, blur_size, factor):
        t = images.detach().clone() + 0.1
        ref = reference.detach().clone() + 0.1

        if ref.shape[0] < t.shape[0]:
            ref = ref[0].unsqueeze(0).repeat(t.shape[0], 1, 1, 1)

        if blur_size == 0:
            mean = torch.mean(t, (1, 2), keepdim=True)
            mean_ref = torch.mean(ref, (1, 2), keepdim=True)

            for i in range(t.shape[0]):
                for c in range(3):
                    t[i, :, :, c] /= mean[i, 0, 0, c]
                    t[i, :, :, c] *= mean_ref[i, 0, 0, c]
        else:
            d = blur_size * 2 + 1

            if blur_type == "blur":
                blurred = cv_blur_tensor(t, d, d)
                blurred_ref = cv_blur_tensor(ref, d, d)
            elif blur_type == "guidedFilter":
                blurred = guided_filter_tensor(t, t, d, 0.01)
                blurred_ref = guided_filter_tensor(ref, ref, d, 0.01)

            for i in range(t.shape[0]):
                for c in range(3):
                    t[i, :, :, c] /= blurred[i, :, :, c]
                    t[i, :, :, c] *= blurred_ref[i, :, :, c]

        t = t - 0.1
        torch.clamp(torch.lerp(images, t, factor), 0, 1)
        return (t,)
