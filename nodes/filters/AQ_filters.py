import torch
import cv2

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
# guided filter a tensor image batch in format [B x H x W x C] on H/W (spatial, per-image, per-channel)
def guided_filter_tensor(ref, images, d, s):
    try:
        from cv2.ximgproc import guidedFilter
    except ImportError:
        print("\033[33mUnable to import guidedFilter, make sure you have only opencv-contrib-python or run the import_error_install.bat script\033[m")

    if d > 100:
        np_img = (
            torch.nn.functional.interpolate(
                images.detach().clone().movedim(-1, 1),
                scale_factor=0.1,
                mode="bilinear",
            )
            .movedim(1, -1)
            .cpu()
            .numpy()
        )
        np_ref = (
            torch.nn.functional.interpolate(
                ref.detach().clone().movedim(-1, 1), scale_factor=0.1, mode="bilinear"
            )
            .movedim(1, -1)
            .cpu()
            .numpy()
        )
        for index, image in enumerate(np_img):
            np_img[index] = guidedFilter(np_ref[index], image, d // 20 * 2 + 1, s)
        return torch.nn.functional.interpolate(
            torch.from_numpy(np_img).movedim(-1, 1),
            size=(images.shape[1], images.shape[2]),
            mode="bilinear",
        ).movedim(1, -1)
    else:
        np_img = images.detach().clone().cpu().numpy()
        np_ref = ref.cpu().numpy()
        for index, image in enumerate(np_img):
            np_img[index] = guidedFilter(np_ref[index], image, d, s)
        return torch.from_numpy(np_img)


# gaussian blur a tensor image batch in format [B x H x W x C] on H/W (spatial, per-image, per-channel)
def cv_blur_tensor(images, dx, dy):
    if min(dx, dy) > 100:
        np_img = (
            torch.nn.functional.interpolate(
                images.detach().clone().movedim(-1, 1),
                scale_factor=0.1,
                mode="bilinear",
            )
            .movedim(1, -1)
            .cpu()
            .numpy()
        )
        for index, image in enumerate(np_img):
            np_img[index] = cv2.GaussianBlur(
                image, (dx // 20 * 2 + 1, dy // 20 * 2 + 1), 0
            )
        return torch.nn.functional.interpolate(
            torch.from_numpy(np_img).movedim(-1, 1),
            size=(images.shape[1], images.shape[2]),
            mode="bilinear",
        ).movedim(1, -1)
    else:
        np_img = images.detach().clone().cpu().numpy()
        for index, image in enumerate(np_img):
            np_img[index] = cv2.GaussianBlur(image, (dx, dy), 0)
        return torch.from_numpy(np_img)
