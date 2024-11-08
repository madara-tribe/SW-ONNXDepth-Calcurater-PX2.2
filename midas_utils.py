import cv2
from midas.transforms import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose

def read_image(path):
    """Read image and output RGB image (0-1).

    Args:
        path (str): path to file

    Returns:
        array: RGB image (0-1)
    """
    img = cv2.imread(path)

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

    return img

def call_transform(model_type="dpt_large_384"):
    net_w, net_h = 384, 384
    keep_aspect_ratio = False
    if model_type == "midas_v21_384" or model_type == "midas_v21_small_256":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        resize_mode = "upper_bound"
    else:
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        resize_mode = "minimal"
    normalization = NormalizeImage(
            mean=mean, std=std
        )
    transform = Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=keep_aspect_ratio,
                ensure_multiple_of=32,
                resize_method=resize_mode,
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet(),
        ]
    )
    return transform, net_w, net_h
