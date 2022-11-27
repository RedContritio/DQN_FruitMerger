import typing
import cv2
import numpy as np


def mix(background: np.ndarray, foreground: np.ndarray, alpha: float = 1.0) -> None:
    """
    mix foreground image to background image, modify background inplace
    """

    alpha_back = background[:, :, 3] / 255.0
    alpha_fore = (foreground[:, :, 3] / 255.0) * alpha

    for c in range(3):
        background[:, :, c] = np.ubyte(
            alpha_fore * foreground[:, :, c]
            + alpha_back * background[:, :, c] * (1 - alpha_fore)
        )

    background[:, :, 3] = np.ubyte(
        (1 - (1 - alpha_fore) * (1 - alpha_back)) * 255)


def cover(background: np.ndarray, foreground: np.ndarray, alpha: float = 1.0) -> None:
    """
    cover foreground image to background image, modify background inplace
    ref: https://stackoverflow.com/a/71701023
    """
    assert (
        background.shape == foreground.shape
    ), f"background and foreground should have the same shape. found: {background.shape} and {foreground.shape}"
    h, w, channels = background.shape

    assert (
        channels == 4
    ), f"image should have exactly 4 channels (RGBA). found:{channels}"

    foreground_colors = foreground[:, :, :3]
    alpha_channel = foreground[:, :, 3] / 255.0  # 0-255 => 0.0-1.0
    alpha_channel = alpha_channel * alpha
    alpha_mask = np.dstack((alpha_channel, alpha_channel, alpha_channel))

    # combine the background with the overlay image weighted by alpha
    background[:, :, :3] = (
        background[:, :, :3] * (1 - alpha_mask) +
        foreground_colors * alpha_mask
    )


def intersectRect(rect1: typing.List[int], rect2: typing.List[int]) -> typing.List[int]:
    """
    intersect two rectangles, return the intersected rectangle

    params:
        - rect1: [left, top, width, height]
        - rect2: [left, top, width, height]

    return:
        intersect rect: [left, top, width, height]
    """
    l1, t1, b1, h1 = rect1
    r1, b1 = l1 + b1, t1 + h1
    l2, t2, b2, h2 = rect2
    r2, b2 = l2 + b2, t2 + h2

    l, r = max(l1, l2), min(r1, r2)
    t, b = max(t1, t2), min(b1, b2)

    return [l, t, max(0, r - l), max(0, b - t)]


def putText2(
    image: np.ndarray,
    text: str,
    center: typing.List[int],
    font_face: int = 0,
    font_scale: float = 1.0,
    color: typing.List[int] = (255, 255, 255),
    thickness: int = 1,
) -> None:
    """
    put text on image, modify image inplace
    """
    INNER_LINE_MARGIN = 5
    x, y = center
    lines = text.splitlines()

    sizes = [
        cv2.getTextSize(line, font_face, font_scale, thickness)[0] for line in lines
    ]

    h_sum = sum([size[1] for size in sizes]) + \
        (len(sizes) - 1) * INNER_LINE_MARGIN
    w_max = max([size[0] for size in sizes])

    y_base = y - h_sum // 2

    for i, (w, h) in enumerate(sizes):
        cv2.putText(
            image,
            lines[i],
            (x - w // 2, y_base),
            font_face,
            font_scale,
            color,
            thickness,
        )
        y_base += h + INNER_LINE_MARGIN


def putInverseColorText(
    image: np.ndarray,
    text: str,
    pos: typing.List[int],
    font_face: int = 0,
    font_scale: float = 1.0,
    thickness: int = 1,
    putTextFunc: typing.Callable = putText2,
) -> None:
    """
    put text on image, modify image inplace
    """
    mask = np.zeros((*image.shape[:2], 3), dtype=np.uint8)
    putTextFunc(mask, text, pos, font_face,
                font_scale, (255, 255, 255), thickness)

    # 0 -> 1, 1 -> -1: (2 * (0.5 - mask / 255.))
    # 1 -> 1,
    image[:, :, :3] = mask + (2 * (0.5 - mask / 255.0)) * image[:, :, :3]
