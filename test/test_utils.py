import cv2
import numpy as np
from PIL import Image


def colorize_image_auto(img,
                        saturation=230,
                        hue_range=(180, 180),
                        blend_strength=1.0,
                        lowc_th_bg=0.05,
                        color_th_bg=0.12,
                        ellipse_size=(0.22, 0.28)):
    """
    Automatically detect subject and apply colorful gradient.

    Args:
        img: PIL Image (input image)
        saturation: Color saturation (0-255). Higher = more vibrant colors
        hue_range: Tuple of (x_hue_range, y_hue_range) for gradient variation
        blend_strength: How much gradient to apply (0.0-1.0). 1.0 = full color
        lowc_th_bg: Low contrast threshold for background detection
        color_th_bg: Color similarity threshold for background detection
        ellipse_size: Tuple of (rx_ratio, ry_ratio) for foreground seed ellipse

    Returns:
        PIL Image with colorized subject
    """
    # Convert PIL Image to numpy array
    img_np = np.array(img)

    # Convert RGB to BGR for OpenCV
    if len(img_np.shape) == 3 and img_np.shape[2] == 3:
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = img_np

    h, w = img_bgr.shape[:2]

    # Helper functions
    def border_connected_only(binary):
        """Keep only border-connected components."""
        num, labels = cv2.connectedComponents(binary.astype(np.uint8))
        edge_labels = set(np.unique(np.concatenate([
            labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]
        ])))
        keep = np.zeros_like(binary, np.uint8)
        for lb in edge_labels:
            if lb != 0:
                keep[labels == lb] = 1
        return keep

    def fill_holes(mask_u8):
        """Fill holes in binary mask."""
        inv = cv2.bitwise_not(mask_u8)
        ff = inv.copy()
        flood_mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(ff, flood_mask, (0, 0), 255)
        holes = cv2.bitwise_not(ff)
        filled = cv2.bitwise_or(mask_u8, holes)
        return filled

    def largest_component(mask_u8):
        """Keep only the largest connected component."""
        num, labels, stats, _ = cv2.connectedComponentsWithStats(
            (mask_u8 > 0).astype(np.uint8), connectivity=8
        )
        if num <= 1:
            return mask_u8
        areas = stats[1:, cv2.CC_STAT_AREA]
        max_idx = 1 + np.argmax(areas)
        largest = (labels == max_idx).astype(np.uint8) * 255
        return largest

    # 1) Calculate background color from border pixels
    B = 10
    border_pixels = np.concatenate([
        img_bgr[:B].reshape(-1, 3), img_bgr[-B:].reshape(-1, 3),
        img_bgr[:, :B].reshape(-1, 3), img_bgr[:, -B:].reshape(-1, 3)
    ], axis=0).astype(np.float32)
    bg_col = np.median(border_pixels, axis=0)
    d = np.linalg.norm(img_bgr.astype(np.float32) - bg_col[None, None, :], axis=2)
    d_norm = (d - d.min()) / (np.ptp(d) + 1e-6)

    # 2) Calculate local contrast (gradient magnitude)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-6)

    # 3) Create sure background seed (low contrast + similar to border color + border-connected)
    seed_bg = ((mag < lowc_th_bg) & (d_norm < color_th_bg)).astype(np.uint8)
    sure_bg = border_connected_only(seed_bg)
    kernel = np.ones((5, 5), np.uint8)
    sure_bg = cv2.morphologyEx(sure_bg, cv2.MORPH_DILATE, kernel, iterations=4)

    # 4) Create sure foreground seed (center ellipse)
    Y, X = np.ogrid[:h, :w]
    cx, cy = w / 2, h / 2
    rx, ry = w * ellipse_size[0], h * ellipse_size[1]
    sure_fg = ((((X - cx) ** 2) / (rx * rx) + ((Y - cy) ** 2) / (ry * ry)) <= 1.0).astype(np.uint8)

    # 5) Initialize GrabCut mask and run
    gc_mask = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    gc_mask[sure_bg == 1] = cv2.GC_BGD
    gc_mask[sure_fg == 1] = cv2.GC_FGD

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    cv2.grabCut(img_bgr, gc_mask, None, bgdModel, fgdModel, 6, cv2.GC_INIT_WITH_MASK)

    fg = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8) * 255

    # 6) Post-process mask: fill holes → largest component → trim border leaks
    fg = fill_holes(fg)
    fg = largest_component(fg)

    # Trim background-like border-connected regions
    LOWC_TH_TRIM = 0.07
    COLOR_TH_TRIM = 0.15
    bg_like = ((d_norm < COLOR_TH_TRIM) & (mag < LOWC_TH_TRIM)).astype(np.uint8)
    bg_border = border_connected_only(bg_like)
    fg = cv2.bitwise_and(fg, cv2.bitwise_not((bg_border * 255).astype(np.uint8)))

    # Soft edges
    fg = cv2.GaussianBlur(fg, (7, 7), 0)
    alpha = (fg.astype(np.float32) / 255.0) * blend_strength
    alpha3 = np.dstack([alpha] * 3)

    # 7) Create colorful gradient (inject H, S; V based on original luminance)
    xx = np.tile(np.linspace(0, 1, w, dtype=np.float32), (h, 1))
    yy = np.tile(np.linspace(0, 1, h, dtype=np.float32), (w, 1)).T
    hue = (xx * hue_range[0] + yy * hue_range[1]) % 180.0
    sat = np.full((h, w), saturation, np.float32)
    v = np.clip(gray * 255.0 * (0.95 + 0.15 * alpha), 0, 255)

    hsv = np.dstack([hue, sat, v]).astype(np.uint8)
    grad_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # 8) Composite: foreground = gradient, background = original
    out = (grad_bgr * alpha3 + img_bgr * (1.0 - alpha3)).astype(np.uint8)

    # Convert BGR back to RGB and return as PIL Image
    out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    return Image.fromarray(out_rgb)


def colorize_subject(img, mask,
                     saturation=230,
                     hue_range=(180, 120),
                     blend_strength=1.0):
    """
    Apply colorful gradient to the subject in the image.

    Args:
        img: PIL Image (input image)
        mask: PIL Image (foreground mask, grayscale)
        saturation: Color saturation (0-255). Higher = more vibrant colors
        hue_range: Tuple of (x_hue_range, y_hue_range) for gradient variation
        blend_strength: How much gradient to apply (0.0-1.0). 1.0 = full color

    Returns:
        PIL Image with colorized subject
    """
    # Convert PIL Images to numpy arrays
    img_np = np.array(img)
    mask_np = np.array(mask.convert('L'))  # Convert mask to grayscale if not already

    # Convert RGB to BGR for OpenCV
    if len(img_np.shape) == 3 and img_np.shape[2] == 3:
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = img_np

    h, w = img_bgr.shape[:2]
    fg = mask_np

    # Recompute d_norm and gradient
    b = 10
    border_pixels = np.concatenate([
        img_bgr[:b].reshape(-1, 3), img_bgr[-b:].reshape(-1, 3),
        img_bgr[:, :b].reshape(-1, 3), img_bgr[:, -b:].reshape(-1, 3)
    ], axis=0).astype(np.float32)
    bg_col = np.median(border_pixels, axis=0)
    d = np.linalg.norm(img_bgr.astype(np.float32) - bg_col[None, None, :], axis=2)
    d_norm = (d - d.min()) / (np.ptp(d) + 1e-6)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, 3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, 3)
    mag = np.sqrt(gx * gx + gy * gy)
    mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-6)

    bg_like = ((d_norm < 0.15) & (mag < 0.07)).astype(np.uint8)

    # Keep only border-connected components of bg_like
    num, labels = cv2.connectedComponents(bg_like)
    border_labels = set(np.unique(np.concatenate([
        labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]
    ])))
    bg_border = np.zeros_like(bg_like)
    for lb in border_labels:
        if lb != 0:
            bg_border[labels == lb] = 1

    # Subtract these from the foreground mask
    fg_clean = cv2.bitwise_and(fg, cv2.bitwise_not((bg_border * 255).astype(np.uint8)))

    # Soft edges
    fg_clean = cv2.GaussianBlur(fg_clean, (7, 7), 0)

    # Recolorize with adjustable parameters
    m = (fg_clean.astype(np.float32) / 255.0) * blend_strength
    m3 = np.dstack([m] * 3)

    xx = np.tile(np.linspace(0, 1, w, dtype=np.float32), (h, 1))
    yy = np.tile(np.linspace(0, 1, h, dtype=np.float32), (w, 1)).T
    hue = (xx * hue_range[0] + yy * hue_range[1]) % 180
    sat = np.full((h, w), saturation, np.float32)
    v = np.clip(gray * 255 * (0.95 + 0.15 * m), 0, 255)
    hsv = np.dstack([hue, sat, v]).astype(np.uint8)
    grad = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    out = (grad * m3 + img_bgr * (1 - m3)).astype(np.uint8)

    # Convert BGR back to RGB and return as PIL Image
    out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    return Image.fromarray(out_rgb)