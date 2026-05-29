from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


SUPPORTED_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
PDF_VARIANTS = {"balanced", "print-soft", "print"}
A4_SIZE_MM = (210.0, 297.0)


@dataclass(frozen=True)
class ProcessedDocument:
    balanced: np.ndarray
    print_soft: np.ndarray
    print: np.ndarray


def load_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else image
    Image.fromarray(rgb).save(path, quality=95)


def save_pdf(path: Path, image: np.ndarray, *, dpi: int = 300, margin_mm: float = 12.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    page_width, page_height = _a4_pixel_size(dpi)
    image_height, image_width = image.shape[:2]
    if image_width > image_height:
        page_width, page_height = page_height, page_width

    margin_px = round(margin_mm * dpi / 25.4)
    content_width = max(1, page_width - (2 * margin_px))
    content_height = max(1, page_height - (2 * margin_px))
    scale = min(content_width / image_width, content_height / image_height)
    resized_size = (max(1, round(image_width * scale)), max(1, round(image_height * scale)))

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else image
    page = Image.new("RGB", (page_width, page_height), "white")
    document = Image.fromarray(rgb).convert("RGB").resize(resized_size, Image.Resampling.LANCZOS)
    offset = ((page_width - resized_size[0]) // 2, (page_height - resized_size[1]) // 2)
    page.paste(document, offset)
    page.save(path, "PDF", resolution=dpi)


def process_document(image: np.ndarray) -> ProcessedDocument:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("expected a BGR color image")

    corrected = _correct_perspective_conservatively(image)
    gray_for_angle = _normalize_illumination(cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY))
    angle = _estimate_skew_angle(gray_for_angle)
    if abs(angle) >= 0.15:
        corrected = _rotate_bound(corrected, -angle)

    gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)
    normalized = _normalize_illumination(gray)
    balanced = _make_balanced(normalized)
    print_soft = _make_soft_print(balanced, reference_gray=gray)
    print_ready = _make_print_ready(balanced, reference_gray=gray)
    return ProcessedDocument(
        balanced=cv2.cvtColor(balanced, cv2.COLOR_GRAY2BGR),
        print_soft=cv2.cvtColor(print_soft, cv2.COLOR_GRAY2BGR),
        print=cv2.cvtColor(print_ready, cv2.COLOR_GRAY2BGR),
    )


def output_paths(input_path: Path, output_dir: Path) -> tuple[Path, Path, Path]:
    stem = input_path.stem
    return (
        output_dir / f"{stem}_balanced.png",
        output_dir / f"{stem}_print_soft.png",
        output_dir / f"{stem}_print.png",
    )


def pdf_output_path(input_path: Path, output_dir: Path) -> Path:
    return output_dir / f"{input_path.stem}_print.pdf"


def select_pdf_variant(processed: ProcessedDocument, variant: str) -> np.ndarray:
    if variant == "balanced":
        return processed.balanced
    if variant == "print-soft":
        return processed.print_soft
    if variant == "print":
        return processed.print
    expected = ", ".join(sorted(PDF_VARIANTS))
    raise ValueError(f"unknown PDF variant {variant!r}; expected one of: {expected}")


def _a4_pixel_size(dpi: int) -> tuple[int, int]:
    width_mm, height_mm = A4_SIZE_MM
    return (round(width_mm * dpi / 25.4), round(height_mm * dpi / 25.4))


def _correct_perspective_conservatively(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    max_side = max(width, height)
    scale = 1200 / max_side if max_side > 1200 else 1.0
    small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else image

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    image_area = small.shape[0] * small.shape[1]
    candidates = sorted(contours, key=cv2.contourArea, reverse=True)[:8]
    for contour in candidates:
        area = cv2.contourArea(contour)
        if area < image_area * 0.45:
            continue
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue

        points = approx.reshape(4, 2).astype(np.float32) / scale
        if not _quad_is_reasonable(points, width, height):
            continue
        return _warp_quad(image, points)

    return image


def _quad_is_reasonable(points: np.ndarray, width: int, height: int) -> bool:
    area = cv2.contourArea(points)
    if area < width * height * 0.45:
        return False

    ordered = _order_points(points)
    sides = np.array(
        [
            np.linalg.norm(ordered[1] - ordered[0]),
            np.linalg.norm(ordered[2] - ordered[1]),
            np.linalg.norm(ordered[3] - ordered[2]),
            np.linalg.norm(ordered[0] - ordered[3]),
        ]
    )
    if np.any(sides < min(width, height) * 0.25):
        return False

    top, right, bottom, left = sides
    if max(top, bottom) / max(1.0, min(top, bottom)) > 1.45:
        return False
    if max(left, right) / max(1.0, min(left, right)) > 1.45:
        return False
    return True


def _warp_quad(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    ordered = _order_points(points)
    tl, tr, br, bl = ordered
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    target_width = int(round(max(width_a, width_b)))
    target_height = int(round(max(height_a, height_b)))
    if target_width < 100 or target_height < 100:
        return image

    destination = np.array(
        [[0, 0], [target_width - 1, 0], [target_width - 1, target_height - 1], [0, target_height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(ordered, destination)
    return cv2.warpPerspective(
        image,
        matrix,
        (target_width, target_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _order_points(points: np.ndarray) -> np.ndarray:
    ordered = np.zeros((4, 2), dtype=np.float32)
    summed = points.sum(axis=1)
    diff = np.diff(points, axis=1).reshape(-1)
    ordered[0] = points[np.argmin(summed)]
    ordered[2] = points[np.argmax(summed)]
    ordered[1] = points[np.argmin(diff)]
    ordered[3] = points[np.argmax(diff)]
    return ordered


def _normalize_illumination(gray: np.ndarray) -> np.ndarray:
    kernel = _odd_kernel_size(min(gray.shape[:2]) // 18)
    background = cv2.medianBlur(gray, kernel)
    normalized = cv2.divide(gray, background, scale=255)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    return clahe.apply(normalized)


def _make_balanced(gray: np.ndarray) -> np.ndarray:
    low, high = np.percentile(gray, (0.6, 99.6))
    if high <= low:
        return gray
    stretched = np.clip((gray.astype(np.float32) - low) * 255.0 / (high - low), 0, 255).astype(np.uint8)
    white_mask = stretched > 218
    stretched[white_mask] = np.clip(stretched[white_mask].astype(np.int16) + 22, 0, 255).astype(np.uint8)
    return stretched


def _make_soft_print(gray: np.ndarray, *, reference_gray: np.ndarray | None = None) -> np.ndarray:
    return _compose_text_protected_print(
        gray,
        foreground_strength=0.78,
        background_floor=238,
        reference_gray=reference_gray,
        aggressive_blank_cleanup=False,
    )


def _make_print_ready(gray: np.ndarray, *, reference_gray: np.ndarray | None = None) -> np.ndarray:
    return _compose_text_protected_print(
        gray,
        foreground_strength=1.05,
        background_floor=246,
        reference_gray=reference_gray,
        aggressive_blank_cleanup=True,
    )


def _compose_text_protected_print(
    gray: np.ndarray,
    *,
    foreground_strength: float,
    background_floor: int,
    reference_gray: np.ndarray | None = None,
    aggressive_blank_cleanup: bool,
) -> np.ndarray:
    whitened = _whiten_background_safely(gray, background_floor=background_floor)
    foreground = _text_foreground_layer(gray, strength=foreground_strength)
    protection = _text_protection_alpha(gray)
    composed = whitened.astype(np.float32) * (1.0 - protection) + foreground.astype(np.float32) * protection
    print_ready = np.clip(composed, 0, 255).astype(np.uint8)
    return _suppress_show_through_in_blank_regions(
        gray if reference_gray is None else reference_gray,
        print_ready,
        background_floor=background_floor,
        aggressive=aggressive_blank_cleanup,
    )


def _whiten_background_safely(gray: np.ndarray, *, background_floor: int) -> np.ndarray:
    result = gray.astype(np.float32)
    local_background = cv2.GaussianBlur(gray, (0, 0), sigmaX=10, sigmaY=10).astype(np.float32)
    local_contrast = local_background - result

    plain_background = (result > 172) & (local_contrast < 10)
    result[plain_background] = np.maximum(result[plain_background], background_floor)

    very_light = result > 214
    result[very_light] = np.clip(result[very_light] + 24, 0, 255)
    return np.clip(result, 0, 255).astype(np.uint8)


def _text_foreground_layer(gray: np.ndarray, *, strength: float) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=0.65)
    sharpened = cv2.addWeighted(gray, 1.0 + (0.32 * strength), blur, -(0.32 * strength), 0)
    local_background = cv2.GaussianBlur(gray, (0, 0), sigmaX=7, sigmaY=7)
    local_contrast = np.maximum(local_background.astype(np.int16) - gray.astype(np.int16), 0)
    darkened = sharpened.astype(np.int16) - np.rint(local_contrast * 0.75 * strength).astype(np.int16)
    return np.clip(darkened, 0, 255).astype(np.uint8)


def _text_protection_alpha(gray: np.ndarray) -> np.ndarray:
    small_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    large_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    blackhat_small = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, small_kernel)
    blackhat_large = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, large_kernel)
    blackhat = np.maximum(blackhat_small, blackhat_large).astype(np.float32)

    local_background = cv2.GaussianBlur(gray, (0, 0), sigmaX=9, sigmaY=9).astype(np.float32)
    local_contrast = local_background - gray.astype(np.float32)
    darkness = np.maximum(184.0 - gray.astype(np.float32), 0.0) * 0.33

    evidence = np.maximum.reduce([blackhat * 1.15, local_contrast, darkness])
    alpha = np.clip((evidence - 3.0) / 22.0, 0.0, 1.0)
    return cv2.GaussianBlur(alpha, (3, 3), sigmaX=0.35)


def _suppress_show_through_in_blank_regions(
    source_gray: np.ndarray,
    print_gray: np.ndarray,
    *,
    background_floor: int,
    aggressive: bool,
) -> np.ndarray:
    front_mask = _front_content_mask(source_gray)
    line_mask = _thin_horizontal_line_mask(source_gray)
    halo_size = 15 if aggressive else 9
    protected = cv2.dilate(
        (front_mask | line_mask).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (halo_size, halo_size)),
        iterations=1,
    ).astype(bool)

    blank_paper = _blank_paper_mask(source_gray, protected, aggressive=aggressive)
    sigma = 15 if aggressive else 11
    local_background = cv2.GaussianBlur(source_gray, (0, 0), sigmaX=sigma, sigmaY=sigma).astype(np.int16)
    source_contrast = local_background - source_gray.astype(np.int16)
    low_contrast_limit = 76 if aggressive else 65
    minimum_print_value = min(255, background_floor + (9 if aggressive else 8))
    dark_fragment = (
        blank_paper
        & (source_gray > (104 if aggressive else 118))
        & (source_contrast >= 3)
        & (source_contrast <= low_contrast_limit)
        & (print_gray < max(252, minimum_print_value))
    )

    cleaned = print_gray.copy()
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(dark_fragment.astype(np.uint8), 8)
    for label in range(1, component_count):
        x, y, width, height, area = stats[label]
        if not _looks_like_backside_fragment(width, height, area, aggressive=aggressive):
            continue

        component = labels == label
        guard = 4 if aggressive else 2
        if np.any(protected[max(0, y - guard) : y + height + guard, max(0, x - guard) : x + width + guard]):
            continue

        cleaned[component] = 255 if aggressive else np.maximum(cleaned[component], min(255, background_floor + 10))

    strong_blank = blank_paper & (source_contrast < (18 if aggressive else 7)) & (cleaned > (188 if aggressive else 210))
    cleaned[strong_blank] = 255 if aggressive else np.maximum(cleaned[strong_blank], min(255, background_floor + 12))
    if aggressive:
        cleaned = _whiten_low_contrast_blank_paper(source_gray, cleaned, blank_paper, protected)
        cleaned = _normalize_confident_blank_regions(source_gray, cleaned, blank_paper, protected)
    return cleaned


def _front_content_mask(gray: np.ndarray) -> np.ndarray:
    local_background = cv2.GaussianBlur(gray, (0, 0), sigmaX=8, sigmaY=8).astype(np.int16)
    local_contrast = local_background - gray.astype(np.int16)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13)))
    confident = (gray < 82) | ((local_contrast > 10) & (gray < 170)) | ((blackhat > 14) & (gray < 185))
    return cv2.morphologyEx(confident.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8)).astype(bool)


def _thin_horizontal_line_mask(gray: np.ndarray) -> np.ndarray:
    local_background = cv2.GaussianBlur(gray, (0, 0), sigmaX=7, sigmaY=3).astype(np.int16)
    local_contrast = local_background - gray.astype(np.int16)
    dark_thin = ((local_contrast > 5) & (gray < 215)).astype(np.uint8)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    return cv2.morphologyEx(dark_thin, cv2.MORPH_OPEN, horizontal_kernel).astype(bool)


def _blank_paper_mask(gray: np.ndarray, protected: np.ndarray, *, aggressive: bool) -> np.ndarray:
    local_std = _local_stddev(gray, sigma=9 if aggressive else 7)
    if aggressive:
        normalized = _normalize_illumination(gray)
        local_mean = cv2.GaussianBlur(normalized, (0, 0), sigmaX=15, sigmaY=15)
        paper_like = (gray > 96) & (normalized > 146) & (local_mean > 165) & (local_std < 43)
        distance_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    else:
        local_mean = cv2.GaussianBlur(gray, (0, 0), sigmaX=15, sigmaY=15)
        paper_like = (gray > 168) & (local_mean > 184)
        distance_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
    far_from_front = ~cv2.dilate(
        protected.astype(np.uint8),
        distance_kernel,
        iterations=1,
    ).astype(bool)
    return paper_like & far_from_front


def _looks_like_backside_fragment(width: int, height: int, area: int, *, aggressive: bool) -> bool:
    if area <= 2:
        return True
    if area > (3200 if aggressive else 900):
        return False
    aspect = width / max(1, height)
    if height <= 2 and width >= 18:
        return False
    if aspect >= 7.0 and width >= 24:
        return False
    limit = 110 if aggressive else 60
    return width <= limit and height <= limit


def _whiten_low_contrast_blank_paper(
    source_gray: np.ndarray,
    cleaned: np.ndarray,
    blank_paper: np.ndarray,
    protected: np.ndarray,
) -> np.ndarray:
    local_background = cv2.GaussianBlur(source_gray, (0, 0), sigmaX=21, sigmaY=21).astype(np.int16)
    source_contrast = local_background - source_gray.astype(np.int16)
    local_std = _local_stddev(source_gray, sigma=13)
    confident_blank = (
        blank_paper
        & ~protected
        & (source_gray > 138)
        & (source_contrast < 34)
        & (local_std < 24)
        & (cleaned > 178)
    )
    confident_blank = cv2.morphologyEx(
        confident_blank.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=1,
    ).astype(bool) & blank_paper
    result = cleaned.copy()
    result[confident_blank] = 255

    remaining_haze = blank_paper & ~protected & (source_gray > 126) & (source_contrast < 48) & (local_std < 29) & (result > 198)
    result[remaining_haze] = np.maximum(result[remaining_haze], 253)
    return result


def _normalize_confident_blank_regions(
    source_gray: np.ndarray,
    cleaned: np.ndarray,
    blank_paper: np.ndarray,
    protected: np.ndarray,
) -> np.ndarray:
    large_background = cv2.GaussianBlur(source_gray, (0, 0), sigmaX=33, sigmaY=33).astype(np.int16)
    low_frequency_contrast = large_background - source_gray.astype(np.int16)
    broad_std = _local_stddev(source_gray, sigma=23)
    residual_print_std = _local_stddev(cleaned, sigma=11)

    distant_protected = cv2.dilate(
        protected.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (37, 37)),
        iterations=1,
    ).astype(bool)
    confident_blank = (
        blank_paper
        & ~distant_protected
        & (source_gray > 122)
        & (large_background > 158)
        & (low_frequency_contrast < 64)
        & (broad_std < 42)
        & (cleaned > 166)
    )
    confident_blank = cv2.morphologyEx(
        confident_blank.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)),
        iterations=1,
    ).astype(bool) & blank_paper & ~distant_protected

    result = cleaned.copy()
    result[confident_blank] = 255

    paper_texture = (
        blank_paper
        & ~distant_protected
        & (source_gray > 134)
        & (low_frequency_contrast < 82)
        & (broad_std < 48)
        & (residual_print_std < 36)
        & (result > 184)
    )
    result[paper_texture] = 255

    normalized = _normalize_illumination(source_gray)
    normalized_mean = cv2.GaussianBlur(normalized, (0, 0), sigmaX=21, sigmaY=21)
    stroke_guard = _front_content_mask(source_gray) | _thin_horizontal_line_mask(source_gray)
    broad_paper_tone = (
        ~stroke_guard
        & (cleaned > 92)
        & (normalized > 126)
        & (normalized_mean > 166)
        & (broad_std < 64)
    )
    result[broad_paper_tone] = 255
    return result


def _local_stddev(gray: np.ndarray, *, sigma: float) -> np.ndarray:
    gray_float = gray.astype(np.float32)
    mean = cv2.GaussianBlur(gray_float, (0, 0), sigmaX=sigma, sigmaY=sigma)
    mean_square = cv2.GaussianBlur(gray_float * gray_float, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return np.sqrt(np.maximum(mean_square - mean * mean, 0.0))


def _estimate_skew_angle(gray: np.ndarray) -> float:
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
    coords = np.column_stack(np.where(binary > 0))
    if len(coords) < gray.size * 0.002:
        return 0.0

    angle = cv2.minAreaRect(coords.astype(np.float32))[2]
    if angle < -45:
        angle = 90 + angle
    if angle > 45:
        angle = angle - 90
    if abs(angle) > 3.0:
        return 0.0
    return float(angle)


def _rotate_bound(image: np.ndarray, angle: float) -> np.ndarray:
    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_width = int((height * sin) + (width * cos))
    new_height = int((height * cos) + (width * sin))
    matrix[0, 2] += (new_width / 2) - center[0]
    matrix[1, 2] += (new_height / 2) - center[1]
    return cv2.warpAffine(
        image,
        matrix,
        (new_width, new_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def _odd_kernel_size(value: int) -> int:
    value = max(15, value)
    if value % 2 == 0:
        value += 1
    return value
