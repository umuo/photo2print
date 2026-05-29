from __future__ import annotations

import cv2
import numpy as np

from photo2print.processing import (
    merged_pdf_output_path,
    output_paths,
    pdf_output_path,
    process_document,
    reconstructed_output_path,
    save_pdf,
    save_pdf_pages,
    select_pdf_variant,
)


def test_output_paths_use_input_stem(tmp_path):
    balanced, print_soft, print_ready = output_paths(tmp_path / "worksheet.photo.jpg", tmp_path / "out")

    assert balanced == tmp_path / "out" / "worksheet.photo_balanced.png"
    assert print_soft == tmp_path / "out" / "worksheet.photo_print_soft.png"
    assert print_ready == tmp_path / "out" / "worksheet.photo_print.png"


def test_pdf_output_path_uses_input_stem(tmp_path):
    assert pdf_output_path(tmp_path / "worksheet.photo.jpg", tmp_path / "out") == tmp_path / "out" / "worksheet.photo_print.pdf"


def test_reconstructed_output_path_uses_input_stem(tmp_path):
    assert reconstructed_output_path(tmp_path / "worksheet.photo.jpg", tmp_path / "out") == tmp_path / "out" / "worksheet.photo_reconstructed.png"


def test_merged_pdf_output_path_uses_output_dir(tmp_path):
    assert merged_pdf_output_path(tmp_path / "out") == tmp_path / "out" / "merged_print.pdf"


def test_select_pdf_variant_returns_requested_processed_image():
    processed = process_document(np.full((120, 90, 3), 235, dtype=np.uint8))

    assert select_pdf_variant(processed, "balanced") is processed.balanced
    assert select_pdf_variant(processed, "print-soft") is processed.print_soft
    assert select_pdf_variant(processed, "print") is processed.print
    assert select_pdf_variant(processed, "reconstructed") is processed.reconstructed


def test_save_pdf_writes_a4_raster_pdf(tmp_path):
    pdf_path = tmp_path / "out" / "page.pdf"
    image = np.full((180, 120, 3), 255, dtype=np.uint8)
    cv2.putText(image, "A4", (24, 94), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2, cv2.LINE_AA)

    save_pdf(pdf_path, image)

    data = pdf_path.read_bytes()
    assert data.startswith(b"%PDF")
    assert b"/MediaBox [ 0 0 595" in data
    assert b"841" in data


def test_save_pdf_pages_writes_multiple_a4_pages(tmp_path):
    pdf_path = tmp_path / "out" / "merged.pdf"
    first = np.full((180, 120, 3), 255, dtype=np.uint8)
    second = np.full((120, 180, 3), 255, dtype=np.uint8)
    cv2.putText(first, "1", (44, 102), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(second, "2", (72, 74), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2, cv2.LINE_AA)

    save_pdf_pages(pdf_path, [first, second])

    data = pdf_path.read_bytes()
    assert data.startswith(b"%PDF")
    assert data.count(b"/Type /Page") >= 2


def test_process_document_returns_readable_cleaned_variants():
    image = np.full((360, 260, 3), (214, 219, 184), dtype=np.uint8)
    cv2.rectangle(image, (18, 18), (242, 342), (235, 232, 205), -1)
    cv2.putText(image, "Worksheet", (40, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (55, 55, 55), 2, cv2.LINE_AA)
    cv2.putText(image, "2 + 3 = ____", (40, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (45, 45, 45), 2, cv2.LINE_AA)
    cv2.circle(image, (205, 235), 34, (150, 145, 120), -1)

    processed = process_document(image)

    assert processed.balanced.shape[:2] == processed.print_soft.shape[:2] == processed.print.shape[:2]
    assert processed.balanced.ndim == 3
    assert processed.print_soft.ndim == 3
    assert processed.print.ndim == 3
    assert processed.reconstructed.ndim == 3
    assert processed.balanced.mean() > image.mean()
    assert processed.print_soft.mean() > 215
    assert processed.print.mean() > 220
    assert processed.print.min() < 30


def test_process_document_builds_reconstructed_a4_white_page():
    image = np.full((420, 300, 3), (205, 212, 184), dtype=np.uint8)
    cv2.rectangle(image, (26, 24), (274, 396), (236, 234, 220), -1)
    for index, y in enumerate(range(80, 255, 34)):
        cv2.putText(image, f"{index + 1}. 12 + {index} = ____", (48, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (45, 45, 45), 1, cv2.LINE_AA)
    cv2.line(image, (50, 304), (230, 304), (112, 112, 112), 1, cv2.LINE_AA)
    cv2.putText(image, "show", (178, 176), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (190, 190, 190), 1, cv2.LINE_AA)

    processed = process_document(image)
    reconstructed = cv2.cvtColor(processed.reconstructed, cv2.COLOR_BGR2GRAY)
    height, width = reconstructed.shape

    assert abs((height / width) - (297 / 210)) < 0.025
    assert reconstructed.mean() > 246
    assert np.count_nonzero(reconstructed < 170) > 80
    assert np.mean(reconstructed == 255) > 0.86


def test_process_document_preserves_tiny_faint_marks():
    image = np.full((220, 260, 3), 236, dtype=np.uint8)
    cv2.rectangle(image, (16, 16), (244, 204), (242, 242, 232), -1)
    cv2.putText(image, "cm3 dm3", (42, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (70, 70, 70), 2, cv2.LINE_AA)
    cv2.putText(image, "3", (88, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (92, 92, 92), 1, cv2.LINE_AA)
    cv2.putText(image, ".", (174, 73), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1, cv2.LINE_AA)
    cv2.line(image, (44, 134), (164, 134), (118, 118, 118), 1, cv2.LINE_AA)
    cv2.line(image, (44, 154), (116, 154), (146, 146, 146), 1, cv2.LINE_AA)

    processed = process_document(image)
    print_gray = cv2.cvtColor(processed.print, cv2.COLOR_BGR2GRAY)
    soft_gray = cv2.cvtColor(processed.print_soft, cv2.COLOR_BGR2GRAY)

    superscript_window = print_gray[52:76, 82:102]
    punctuation_window = print_gray[58:84, 166:186]
    faint_line_window = soft_gray[148:160, 42:120]
    background_window = print_gray[22:48, 26:70]

    assert superscript_window.min() < 145
    assert punctuation_window.min() < 170
    assert faint_line_window.min() < 210
    assert background_window.mean() > 240


def test_process_document_suppresses_backside_show_through_in_blank_regions():
    image = np.full((260, 360, 3), 242, dtype=np.uint8)
    cv2.rectangle(image, (12, 12), (348, 248), (246, 246, 238), -1)

    cv2.putText(image, "x + 1 =", (38, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (55, 55, 55), 2, cv2.LINE_AA)
    cv2.putText(image, "2", (74, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (92, 92, 92), 1, cv2.LINE_AA)
    cv2.putText(image, ".", (132, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (96, 96, 96), 1, cv2.LINE_AA)
    cv2.line(image, (42, 142), (156, 142), (130, 130, 130), 1, cv2.LINE_AA)

    cv2.putText(image, "808", (210, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (188, 188, 188), 1, cv2.LINE_AA)
    cv2.putText(image, "ABC", (206, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (196, 196, 196), 1, cv2.LINE_AA)
    for center in [(230, 160), (244, 166), (270, 154), (304, 175)]:
        cv2.circle(image, center, 1, (190, 190, 190), -1)

    processed = process_document(image)
    print_gray = cv2.cvtColor(processed.print, cv2.COLOR_BGR2GRAY)
    soft_gray = cv2.cvtColor(processed.print_soft, cv2.COLOR_BGR2GRAY)

    show_through_window = print_gray[70:140, 200:330]
    superscript_window = print_gray[48:70, 68:88]
    punctuation_window = print_gray[54:76, 126:142]
    answer_line_window = print_gray[136:148, 40:160]

    assert show_through_window.mean() == 255
    assert show_through_window.std() == 0
    assert np.count_nonzero(show_through_window < 255) == 0
    assert np.count_nonzero(soft_gray[70:140, 200:330] < 253) > 0
    assert superscript_window.min() < 145
    assert punctuation_window.min() < 170
    assert answer_line_window.min() < 210


def test_aggressive_print_whitens_shaded_blank_paper_more_decisively():
    image = np.full((300, 420, 3), 238, dtype=np.uint8)
    cv2.rectangle(image, (14, 14), (406, 286), (244, 244, 236), -1)
    cv2.putText(image, "Solve:", (42, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (54, 54, 54), 2, cv2.LINE_AA)
    cv2.putText(image, "3x - 1 = 8", (42, 126), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (58, 58, 58), 2, cv2.LINE_AA)
    cv2.line(image, (46, 198), (180, 198), (120, 120, 120), 1, cv2.LINE_AA)

    yy, xx = np.indices((116, 162))
    haze = 152 + 12 * np.sin(xx / 13.0) + 9 * np.cos(yy / 11.0)
    blank_patch = np.clip(haze, 126, 178).astype(np.uint8)
    image[154:270, 226:388] = cv2.cvtColor(blank_patch, cv2.COLOR_GRAY2BGR)

    processed = process_document(image)
    print_gray = cv2.cvtColor(processed.print, cv2.COLOR_BGR2GRAY)

    shaded_blank = print_gray[170:252, 244:370]
    answer_line_window = print_gray[192:204, 42:184]
    equation_window = print_gray[102:136, 36:232]

    assert np.mean(shaded_blank >= 253) > 0.97
    assert shaded_blank.std() < 16
    assert answer_line_window.min() < 210
    assert equation_window.min() < 80
