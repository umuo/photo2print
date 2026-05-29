from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from photo2print.cli import main


def test_cli_writes_balanced_and_print_images(tmp_path, capsys):
    input_path = tmp_path / "input.jpg"
    output_dir = tmp_path / "generated"
    image = np.full((180, 130, 3), 235, dtype=np.uint8)
    cv2.putText(image, "ABC", (18, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.imwrite(str(input_path), image)

    main([str(input_path), "--output-dir", str(output_dir)])

    assert (output_dir / "input_balanced.png").exists()
    assert (output_dir / "input_print_soft.png").exists()
    assert (output_dir / "input_print.png").exists()
    assert (output_dir / "input_print.pdf").exists()
    stdout = capsys.readouterr().out
    assert "Wrote balanced image" in stdout
    assert "Wrote soft print image" in stdout
    assert "Wrote printable PDF" in stdout


def test_cli_can_skip_pdf_export(tmp_path, capsys):
    input_path = tmp_path / "input.jpg"
    output_dir = tmp_path / "generated"
    image = np.full((140, 100, 3), 240, dtype=np.uint8)
    cv2.putText(image, "PDF", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.imwrite(str(input_path), image)

    main([str(input_path), "--output-dir", str(output_dir), "--no-pdf"])

    assert (output_dir / "input_balanced.png").exists()
    assert (output_dir / "input_print_soft.png").exists()
    assert (output_dir / "input_print.png").exists()
    assert not (output_dir / "input_print.pdf").exists()
    assert "Wrote printable PDF" not in capsys.readouterr().out


def test_cli_can_write_reconstructed_image(tmp_path, capsys):
    input_path = tmp_path / "input.jpg"
    output_dir = tmp_path / "generated"
    image = np.full((180, 130, 3), 235, dtype=np.uint8)
    cv2.putText(image, "ABC", (18, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.imwrite(str(input_path), image)

    main([str(input_path), "--output-dir", str(output_dir), "--reconstruct"])

    assert (output_dir / "input_reconstructed.png").exists()
    assert "Wrote reconstructed image" in capsys.readouterr().out


def test_cli_can_use_reconstructed_pdf_variant_without_png_flag(tmp_path):
    input_path = tmp_path / "input.jpg"
    output_dir = tmp_path / "generated"
    image = np.full((180, 130, 3), 235, dtype=np.uint8)
    cv2.putText(image, "PDF", (18, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.imwrite(str(input_path), image)

    main([str(input_path), "--output-dir", str(output_dir), "--pdf-variant", "reconstructed"])

    assert (output_dir / "input_print.pdf").exists()
    assert not (output_dir / "input_reconstructed.png").exists()


def test_cli_processes_multiple_positional_images_and_writes_merged_pdf(tmp_path, capsys):
    first_path = tmp_path / "first.jpg"
    second_path = tmp_path / "second.jpg"
    output_dir = tmp_path / "generated"
    first = np.full((140, 100, 3), 240, dtype=np.uint8)
    second = np.full((120, 160, 3), 238, dtype=np.uint8)
    cv2.putText(first, "1", (35, 78), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (25, 25, 25), 2, cv2.LINE_AA)
    cv2.putText(second, "2", (68, 72), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (25, 25, 25), 2, cv2.LINE_AA)
    cv2.imwrite(str(first_path), first)
    cv2.imwrite(str(second_path), second)

    main([str(first_path), str(second_path), "--output-dir", str(output_dir)])

    assert (output_dir / "first_balanced.png").exists()
    assert (output_dir / "first_print_soft.png").exists()
    assert (output_dir / "first_print.png").exists()
    assert not (output_dir / "first_print.pdf").exists()
    assert (output_dir / "second_balanced.png").exists()
    assert (output_dir / "second_print_soft.png").exists()
    assert (output_dir / "second_print.png").exists()
    assert not (output_dir / "second_print.pdf").exists()
    assert (output_dir / "merged_print.pdf").exists()
    stdout = capsys.readouterr().out
    assert "Wrote merged printable PDF" in stdout
    assert "Wrote printable PDF" not in stdout


def test_cli_batch_can_write_per_image_pdfs(tmp_path):
    first_path = tmp_path / "first.jpg"
    second_path = tmp_path / "second.jpg"
    output_dir = tmp_path / "generated"
    image = np.full((120, 90, 3), 240, dtype=np.uint8)
    cv2.putText(image, "PDF", (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (25, 25, 25), 2, cv2.LINE_AA)
    cv2.imwrite(str(first_path), image)
    cv2.imwrite(str(second_path), image)

    main([str(first_path), str(second_path), "--output-dir", str(output_dir), "--per-image-pdf"])

    assert (output_dir / "first_print.pdf").exists()
    assert (output_dir / "second_print.pdf").exists()
    assert (output_dir / "merged_print.pdf").exists()


def test_cli_processes_input_dir_by_filename_order(tmp_path, capsys):
    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "generated"
    input_dir.mkdir()
    for name, text in [("b.jpg", "B"), ("a.jpg", "A")]:
        image = np.full((120, 90, 3), 240, dtype=np.uint8)
        cv2.putText(image, text, (28, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (25, 25, 25), 2, cv2.LINE_AA)
        cv2.imwrite(str(input_dir / name), image)

    main(["--input-dir", str(input_dir), "--output-dir", str(output_dir)])

    assert (output_dir / "a_print.png").exists()
    assert (output_dir / "b_print.png").exists()
    assert (output_dir / "merged_print.pdf").exists()
    stdout = capsys.readouterr().out
    assert stdout.index("a_balanced.png") < stdout.index("b_balanced.png")


def test_cli_rejects_missing_input(tmp_path):
    missing = tmp_path / "missing.jpg"

    try:
        main([str(missing)])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("missing input should fail")
