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


def test_cli_rejects_missing_input(tmp_path):
    missing = tmp_path / "missing.jpg"

    try:
        main([str(missing)])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("missing input should fail")
