from __future__ import annotations

import argparse
from pathlib import Path

from .processing import (
    PDF_VARIANTS,
    SUPPORTED_EXTENSIONS,
    load_image,
    output_paths,
    pdf_output_path,
    process_document,
    save_image,
    save_pdf,
    select_pdf_variant,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="photo2print",
        description="Convert a photographed document page into print-friendly cleaned images.",
    )
    parser.add_argument("image", type=Path, help="Path to one photographed document or worksheet image.")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for generated PNG files. Defaults to ./outputs.",
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Skip the default printable PDF export.",
    )
    parser.add_argument(
        "--pdf-variant",
        choices=sorted(PDF_VARIANTS),
        default="print",
        help="Processed variant to place in the PDF. Defaults to print.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    image_path = args.image.expanduser()
    output_dir = args.output_dir.expanduser()

    if not image_path.exists():
        parser.error(f"input image does not exist: {image_path}")
    if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        parser.error(f"unsupported image type {image_path.suffix!r}; expected one of: {supported}")

    image = load_image(image_path)
    processed = process_document(image)
    balanced_path, print_soft_path, print_path = output_paths(image_path, output_dir)
    save_image(balanced_path, processed.balanced)
    save_image(print_soft_path, processed.print_soft)
    save_image(print_path, processed.print)
    pdf_path = pdf_output_path(image_path, output_dir)
    if not args.no_pdf:
        save_pdf(pdf_path, select_pdf_variant(processed, args.pdf_variant))

    print(f"Wrote balanced image: {balanced_path}")
    print(f"Wrote soft print image: {print_soft_path}")
    print(f"Wrote print image: {print_path}")
    if not args.no_pdf:
        print(f"Wrote printable PDF: {pdf_path}")
