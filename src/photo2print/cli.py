from __future__ import annotations

import argparse
from pathlib import Path

from .processing import (
    PDF_VARIANTS,
    SUPPORTED_EXTENSIONS,
    load_image,
    merged_pdf_output_path,
    output_paths,
    pdf_output_path,
    process_document,
    reconstructed_output_path,
    save_image,
    save_pdf,
    save_pdf_pages,
    select_pdf_variant,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="photo2print",
        description="Convert a photographed document page into print-friendly cleaned images.",
    )
    parser.add_argument("images", type=Path, nargs="*", help="Path to one or more photographed document images.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Directory containing images to process as a batch. Files are processed by name.",
    )
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
    parser.add_argument(
        "--reconstruct",
        action="store_true",
        help="Also write a reconstruction-oriented PNG for text-heavy worksheet pages.",
    )
    parser.add_argument(
        "--merged-pdf",
        type=Path,
        help="Path for the merged batch PDF. Defaults to OUTPUT_DIR/merged_print.pdf.",
    )
    parser.add_argument(
        "--per-image-pdf",
        action="store_true",
        help="In batch mode, also write one printable PDF next to each image's PNG outputs.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    output_dir = args.output_dir.expanduser()
    image_paths = _resolve_input_paths(parser, args.images, args.input_dir)
    batch_mode = args.input_dir is not None or len(image_paths) > 1

    pdf_pages = []
    for image_path in image_paths:
        image = load_image(image_path)
        processed = process_document(image)
        balanced_path, print_soft_path, print_path = output_paths(image_path, output_dir)
        save_image(balanced_path, processed.balanced)
        save_image(print_soft_path, processed.print_soft)
        save_image(print_path, processed.print)
        reconstructed_path = reconstructed_output_path(image_path, output_dir)
        if args.reconstruct:
            save_image(reconstructed_path, processed.reconstructed)

        pdf_image = select_pdf_variant(processed, args.pdf_variant)
        pdf_pages.append(pdf_image)
        pdf_path = pdf_output_path(image_path, output_dir)
        if not args.no_pdf and (not batch_mode or args.per_image_pdf):
            save_pdf(pdf_path, pdf_image)

        print(f"Wrote balanced image: {balanced_path}")
        print(f"Wrote soft print image: {print_soft_path}")
        print(f"Wrote print image: {print_path}")
        if args.reconstruct:
            print(f"Wrote reconstructed image: {reconstructed_path}")
        if not args.no_pdf and (not batch_mode or args.per_image_pdf):
            print(f"Wrote printable PDF: {pdf_path}")

    if batch_mode and not args.no_pdf:
        merged_pdf_path = args.merged_pdf.expanduser() if args.merged_pdf else merged_pdf_output_path(output_dir)
        save_pdf_pages(merged_pdf_path, pdf_pages)
        print(f"Wrote merged printable PDF: {merged_pdf_path}")


def _resolve_input_paths(
    parser: argparse.ArgumentParser,
    positional_images: list[Path],
    input_dir: Path | None,
) -> list[Path]:
    image_paths = [path.expanduser() for path in positional_images]
    if input_dir is not None:
        directory = input_dir.expanduser()
        if not directory.exists():
            parser.error(f"input directory does not exist: {directory}")
        if not directory.is_dir():
            parser.error(f"input directory is not a directory: {directory}")
        image_paths.extend(
            sorted(
                (path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS),
                key=lambda path: path.name,
            )
        )

    if not image_paths:
        parser.error("expected at least one input image or --input-dir")

    unique_paths = list(dict.fromkeys(image_paths))
    for image_path in unique_paths:
        if not image_path.exists():
            parser.error(f"input image does not exist: {image_path}")
        if not image_path.is_file():
            parser.error(f"input image is not a file: {image_path}")
        if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            parser.error(f"unsupported image type {image_path.suffix!r}; expected one of: {supported}")
    return unique_paths
