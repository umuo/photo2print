# Photo2Print

Convert photographed worksheet/document images into print-friendly electronic pages with white background and dark text.

## Install

This project is managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

## Usage

Run the CLI with one input image path:

```bash
uv run photo2print samples/input.jpg --output-dir sample_outputs
```

For an input named `input.jpg`, the command writes:

- `sample_outputs/input_balanced.png`: a cleaned grayscale page that preserves softer detail.
- `sample_outputs/input_print_soft.png`: a printable version that whitens clear background areas while keeping more of the original grayscale stroke detail.
- `sample_outputs/input_print.png`: an aggressive exam-paper style print version with near-solid white blank areas and darker foreground, tuned to avoid eating small characters and faint marks.
- `sample_outputs/input_print.pdf`: a printable A4 PDF, centered with margins and generated from the aggressive `*_print.png` variant by default.

For text-heavy worksheets, write an additional reconstruction-oriented page:

```bash
uv run photo2print samples/input.jpg --output-dir sample_outputs --reconstruct
```

This adds:

- `sample_outputs/input_reconstructed.png`: a rebuilt A4-shaped white page containing extracted dark foreground content.

Skip PDF export when you only want PNG files:

```bash
uv run photo2print samples/input.jpg --output-dir sample_outputs --no-pdf
```

Choose which processed variant is placed in the PDF:

```bash
uv run photo2print samples/input.jpg --output-dir sample_outputs --pdf-variant print-soft
```

Supported PDF variants are `print`, `print-soft`, `balanced`, and `reconstructed`.

To export a PDF from the rebuilt page:

```bash
uv run photo2print samples/input.jpg --output-dir sample_outputs --pdf-variant reconstructed
```

Process multiple images in one command by passing more than one input path:

```bash
uv run photo2print samples/page1.jpg samples/page2.jpg --output-dir sample_outputs
```

Or process all supported images in a directory, sorted by filename:

```bash
uv run photo2print --input-dir samples --output-dir sample_outputs
```

Batch mode writes the same per-image PNG outputs for every input image. By default it also writes one merged A4 PDF:

- `sample_outputs/merged_print.pdf`: all processed pages combined in input order.

To choose a different merged PDF path:

```bash
uv run photo2print --input-dir samples --output-dir sample_outputs --merged-pdf sample_outputs/worksheets.pdf
```

Per-image PDFs are optional in batch mode:

```bash
uv run photo2print samples/page1.jpg samples/page2.jpg --output-dir sample_outputs --per-image-pdf
```

The processor normalizes uneven lighting, boosts contrast, lightly deskews, and only applies perspective correction when it detects a clear page outline. The print outputs no longer use destructive opening/closing cleanup after thresholding. Instead, they combine a safely whitened background layer with a protected foreground layer derived from local text contrast, then run a region-aware cleanup pass that treats pale back-side show-through in otherwise blank paper as background contamination. The aggressive `*_print.png` path uses larger text and answer-line protection halos, removes larger low-contrast fragments in blank zones, and pushes confident blank paper to solid white. The softer `*_print_soft.png` path keeps a more conservative cleanup for pages with especially faint pencil, tiny annotations, or delicate strokes.

The reconstructed mode does not use OCR and does not retype or understand the worksheet. It detects the dominant dark foreground, removes photographed paper/background tone, crops around the detected layout, and places the extracted marks onto a synthetic white A4-shaped canvas. This is useful for text-heavy worksheets that should look more like a clean electronic sheet, but it can drop very pale pencil, colored content, light gray diagrams, or content that looks like back-side show-through.

For photographed worksheets and textbook pages, prefer `*_print.png` when you want a whiter printout and `*_print_soft.png` when the page contains especially faint pencil, tiny annotations, or delicate strokes.

## Development

```bash
uv run pytest
```
