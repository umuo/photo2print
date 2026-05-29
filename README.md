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

The processor normalizes uneven lighting, boosts contrast, lightly deskews, and only applies perspective correction when it detects a clear page outline. The print outputs no longer use destructive opening/closing cleanup after thresholding. Instead, they combine a safely whitened background layer with a protected foreground layer derived from local text contrast, then run a region-aware cleanup pass that treats pale back-side show-through in otherwise blank paper as background contamination. The aggressive `*_print.png` path uses larger text and answer-line protection halos, removes larger low-contrast fragments in blank zones, and pushes confident blank paper to solid white. The softer `*_print_soft.png` path keeps a more conservative cleanup for pages with especially faint pencil, tiny annotations, or delicate strokes.

For photographed worksheets and textbook pages, prefer `*_print.png` when you want a whiter printout and `*_print_soft.png` when the page contains especially faint pencil, tiny annotations, or delicate strokes.

## Development

```bash
uv run pytest
```
