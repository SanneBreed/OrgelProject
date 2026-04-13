# marcussen

Minimal Python project for parsing and comparing Marcussen organ recordings from `.flac` filenames.

## Install `uv` (one-time)

macOS/Linux (bash/zsh):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows (PowerShell):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Verify:

```bash
uv --version
```

## Setup (from `pyproject.toml`)

macOS/Linux (bash/zsh):

```bash
uv venv .venv
source .venv/bin/activate
uv sync --extra librosa
```

Windows (PowerShell):

```powershell
uv venv .venv
.\.venv\Scripts\Activate.ps1
uv sync --extra librosa
```

This installs dependencies directly from `pyproject.toml` (including `fadtk`) plus the `librosa` extra used for audio loading and low-level audio processing in comparisons.

`fadtk` is available at [microsoft/fadtk](https://github.com/microsoft/fadtk?tab=readme-ov-file) and provides a machine-learning-based metric between two audio fragments.

## Dataset root

Set a default dataset location with `MARCUSSEN_DATASET_ROOT`.

macOS/Linux (bash/zsh):

```bash
export MARCUSSEN_DATASET_ROOT="Marcussen 1945-1975_FLAC_Dataset"
```

Windows (PowerShell):

```powershell
$env:MARCUSSEN_DATASET_ROOT = "Marcussen 1945-1975_FLAC_Dataset"
```

You can always override with `--root` per command.

## Commands

Parse all files and write `filepath + group key` columns:

```bash
marcussen parse --root "Marcussen 1945-1975_FLAC_Dataset" --out outputs/parsed_group_keys.csv
```

Run within-group pairwise comparisons (streamed row-by-row to CSV):

```bash
marcussen compare --root "Marcussen 1945-1975_FLAC_Dataset" --out outputs/pairs.csv --metric placeholder
```

Run a short debug compare and stop after the first 20 pairs:

```bash
marcussen compare --root "Marcussen 1945-1975_FLAC_Dataset" --out outputs/pairs_debug_20.csv --metric placeholder --max-pairs 20
```

Prepare the listening-experiment dataset. This exports processed toot clips as WAV files, writes cross-organ pairs from `toot_1`, and appends same-organ `toot_1` vs `toot_2` control pairs at the end of `pairs.csv`:

```bash
marcussen prepare_listening_dataset --root "Marcussen 1945-1975_FLAC_Dataset" --out-dir outputs/listening_experiment_pairs --trim --normalize
```

Export a centered steady-state window after toot detection and trimming:

```bash
marcussen prepare_listening_dataset --root "Marcussen 1945-1975_FLAC_Dataset" --out-dir outputs/listening_experiment_pairs_steady --trim --steady-state-seconds 1.5 --normalize
```

Restrict any command to `CLOSE` microphone recordings only by adding `--close-only`:

```bash
marcussen prepare_listening_dataset --root "Marcussen 1945-1975_FLAC_Dataset" --out-dir outputs/listening_experiment_pairs_close --close-only --trim --normalize
```

Useful `prepare_listening_dataset` options:

- `--trim` removes leading and trailing silence from each exported clip while preserving internal gaps.
- `--steady-state-seconds N` extracts a centered active window of `N` seconds after trimming.
- `--normalize` peak-normalizes each exported clip to `-1 dBFS`.
- `--csv-name NAME.csv` changes the output CSV filename inside `--out-dir`.
- `--max-pairs N` caps the total number of written rows for debugging.
- `--debug-first-50-per-batch` writes only the first 50 rows from each listening batch and prunes unreferenced WAV files.

## Audio Prep

`prepare_listening_dataset` uses the audio-prep pipeline in `src/marcussen/audio_prep.py` to transform each source recording before export.

- Source files are loaded from FLAC and written back out as WAV.
- When toot expansion is enabled, the prep pipeline detects up to three toot regions in each source recording.
- The listening-dataset export currently keeps `toot_1` for cross-organ comparisons and uses `toot_1` plus `toot_2` to build same-organ control pairs.
- Exported filenames include the toot index and any enabled processing steps.
- The output CSV records the exported WAV paths, source toot indices, batch, and `processing_chain` for each pair.

## Notes

- Filename parsing is tolerant: it collects warnings and unknown tokens instead of failing.
- Default comparison classes are keyed by `family`, `registration_raw`, `division`, `pitch`, and `mic_location`.
- Comparisons are run within those classes, and only across different `organ_id` values.
- `librosa` is required in this project for audio loading and low-level audio processing.
- `prepare_listening_dataset` uses reusable audio-prep functions to detect toots, trim clips, extract steady-state windows, normalize audio, and write WAV exports.
- The listening-dataset CSV groups exported pairs by batch, records which toot index from each source file was used, and appends same-organ control pairs at the end.
