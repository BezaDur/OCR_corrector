# Vendored: Amharic morphological segmenter suite

A frozen snapshot of the external morphological instruments, copied here so
that `amharic_ocr_corrector.py` can actually import them.

## Why

The corrector has always tried `from segmenter import AmharicMorphologicalSegmenter`,
resolved through `MORPH_PATHS`. Those paths pointed outside the repo (a machine-local
checkout), so **a fresh clone could never import the analyzer**. The import failure was
logged once at WARNING level and swallowed, which meant:

- `self.segmenter is None`
- `analyze()` returned `known=False` for every out-of-vocabulary token
- the entire `lambda_morph` scoring term silently cancelled out of the ranking

In other words the system ran as if the morphology module did not exist. Vendoring a
snapshot makes the integration reproducible and testable.

## Provenance

Copied from the Google Drive upload of the private `DesBw/segmenter` project (the repo
itself is not reachable from the build sandbox — `git clone` returns 403).

**Not vendored:** `data/main_i7.txt` (103 MB). The segmenter imports
`MAIN_DICTIONARY_PATH` but resolves it lazily; `AmharicMorphologicalSegmenter()`
instantiates in ~0 s and `segment_word()` works without it, so the snapshot is
functional for segmentation. If a code path later needs the full lexicon, copy it in
(or point `AMHARIC_MORPH_PATHS` at a live checkout).

## Modifications

Two changes, both to avoid a module-name collision. The suite ships its own
`normalizer.py`, and this repo **also** has a root-level `normalizer.py` exporting an
`AmharicNormalizer` class. Putting the suite on `sys.path` would shadow the corrector's
own normalizer and silently change normalization behaviour across the whole process.

1. `normalizer.py` → **`seg_normalizer.py`**.
2. The two importers of it, `maximal_lemma_arbitrator.py` and `segmenter.py`, updated
   to `from seg_normalizer import AmharicNormalizer`.

Nothing else was modified.

## Precedence

`_DEFAULT_MORPH_PATHS` lists the vendored copy **first**, and the loader inserts each
path at `sys.path[0]` in order — so later entries end up with higher priority. The
vendored copy is therefore the lowest-priority fallback: a live checkout on
`AMHARIC_MORPH_PATHS` or on the machine paths still wins, and the snapshot is used only
when nothing else is available. The same applies to `data/morphemes.json`, which falls
back to the vendored copy only when the *default* path is missing — an explicitly
passed `morphemes_json_path=` is honoured even if it does not exist, so
`require_morphemes_json` keeps its teeth.

## Keeping it in sync

This is a *frozen* copy, so it will drift from the live segmenter. Refresh by replacing
the `.py` files and re-applying modification (1) and (2) above. Treat any drift as a
behaviour change worth measuring, not a routine bump.

## Dependencies

`morph_types.py` requires `pydantic` (it raises a clear `ImportError` without it).
`phonology.py`, `pos_tags.py`, `seg_normalizer.py` have no third-party dependencies.
