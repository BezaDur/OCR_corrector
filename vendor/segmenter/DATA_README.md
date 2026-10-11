# Shared data

This directory is the single source of input data shared by the main segmenter
and the standalone tools in `verbal_type/`. Paths are declared in the repository
root's `project_paths.py`; do not keep subsystem-local copies of these files.

| File | Purpose |
|---|---|
| `main_i7.txt` | Word-frequency dictionary |
| `2_char_true_words.tsv` | Two-character word registry, consumed as flat raw text (any non-empty 2-character field licenses that word; column meaning is not assumed) |
| `morphemes.json` | Declarative morpheme inventories and lexical exceptions |
| `morphemes_clusters.json` | Segmenter macro-affix clusters |
| `leslau_templates.json` | Segmenter verbal templates |
| `leslau_templates_xx.json` | Archived/alternate template data |
| `10_column_verbs.tsv` | Shared source verb-paradigm evaluation corpus |
| `layer1_verb_gold.tsv` | Generated row-aware gold; repeated forms within one paradigm row retain the union of all column labels |

`10_column_verbs.tsv` is also the **paradigm matrix** used by the dual licensor:
its column headers name the reading (`Imperative`, `Simple Perfect`, `Coverb`, …)
that a form licensed in that cell can carry, mapped to canonical tags in
`morphology._PARADIGM_COLUMN_TAGS`.

Regenerate `layer1_verb_gold.tsv` after changing the source matrix by running
`python generate_layer1_gold.py` from the repository root. The generator keeps
identical spellings from different paradigm rows separate. When updating a shared
input, update this canonical file. Runtime defaults are anchored to the repository
location, so commands do not depend on the current working directory.
