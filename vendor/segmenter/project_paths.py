"""Canonical repository paths for code and shared linguistic data.

All paths are anchored to this file rather than the process working directory,
so both the segmenter and the standalone ``verbal_type`` tools resolve the same
source data when invoked from another directory.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"

# Shared corpus / lexical resources.
MAIN_DICTIONARY_PATH = DATA_DIR / "main_i7.txt"
TWO_CHAR_WORDS_PATH = DATA_DIR / "2_char_true_words.tsv"
MORPHEMES_PATH = DATA_DIR / "morphemes.json"

# Segmenter morphology resources.
MORPHEMES_CLUSTERS_PATH = DATA_DIR / "morphemes_clusters.json"
LESLAU_TEMPLATES_PATH = DATA_DIR / "leslau_templates.json"
LESLAU_TEMPLATES_ARCHIVE_PATH = DATA_DIR / "leslau_templates_xx.json"

# Shared gold evaluation corpora.
GOLD_VERBS_PATH = DATA_DIR / "10_column_verbs.tsv"
#: Shipped verb paradigm matrix; also licenses the verbal side of bare
#: nominal/verbal homographs (see ``morphology.is_paradigm_matrix_form``).
PARADIGM_MATRIX_PATH = GOLD_VERBS_PATH
LAYER1_VERB_GOLD_PATH = DATA_DIR / "layer1_verb_gold.tsv"
LAYER1_MACRO_GOLD_PATH = DATA_DIR / "layer1_macro_gold.tsv"

# Derived disambiguation resources (regenerate with tools/build_contingent_jussive_policy.py).
CONTINGENT_JUSSIVE_POLICY_PATH = DATA_DIR / "contingent_jussive_policy.json"
PERFECTIVE_IMPERATIVE_POLICY_PATH = DATA_DIR / "perfective_imperative_policy.json"
#: Curated Amharic experiencer (psych) verb list.
EXPERIENCER_VERBS_PATH = DATA_DIR / "experiencer_verbs.json"
