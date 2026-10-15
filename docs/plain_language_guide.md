# A Plain-Language Guide to This Repository

**Who this is for:** someone who has never written code, never trained a machine-learning
model, and may not know what "OCR" means beyond "scanning a document." This guide explains
the entire repository from the top down — why it exists, what each piece does, and what
actually works today.

**How to read it:** Parts 1–3 give you the whole story and can be read in half an hour.
Parts 4–6 are reference material: one entry per folder, per module, per script. Read those
when you need to know what a specific file does. Part 7 tells you what runs today and what
does not. Part 8 is a glossary.

**A note on honesty.** This repository's older documentation contains numbers that do not
reproduce when you run the code. Every figure in *this* guide was produced by running the
repository's own scripts on this machine, and is marked:

- ✅ **verified** — reproduced by running the code
- ⚠️ **claimed** — stated in an older document but not reproduced
- ❌ **does not work** — the command fails, and why is stated

Where a verified number contradicts an older document, this guide gives the verified one
and says so. The full evidence trail, with the exact commands, is in
[`readme_claims_audit.md`](readme_claims_audit.md).

---

## Contents

1. [The problem, in plain language](#part-1-the-problem-in-plain-language)
2. [Two projects, and why they are separate](#part-2-two-projects-and-why-they-are-separate)
3. [The whole pipeline at a glance](#part-3-the-whole-pipeline-at-a-glance)
4. [The parent project, stage by stage](#part-4-the-parent-project-stage-by-stage)
5. [The child project, in depth](#part-5-the-child-project-in-depth)
6. [Shared plumbing: folders that support everything else](#part-6-shared-plumbing)
7. [What actually runs today](#part-7-what-actually-runs-today)
8. [Glossary](#part-8-glossary)
9. [How to check any claim in this guide yourself](#part-9-how-to-check-any-claim-yourself)

---

# Part 1: The problem, in plain language

## What OCR is

**OCR** stands for *Optical Character Recognition*. It is the software that turns a
photograph or scan of a page into actual text you can search, copy, and edit. When you
photograph a document with your phone and the phone lets you copy the words off it, that
was OCR.

OCR software looks at the dark marks on the page and asks: *which letter is this?* It works
well for English printed in a modern font on clean white paper. It works much less well
when the paper is old, the print is faint, the scan is crooked, or the writing system is
unusual.

## What goes wrong

OCR makes mistakes. The important thing to understand is that **its mistakes are not
random**. They are *visual* mistakes — the software confuses two marks that look alike to a
scanner, even though they are completely different letters to a reader.

Here is the Amharic alphabet's core problem. Amharic (the working language of Ethiopia,
written in the **Ge'ez** script, also called **Ethiopic**) does not have an alphabet of ~26
letters. It has a **syllabary**: each consonant has up to seven different written forms,
one per vowel sound. So the consonant "l" is not one letter but seven:

| order | 1st | 2nd | 4th | 6th |
|---|---|---|---|---|
| the "l" family | ለ (le) | ሉ (lu) | ላ (la) | ል (li) |
| the "r" family | ረ (re) | ሩ (ru) | ራ (ra) | ር (ri) |

That is roughly **200+ base shapes times up to 7 forms each** — around 300 distinct
characters a scanner must tell apart, many of which differ only by the length of a stroke or
the presence of a small leg. Multiply that by old paper, faded ink, and low-resolution
scans, and confusions become constant.

The single most common failure this repository was built around is called the
**"sadis-to-ge'ez shift"** — the 6th form of a consonant getting read as the 1st form, or
vice versa. `ሪ` becoming `ረ`. One small stroke, and a real word becomes a different real
word, or becomes nothing at all.

## Why this is harder than ordinary spell-checking

Your phone's keyboard has a spell-checker. It knows a list of real words, and when you type
something not on the list, it suggests the nearest real word. That approach handles most
English typos, and it handles the *easy* half of Amharic OCR errors too.

But OCR errors split into two very different kinds, and this distinction drives the entire
design of this repository:

**Kind 1 — the "non-word" error.** The scanner produces something that is not a word in the
language at all. `በሸታ` is not Amharic. A dictionary catches this instantly: *not on the
list → suspicious.* These are easy.

**Kind 2 — the "real-word" error.** The scanner produces something that *is* a perfectly
good word, just not the word that was on the page. This is the hard case, and it is why
this project exists:

- The page said `በሽታ` (*beshita*, "disease").
- The scanner read `በሸታ` — not a word. Dictionary catches it. Easy.
- But the page said `ሰራ` (*serra*, "he worked") and the scanner read `ሰረ` (*serre*) — also
  a real word. **Nothing in a dictionary can tell you this is wrong.** The only evidence is
  that `ሰረ` makes no sense in that sentence.

This is exactly what the repository's own analysis script
(`3_training_evaluation/09_analyze_error_composition.py`) was written to measure: *what
fraction of errors are non-words, and what fraction are real words?* Because the answer
determines whether a dictionary is enough, or whether you need something that reads the
sentence for meaning.

## The trap: fixing text can break text

Here is the problem that makes this project genuinely difficult, and that a casual reader
will miss.

Suppose the software is unsure whether a word is correct. It has two options: leave it
alone, or try to fix it. If it tries to fix a word that was **already correct**, it has
*damaged* good text. And damage is worse than a missed repair, because:

- A missed repair leaves the text exactly as the scanner produced it — bad, but no worse.
- A wrong repair *changes* the text into something the author never wrote, and now looks
  authoritative, because a machine "corrected" it.

So the central engineering question is not *"how many errors can we fix?"* It is
***"how do we fix errors without damaging the words that were already right?"***

This repository's own benchmark answers that question, and the answer is uncomfortable:

> ⚠️ The committed benchmark at
> `5_correction_evaluation/benchmarks/benchmark_full_symspell.json` records that running the
> full corrector made the text **worse overall** — character accuracy fell from 1.178% error
> to 1.930% error, and **2,462 of 60,596 already-correct words were damaged**.

That single fact is the reason the child project exists. Read on.

## A worked example: what happens to one word

Let us follow a single word all the way through, because this makes every later section
concrete.

A page of Amharic text is scanned. The author wrote `በሽታ` ("disease"). The scanner produced
`በሸታ`, which is not an Amharic word.

1. **The detector (child project) looks at it.** It gathers about fourteen pieces of
   evidence: *is this in the dictionary?* (no) *how common would it be if it were?* (very
   rare) *does changing exactly one character turn it into a real word?* (yes — `በሽታ`)
   *is that one-character change a confusion a scanner actually makes?* (yes — `ሸ`/`ሽ` are
   a known visual pair) *how many times does this word appear on this page?* (once) and so
   on.
2. **It produces a verdict.** One of three words: `pass` (leave it alone), `gray` (unsure —
   look at it, but be careful), `flag` (this is wrong — fix it). For `በሸታ` the current
   model says **`gray`**, with a suspicion score of **0.154** out of 1. ✅ *verified by
   running it.*
3. **The corrector (parent project) receives that verdict.** If the verdict had been `pass`,
   it would not touch the word at all. Because it is `gray` or `flag`, it generates
   candidate replacements — dictionary words one edit away, prioritising edits that match
   known scanner confusions. `በሽታ` is a candidate.
4. **A language model scores each candidate in context.** The question is not "is `በሽታ` a
   word?" but "given the words around it, which candidate makes the best sentence?" This is
   the step that can catch Kind 2 errors, and it is the step that is currently **missing
   from this repository** (see Part 7).
5. **The winner replaces the original** — but only if it beats the original by a safety
   margin. That margin exists precisely to avoid damaging correct text.

Every folder in this repository is some part of that story.

---

# Part 2: Two projects, and why they are separate

This repository contains **two projects**, and understanding the split is the single most
important thing in this guide.

## The parent project: the fixer

Folders `1_data_preparation` through `5_correction_evaluation`.

This is the whole system: train your own OCR engine, measure how bad it is, learn exactly
*which* characters it confuses with *which* others, then build a corrector that repairs the
damage. It is large, slow, and powerful.

## The child project: the triage nurse

Folder `OCR_error_detection`.

This is one small, self-contained component with one job: **look at each word and decide
whether it is worth touching.** It does not fix anything. It answers one question per word
and answers it in **under half a millisecond** — about 2,450 words per second. ✅ *verified*

## The analogy that makes this click

Think of a hospital emergency department.

- The **child project is the triage nurse.** Every person who walks in gets seen by the
  nurse first. The nurse does not perform surgery. The nurse's entire job is to sort people
  into three groups: *you're fine, go home* / *something might be wrong, wait here* /
  *this is serious, see a doctor now.* The nurse must be **fast**, because everyone passes
  through, and **careful about false alarms**, because crying wolf wastes the surgeons.

- The **parent project is the surgical team.** Expensive, slow, highly skilled. You do not
  send every person with a sniffle to surgery — that would harm them and exhaust the team.

Now recall the uncomfortable fact from Part 1: the corrector, when applied broadly,
**damaged 2,462 words that were already correct**. That is surgery on people with sniffles.

The child project exists to stop that. Its job is to say *"leave this word alone"* about
**84% of the time** ✅ *verified*, so the expensive and risky corrector only ever runs on
the words that genuinely need it.

## Why the child is a separate, self-contained folder

This was a deliberate design decision, and it has real consequences:

- The child has **its own tests** (60 of them ✅), **its own evaluation data** (100 pages),
  **its own dictionary files**, and **its own trained model**. It does not import anything
  from the parent.
- You can therefore run, test, and evaluate the child **without any of the parent's
  heavyweight dependencies** — no Tesseract, no KenLM, no external hard drives.
- The parent talks to the child through exactly one file, `OCR_error_detection/adapter.py`,
  which is a translator between the two.

**This matters practically:** in the environment this guide was written in, the child runs
perfectly and the parent does not run at all (missing external dependencies). Because they
are separated, that is not a catastrophe — the detector can still be developed, measured,
and improved on its own.

---

# Part 3: The whole pipeline at a glance

## The five stages

```
   ┌─────────────────────────────────────────────────────────────────────┐
   │  STAGE 1   1_data_preparation                                       │
   │  "Make the practice material."                                      │
   │  Take real Amharic text, print it in 20 different fonts, damage the │
   │  images to look like bad scans. Now we have pages whose correct     │
   │  text we know exactly.                                              │
   └────────────────────────────┬────────────────────────────────────────┘
                                ▼
   ┌─────────────────────────────────────────────────────────────────────┐
   │  STAGE 2   2_training                                               │
   │  "Teach the scanner."                                               │
   │  Train Tesseract (the OCR engine) on those pages for several epochs │
   │  so it learns to read Amharic better than the stock Google model.   │
   └────────────────────────────┬────────────────────────────────────────┘
                                ▼
   ┌─────────────────────────────────────────────────────────────────────┐
   │  STAGE 3   3_training_evaluation                                    │
   │  "Grade the scanner, and learn its habits."                         │
   │  Run the trained OCR on the test pages, compare to the known-correct│
   │  text, and measure the error rate. Crucially: record WHICH character│
   │  got confused with WHICH OTHER character, and how often.            │
   └────────────────────────────┬────────────────────────────────────────┘
                                ▼
   ┌─────────────────────────────────────────────────────────────────────┐
   │  STAGE 4   4_ocr_correction                                         │
   │  "Build the fixer."                                                 │
   │  Turn Stage 3's confusion records into a lookup table, then build   │
   │  the corrector: dictionary + confusion table + morphology +         │
   │  language model, working together.                                  │
   │                                                                     │
   │  ◄──── uses the CHILD as its triage nurse ────┐                     │
   └────────────────────────────┬──────────────────┼─────────────────────┘
                                ▼                  │
   ┌───────────────────────────────────────────┐   │
   │  STAGE 5   5_correction_evaluation        │   │
   │  "Did the fixer help or hurt?"            │   │
   │  Compare original OCR vs corrected text   │   │
   │  vs known-correct text. Count repairs,    │   │
   │  count damage.                            │   │
   └───────────────────────────────────────────┘   │
                                                   │
   ┌───────────────────────────────────────────────┴─────────────────────┐
   │  THE CHILD   OCR_error_detection                                    │
   │  "Is this word worth touching?"                                     │
   │  Self-contained. Own model, own dictionaries, own tests, own        │
   │  evaluation data. Answers pass / gray / flag in ~0.3 ms per word.   │
   └─────────────────────────────────────────────────────────────────────┘
```

## About the script numbers — this confuses everyone

Scripts are named with numbers: `01_generate_eval_pages.py`, `02_run_ocr_eval.py`,
`03_build_confusion_matrix.py`, … up to `15_fit_stage1_scores.py`.

**These numbers are the order the work was originally done in, not the order the folders
are arranged in.** They do not match. For example:

| Step number | Script | Lives in folder |
|---|---|---|
| 01 | `01_generate_eval_pages.py` | Stage 1 |
| 02 | `02_run_ocr_eval.py` | **Stage 3** |
| 03 | `03_build_confusion_matrix.py` | **Stage 4** |
| 04 | `04_evaluate_full_corrector.py` | **Stage 5** |
| 05 | `05_collect_error_cases.py` | **Stage 5** |
| 06 | `06_diagnose_ocr_engine.py` | **Stage 3** |
| 07 | `07_generate_training_boxes.py` | **Stage 1** |
| 08 | `08_compile_lstmf_and_split.py` | **Stage 1** |
| 09 | `09_train_epochs.sh` | **Stage 2** |
| 09 | `09_analyze_error_composition.py` | **Stage 3** (a different step 09!) |
| 10 | `10_package_best_and_fast.sh` | **Stage 2** |
| 10 | `10_compare_ocr_outputs.py` | **Stage 3** (a different step 10!) |
| 11, 11b, 12 | benchmarks and audits | **Stage 3** |
| 13, 14, 15 | decision audits | **Stage 5** |

So: **use the folder to know what a script is for, and the number only as a loose hint
about history.** Two scripts even share a number (there are two step 09s and two step 10s).

---

# Part 4: The parent project, stage by stage

For every script below you get: what it does in plain language, what it needs, what it
produces, and whether it runs today.

> **A warning that applies to most of Stage 1–3 and all of Stage 5:** these scripts were
> written on one person's computer. They read and write large data that lives outside this
> repository — on drives mounted at `/media/des` and `/mnt/mintdata`. Those drives do not
> exist on a fresh machine. The repository has a helper (`repo_paths.py`, see Part 6) that
> lets you point the scripts at your own copies via environment variables, but the data
> itself is not here. **Treat Stages 1–3 as documentation of what was done, not as things
> you can re-run today.**

---

## Stage 1 — `1_data_preparation`

**Purpose in one sentence:** manufacture practice material, because real scanned Amharic
documents with known-correct text are hard to obtain in quantity.

**The core idea.** You cannot train or measure an OCR engine without pages whose true text
you know. Real scans do not come with answer keys. So instead you *make* the pages: take
real Amharic text, print it in many fonts, then deliberately damage the images the way a
bad scanner would — uneven lighting, ink bleeding, curved spines, sensor noise. Now you
have an unlimited supply of pages where you know the right answer exactly.

This is called **synthetic data**, and it is standard practice in OCR research.

### The scripts

| Script | What it does |
|---|---|
| **`01_generate_eval_pages.py`** (360 lines) | The main document-synthesis engine. Renders real Amharic text into page images using OpenCV and Pillow, with *targeted font fallback* so a missing character never becomes a hollow box, and physically-modelled degradation: spine curvature, ink bleed, non-uniform lighting, sensor noise. |
| **`build_final_training_data.py`** (174) | Builds the master list of characters the OCR engine must learn, and extracts 150,000 clean lines of training text. Notable rule: **zero Latin letters (A–Z, a–z)**, because including them teaches the engine to confuse `T` with `ፕ`. |
| **`discover_character_inventory.py`** (108) | Scans the raw text corpus to discover which Amharic characters *actually occur* in real writing, rather than guessing from a chart. Categorises every character found. |
| **`rare_character_miner.py`** (120) | Searches a 4.9 GB corpus for every authentic sentence containing an under-represented character — the rare ones, the labiovelars (consonants with a "w" sound built in), Ethiopic numerals, and publication symbols. Rare characters get ignored by training unless you deliberately over-sample them. |
| **`assemble_balanced_training_data.py`** (110) | Combines the mined rare sentences with ordinary prose to produce a **70,000-line** training set with *balanced* character representation, and compiles the final character inventory file (`amh.unicharset`). |
| **`07_generate_training_boxes.py`** (247) | Generates the "answer key" for training. Tesseract needs, for each training image, a `.box` file saying exactly which character is at which pixel. This uses 20 curated fonts and verifies one-to-one that **zero glyphs were dropped**. |
| **`08_compile_lstmf_and_split.py`** (138) | Converts the image/box pairs into `.lstmf` tensor files (the format Tesseract's neural network actually reads), using 4 CPU cores in parallel, then splits them 90/10 into training and evaluation lists. |
| **`audit_training_text.py`** (105) | Quality check: finds characters the engine is expected to know that **never appear** in the training text (a "zero-shot blind spot" — it will never learn them), counts occurrences of critical rare characters, and checks line-length distribution. |
| **`audit_rare_glyphs.py`** (94) | Quality check: counts exact frequencies of rare characters, labiovelars, numerals, and publication symbols. |
| **`inspect_and_install_fonts.py`** (126) | Installs the fonts into the operating system, rebuilds the font cache, then **tests each font directly** to confirm the rendering system can actually find it. Catches the classic failure where a font is present but unusable. |
| **`clean_font_directories.py`** (90) | Deletes redundant, condensed, and unsupported font files, leaving strictly the 20 curated production fonts. |
| **`diagnose_font_cmaps.py`** (64) | Inspects each font's internal character map to find fonts that *lack* Arabic digits, quotes, colons, or Ethiopic punctuation. A font missing those silently produces blank or box characters. |
| **`diagnose_font_accuracy.py`** (97) | Tests Google's official Amharic model **font by font**, revealing which font families Google trained on and which ones cause it to fail. This is how the project knew a custom model was worth building. |
| **`check_pango_names.py`** (38) | Small helper: lists the names the rendering system (Pango) knows each font by, which often differ from the filename. |
| **`generate_eval_dataset.py`** (200) | An *alternative* evaluation-set builder: instead of rendering images, it corrupts text directly using the real confusion matrix, with realistic "burstiness" (errors cluster, they don't spread evenly). Faster, no images. |
| **`generate_eval_pages.py`** (176) | Older/simpler version of `01_generate_eval_pages.py`. |

### What Stage 1 leaves behind in the repository

| File | Size / extent ✅ | What it is |
|---|---|---|
| `data/amh_final_training_text.txt` | **58.2 MB**, 4,420,102 words, 359,776 lines | The main training text. This is the "4.4M tokens" the README mentions. |
| `data/mined_rare_sentences.txt` | **58.2 MB** | Output of the rare-character miner. (Coincidentally the same byte size as the file above, but *not* the same content — verified by checksum.) |
| `data/amh.training_text` | **24.9 MB** | Training text with character annotations |
| `data/amh.unicharset` | 20,163 bytes | The authoritative character inventory |
| `data/amh.lstm-unicharset` | 20,163 bytes | Variant used by the neural network |
| `data/google_raw.lstm-unicharset` | 18,906 bytes | Google's inventory, for comparison |
| `fonts.conf` | — | Font configuration |

*(The 20 actual font files live in the top-level `fonts/` folder — see Part 6.)*

---

## Stage 2 — `2_training`

**Purpose in one sentence:** actually train the OCR engine.

**What "training" means here.** Tesseract is an open-source OCR engine. Its modern version
uses a neural network called an **LSTM** (Long Short-Term Memory — a type of network good
at sequences, like a line of text). Training means showing it the images and answer keys
from Stage 1, repeatedly, and letting it adjust itself until it reads them correctly.

One pass through all the training material is called an **epoch**. This project ran five.

### The scripts

All three are **shell scripts** (`.sh`) — text files of commands for the Linux terminal,
not Python programs.

| Script | What it does |
|---|---|
| **`09_train_epochs.sh`** | Runs five sequential training epochs, saving a checkpoint after each so training can resume if interrupted. |
| **`09_run_epoch2.sh`** | A "fail-safe" runner for epoch 2 specifically, targeting sub-1% character error and convergence on two particular difficult characters (`ቯ` and `ጰ`). Exists because that epoch kept failing. |
| **`10_package_best_and_fast.sh`** | Finds the checkpoint with the lowest error rate and packages it into two finished OCR models: a **"best"** one (most accurate) and a **"fast"** one (smaller, quicker, slightly less accurate). |
| **`diagnose_job_failure.py`** (86) | A debugging helper: when a training run dies, this reads the logs and explains why. |

### What Stage 2 leaves behind

| File | Size ✅ | What it is |
|---|---|---|
| `models/best/amh.traineddata` | **8.39 MB** | The best finished model |
| `models/best/amh.lstm` | **3.36 MB** | Just the neural network part |
| `google_amh.lstm` | **437 KB** | Google's network, as the baseline to beat |
| `config/confusion_matrix.json` | **65.8 KB** | A copy of the confusion data (see the note below) |
| `config/amh.unicharset` | 20,163 bytes | A copy of the character inventory |

> ⚠️ **Note:** the repository contains **three different confusion files in three places**,
> all with different contents and different sizes ✅:
>
> | File | Size |
> |---|---|
> | `2_training/config/confusion_matrix.json` | 65,839 bytes |
> | `OCR_error_detection/Model/confusion_matrix.json` | 47,847 bytes ← **the one the detector uses** |
> | `4_ocr_correction/confusion_matrix_patch.json` | 33,843 bytes |
>
> The root README claims "single source, no duplicates." That is not accurate, and it is
> worth knowing which file a given number came from.

### Where the finished models ended up

The two packaged models are in the **top-level `models/` folder**, not here:

| File | Size | What it is |
|---|---|---|
| `models/amh_best.traineddata` | **4.87 MB** ✅ | The accurate model |
| `models/amh_fast.traineddata` | **658 KB** ✅ | The fast model |

These **are** in the repository. What is missing is the `tesseract` program itself, which is
what loads them (see Part 7).

---

## Stage 3 — `3_training_evaluation`

**Purpose in one sentence:** grade the trained scanner, and — more importantly — learn
exactly *which* characters it confuses with *which* others.

This is the stage that produces the repository's most valuable single artifact: the
**confusion matrix**.

### What a confusion matrix is

Imagine a giant table. Down one side, every character. Across the top, every character. In
each cell, a number: *how many times did the scanner read the row-character as the
column-character?*

Almost every cell is zero. But a few cells have big numbers, and those cells are **gold**.
They say: "when you see `ሸ` in a scan, there is a strong chance the page actually said
`ሽ`."

That transforms correction from guessing into evidence. Instead of "what dictionary word is
one edit away?", you can ask "what dictionary word is one edit away **using an edit the
scanner is known to make**?" That is a vastly shorter, vastly more reliable list.

### The scripts

| Script | What it does |
|---|---|
| **`02_run_ocr_eval.py`** (129) | Runs the trained Tesseract over all the synthetic evaluation pages, using multiple CPU processes at once for speed. Produces the raw OCR text that everything else grades. |
| **`03_build_confusion_matrix.py`** (244) | *(lives in Stage 4's folder, but is conceptually Stage 3's output)* Aligns the OCR text against the known-correct text character by character, computes the error rates, and **exports the optical substitution costs to `confusion_matrix.json`**. This is the single most important script in the pipeline. |
| **`09_analyze_error_composition.py`** (317) | The strategic analysis. Answers the two questions from Part 1: **what fraction of errors produce a non-word** (easy to catch with a dictionary) versus **a real word** (invisible to a dictionary), and **what fraction of errors are even reachable** by a one-edit correction. Its own docstring says it plainly: if the second number is low, *"no amount of detection tuning will help — the generator cannot reach the answer."* |
| **`06_diagnose_ocr_engine.py`** (149) | Audits the raw OCR output *before* any correction, to pinpoint three specific failure modes: wrong spacing (words split or fused), the sadis/ge'ez order erosion, and dropped abbreviation punctuation. |
| **`10_compare_ocr_outputs.py`** (137) | Compares several OCR outputs against ground truth. Needs no model, no GPU, no images — text files only, so it runs anywhere. Its docstring contains an important warning: the pipeline is only ever evaluated against *one* OCR output, and if that output came from a bad model, every conclusion of the form "the corrector makes things worse" really means "we measured against bad input." |
| **`11_benchmark_new_model.py`** (137) | Full benchmark of the newly trained model over 150 stress-tested pages. Re-runs the OCR. |
| **`11b_benchmark_existing_ocr.py`** (286) | The *scoring half* of `11`, with the OCR step removed. Use when the OCR text already exists and re-recognising would cost hours for no new information. Written to answer a specific contradiction its author hit: *"step 11 told me 98% but step 04 reports 34.39% CER. Which is real?"* |
| **`12_audit_raw_ocr_sentences.py`** (114) | Robust word-level alignment and failure-case extraction. Uses global dynamic-programming alignment so that line wraps and blank lines **never** cause the comparison to slip out of step — a subtle bug that silently corrupts results. |

### The measurement modules (shared by Stages 3, 4 and 5)

These are libraries, not runnable scripts — other scripts import them.

| Module | What it provides |
|---|---|
| **`ocr_metrics.py`** (256) | The canonical definitions of the error metrics, independent of any language model. Notably careful: a position is only counted as "clean" if it matches in *every* equally-good alignment, because when two alignments tie, claiming a word was correct is a guess. Ambiguous positions are excluded and counted separately, not silently assigned. |
| **`tsv_gt.py`** (103) | Adds ground truth to Tesseract's per-word data file. Tesseract emits a **TSV** (tab-separated values — a simple text table) with one row per word and a confidence score. This adds three columns: the comparable core of the word, the true word it matched, and a status of `clean` / `error` / `ambiguous`. |
| **`tsv_confidence.py`** (84) | Extracts per-word confidence scores from those TSV files and aligns them to text lines. Careful about Tesseract's file structure (levels 1–4 are containers with no confidence; only level 5 rows carry words) and refuses to guess when counts disagree — the line simply gets no confidence, and the mismatch is counted. |
| **`tsv_gt_stats.py`** (59) | Cross-tabulates confidence against truth: how many *correct* words had low confidence, how many *wrong* words had high confidence. This is what tells you whether Tesseract's own confidence can be trusted — and the answer, measured below, is "only partly." |
| **`query_raw_characters.py`** | Audits the original raw dataset for exact counts and origins of six specific rare characters (`ቌ ኈ ኊ ኌ ጔ ፼`). |

### What Stage 3 discovered — the confidence finding

One of the most practically useful results in the repository, ✅ **verified** from the
child project's 100 evaluation pages:

| | Mean Tesseract confidence |
|---|---|
| Words that were **correct** | **87.0** out of 100 |
| Words that were **wrong** | **60.8** out of 100 |

Useful, but not enough:

- **18.9% of the errors scored above 85** — confidently wrong.
- **7.4% of the correct words scored below 70** — uncertainly right.

**This is why the child project's model is "conf-free."** An earlier design fed Tesseract's
confidence into the detector. It was removed, because a signal that is wrong 18.9% of the
time on the cases you most need to catch is not a foundation to build on. The trained model
still has a slot for it, and that slot has **exactly zero influence** ✅ *verified: gain
0.0, used in 0 of the model's decisions.*

---

## Stage 4 — `4_ocr_correction`

**Purpose in one sentence:** the fixer itself.

### `amharic_ocr_corrector.py` — 1,817 lines, 87 KB ✅

This is the largest and most important file in the parent project. (⚠️ The Stage 4 README
calls it "80K lines" — it is 87 KB on disk and **1,817 lines**. The README is wrong.)

Its architecture is five components working in sequence, each doing what the previous one
cannot:

**1. `EthiopicPunctuationManager` — get punctuation out of the way.**
Amharic text runs words together with a punctuation mark `፡` between them, and words often
arrive with punctuation attached: `ነው።`. Before you can look anything up in a dictionary you
must separate `ነው` from `።`. This class strips leading and trailing punctuation while
deliberately **preserving internal abbreviation markers** (`/` and `.`), which are part of
the word, not decoration.

> ⚠️ **A live bug found in this class.** It strips punctuation but **not digits** — digits
> are not in its punctuation set. So a token like `ከ20` reaches the detector as `ከ20`. That
> single fact caused most of the detector's false alarms; see Part 5 for the full story and
> the fix.

**2. `AmharicNormalizer` — collapse spelling variants.**
Ge'ez script has historical spelling variants that are pronounced identically: `ሐ` and `ሀ`
both sound like "h"; `ሠ` and `ሰ` both sound like "s"; `ዐ` and `አ` both sound like "ʔa". A
dictionary indexed on one form will not find the other. This normaliser maps every variant
to a single canonical form for *lookup*, while keeping a reverse map so the output can be
written back in the spelling the author actually used.

**3. SymSpell + reverse optical mapping — generate candidates.**
**SymSpell** is a well-known fast spell-checking algorithm. Its trick: instead of comparing
your misspelling against every dictionary word (slow), it pre-computes "deletions" of every
dictionary word at index time, so lookup is nearly instant. Here it is combined with the
confusion matrix, so candidate generation is driven by *edits the scanner is known to
make* rather than arbitrary edits.

**4. `AmharicMorphologicalSegmenter` — handle words not in the dictionary.**
Amharic is **agglutinative**: it builds long words by gluing prefixes and suffixes onto a
root. `በመሄድ` ("by going") is one word made of several pieces. Such a word may not appear in
any dictionary, yet be perfectly correct. This component breaks a word into its
morphological pieces; if it decomposes cleanly into real pieces, that is evidence the word
is legitimate even though the whole form is not listed.

> ⚠️ **This component silently does not work in a fresh clone.** It depends on a Python
> library called `pydantic`, which is **not listed in `requirements.txt`**. The import fails,
> the failure is logged once as a warning and swallowed, and the system carries on with
> `segmenter = None` — which makes the entire morphology term in the scoring formula cancel
> out to zero. The repository ran as if this module did not exist. The segmenter code itself
> has now been copied into `vendor/segmenter/` (Part 6) so the import can succeed, but
> `pydantic` is still not listed as a dependency. ❌
>
> To be fair to `requirements.txt`: it *does* document KenLM properly, explaining that it is
> not on PyPI, must be installed separately, and that a model file must be supplied. It lists
> numpy, joblib, lightgbm, scikit-learn, symspellpy, pandas, opencv-python, Pillow and
> fonttools. `pydantic` is simply absent — a one-line omission with a large consequence.

**5. KenLM 5-gram Viterbi beam decoder — use context.**
This is the component that can catch **Kind 2 errors** (real-word errors), because it is the
only one that reads the *sentence* rather than the word.

- A **5-gram language model** is a statistical model trained on a large body of Amharic text
  that knows, for any four words, how likely various fifth words are. It encodes what
  Amharic *sounds like*.
- **Viterbi beam decoding** means: rather than deciding each word in isolation, it searches
  for the best *sequence* of word choices across the sentence, keeping the top few
  possibilities ("the beam") alive at each step instead of committing early.
- It includes a **protective anchor barrier** — a deliberate penalty structure that makes it
  hard to replace a word unless the replacement is clearly better in context. This is the
  anti-damage mechanism.

> ❌ **This component cannot run at all in a fresh clone.** KenLM is a separate C++ library
> with Python bindings. It is not installed, and **no language model file exists anywhere in
> this repository** (searched for `*.klm`, `*.binary`, `*.arpa`, `*.lm` — zero results). The
> corrector therefore refuses to start:
> `ValueError: AmharicOCRCorrector needs a KenLM language model`. This is why both "How to
> use" examples in the root README fail.

### The other Stage 4 files

| File | What it does |
|---|---|
| **`03_build_confusion_matrix.py`** (244) | Described in Stage 3. Builds the confusion matrix from OCR-vs-truth alignment. |
| **`confusion_matrix_coverage_audit.py`** (132) | Quality check on the confusion matrix. Audits coverage of the "same consonant, different order" confusions (`ር`/`ሮ`, `ሁ`/`ሆ`, `ለ`/`ሎ`) — the single most common class of Amharic OCR error — and reports which families are missing. |
| **`confusion_matrix_patch.json`** | Manual additions to the confusion matrix. |
| **`normalizer.py`** (311) | The parent's normaliser. Larger than the child's: it also handles punctuation, numerals, and abbreviations. |
| **`amharic_canonical_normalizer.py`** (83) | Standardises punctuation *equivalences* so that neither the metrics nor the corrector penalises a keyboard approximation (for instance treating several quote styles as the same thing). |
| **`suspicion.py`** (424) | **The old, rule-based triage nurse — deprecated.** Its own docstring is blunt about why it was replaced: *"The corrector currently generates candidates and runs the LM over EVERY token. At an 8% base rate that is ~92% wasted work, and worse, it gives the system ~92% opportunities to damage text that was already right."* Kept only as a fallback for when the machine-learning model cannot load. |
| **`trace_candidate.py`** (334) | A forensic debugging tool. Give it an observed OCR token and the word you expected, and it traces *every* step: was a candidate generated? was it filtered out? by which gate? with what cost? Indispensable when a correction that should have happened did not. |
| **`test_corrector.py`** (51) | Smoke test for the corrector. |
| **`test_page1_fixes.py`** (99) | Spot-checks the corrector on five known failure patterns from page 1. Its docstring contains a valuable practical warning: the full multi-million-word dictionary costs about **1.03 KB per entry** to load into SymSpell, so loading it uncapped needs **~6 GB** and will be killed on most machines. It also reports the measured trade-off: the top 300,000 entries cover 92.4% of running text, the top 500,000 cover 96.9%, and **65% of dictionary entries occur exactly once** — and in an OCR-mined dictionary most of those singletons *are themselves OCR errors*. So capping the dictionary usually improves accuracy as well as saving memory. |

---

## Stage 5 — `5_correction_evaluation`

**Purpose in one sentence:** did the fixer actually help?

This is the stage that answers the question the whole project hinges on, and it is the stage
whose result is least reflected in the repository's promotional documents.

### The scripts

| Script | What it does |
|---|---|
| **`04_evaluate_full_corrector.py`** (420) | The end-to-end benchmark. Runs the complete corrector over the evaluation pages and measures the result. Can also emit a per-word verdict log (`--verdict-dir`) recording what the detector thought about each token — the input for scripts 14 and 15. |
| **`05_collect_error_cases.py`** (133) | Extracts the failure cases. Needs no KenLM and no morphology — it works purely from alignment, so it runs even when the full corrector cannot. |
| **`13_audit_correction_decisions.py`** (170) | The most important script in this stage. Compares three versions of every word — **ground truth**, **raw OCR**, **corrected output** — and sorts every decision the corrector made into four categories: **TRUE_POSITIVE** (a legitimate repair), **FALSE_POSITIVE** (damage to a word that was already correct), **FALSE_NEGATIVE** (an error left unrepaired), **SUBOPTIMAL_FLIP** (changed, but to the wrong word). Exports a side-by-side table and prints the most common failure patterns. |
| **`14_stage1_precision.py`** (64) | Measures the detector's precision: of the words it flagged, how many were really errors? Sweeps thresholds and reports the lowest one that reaches a target precision. |
| **`15_fit_stage1_scores.py`** (84) | Asks whether a *learned* score beats the *hand-tuned* one. Reports how well each individual signal separates errors from clean words (0.5 = useless, 1.0 = perfect), then trains a simple model on all pages but one and tests it on the held-out page — so no page ever scores itself. This script is essentially the research that led to the child project's LightGBM model. |

### The shared libraries

| Module | What it provides |
|---|---|
| **`stage1_labels.py`** (81) | Labels every detector verdict as `clean`, `error`, or `ambiguous` against ground truth. The `ambiguous` category matters: when two alignments tie, a token may have matched in *some* optimal alignments and not others. Those are excluded rather than guessed at. |
| **`stage1_fit.py`** (85) | Fits a scoring model in **pure Python with no extra packages** — including a hand-written logistic regression by gradient descent, and leave-one-page-out cross-validation. Written so the analysis can run in a bare environment. |
| **`smoke_eval.sh`** | Quick end-to-end sanity check. |

### The result — read this carefully

`benchmarks/benchmark_full_symspell.json` is a committed, finished benchmark. ✅ **verified
by reading the file.** Running the full corrector over the evaluation set:

| Measure | Raw OCR | After correction | Change |
|---|---|---|---|
| Character error rate (strict) | 1.178% | 1.930% | **64% worse** |
| Word error rate | 4.674% | 6.293% | **35% worse** |
| Character error reduction | — | **−63.8%** | negative = harm |
| Already-correct words damaged | — | **2,462 of 60,596** | 4.1% of good text |

Under a more forgiving comparison that ignores punctuation and spelling variants:

| Measure | Raw OCR | After correction |
|---|---|---|
| Character error rate (canonical) | 1.059% | 1.140% |
| Character error reduction | — | **−7.68%** |

**In plain terms: on this data, the corrector currently makes the text worse.** It damages
more good words than it repairs. Even on the most forgiving measurement it is net-negative.

> ⚠️ **No README in this repository says this.** It is recorded in a JSON file in a
> `benchmarks/` subfolder that no top-level document points to.

This is not a reason to abandon the project — it is exactly the finding that tells you where
to work. It says: *detection is not yet selective enough, and correction is not yet careful
enough.* Which is precisely the problem the child project was built to solve, and why the
rest of this guide pays so much attention to it.

Two important caveats, both from the repository's own scripts:

1. `10_compare_ocr_outputs.py` warns that if the OCR output being corrected came from a weak
   model, "the corrector is net-negative" may really mean "we measured against bad input."
2. The corrector cannot currently run at all without KenLM (Part 7), so this benchmark
   reflects a specific earlier configuration.

---

# Part 5: The child project, in depth

Folder: `OCR_error_detection`. This is the part of the repository that **fully works today**
and the part this guide can describe most precisely, because every number below was measured
by running it.

## What it does

Given one word (and optionally, how often that word appears on its page), it returns:

- a **verdict**: `pass`, `gray`, or `flag`
- a **suspicion score**: a number from 0 to 1, where 1 means "certainly wrong"
- the **evidence** it used, so a human can check its reasoning

It runs in about **0.4 milliseconds per word** ✅ (roughly 2,450 words per second) and takes
about **3 seconds to start up** ✅ (it loads a 54,057-word dictionary and a trained model).
Running the full 100-page evaluation, which also does the alignment work and reads the files
from disk, averages ~1.4 ms per word and takes about 85 seconds.

## The three verdicts

| Verdict | Meaning | What the corrector does with it | Share of clean words ✅ | Share of real errors ✅ |
|---|---|---|---|---|
| **`pass`** | Confident this word is fine | Leave it completely alone | **84.1%** | 7.1% (these are misses) |
| **`gray`** | Unsure | Investigate, but correct only with a wide safety margin | 15.8% | **90.2%** |
| **`flag`** | Confident this word is wrong | Correct it | **0.10%** | 2.6% |

Two things to notice:

- The verdicts are **lowercase** in the code. Several documents write `"FLAG"` in capitals
  and then compare against it — that comparison never matches. ⚠️
- **Almost every real error lands in `gray`, not `flag`.** Only 2.6% of errors reach `flag`.
  This is a known limitation, analysed at the end of this Part.

## How it decides: the fourteen clues

The detector is a **LightGBM model** — a type of machine learning called *gradient-boosted
decision trees*. The intuition: it is a committee of 529 small flowcharts ✅, each of which
asks simple yes/no questions about the word ("Is it in the dictionary?" "Is it longer than
five characters?"). Each flowchart is trained to fix the mistakes the previous ones made.
Their votes are summed into a single suspicion score.

The committee is only allowed to look at **fourteen clues**, and the design rule is that
every clue must be computable at both training time and run time *in exactly the same way*.
That rule exists to prevent a specific and common disaster, which this repository actually
suffered from — see "The bug" below.

| # | Clue | Plain-language question | Importance ✅ |
|---|---|---|---|
| 1 | `log_corpus_freq` | How common is this word in 4.4 million words of real Amharic? (Rarer = more suspicious.) | **67.6%** |
| 2 | `in_lexicon` | Is this word in the dictionary at all? | **12.7%** |
| 3 | `token_len` | How long is the word? | 6.3% |
| 4 | `doc_freq` | How many times does it appear on *this* page? | 2.8% |
| 5 | `is_doc_singleton` | Does it appear exactly once on this page? (A one-off odd word is more suspicious than a repeated one.) | 2.0% |
| 6 | `rare_glyph_ratio` | What fraction of its characters come from the rare parts of the syllabary? | 1.9% |
| 7 | `one_flip_count` | How many real words can you make by changing exactly one character? | 1.9% |
| 8 | `is_high_freq` | Is it a very common word? (A shortcut flag.) | 1.6% |
| 9 | `has_one_flip` | Is there *at least one* real word one character away? | 1.5% |
| 10 | `has_digit` | Does it contain a numeral? | 1.1% |
| 11 | `len_ratio` | How long is it compared to the average Amharic word (4.14 characters)? | 0.6% |
| 12 | `char_kenlm_score` | How plausible is this character sequence? | **0.0%** — unused |
| 13 | `n_neighbors` | How many nearby dictionary words? | **0.0%** — unused |
| 14 | `mixed_script` | Does it mix Ge'ez with Latin or other scripts? | **0.0%** — unused |

**Read that last column carefully.** Three of the fourteen clues have **zero influence** —
the model never uses them. Two of them (`char_kenlm_score`, `n_neighbors`) require
components that are not present in this repository. This is honest and documented: the model
is described as "conf-free" and "KenLM OFF," and the zero importance figures ✅ confirm that
description is accurate.

The first two clues — how common the word is, and whether it is in the dictionary — together
account for **80% of the model's decisions**. This is why a bug in dictionary loading is
catastrophic rather than cosmetic, and why one of the fixes described below makes a missing
dictionary file a hard error instead of a silent downgrade.

## The modules

| File | Lines | What it does |
|---|---|---|
| **`detector.py`** | 654 | The front door. `DetectionEngine` loads the dictionary, the model and the confusion matrix, computes the fourteen clues for a word, gets the committee's vote, converts it into a calibrated probability, and applies the two thresholds to produce `pass`/`gray`/`flag`. |
| **`features.py`** | 324 | **The single definition of the fourteen clues**, shared by training and by running. This file is the guarantee against train/serve skew — the same code computes the clues in both places, so they cannot drift apart. Contains a long comment block explaining the rules each clue follows. |
| **`signals.py`** | 214 | The individual checks, and the two data structures that carry results: `TokenSignals` (all the clues about one word) and `Verdict` (the answer returned to the caller). |
| **`scoring.py`** | 51 | The *old* hand-weighted way of combining clues into one number. Kept as the fallback used when the machine-learning model cannot load. |
| **`confusion.py`** | 86 | Loads `confusion_matrix.json` and builds the **reverse substitution map**: given a character the scanner produced, which characters might the page actually have had? Also provides the cost of each insertion, deletion and substitution. |
| **`normalizer.py`** | 90 | The child's copy of the spelling-variant collapser (`ሐ`→`ሀ`, `ሠ`→`ሰ`, `ዐ`→`አ`). Deliberately duplicated so the child stays self-contained. |
| **`utils.py`** | 82 | Character sets and confusion classes. Contains an important warning: there are **two** lists of "rare glyphs", a wide one for the rule-based fallback and a narrow one the model was actually trained on. They must not be merged — adding the `ሸ` series to the model's list was measured to **multiply the false-alarm rate by 13** (0.027% → 0.353%). |
| **`alignment.py`** | 72 | The word-alignment engine. Given two sequences of words, finds the best way to line them up, and — importantly — tracks *every* equally-good alignment, so it can tell you which positions are certain and which are genuinely ambiguous. |
| **`adapter.py`** | 265 | **The only connection between the two projects.** Translates between the parent corrector's interface and the child's, and lets the child *reuse the dictionary the parent already loaded* instead of loading a second copy (which would double memory use). If anything goes wrong it falls back to the rule-based scorer and records why. |
| **`run_ocr.py`** | 153 | Runs Tesseract over the 100 evaluation images to produce the OCR text and per-word confidence files. One-time setup step; needs Tesseract installed. |
| **`__init__.py`** | 42 | Decides what `from OCR_error_detection import ...` gives you. Exports `DetectionEngine`, `CorrectorDetectionAdapter`, `TokenSignals`, `Verdict`, `ConfusionMatrix`, `AmharicNormalizer`, and the scoring helpers. |

## The data folder — `Model/`

| File | Size | What it is |
|---|---|---|
| **`detector_lgbm_conffree.pkl`** | 3.66 MB ✅ | **The trained detector.** A `.pkl` (pickle) is Python's format for saving a finished object to disk. Contains the 529-tree committee, the calibration curve, the thresholds, and notes about how it was trained. |
| **`main_i7_frq10.txt`** | 15 MB, 737,106 lines ✅ | The main dictionary: one word per line with how often it occurs in a large corpus. "frq10" means words occurring fewer than 10 times were already removed. |
| **`local_lexicon.txt`** | 750,870 lines ✅ | An alternative dictionary, built from a smaller corpus (max frequency 60,761 vs 5.6 million). |
| **`confusion_matrix.json`** | 47.8 KB ✅ | The scanner's known confusion habits, distilled to **304 reverse-substitution entries**. |
| **`harmonized_corpus_jiwer.tsv`** | 48 MB, **946,057 rows across 1,367 pages** ✅ | The training data. One row per word, with the page it came from, how often it occurs on that page, and whether it was an error. Contains **61,081 real errors** (6.46%). Built with a library called `jiwer`, which does the alignment. |
| **`train_LightGBM.py`** | 662 | **The trainer.** Builds the committee from the corpus above. |
| **`train_final.py`** | — | A convenience wrapper: "train the exact recipe the README calls authoritative," in one command. |
| **`train_sweep_report.json`** | — | A saved report from a comparison run (explained below). |

> ⚠️ `OCR_error_detection/main_i7_frq10.txt` (in the child's root, not in `Model/`) is a
> **byte-for-byte duplicate** of the one in `Model/` — 15 MB stored twice. Only one fallback
> code path references it.

## How the model is trained — `train_LightGBM.py`

Worth understanding, because it explains what the reported numbers mean.

1. **Load the corpus** — 946,057 words from 1,367 pages.
2. **Hold out 100 pages completely.** Pages 1–100 are set aside and never trained on. These
   are the same pages in `Evaluation/`. Everything reported about them is therefore an
   honest measurement on data the model never saw.
3. **Group the rest by page.** This matters. If you shuffle individual *words* into training
   and testing piles, words from the same page land on both sides — and words on the same
   page resemble each other, so the model looks better than it is. Grouping by page prevents
   that leakage.
4. **Five-fold cross-validation.** Split the 1,267 remaining pages into five piles. Train
   five times, each time leaving out a different pile and predicting it. Every word gets
   predicted by a model that never saw it.
5. **Calibrate.** The committee's raw vote is not a real probability. An **isotonic
   calibrator** learns a curve converting raw votes into honest probabilities, fitted on
   those out-of-fold predictions.
6. **Train the final model** on all 1,267 pages.
7. **Measure on the 100 held-out pages** and save everything.

It can also run in **sweep mode** (`--sweep-freqs`), which trains several dictionary
configurations and prints a comparison table — but that mode **does not save a model**, and
it measures on validation pages drawn from the *training* distribution, not on the held-out
100. That distinction turns out to matter a great deal (see below).

## The evaluation data — `Evaluation/`

| Folder / file | Contents |
|---|---|
| `GroundTruth/` | **100 files**, `page_NNNN.gt.txt` — the correct text of each page |
| `OCRed/` | **200 files** — for each page, `page_NNNN.ocr.txt` (what the scanner read) and `page_NNNN.conf.tsv` (its confidence per word) |
| `images/` | **100 files**, `page_NNNN.png` — the page images themselves |
| `analysis.tsv` | **62,521 rows** — a committed per-word analysis |

## Measured performance ✅

Running `scripts/evaluate_child.py` over all 100 pages:

```
Pages 100, Ground-truth words 62,427, OCR words 62,479, Aligned 62,385
Word error rate 6.84% (4,267 substitutions), Character error rate 1.90%

ERROR words (4,267):   FLAG 113 (2.6%)   GRAY 3,849 (90.2%)   PASS 305 (7.1%)
                       → 92.9% of errors were noticed

CLEAN words (58,118):  PASS 48,896 (84.1%)   GRAY 9,163   FLAG 59
                       → 0.10% of correct words were falsely accused
```

So, in plain language, per 1,000 words on a typical page:

- About **68 words are scanner errors.** The detector notices about **63** of them and
  misses about 5.
- About **932 words are correct.** The detector leaves **784** completely alone, marks
  **147** as "worth a careful look," and wrongly accuses about **1**.

⚠️ Older documents claim a false-accusation rate of **0.01%** (1 in 10,000) and error
detection of **94.2%**. Neither reproduces. The verified figures are **0.10%** and
**92.9%**. The false-accusation figure improved tenfold during the work described in the
next section — before the fix it was 0.72%.

## The bug that was found and fixed

This is worth understanding even if you read nothing else technical, because it explains how
a system can look fine and be badly broken.

**The clue in question:** `has_digit` — "does this word contain a numeral?"

**What went wrong.** Amharic has its own numerals: `፩ ፪ ፫ ፬ ፭ ፮ ፯ ፰ ፱`. These behave
differently from `0 1 2 3` in a very obscure way:

- The cleaning step that prepares a word for dictionary lookup **removes `0`–`9`** but
  **keeps `፩`–`፱`**. (A technicality of how the two kinds of numeral are classified in
  Unicode.)
- `has_digit` was the *one* clue computed from the word **before** cleaning, on the reasoning
  that cleaning removes digits so the cleaned word "can't answer the question."

Now look at what was in the training data. ✅ **verified:**

- **Zero** of the 946,057 training words contained an Arabic digit or any punctuation.
- Exactly **812** of them (0.086%) contained an *Ethiopic* numeral — words like `መሆ፭ም` and
  `፬ዎቹ`, where the numeral is genuinely part of the word.

So the committee learned "contains a numeral" from **one tiny, peculiar group of 812 real
Amharic words**, and gave that clue enough weight to use it in 164 decisions.

Then, in production, something completely different arrived. Recall from Stage 4 that the
punctuation manager strips punctuation but **not digits**. So real OCR output like `ከ20`
("from 20") or `በ5ሺህ` ("in 5 thousand") reached the detector with its digits intact. The
cleaning removed the `2` and `0` from every *other* clue — but `has_digit` stayed switched
on. The word fell into the branch the committee had learned from rare Ethiopic-numeral
vocabulary, and came back screaming:

| word shown to the detector | only differing clue | score for the clean form | score as actually sent |
|---|---|---|---|
| `ከ20` vs `ከ` | `has_digit` off→on | 0.012 → **pass** | 0.828 → **flag** |
| `በ5ሺህ` vs `በሺህ` | `has_digit` off→on | 0.015 → **pass** | 0.805 → **flag** |
| `“3ኛውን` vs `ኛውን` | `has_digit` off→on | 0.360 → gray | 1.000 → **flag** |

**420 of the 62,479 evaluation tokens contained an Arabic digit, and they accounted for
essentially all of the detector's false alarms.**

**The fix:** compute `has_digit` from the cleaned word, like every other clue. This is
provably identical on the training data — because cleaning changed *nothing* in any of the
946,057 training words, both definitions flag exactly the same 812 rows. So **no retraining
was needed and no saved model changed behaviour on any input it had ever seen.**

**The result:**

| | before fix | after fix |
|---|---|---|
| False alarms on clean words | 419 (0.72%) | **59 (0.10%)** |
| Errors noticed | 92.9% | 92.9% (unchanged) |

**And a stronger check.** The repository has two independent ways of measuring the same 100
pages, and before the fix they disagreed by a factor of 3 to 9 *on the same model* — a
warning sign that something was inconsistent. After the fix they agree to three decimal
places:

| model configuration | measurement method A | method B before fix | method B after fix |
|---|---|---|---|
| local dictionary, freq ≥ 10 | 0.0809% | 0.730% | **0.081%** ✓ |
| local dictionary, freq ≥ 3 | 0.4818% | 0.762% | **0.482%** ✓ |
| main dictionary, freq ≥ 1000 | 0.0705% | 0.692% | **0.071%** ✓ |
| main dictionary, freq ≥ 500 | 0.2684% | 0.919% | **0.268%** ✓ |

**One more thing worth noting.** The child's test suite contained a line
`RAW_ONLY_DIFF = {"has_digit"}` — the parity test had been *told* to excuse that one clue
from the "training and running must match" check. The test didn't fail to catch the bug; it
was configured to permit it. That line is now empty, so all fourteen clues are checked.

## The other child fixes

| Fix | Why it mattered |
|---|---|
| **A missing dictionary file is now a hard error** | If you passed a dictionary path that did not exist, the detector silently fell back to an **eight-word** dictionary and carried on producing confident-looking verdicts. Since two clues carry **80%** of the model's decisions, every answer was meaningless — and nothing warned you. A common cause is a typo: the child README documents `main_i7_freq10.txt`, but the real file is `main_i7_frq10.txt`. The error message now names the real file. |
| **Gates can be retuned without rewriting the model** | The two thresholds were stored inside the model file and always overrode anything the caller passed, so there was no way to try a different operating point. New opt-in switch `gates_override_model=True`. It is opt-in because the parent passes its own values (0.30/0.70) through the adapter, and honouring those unconditionally would silently change the parent's behaviour. |
| **The trainer no longer mislabels its own output** | The note saved inside each model said "main_i7" even when the model had been trained from `local_lexicon.txt`. |
| **The evaluation script can test any model** | `evaluate_child.py` always used the shipped model, so no alternative could be measured with the documented tool. New `--model`, `--lexicon`, `--min-freq`. |

## Which dictionary should the detector use? (answered)

The README claimed the shipped configuration was decisively best, with "7× better false
alarms than the local dictionary." That claim was tested properly: four candidate models
were trained with the documented recipe, and **all five** (four new plus the shipped one)
were run through one identical harness on the held-out 100 pages. ✅

| configuration | dictionary size | errors noticed | false alarms | overall score |
|---|---|---|---|---|
| **shipped model** (main, freq ≥ 500) | 54,057 | **92.85%** | 0.102% | **0.9276** |
| local dictionary, freq ≥ 10 | 52,494 | 92.24% | **0.081%** | 0.9217 |
| main dictionary, freq ≥ 1000 | 32,402 | 92.03% | **0.071%** | 0.9197 |
| main dictionary, freq ≥ 500 (retrained) | 54,057 | 91.00% | 0.268% | 0.9076 |
| local dictionary, freq ≥ 3 | 162,893 | 91.07% | 0.482% | 0.9063 |

**Conclusions:**

1. **The shipped model wins. Do not switch dictionaries.** The in-distribution comparison
   that ranked it 4th of 7 does not survive contact with genuinely held-out pages.
2. **Dictionary choice barely matters.** All five span 1.85 points of error detection, and
   two alternatives have *lower* false-alarm rates than the shipped model. The "7× better"
   claim was noise between two estimates made on training-like data.
3. **The shipped model cannot be regenerated from the documented recipe.** Retraining the
   identical configuration produces a *worse* model (0.9076 vs 0.9276), because the shipped
   file predates a change to how missing values are handled. This is a reproducibility
   problem, recorded but not fixed.

## Why almost nothing reaches `flag` — and what was done about it

The three-way verdict is supposed to be its main feature, but only **2.6% of real errors**
reach `flag`; ~90% sit in `gray`. ⚠️ A committed analysis file records **60.4%** of errors
reaching `flag` on these same pages, so behaviour has also drifted substantially at some
point.

A dedicated tool (`scripts/tune_gates.py`) was written to find better thresholds. Three
things it had to solve:

- **The obvious ranking metric is unusable.** The README ranks configurations by
  `score = recall × (1 − false-flag rate)`. But `recall` counts `gray` as "caught" while the
  false-flag term counts only `flag`. So setting the pass threshold to zero — passing
  *nothing*, sending every word in the document to the corrector — scores a perfect 100%
  recall at no measured cost. The metric rewards the degenerate answer.
- **The two thresholds must be chosen separately,** each against its own explicit budget,
  never combined into one number.
- **Choosing thresholds on the same pages you then report is cheating.** So thresholds are
  selected on four-fifths of the pages and measured on the fifth, five times over.

**The root constraint:** the calibration step produces only **59 distinct probability
values** across 62,385 words, and **only 0.276% of all words ever score above 0.65**. The
threshold is not badly placed — the scores simply do not reach it.

**Recommendation ✅:**

- **Keep the pass threshold at 0.10.** Lowering it to 0.05 would catch 98 more errors but
  send **1,087 more correct words** to the corrector, and that extra band is only **8.3%
  errors**. Given that correction is currently net-harmful (Stage 5), spending it on a
  92%-clean band is the wrong trade. Cross-validation agreed: the same threshold was chosen
  in **5 of 5** folds.
- **Raise the flag threshold from 0.65 to 0.85.** This is close to free: **109 vs 113** true
  errors flagged, against **31 vs 59** correct words damaged — a 47% reduction in damage for
  a 3.5% reduction in coverage. Precision rises from 65.7% to 77.9%. Chosen in 4 of 5 folds.
  Note the optimum is model-specific: the two alternative models are *already* optimal at
  0.65, so this must not be hard-coded.
- **Threshold tuning cannot rescue the three-way design.** At any threshold with usable
  precision, `flag` covers only 2.3–4.5% of errors across all models; reaching 8% requires
  dropping to where precision is ~50% — a coin flip, damaging as many words as it fixes. The
  honest options are to accept a small high-precision `flag` band, collapse to two verdicts,
  or retrain the calibration. That is a product decision, and it is deliberately left open.

## The child's scripts

| Script | Lines | What it does |
|---|---|---|
| **`evaluate_child.py`** | 250 | **The main measurement.** Runs the detector over all 100 held-out pages and reports word/character error rates, how many errors were noticed, how many clean words were falsely accused, and the full pass/gray/flag breakdown. Can act as a quality gate: `--min-recall 90 --max-false-flag 1` makes it exit with failure if the detector regresses. Now also accepts `--model`, `--lexicon`, `--min-freq`. |
| **`compare_lexicon_configs.py`** | 261 | Runs several model-and-dictionary configurations through one identical harness and prints a ranked comparison table. This produced the dictionary table above. |
| **`tune_gates.py`** | 545 | Chooses the two thresholds defensibly. Collects every word's probability once, then sweeps thresholds exactly, reports the full trade-off curves, and cross-validates the *selection* over pages. |
| **`audit_decision.py`** | 172 | Compares ground truth, raw OCR and corrected text to categorise each decision. Deliberately conservative: it only assigns a category where the alignment is unambiguous in all comparisons, and leaves other rows visible rather than forcing them into a bucket. |
| **`test_child.py`** | 40 | Runs the child's whole test suite from any directory. Exits non-zero if anything fails. |

## The child's tests

`tests/test_detector.py` (494 lines) and `tests/test_features.py` (267 lines). **60 tests,
all passing** ✅.

They are designed to run **without** KenLM, without the external morphological segmenter,
and without the production dictionary — using tiny on-disk dictionaries and the real
confusion matrix. That is why they work in a bare environment.

The most important group is `TestTrainServeParity`: it builds the same word's clue vector
via the *training* path and via the *running* path and asserts they are identical, clue by
clue. That is the guard rail against the class of bug described above — and, as noted, it
had been configured to excuse exactly the clue that broke.

## The child's archived experiments — `archive/`

Nine older scripts kept for reference: `train_improved.py`, `train_v3.py`, `train_v4.py`,
`test_conf_free.py`, `evaluate_detection.py`, `evaluate_v2.py`, `analyze_methods.py`,
`check_alignment.py`, `demo_improvement.py`. These are the project's history — earlier
attempts at the detector. **Nothing imports them and they are not part of the working
system.** Useful for understanding why decisions were made; do not run them expecting
current behaviour.

---

# Part 6: Shared plumbing

## `repo_paths.py` (183 lines) — the single most useful file for a newcomer

**The problem it solves.** Every stage script originally had the author's hard-drive paths
typed directly into it. Nothing could run on any other computer, and failures were cryptic
(`PermissionError: /media/des`).

**What it does.** Every large data location is now read from an **environment variable**
(a setting you give the operating system) with the original path as the default. The
original machine keeps working unchanged; any other machine sets only the variables it
actually has data for.

| Variable | Default location | Contains |
|---|---|---|
| `AMHARIC_DATA_ROOT` | `/mnt/mintdata/Users/Des/Mint` | Language models, dictionaries, raw text corpora |
| `TESSERACT_WORKSPACE` | `/media/des/Tess/Tesseract2` | Synthetic evaluation pages, training output |
| `TESSERACT_TRAINING` | `/media/des/Tess/Tesseract/TessTraining` | Fonts, language data |
| `AMHARIC_GOOGLE_BEST` | `/media/des/Tess/Tesseract/Google_best` | Google's reference model |
| `AMHARIC_WORD_KENLM` | *(under DATA_ROOT)* | The 5-gram word language model |
| `AMHARIC_CHAR_KENLM` | *(under DATA_ROOT)* | The character language model |
| `AMHARIC_MAIN_I7_DICT` | *(under DATA_ROOT)* | The multi-million-word dictionary |
| `AMHARIC_RAW_DATASET` | *(under DATA_ROOT)* | The 4.9 GB raw text corpus |

Running `python repo_paths.py` prints where everything resolved to — the first thing to do
when a script cannot find its data. It also provides `require()`, which produces a clear
"the font directory is missing at /path" instead of an obscure crash.

## The root-level wrapper files

You will notice the repository root contains files with the same names as files inside the
stage folders: `amharic_ocr_corrector.py`, `normalizer.py`, `suspicion.py`, `ocr_metrics.py`,
`stage1_fit.py`, `stage1_labels.py`, `tsv_confidence.py`, `tsv_gt.py`, `tsv_gt_stats.py`,
`generate_eval_dataset.py`, `generate_eval_pages.py`, `amharic_canonical_normalizer.py`,
`ocr_decision_audit.py`.

**These are all tiny compatibility shims, 10–20 lines each.** The real code moved into the
stage folders; these files exist so that old commands like
`from amharic_ocr_corrector import AmharicOCRCorrector` keep working. Each one loads the real
module from its new home and installs it under the old name.

**If you are reading the code, ignore these and go to the real file inside the stage
folder.**

## `vendor/segmenter/` — the copied-in morphology library

Thirteen Python files, about **12,100 lines**, including `morphology.py` (2,700),
`segmenter.py` (1,999), `pos_tagger.py` (1,934), `affix_peeler.py` (1,362),
`maximal_lemma_arbitrator.py` (1,180), and others, plus a `data/` folder and two README files
(`VENDORING.md`, `DATA_README.md`).

**Why it exists**, in its own words: the corrector always tried to import a morphological
analyser from paths pointing *outside the repository* — a checkout on the author's machine.
So a fresh clone could never import it. The failure was logged once as a warning and
swallowed, meaning `segmenter = None`, every out-of-vocabulary word was reported as unknown,
and the morphology term silently cancelled out of the ranking. **The system ran as if the
module did not exist.**

Vendoring (copying a dependency into your own repository rather than fetching it) makes the
integration reproducible and testable. Its `VENDORING.md` documents the provenance and notes
what was *not* copied: `data/main_i7.txt`, a 103 MB dictionary.

⚠️ Copying the code in does not by itself fix the problem — the segmenter needs `pydantic`,
which is still missing from `requirements.txt`. ❌

## `fonts/` — 20 curated Amharic fonts ✅

`AbyssinicaSIL-R.ttf`, `Brana-Regular.ttf`, `EthiopicAbay-Regular.ttf`,
`EthiopicLessan-Regular.ttf`, `EthiopicSadiss-Regular.ttf`, `FreeSerif.ttf`, the
`NotoSansEthiopic` family in several weights, and others. These are the fonts Stage 1
renders with, deliberately reduced to exactly 20 verified families by
`clean_font_directories.py`.

## `models/` — the finished OCR engines ✅

`amh_best.traineddata` (4.87 MB) and `amh_fast.traineddata` (658 KB). A `.traineddata` file
is a complete, ready-to-use Tesseract language model. These are the output of Stage 2 and
are present in the repository.

## `tests/` — the parent's test suite

Fourteen files, **159 tests, all passing, 24 skipped** ✅ (the skips are tests needing
dependencies absent here).

| File | Lines | What it covers |
|---|---|---|
| `test_head_contract.py` | 987 | The largest: the contract between the corrector's decoding stages |
| `test_ocr_corrector.py` | 330 | The corrector's core behaviour |
| `test_detection.py` | 189 | Detection integration |
| `test_principles.py` | 185 | Design invariants the whole system must respect |
| `test_decision_audit.py` | 175 | The decision-categorisation logic |
| `test_char_lm.py` | 141 | Character language model handling |
| `test_tsv_confidence.py` | 126 | Confidence extraction from Tesseract output |
| `test_evaluator.py` / `test_evaluator_limit.py` | 123 / 108 | The evaluation harness |
| `test_phase_a.py` | 101 | Phase-A behaviour |
| plus five more | | punctuation, normalisation, alignment, metrics, labels |

## `docs/` — the existing documentation

| File | What it is |
|---|---|
| `readme_claims_audit.md` | **The verification report behind this guide.** Every claim in the older documents, tested, with reproduction commands. Read this if you want to check a number. |
| `architecture.md`, `project_structure.md`, `components_guide.md`, `repo_layout.md` | Descriptive documentation. ⚠️ Contains the unreproduced figures this guide corrects. |
| `corrector_usage.md` | How to use the corrector. |
| `decision_audit.md` | The decision-categorisation methodology. |
| `segmenter_integration.md` | How the morphology library is wired in, including memory guidance. |
| `lgbm_detector_review.md`, `lgbm_step1_changes.md`, `lgbm_step2_gate.md` | Reviews of the detector's development. |
| `lgbm_review/` | **Fourteen analysis scripts** written during those reviews — synthetic set builders, honest cross-validation, label checks, normalisation-bug probes, dictionary coverage checks. These are how the detector's claims were independently examined. |
| `architecture_review.md`, `ocr_corrector_review.md`, `principles_review.md` | Critical reviews of the design. |
| `two_phase_benchmark.md`, `new_test_set_plan.md` | Benchmarking methodology and plans. |

## `archive/` — the parent's history

`detection_old/`, `eval_data_old/`, `models/`, `models_old/`, `reports/`, plus a file named
`amharic_ocr_corrector (copy).py`. Old versions kept for reference. **Not part of the
working system.**

Note `archive/models/detector_lgbm_conffree_main_i7_freq500.pkl` is **byte-identical** to
the shipped `Model/detector_lgbm_conffree.pkl` ✅ — a useful confirmation of which
configuration was shipped.

## Loose files at the root

| File | What it is |
|---|---|
| `telegram_c4.txt` | **40 MB** of real Amharic text, scraped from Telegram channels and the C4 corpus. Raw source material. Referenced only by `docs/lgbm_review/build_synthetic_source.py` and `docs/new_test_set_plan.md`. |
| `requirements.txt` | The dependency list. ⚠️ **Incomplete — missing `pydantic`.** |
| `README.md` | The main entry point. ⚠️ Contains unreproduced figures and two examples that fail. |
| `.gitignore` | Tells git which files not to track (virtual environments, caches, etc.). |
| `augraphy_cache/` | Cached sample images from **Augraphy**, a library that simulates document degradation. |
| `font_diag_temp/` | Scratch output from font diagnostics. |

---

# Part 7: What actually runs today

This is the most practically important part of the guide. Everything below was tested.

## ✅ Works today, in a bare environment

Install six ordinary Python packages and these all run:

```bash
python3 -m venv .venv
.venv/bin/pip install numpy joblib lightgbm scikit-learn symspellpy==6.10.0 pandas pytest
```

| What | Command | Result |
|---|---|---|
| The whole child detector | `from OCR_error_detection import DetectionEngine` | Loads in ~2–3 s, answers in ~0.3 ms/word |
| Child tests | `python -m pytest OCR_error_detection/tests/ -q` | **60 passed** |
| Parent tests | `python -m unittest discover -s tests -p 'test_*.py'` | **159 passed, 24 skipped** |
| Held-out evaluation | `python OCR_error_detection/scripts/evaluate_child.py` | 92.9% recall, 0.10% false alarms, ~85 s |
| Configuration comparison | `python OCR_error_detection/scripts/compare_lexicon_configs.py` | Ranked table |
| Threshold tuning | `python OCR_error_detection/scripts/tune_gates.py` | Trade-off curves + cross-validated recommendation |
| Retraining the detector | `python OCR_error_detection/Model/train_LightGBM.py --sweep-freqs 10,100,500,1000` | Reproduces the committed report **to the digit** ✅ |
| Path diagnostics | `python repo_paths.py` | Prints resolved locations |

**The child project is fully self-sufficient.** This is the payoff of the separation
described in Part 2.

## ❌ Does not work, and exactly why

| What fails | Error | Root cause |
|---|---|---|
| Both "How to use" examples in the root README that involve the corrector | `ValueError: AmharicOCRCorrector needs a KenLM language model` | **No language model file exists anywhere in the repository.** Searched for `*.klm`, `*.binary`, `*.arpa`, `*.lm` — zero results. The 40 MB `telegram_c4.txt` is raw text, not a model. |
| Any morphological analysis | silently skipped | `pydantic` missing from `requirements.txt` → the vendored segmenter cannot import → `segmenter = None` → the morphology term cancels out |
| Stage 1 scripts | `PermissionError: [Errno 13] Permission denied: '/media/des'` | External drives not mounted; and no `tesseract` binary |
| Stage 2 shell scripts | fail immediately | Need the Tesseract training toolchain and the external workspace |
| Stage 3 scripts | fail | Need the `tesseract` binary — **not installed** |
| Stage 5 (`04_evaluate_full_corrector.py`) | fails | Needs KenLM *and* `/media/des` |
| `run_ocr.py` | fails | Needs the `tesseract` binary |

Three of these are fixable by the user in minutes if the data exists elsewhere:

1. **Add `pydantic` to `requirements.txt`.** One line.
2. **Point the scripts at your own data** with the environment variables from `repo_paths.py`.
3. **Install `tesseract`** (`apt install tesseract-ocr`) and, for the corrector, **build
   KenLM** and supply a trained Amharic 5-gram model.

The KenLM model is the hard one: it is not in the repository and cannot be reconstructed
from what is here. Training one requires the `kenlm` toolchain and a large Amharic corpus —
`telegram_c4.txt` is a plausible starting point, but that is a project of its own.

## ⚠️ Runs, but its numbers should not be trusted

| Document | Problem |
|---|---|
| `README.md` and `OCR_error_detection/README.md` | The headline "94.2% recall, 0.01% false alarms" does not reproduce. Verified: **92.9% and 0.10%**. |
| Both "How to use" examples | The child example prints `gray 0.154`, not the documented `FLAG 0.85`. The parent example raises an exception. |
| `Evaluation/analysis.tsv` | A third, different set of numbers for the same 100 pages (60.4% flag rate). No current model reproduces it. Should be regenerated or retired. |
| `docs/components_guide.md` | Shows a worked example whose printed clue vector produces `gray 0.31`, not the documented `flag 0.85`. Also shows dictionary frequencies that do not match the real file. |
| `4_ocr_correction/README.md` | Calls the corrector "80K lines"; it is **1,817 lines** (87 KB). |
| `OCR_error_detection/README.md` | Documents a dictionary filename that does not exist (`main_i7_freq10.txt` for `main_i7_frq10.txt`), and says "top 150k (freq>=3)" where the real count is 162,893. |
| The shipped model file itself | Its internal note reads `"...BEST FF 0.01%"` — the unreproduced claim is baked into the artifact. |

## The three sets of numbers for the same 100 pages

Worth seeing side by side, because it shows why verification mattered:

| Source | Errors noticed (pass+gray) | Errors reaching `flag` | False alarms |
|---|---|---|---|
| `README.md` | ⚠️ 94.2% | not stated | ⚠️ 0.01% |
| committed `Evaluation/analysis.tsv` | ⚠️ 95.64% | ⚠️ 60.37% | ⚠️ 1.194% |
| **running the code today** | ✅ **92.9%** | ✅ **2.6%** | ✅ **0.10%** |

Three documents, three incompatible answers, one set of pages.

---

# Part 8: Glossary

**Agglutinative** — a language that builds long words by gluing prefixes and suffixes onto a
root. Amharic is one; so are Turkish and Finnish. Makes dictionary lookup harder, because
the number of possible word forms is huge.

**Alignment** — lining up two sequences of words to see which corresponds to which. Needed
to compare scanner output against known-correct text. Subtler than it sounds: when the
scanner splits one word into two, or fuses two into one, a naive line-by-line comparison
silently slips out of step and every result after that point is wrong.

**AUC** (Area Under the ROC Curve) — a single-number summary of how well a score separates
two groups, from 0.5 (no better than a coin flip) to 1.0 (perfect). Unlike accuracy it does
not depend on where you set the threshold. The detector's is **0.92** ✅.

**Calibration** — converting a model's raw output into an honest probability. A model might
output 0.7 for a word and be right only 40% of the time; calibration learns the correction.
**Isotonic calibration** learns a flexible staircase-shaped curve from data rather than
assuming a mathematical form. Side effect: it produces only a limited set of distinct output
values (here, 59), which is why some thresholds behave identically.

**CER / WER** — Character Error Rate and Word Error Rate. The standard OCR measures. CER
counts wrong characters as a fraction of all characters; WER counts wrong words. CER is
always lower than WER, because one wrong character makes a whole word wrong. Measured here:
**CER 1.90%, WER 6.84%** ✅.

**Confusion matrix** — the table of "the scanner read character X when the page said
character Y, and it did so N times." The most valuable artifact in this repository, because
it turns correction from guessing into evidence.

**Cross-validation** — measuring honestly by holding data back. Split into piles, train on
some, test on the rest, rotate. **Leave-one-page-out** holds back a whole page at a time,
which matters here because words on the same page resemble each other.

**Epoch** — one complete pass through the training material. This project ran five.

**Gradient-boosted decision trees (LightGBM)** — a machine-learning method that builds many
small flowcharts in sequence, each one trained to correct the errors of all the previous
ones, then sums their votes. Fast, robust, and works well on tabular clue data like the
fourteen clues. LightGBM is Microsoft's implementation.

**Ground truth** — the known-correct answer. Everything in this repository is measured
against it. Obtaining it is why Stage 1 manufactures synthetic pages: real scans have no
answer key.

**KenLM** — a fast library for querying large statistical language models. **Not present in
this repository**, and the corrector cannot start without it.

**Lexicon** — a dictionary: a list of real words with how often each occurs. "Frequency
≥ 500" means only words appearing at least 500 times in the source corpus are included,
which filters out junk. Measured: the main dictionary has **737,106 lines**, of which
**54,057** survive a frequency ≥ 500 filter ✅.

**LSTM** — Long Short-Term Memory, a neural network design for sequences. Used inside
Tesseract to read lines of text.

**`.lstmf`** — the compiled tensor format Tesseract's LSTM reads during training. Produced
by `08_compile_lstmf_and_split.py`.

**Morphology / morpheme** — the structure of words. A **morpheme** is the smallest
meaningful piece: `በመሄድ` ("by going") is three morphemes. Morphological analysis breaks a
word into its pieces to judge whether it is legitimate even if the whole form is not in any
dictionary.

**One-flip** — changing exactly one character. "Does this word have a one-flip to a real
word?" is one of the strongest clues the detector has, because most OCR errors are single
character confusions.

**Pickle (`.pkl`)** — Python's format for saving a finished object to disk and loading it
back later. The trained detector is one file.

**Precision** — of the things you acted on, how many deserved it. If the detector flags 100
words and 78 were really errors, precision is 78%. The complement — 22 wrongly accused — is
what the false-alarm rate measures. Chosen here as the key criterion for the flag threshold,
because a low-precision flag sends correct words into a corrector that damages them.

**Recall** — of the things you should have acted on, how many did you catch. If there were
1,000 errors and you noticed 929, recall is 92.9%.

**SymSpell** — a fast spell-checking algorithm. Pre-computes deletions of every dictionary
word at index time so lookup is nearly instant instead of comparing against the whole
dictionary. Costs roughly **1 KB of memory per dictionary entry**, which is why loading a
multi-million-word dictionary needs ~6 GB.

**Synthetic data** — data you manufacture rather than collect. Here: real Amharic text
rendered in 20 fonts and deliberately degraded. The advantage is that you know the correct
answer exactly, which is what makes measurement possible.

**Tesseract** — the open-source OCR engine, originally from HP, now maintained by Google.
Reads the page images and produces text. `.traineddata` files are its language models; two
are included in `models/`. **The `tesseract` program itself is not installed here.**

**Train/serve skew** — when the code that prepares data for *training* differs from the code
that prepares data at *run time*. The model then sees inputs in production that it never saw
in training, and behaves unpredictably. This is exactly the bug described in Part 5, and it
is why `features.py` exists as one shared definition.

**TSV** — Tab-Separated Values. A plain-text table where columns are separated by tab
characters. Tesseract emits one per page, with a row per word and its confidence.

**Unicharset** — Tesseract's file listing every character the model must recognise, with
properties of each. Built in Stage 1.

**Unicode** — the standard assigning a unique number to every character in every writing
system. Ethiopic occupies `U+1200`–`U+137F`. The `has_digit` bug was ultimately a fact about
Unicode: Ethiopic numerals are classified differently from Arabic ones, so a regular
expression that strips "digits" strips one kind and not the other.

**Viterbi beam decoding** — searching for the best *sequence* of choices rather than the
best choice at each step, while keeping only the top few partial sequences ("the beam")
alive to stay tractable. This is what lets the corrector use sentence context.

---

# Part 9: How to check any claim yourself

Everything in this guide is reproducible. The pattern is always the same: create an isolated
Python environment, install six packages, run a script, read the output.

```bash
cd OCR_corrector

# 1. Create an isolated environment (does not touch your system Python)
python3 -m venv .venv
.venv/bin/pip install numpy joblib lightgbm scikit-learn symspellpy==6.10.0 pandas pytest

# 2. Does the detector work at all?
.venv/bin/python -c "
from OCR_error_detection import DetectionEngine
d = DetectionEngine()
r = d.verdict('በሸታ', doc_freq=2)
print(r.label, round(r.suspicion, 4))"
# verified output:  gray 0.1537
# (the README claims "FLAG 0.85" — it does not reproduce)

# 3. Run the child's tests
.venv/bin/python -m pytest OCR_error_detection/tests/ -q          # 60 passed

# 4. Measure the detector on the 100 held-out pages (~85 seconds)
.venv/bin/python OCR_error_detection/scripts/evaluate_child.py

# 5. Reproduce the dictionary comparison table the README prints
.venv/bin/python OCR_error_detection/Model/train_LightGBM.py \
    --sweep-freqs 10,100,500,1000 --report /tmp/sweep.json
# then compare /tmp/sweep.json against OCR_error_detection/Model/train_sweep_report.json
# — they match to the digit

# 6. Compare configurations on held-out pages instead
.venv/bin/python OCR_error_detection/scripts/compare_lexicon_configs.py

# 7. See the threshold trade-offs and the cross-validated recommendation
.venv/bin/python OCR_error_detection/scripts/tune_gates.py

# 8. Confirm what the corrector cannot do
.venv/bin/python -c "
from importlib.util import spec_from_file_location, module_from_spec
s = spec_from_file_location('m', '4_ocr_correction/amharic_ocr_corrector.py')
m = module_from_spec(s); s.loader.exec_module(m)
print(m.AmharicOCRCorrector().correct_text('ሰላም በሸታ'))"
# verified output: ValueError: AmharicOCRCorrector needs a KenLM language model

# 9. See where every script expects its data to be
.venv/bin/python repo_paths.py
```

For the full audit — every claim tested, with the exact command and the exact output — see
[`readme_claims_audit.md`](readme_claims_audit.md).

---

## If you remember only five things

1. **There are two projects.** The parent fixes text; the child decides which words are
   worth fixing. The child exists because fixing everything damages more than it repairs —
   a committed benchmark records **2,462 correct words damaged**.

2. **The child works; the parent does not run.** The child is self-contained and fully
   measurable today. The parent needs a KenLM language model that **is not in this
   repository** and a `pydantic` dependency that **is not in `requirements.txt`**.

3. **The child's real numbers are 92.9% of errors noticed and 0.10% of correct words
   falsely accused** — not the 94.2% and 0.01% the README claims.

4. **A real bug was found and fixed.** One of fourteen clues was computed differently at
   training time than at run time. It caused **~86% of all false alarms**, and fixing it
   required no retraining because the fix was provably identical on all 946,057 training
   words.

5. **Don't switch dictionaries.** Properly tested on held-out pages, the shipped
   configuration wins. The claimed 7× advantage over the alternative was noise.
