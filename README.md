# disentangling-trasyanka

Word-level language identification for **Trasyanka**, the mixed Belarusian–Russian
spoken variety. Models are trained on standard Russian and Belarusian text and tested
on transcribed Trasyanka speech, labelling every word as `BE`, `RU`, `SHARED` or `HYBRID`.

## Data (`data/raw/`)

| Dataset | Use | Labels |
|---|---|---|
| UD Russian-SynTagRus (`ru_syntagrus-ud-*.conllu`) | training | pseudo-labels |
| UD Belarusian-HSE (`be_hse-ud-*.conllu`) | training | pseudo-labels |
| Oldenburg Trasyanka corpus (`fam_*.cha`, CHAT format) | test | gold, per word |

All three are converted to one shared Latin spelling, the one used by the Trasyanka
transcripts (e.g. `гаварыў` → `havaryú`), so training and test words look alike.

### Getting the data

`data/raw/` is not tracked in git. Put these files directly in `data/raw/` (no subfolders
needed for the treebanks):

1. **UD Russian-SynTagRus**: from [UD_Russian-SynTagRus](https://github.com/UniversalDependencies/UD_Russian-SynTagRus),
   copy `ru_syntagrus-ud-train-a.conllu`, `-train-b.conllu`, `-train-c.conllu` and `ru_syntagrus-ud-test.conllu`.
2. **UD Belarusian-HSE**: from [UD_Belarusian-HSE](https://github.com/UniversalDependencies/UD_Belarusian-HSE),
   copy `be_hse-ud-train.conllu` and `be_hse-ud-test.conllu`.
3. **Oldenburg corpus of Belarusian-Russian mixed speech** (OC-BRMS / OK-WRGR;
   Hentschel, Zeller & Tesch, University of Oldenburg, BIS-Verlag, ISBN 978-3-8142-2305-6):
   the [project page](https://uol.de/en/slavistik/forschung/sprachwissenschaft/bereich-mischvarietaeten/oc-brms)
   links to the download and the documentation. The code expects the CHAT version: copy all
   `fam_*.cha` files (444 transcripts, 7 families) into `data/raw/`.

The file names must match exactly; `src/utils/conllu_parser.py` lists the treebank paths.

## Code (`src/`)

| File | What it does |
|---|---|
| `utils/translit.py` | Cyrillic → shared Latin scheme; cleans the transcription marks in `.cha` words |
| `utils/conllu_parser.py` | Reads the treebanks, builds Latin vocabularies, assigns pseudo-labels |
| `utils/cha_parser.py` | Reads the `.cha` transcripts: gold labels, phonetic labels, POS and lemmas from `%mor` |
| `utils/build_processed.py` | Writes `data/processed/*.jsonl` and `stats.json`, the input for all models |
| `utils/metrics.py` | Macro-F1, per-label report, confusion matrix |
| `baseline/cues.py` | Latin spelling cues and word endings for BE vs RU, chosen from the treebanks |
| `baseline/splits.py` | Training settings, Oldenburg family split, evaluation sets, scoring and `results/*.json` |
| `baseline/ngram_heuristic.py` | Rule-based baseline: cues, endings and vocabulary lookups |
| `baseline/train_crf.py`, `crf_features.py` | CRF baseline (sklearn-crfsuite) |
| `model/data.py` | Character vocabulary, padded batches and masks for the BiLSTM (PyTorch) |
| `model/bilstm.py` | Hierarchical BiLSTM: character BiLSTM per word, word BiLSTM per sentence |

Models train on RU train + official-orthography BE train (RU sampled to the same size) and are
evaluated on Trasyanka, the RU test split, the official BE test split and the Taraškievica sentences.

| Macro-F1 | Trasyanka | RU test | BE official test | Taraškievica |
|---|---|---|---|---|
| Heuristic | 0.3672 | 0.9215 | 0.5503 | 0.5597 |
| CRF | 0.4478 | 0.9997* | 0.5958 | 0.5992 |

\* Circular: the CRF's vocabulary features reproduce the rule that made the RU pseudo-labels.

### Oldenburg family split

The Oldenburg corpus is split by family, never by utterance, so the same speakers and
conversations never appear in both training and test:

| Part | Families | Words | HYBRID |
|---|---|---|---|
| train | `fam_ak`, `fam_ch`, `fam_mi`, `fam_sm` | 121,213 | 2,544 |
| dev | `fam_sh` | 23,069 | 396 |
| test | `fam_ba`, `fam_ra` | 56,475 | 1,205 |

Training settings: **zero-shot** (treebanks only), **in-domain** (Oldenburg train families) and
**mixed** (both), plus a learning curve over 10 / 30 / 70 % of the train-family transcripts.
Every setting is compared on `trasyanka_test`. The Checkpoint 1 table above scores all of Trasyanka.

## Requirements

Python 3.13. The versions are pinned in `requirements.txt`; PyTorch is the CPU build:

```
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

## Running

From the project root:

```
python src/utils/conllu_parser.py     # builds data/dictionaries/ and shows sample labels
python src/utils/cha_parser.py        # parses one transcript as a demo
python src/utils/build_processed.py   # writes data/processed/ (about 1 minute)
python src/baseline/ngram_heuristic.py  # writes results/heuristic.json
python src/baseline/train_crf.py        # writes results/crf.json (about 2 minutes)
```
