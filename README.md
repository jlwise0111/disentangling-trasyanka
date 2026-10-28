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

## Code (`src/`)

| File | What it does |
|---|---|
| `utils/translit.py` | Cyrillic → shared Latin scheme; cleans the transcription marks in `.cha` words |
| `utils/conllu_parser.py` | Reads the treebanks, builds Latin vocabularies, assigns pseudo-labels |
| `utils/cha_parser.py` | Reads the `.cha` transcripts: gold labels, phonetic labels, POS and lemmas from `%mor` |
| `utils/build_processed.py` | Writes `data/processed/*.jsonl` and `stats.json`, the input for all models |
| `utils/metrics.py` | Macro-F1, per-label report, confusion matrix |
| `baseline/cues.py` | Latin spelling cues and word endings for BE vs RU, chosen from the treebanks |
| `baseline/splits.py` | Shared training set, the four evaluation sets, scoring and `results/*.json` |
| `baseline/ngram_heuristic.py` | Rule-based baseline: cues, endings and vocabulary lookups |
| `baseline/train_crf.py`, `crf_features.py` | CRF baseline (sklearn-crfsuite) |

Models train on RU train + official-orthography BE train (RU sampled to the same size) and are
evaluated on Trasyanka, the RU test split, the official BE test split and the Taraškievica sentences.

| Macro-F1 | Trasyanka | RU test | BE official test | Taraškievica |
|---|---|---|---|---|
| Heuristic | 0.3672 | 0.9215 | 0.5503 | 0.5597 |
| CRF | 0.4478 | 0.9997* | 0.5958 | 0.5992 |

\* Circular: the CRF's vocabulary features reproduce the rule that made the RU pseudo-labels.

## Running

From the project root:

```
python src/utils/conllu_parser.py     # builds data/dictionaries/ and shows sample labels
python src/utils/cha_parser.py        # parses one transcript as a demo
python src/utils/build_processed.py   # writes data/processed/ (about 1 minute)
python src/baseline/ngram_heuristic.py  # writes results/heuristic.json
python src/baseline/train_crf.py        # writes results/crf.json (about 2 minutes)
```
