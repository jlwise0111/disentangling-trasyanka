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
| `baseline/ngram_heuristic.py` | Rule-based baseline *(not yet updated for the Latin pipeline)* |
| `baseline/train_crf.py`, `crf_features.py` | CRF baseline *(not yet updated for the Latin pipeline)* |

## Running

From the project root:

```
python src/utils/conllu_parser.py     # builds data/dictionaries/ and shows sample labels
python src/utils/cha_parser.py        # parses one transcript as a demo
python src/utils/build_processed.py   # writes data/processed/ (about 1 minute)
```
