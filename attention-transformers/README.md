# Attention & Transformers

Assignment 2 for the Deeper Neural Networks course. Covers self-attention and transformer-based models.

## Contents

| File | Description |
|------|-------------|
| `part_a_self_attention.py` | Self-attention from scratch: implements Q, K, V projections and attention scores for a single sentence (no training) |
| `part_b_movie_chatbot.py` | T5-based movie chatbot: fine-tunes T5-small on a small IMDb-style QA corpus and provides an interactive Q&A interface |
| `report.tex` | LaTeX report for the assignment |

## Running

From the project root (where `requirements.txt` lives):

```bash
pip install -r requirements.txt
python attention-transformers/part_a_self_attention.py
python attention-transformers/part_b_movie_chatbot.py
```

Part B downloads the pretrained T5-small model on first run and includes an interactive chatbot mode at the end.
