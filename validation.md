# Validation pilot: do my labels match a second labeler's?

This page records a small pilot check of the six comment categories used in `backend/sentiment.py`. Read the Important note first, because it changes what the numbers mean.

## Important note

The second set of labels in this pilot was **not produced by this repo's classifier**. I did not have an API key for the model the tool uses, so after labeling the comments myself, I ran the same comments through Gemini to get a second set of labels. The Gemini model and the exact prompt were not recorded.

So this pilot compares my hand labels to Gemini's labels. It does not measure how accurate `backend/sentiment.py` is. The repo classifier has not been validated yet. The method and scripts are ready for that run (see Next steps).

## Question

On real YouTube comments, how often does a second labeler's category match the category I give the same comment by hand?

This measures agreement with one person. It is not ground truth, and nothing was tuned.

## Data

- Video: https://youtu.be/8VL66ifp1Xc (id `8VL66ifp1Xc`), channel Your Muse Traci
- 35 comments from a CSV export made by an outside tool. They did not come from `get_comments` or the YouTube API. I do not know how the export chose its 35 comments (newest, top, replies or not).
- All 35 were used. No sampling.
- `data/labeling_sheet.csv` has `id`, `comment`, `my_label`. `data/tool_labels.csv` has `id`, `tool_label` (here, the Gemini labels). Both were made by hand outside the scripts.
- `data/` is git-ignored. Raw comment text is never committed.
- Run date: 2026-10-20.

## Category definitions

Copied from the system prompt in `backend/sentiment.py`:

- supportive: specific praise, thanks, or a personal experience that refers to the content.
- question: asks about the content, a product, or the creator.
- critical: disagreement or criticism, including constructive criticism.
- toxic: insults, harassment, or hate.
- generic: low-effort reactions with no specific content ("nice", "first", emoji only).
- spam: promotion, links, giveaways, or bot-like text.

Each comment gets exactly one label.

Rule I added while labeling: default to `generic` for low-effort reactions with slang or informal humor unless explicit praise or support is clear. This rule is stricter than the definitions above and pushed my labels toward `generic`.

## Comparison

`scripts/compare.py` reports overall agreement with a 95% Wilson interval, Cohen's kappa, per category recall and precision, a confusion matrix (my label in rows, the second labeler in columns), and every mismatch.

## Results (my labels vs Gemini)

- **Scored comments:** 35 of 35
- **Overall agreement:** 97.1% (34 of 35)
- **95% interval (Wilson):** 85.5% to 99.5%
- **Cohen's kappa:** 0.79. Kappa is pulled around by how uneven the categories are, so I do not rely on a verbal label for it.

| category | I labeled | Gemini labeled | both | recall | precision |
|---|---|---|---|---|---|
| supportive | 1 | 0 | 0 | 0.0% | n/a |
| question | 1 | 1 | 1 | 100.0% | 100.0% |
| critical | 0 | 0 | 0 | n/a | n/a |
| toxic | 0 | 0 | 0 | n/a | n/a |
| generic | 32 | 33 | 32 | 100.0% | 97.0% |
| spam | 1 | 1 | 1 | 100.0% | 100.0% |

Confusion matrix (rows = my label, columns = Gemini label):

| | supportive | question | critical | toxic | generic | spam | total |
|---|---|---|---|---|---|---|---|
| supportive | 0 | 0 | 0 | 0 | 1 | 0 | 1 |
| question | 0 | 1 | 0 | 0 | 0 | 0 | 1 |
| critical | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| toxic | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| generic | 0 | 0 | 0 | 0 | 32 | 0 | 32 |
| spam | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| total | 0 | 1 | 0 | 0 | 33 | 1 | 35 |

Mismatch (1): ID 2. I labeled it `supportive` and Gemini labeled it `generic`. It is a short, informal comment that can be read as praise or as a casual reaction.

## What this shows, and what it does not

- Both label sets put 32 or 33 of 35 comments in `generic`. A labeler that answered `generic` every time would match me on 32 of 35 (91.4%). The 34 of 35 is above that, but the low end of the interval (85.5%) is below it, so this sample cannot separate Gemini from that baseline.
- On a second read against the definitions, several comments look like they fit other categories than `generic`: ids 5, 6, 23, 32 and 34 read as praise or thanks, ids 8, 11 and 29 ask questions, and ids 15, 17, 19 and 20 trade insults. Both label sets still say `generic` for them. I have not changed the labels, so the figures above are as originally produced.
- So the high agreement mostly shows that both labelers defaulted to `generic`. It says very little about how well anyone finds `supportive`, `question`, `critical` or `toxic` comments, and nothing about the repo classifier.

## Limits

- 35 comments, one video, one human labeler.
- The second labeler is Gemini, not the repo classifier, with an unrecorded model and prompt.
- I labeled all 35 comments myself first and only then ran them through Gemini, so my labels were not influenced by its output. The Gemini prompt was not recorded, so I cannot say whether it included the category definitions.
- Unknown selection method for the 35 comments.
- This checks the category label only, not sentiment, themes, brand-fit, or the Audience Trust Score.

## Next steps

1. Relabel the 35 comments strictly against the definitions, without looking at any tool output.
2. Run the repo classifier with an API key to produce real `data/tool_labels.csv`.
3. Run `python scripts/compare.py` and replace the Results section.
