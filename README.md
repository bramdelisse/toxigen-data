---
annotations_creators:
- expert-generated
language_creators:
- machine-generated
languages:
- en-US
licenses: []
multilinguality:
- monolingual
pretty_name: ToxiGen
size_categories:
- 100K<n<1M
source_datasets:
- original
task_categories:
- text-classification
task_ids:
- hate-speech-detection
---

# Dataset Card for ToxiGen

## Dataset Description

- **Repository:** https://github.com/microsoft/toxigen
- **Paper:** https://arxiv.org/abs/2203.09509

## Dataset Structure

### Data Fields

We release TOXIGEN as a dataframe with the following fields:
- **prompt** is the prompt used for **generation**.
- **generation** is the TOXIGEN generated text.
- **generation_method** denotes whether or not ALICE was used to generate the corresponding generation. If this value is ALICE, then ALICE was used, if it is TopK, then ALICE was not used.
- **prompt_label** is the binary value indicating whether or not the prompt is toxic (1 is toxic, 0 is benign).
- **group** indicates the target group of the prompt.
- **roberta_prediction** is the probability predicted by our corresponding RoBERTa model for each instance.