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

## Table of Contents
- [Dataset Description](#dataset-description)
  - [Dataset Summary](#dataset-summary)
  - [Languages](#languages)
- [Dataset Structure](#dataset-structure)
  - [Data Fields](#data-instances)
- [Additional Information](#additional-information)
  - [Citation Information](#citation-information)

## Sign up for Data Access
To access ToxiGen, first fill out [this form](https://forms.office.com/r/r6VXX8f8vh).

## Dataset Description

- **Repository:** https://github.com/microsoft/toxigen
- **Paper:** https://arxiv.org/abs/2203.09509
- **Point of Contact #1:** [Tom Hartvigsen](tomh@mit.edu)
- **Point of Contact #2:** [Saadia Gabriel](skgabrie@cs.washington.edu)

### Dataset Summary

This dataset is for implicit hate speech detection. All instances were generated using GPT-3 and the methods described in [our paper](https://arxiv.org/abs/2203.09509).

### Languages

All text is written in English.

## Dataset Structure

### Data Fields

We release TOXIGEN as a dataframe with the following fields:
- **prompt** is the prompt used for **generation**.
- **generation** is the TOXIGEN generated text.
- **generation_method** denotes whether or not ALICE was used to generate the corresponding generation. If this value is ALICE, then ALICE was used, if it is TopK, then ALICE was not used.
- **prompt_label** is the binary value indicating whether or not the prompt is toxic (1 is toxic, 0 is benign).
- **group** indicates the target group of the prompt.
- **roberta_prediction** is the probability predicted by our corresponding RoBERTa model for each instance.

### Citation Information

```bibtex
@inproceedings{hartvigsen2022toxigen,
  title={ToxiGen: A Large-Scale Machine-Generated Dataset for Implicit and Adversarial Hate Speech Detection},
  author={Hartvigsen, Thomas and Gabriel, Saadia and Palangi, Hamid and Sap, Maarten and Ray, Dipankar and Kamar, Ece},
  booktitle={Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics},
  year={2022}
}
```
