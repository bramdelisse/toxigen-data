# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import csv
import json
import os
import ast
import datasets
import numpy as np

_CITATION = """\
@inproceedings{hartvigsen2022toxigen,
  title={ToxiGen: A Large-Scale Machine-Generated Dataset for Implicit and Adversarial Hate Speech Detection},
  author={Hartvigsen, Thomas and Gabriel, Saadia and Palangi, Hamid and Sap, Maarten and Ray, Dipankar and Kamar, Ece},
  booktitle={Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics},
  year={2022}
}
"""
_DESCRIPTION = """\
Toxigen is a large-scale dataset containing implicitly toxic and benign sentences mentioning 13 minority groups, and a tool to stress test a given off-the-shelf toxicity classifier. The dataset is generated using a large language model (GPT3). It is intended to be used for training classifiers that learn to detect subtle hate speech that includes no slurs or profanity.
"""

# TODO: Add a link to an official homepage for the dataset here
_HOMEPAGE = "https://github.com/microsoft/TOXIGEN"

_LICENSE = None

_URLS = {"train":"https://huggingface.co/datasets/skg/toxigen-data/resolve/main/toxigen.csv","annotated":"https://huggingface.co/datasets/skg/toxigen-data/resolve/main/annotated_test.csv","annotated-train":"https://huggingface.co/datasets/skg/toxigen-data/resolve/main/annotated_train.csv"}




# TODO: Name of the dataset usually match the script name with CamelCase instead of snake_case
class ToxigenData(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    VERSION = datasets.Version("1.1.0")

    # This is an example of a dataset with multiple configurations.
    # If you don't want/need to define several sub-sets in your dataset,
    # just remove the BUILDER_CONFIG_CLASS and the BUILDER_CONFIGS attributes.

    # If you need to make complex sub-parts in the datasets with configurable options
    # You can create your own builder configuration class to store attribute, inheriting from datasets.BuilderConfig
    # BUILDER_CONFIG_CLASS = MyBuilderConfig

    # You will be able to load one or the other configurations in the following list with
    # data = datasets.load_dataset('my_dataset', 'first_domain')
    # data = datasets.load_dataset('my_dataset', 'second_domain')
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="train", version=VERSION, description="Raw data without annotations"),
        datasets.BuilderConfig(name="annotated", version=VERSION, description="Annotated data from human eval"),
    ]

    DEFAULT_CONFIG_NAME = "annotated"  # It's not mandatory to have a default configuration. Just use one if it make sense.

    def _info(self):
        # TODO: This method specifies the datasets.DatasetInfo object which contains informations and typings for the dataset
        if self.config.name == "train":  # This is the name of the configuration selected in BUILDER_CONFIGS above
            features = datasets.Features(
                {
                    "prompt": datasets.Value("string"),
                    "generation": datasets.Value("string"),
                    "generation_method": datasets.Value("string"),
                    "group": datasets.Value("string"),
                    "prompt_label": datasets.Value("int64"),
                    "roberta_prediction": datasets.Value("float64"),
                    # These are the features of your dataset like images, labels ...
                }
            )
        else:  # This is an example to show how to have different features for "first_domain" and "second_domain"
            features = datasets.Features(
                {
                    "text": datasets.Value("string"),
                    "target_group": datasets.Value("string"),
                    "factual?": datasets.Value("string"),
                    "ingroup_effect": datasets.Value("string"),
                    "lewd": datasets.Value("string"),
                    "framing": datasets.Value("string"),
                    "predicted_group": datasets.Value("string"),
                    "stereotyping": datasets.Value("string"),
                    "intent": datasets.Value("float64"),
                    "toxicity_ai": datasets.Value("float64"),
                    "toxicity_human": datasets.Value("float64"),
                    "predicted_author": datasets.Value("string"),
                    "actual_method": datasets.Value("string"),
                    # These are the features of your dataset like images, labels ...
                }
            )
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features, uncomment supervised_keys line below and
            # specify them. They'll be used if as_supervised=True in builder.as_dataset.
            # supervised_keys=("sentence", "label"),
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # TODO: This method is tasked with downloading/extracting the data and defining the splits depending on the configuration
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLS
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive
        if "annotated" in self.config.name:
           urls = [_URLS[self.config.name],_URLS["annotated-train"]]
        else:
           urls = _URLS[self.config.name]
        data_dir = dl_manager.download(urls)
        if self.config.name == "annotated":
           return [datasets.SplitGenerator(name=datasets.Split.TEST,gen_kwargs={"filepath": data_dir[0],"split": "small_eval_set",},),datasets.SplitGenerator(name=datasets.Split.TRAIN,gen_kwargs={"filepath": data_dir[1],"split": "large_eval_set",},)]
        else:
           return [datasets.SplitGenerator(name=datasets.Split.TRAIN,gen_kwargs={"filepath": data_dir,"split": "full_unannotated",},)]
    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepath, split):
        # TODO: This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.
        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.
        with open(filepath, encoding="utf-8") as f:
                data = [row for row in csv.reader(f)]
                header = data[0]
                data = data[1:]
                if self.config.name == "annotated":
                    # Yields examples as (key, example) tuples
                    for key,d in enumerate(data):
                        if d[0][:2] == "b'" or d[0][:2] == 'b"':
                           d[0] = d[0][2:]
                        if d[0][-1] == "'" or d[0][-1] == '"':
                           d[0] = d[0][:-1]
                        yield key, {
                        "text": d[0],
                        "target_group": d[1],
                        "factual?": d[2],
                        "ingroup_effect": d[3],
                        "lewd": d[4],
                        "framing": d[5],
                        "predicted_group": d[6],
                        "stereotyping": d[7],
                        "intent": float(d[8]),
                        "toxicity_ai": float(d[9]),
                        "toxicity_human": float(d[10]),
                        "predicted_author": d[11],
                        "actual_method": d[header.index("actual_method")].lower(),
                        }
                else:
                    # Yields examples as (key, example) tuples
                    for key,d in enumerate(data):
                        yield key, {
                        "prompt": d[0],
                        "generation": d[1],
                        "generation_method": d[2],
                        "group": d[3],
                        "prompt_label": int(d[4]),
                        "roberta_prediction": float(d[5]),

                        }
