Adding a new dataset
====================

- Do any preprocessing necessary. The dataset should have 'train', 'validation' and 'test' splits.
- Save the processing script in the [`datasets`](../datasets/) directory.
- Upload the dataset to Hugging Face, including the 'train', 'validation' and 'test' splits.
- Create a new dataset container in [`al_llm/dataset_container.py`](../al_llm/dataset_container.py).
    * Subclass `HuggingFaceDatasetContainer`.
    * Write a docstring including a reference to the dataset.
    * Set `DATASET_NAME` to the name of the dataset on Hugging Face, including the 'OxAISH-AL-LLM/' part.
    * Set the `CATEGORIES` ordered dict to the specify the dataset classes.
        - Order the classes as they are in the dataset csv file.
        - The key in the ordered dict is an abbreviation of the class name, used to represent the data in Hugging Face.
        - The value is the human readable name of the class.
    * Set `TOKENIZED_LENGTH_UPPER_QUARTILE` to the upper quartile of the lengths of the sentences in the dataset. This will be the max length of sentences generated for this dataset.
    * Define any preprocessing in `_preprocess_dataset`. This could involve renaming columns or anything else.
- Import the dataset container into [`al_llm/experiment.py`](../al_llm/experiment.py).
- In that file, add an entry in the `Experiment.MAP_DATASET_CONTAINER` dict for the new dataset.
- Import the dataset container into [`al_llm/__init__.py`](../al_llm/__init__.py).
- (Optional, recommended) Create a TAPTed version of GPT-2 for the dataset.
    * Follow the guide in [`scripts/tapt/README.md`](../scripts/tapt/README.md).
    * You'll need to add a new entry in the `DATASET_NAME_MAP` dict in [`scripts/tapt/run_clm.py`](../scripts/tapt/run_clm.py).
    * In Weights and Biases, tag the artifact created with 'default', and also a tag specifying the TAPTing hyperparameters, following the specification, where `DATASET_VERSION` is 'orig' unless you're TAPTing with a modified version of the dataset.

        e{NUM_EPOCHS}-b{BATCH_SIZE}-bl{BLOCK_SIZE}-{DATASET_VERSION}
