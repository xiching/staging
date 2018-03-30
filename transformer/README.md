# Transformer Translation Model
This is an implementation of the Transformer translation model as described in the [Attention is All You Need](https://arxiv.org/abs/1706.03762) paper. Based on the code provided by the authors: [Transformer code](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py) from [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor).

Transformer is a neural network architecture that solves sequence to sequence problems using attention mechanisms. Unlike traditional neural seq2seq models, Transformer does not involve recurrent connections. The attention mechanism learns dependencies between tokens in two sequences. Since attention weights apply to all tokens in the sequences, the Tranformer model is able to easily capture long-distance depedencies (an issue that appears in recurrent models).

Transformer's overall structure follows the standard encoder-decoder pattern. The encoder uses self-attention to compute a representation of the input sequence. The decoder generates the output sequence one token at a time, taking the encoder output and previous decoder-outputted tokens as inputs.

The model also applies embeddings on the input and output tokens, and adds a constant positional encoding. The positional encoding adds information about the position of each token.

## Training the model
1. **Download and preprocess data**

   Run [`data_download.py`](data_download.py)

   Args:
    * `--data_dir`: `~/data/translate_ende` by default. Path TFRecord data, and vocab file will be saved.
    * `--raw_data`: `/tmp/translate_ende_raw` by default. Path where the raw data will be downloaded and extracted if not already there.
   ---
   The data is downloaded from the Workshop of Machine Transtion (WMT) [news translation task](http://www.statmt.org/wmt17/translation-task.html). The following datasets are used:

   * Europarl v7
   * Common Crawl corpus
   * News Commentary v12

   See the [download section](http://www.statmt.org/wmt17/translation-task.html#download) to explore the raw datasets.

   The parameters in this model are tuned to fit the English-German translation data, so the EN-DE texts are extracted from the downloaded compressed files.


2. **Run model training and evaluation**

   Run [`transformer.py`](transformer.py)

   Args:
   * `--data_dir`: `~/data/translate_ende` by default. This should be set to the same directory where the dataset was downloaded and preprocessed.
   * `--model_dir`: `/tmp/transformer_model` by default. Directory to save Transformer model training checkpoints.
   * `--num_cpu_cores`: 4 by default. Number of CPU cores to use in the input pipeline.
   * `--training_step`: 250000 by default. Total number of training steps.
   * `--eval_interval`: 1000 by default. Number of training steps to run between evaluations.
   * `--params`: Parameter set to use when creating and training the model. Options are `base` (default) and `big`.

3. **Translate using the model**
   (TODO)

4. **Compute BLEU score**
   (TODO)

## Benchmarks
(TODO)

## Implementation overview

A brief look at each component in the code:
1. **Data download**
   * [`data_download.py`](data_download.py): Downloads and extracts data, then uses `Subtokenizer` to tokenize strings into arrays of int IDs. The int arrays are converted to `tf.Examples` and saved in the `tf.RecordDataset` format.
   * [`tokenizer.py`](tokenizer.py): Defines the `Subtokenizer` class. During initialization, the raw data is used to generate a vocabulary list containing common subtokens* that appear in the input data. This vocabulary list stays static through training, evaluation, and inference.

2. **Model training and evaluation**
   * [`transformer.py`](transformer.py): defines the model, and creates an `estimator` to train and evaluate the model.
   * [`dataset.py`](dataset.py): contains functions for creating a `dataset` that is passed to the `estimator`

3. **Inference with trained model**
   * [`translate.py`](translate.py): First, uses `Subtokenizer` to tokenize the input. The vocabulary file is the same used to tokenize the training/eval files. Second, uses beam search to find the combination of tokens that maximizes the probability outputted by the model decoder. The tokens are then converted back to strings with `Subtokenizer`.

4. **BLEU computation**
   * [`compute_bleu.py`](compute_bleu.py): (TODO)

**Subtoken**: Words are referred as tokens, and parts of words are referred as 'subtokens'. For example, the word 'inclined' may be split into `['incline', 'd_']`. The '_' indicates the end of the token. The subtoken vocabulary list is guaranteed to contain the alphabet (including numbers and special characters), so all words can be tokenized.


