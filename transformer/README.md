# Transformer Translation Model
This is an implementation of the Transformer translation model as described in the [Attention is All You Need](https://arxiv.org/abs/1706.03762) paper. Based on the code provided by the authors: [Transformer code](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py) from [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor).

Transformer is a neural network architecture that solves sequence to sequence problems using attention mechanisms. Unlike traditional neural seq2seq models, Transformer does not involve recurrent connections. The attention mechanism learns dependencies between tokens in two sequences. Since attention weights apply to all tokens in the sequences, the Tranformer model is able to easily capture long-distance depedencies (an issue that appears in recurrent models).

Transformer's overall structure follows the standard encoder-decoder pattern. The encoder uses self-attention to compute a representation of the input sequence. The decoder generates the output sequence one token at a time, taking the encoder output and previous decoder-outputted tokens as inputs.

The model also applies embeddings on the input and output tokens, and adds a constant positional encoding. The positional encoding adds information about the position of each token.

## Training the model
1. **Download and preprocess data**

   Run `data_download` to download and preprocess the data. After the data is downloaded and extracted, the training data is used to generate a vocabulary of subtokens. The evaluation and training strings are tokenized, and the resulting data is sharded, shuffled, and saved as TFRecords.

   ```
   python data_download.py --data_dir=/path/to/data
   ```

   Arguments:
    * `--data_dir`: Path where the preprocessed TFRecord data, and vocab file will be saved.
    * Use the `--help` or `-h` flag to get a full list of possible arguments.

     1.75GB of compressed data will be downloaded. In total, the raw files (compressed, extracted, and combined files) take up 8.4GB of disk space. The resulting TFRecord and vocabulary files are 722MB. The script takes around 40 minutes to run, with the bulk of the time spent downloading and ~15 minutes spent on preprocessing.

2. **Model training and evaluation**

   Run `transformer.py`, which creates a Transformer model graph using Tensorflow Estimator.

   ```
   python transformer.py --data_dir=/path/to/data --model_dir=/path/to/model --params=base
   ```

   Arguments:
   * `--data_dir`: This should be set to the same directory given to the `data_download`'s `data_dir` argument.
   * `--model_dir`: Directory to save Transformer model training checkpoints.
   * `--params`: Parameter set to use when creating and training the model. Options are `base` (default) and `big`.
   * Use the `--help` or `-h` flag to get a full list of possible arguments.


   Training and evaluation metrics (loss, accuracy, approximate BLEU score, etc.) are saved using `tf.summary`, and can be displayed in the browser using Tensorboard.
   ```
   tensorboard --logdir=/path/to/model
   ```
   The values are displayed at [localhost:6006].

3. **Translate using the model**
   (TODO)

4. **Compute official BLEU score**
   (TODO)

## Benchmarks
(TODO)

## Implementation overview

A brief look at each component in the code:
1. **Data download and preprocessing**
   * [`data_download.py`](data_download.py): Downloads and extracts data, then uses `Subtokenizer` to tokenize strings into arrays of int IDs. The int arrays are converted to `tf.Examples` and saved in the `tf.RecordDataset` format.

     The data is downloaded from the Workshop of Machine Transtion (WMT) [news translation task](http://www.statmt.org/wmt17/translation-task.html). The following datasets are used:

     * Europarl v7
     * Common Crawl corpus
     * News Commentary v12

     See the [download section](http://www.statmt.org/wmt17/translation-task.html#download) to explore the raw datasets.

     The parameters in this model are tuned to fit the English-German translation data, so the EN-DE texts are extracted from the downloaded compressed files.

   * [`tokenizer.py`](tokenizer.py): Defines the `Subtokenizer` class. During initialization, the raw data is used to generate a vocabulary list containing common subtokens* that appear in the training data. Note that the same subtoken vocabulary must be used on all data so that the model learns and outputs consistent IDs.

      The target vocabulary size of the WMT dataset is 32k. The set of subtokens is found through binary search on the minimum number of times a subtoken appears in the data. The actual vocabulary size is 33,708, and is stored in a 324kB file.

2. **Model training and evaluation**
   * [`transformer.py`](transformer.py): defines the model, and creates an `estimator` to train and evaluate the model.
   * [`dataset.py`](dataset.py): contains functions for creating a `dataset` that is passed to the `estimator`

3. **Inference with trained model**
   * [`translate.py`](translate.py): First, `Subtokenizer` tokenizes the input. The vocabulary file is the same used to tokenize the training/eval files. Next, beam search is used to find the combination of tokens that maximizes the probability outputted by the model decoder. The tokens are then converted back to strings with `Subtokenizer`.

4. **BLEU computation**
   * [`compute_bleu.py`](compute_bleu.py): (TODO)

**Subtoken**: Words are referred as tokens, and parts of words are referred as 'subtokens'. For example, the word 'inclined' may be split into `['incline', 'd_']`. The '\_' indicates the end of the token. The subtoken vocabulary list is guaranteed to contain the alphabet (including numbers and special characters), so all words can be tokenized.


