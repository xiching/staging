# Transformer Translation Model
This is an implementation of the Transformer translation model as described in the [Attention is All You Need](https://arxiv.org/abs/1706.03762) paper. Based on the code provided by the authors: [Transformer code](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py) from [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor).

Transformer is a neural network architecture that solves sequence to sequence problems using attention mechanisms. Unlike traditional neural seq2seq models, Transformer does not involve recurrent connections. The attention mechanism learns dependencies between tokens in two sequences. Since attention weights apply to all tokens in the sequences, the Tranformer model is able to easily capture long-distance depedencies (an issue that appears in recurrent models).

Transformer's overall structure follows the standard encoder-decoder pattern. The encoder uses self-attention to compute a representation of the input sequence. The decoder generates the output sequence one token at a time, taking the encoder output and previous decoder-outputted tokens as inputs.

The model also applies embeddings on the input and output tokens, and adds a constant positional encoding. The positional encoding adds information about the position of each token.

## Training the model
1. **Data download**

   Follow  the first part of [tensor2tensor's walkthrough](https://github.com/tensorflow/tensor2tensor#walkthrough) to download the translate_ende_wmt32k dataset.

   ```
   pip install tensor2tensor

   PROBLEM=translate_ende_wmt32k
   MODEL=transformer
   HPARAMS=transformer_base_single_gpu

   DATA_DIR=$HOME/t2t_data
   TMP_DIR=/tmp/t2t_datagen
   TRAIN_DIR=$HOME/t2t_train/$PROBLEM/$MODEL-$HPARAMS

   mkdir -p $DATA_DIR $TMP_DIR $TRAIN_DIR

   # Generate data
   t2t-datagen \
     --data_dir=$DATA_DIR \
     --tmp_dir=$TMP_DIR \
     --problem=$PROBLEM
   ```

2. **Training**

   Run `transformer.py`.
   ```
   transformer.py --data_dir=$DATA_DIR
   ```
   The `--data_dir` argument should be set to the same directory where the dataset was downloaded.

## Benchmarks
(TODO)

## Predictions and BLEU score
(TODO)
