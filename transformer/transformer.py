# Copyright 2018 MLBenchmark Group. All Rights Reserved.
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
# ==============================================================================
"""Defines the Transformer model, and run training loop for the model.

Transformer model code source: https://github.com/tensorflow/tensor2tensor
"""

import argparse
import math
import sys

from six.moves import xrange
import tensorflow as tf
from tensorflow.python.util import nest

import beam_search
import dataset
import metrics
import model_params
import tokenizer


class Transformer(object):
  """Transformer model that inputs and outputs data."""

  def __init__(self, params, train):
    self.train = train
    self.params = params

    self.embedding_softmax_layer = EmbeddingSharedWeights(params.vocab_size,
                                                          params.hidden_size)
    self.encoder = Encoder(params, train)
    self.decoder = Decoder(params, train)

  def __call__(self, inputs, targets=None):
    # Variance scaling is used here because it seems to work in many problems.
    # Other reasonable initializers may also work just as well.
    initializer = tf.variance_scaling_initializer(
        self.params.initializer_gain, mode="fan_avg", distribution="uniform")
    with tf.variable_scope("Transformer", initializer=initializer):
      if targets is None:
        return self.predict(inputs)

      # Prepare encoder inputs and encode
      embedded_inputs, inputs_padding = self.embedding_softmax_layer(inputs)
      embedded_targets, _ = self.embedding_softmax_layer(targets)
      attention_bias = self.get_padding_bias(inputs_padding)
      encoder_outputs = self.encode(
          embedded_inputs, inputs_padding, attention_bias)

      # Pepare decoder inputs and decode
      decoder_inputs, decoder_self_attention_bias = prepare_decoder_inputs(
          embedded_targets, self.params.hidden_size,
          self.params.layer_postprocess_dropout, self.train)
      decoder_outputs = self.decode(
          decoder_inputs, encoder_outputs, attention_bias,
          decoder_self_attention_bias)

      logits = self.embedding_softmax_layer.linear(decoder_outputs)
      return logits

  def get_padding_bias(self, inputs_padding):
    """Calculate attention bias from input padding."""
    # Create bias tensor of size [batch_size, 1, 1, input_len] that is zero
    # everywhere except -1e9 (negative infinity) at the padding locations.
    with tf.name_scope("attention_bias"):
      attention_bias = inputs_padding * -1e9
      attention_bias = tf.expand_dims(
          tf.expand_dims(attention_bias, axis=1), axis=1)
    return attention_bias

  def encode(self, embedded_inputs, inputs_padding, attention_bias):
    """Get encoder output from inputs."""
    encoder_inputs = prepare_encoder_inputs(
        embedded_inputs, self.params.hidden_size,
        self.params.layer_postprocess_dropout, self.train)
    return self.encoder(encoder_inputs, attention_bias, inputs_padding)

  def decode(self, decoder_inputs, encoder_outputs, attention_bias,
      decoder_self_attention_bias, cache=None):
    """Get decoder outputs from target values and encoder output."""
    return self.decoder(
        decoder_inputs, encoder_outputs, decoder_self_attention_bias,
        attention_bias, cache)

  def predict(self, inputs):
    """Predict"""
    # Prepare decoder values
    embedded_inputs, inputs_padding = self.embedding_softmax_layer(inputs)
    attention_bias = self.get_padding_bias(inputs_padding)
    encoder_outputs = self.encode(
        embedded_inputs, inputs_padding, attention_bias)
    max_decode_length = tf.shape(inputs)[1] + self.params.extra_decode_length
    decoder_self_attention_bias = get_decoder_self_attention_bias(
        max_decode_length)

    timing_signal = get_position_encoding(
        max_decode_length + 1, self.params.hidden_size)

    def preprocess_targets(targets, i):
      """Performs preprocessing steps on the targets to prepare for the decoder.

      This includes:
        - Embedding the ids.
        - Flattening to 3D tensor.
        - Optionally adding timing signals.

      Args:
        targets: inputs ids to the decoder. [batch_size, 1]
        i: scalar, Step number of the decoding loop.

      Returns:
        Processed targets [batch_size, 1, hidden_dim]
      """
      targets, _ = self.embedding_softmax_layer(targets)

      # The initial IDs shouldn't matter, so when i=0 set all embedded values to
      # zero.
      targets = tf.cond(
          tf.equal(i, 0), lambda: tf.zeros_like(targets), lambda: targets)

      targets += timing_signal[i:i + 1]
      return targets

    def symbols_to_logits_fn(ids, i, cache):
      """Predict the next set of logits.

      Args:
        ids: Current decoded sequences
          int32 tensor with shape [batch_size * beam_size, i]
        i: Loop index, also the current sequence length.
        cache: dictionary of values storing the encoder output, encoder-decoder
          attention bias, and previous decoder attention values.

      Returns:
        Tuple of
          (logits of candidate IDs for each sequence
             [batch_size * beam_size, vocab_size],
           updated cache values)
      """
      decoder_input = ids[:, -1:]  # Set decoder input as the last generated IDs
      decoder_input = preprocess_targets(decoder_input, i)
      bias = decoder_self_attention_bias[:, :, i:i + 1, :i + 1]

      decoder_outputs = self.decode(
          decoder_input, cache.get("encoder_output"),
          cache.get("encoder_decoder_attention_bias"), bias, cache)

      logits = self.embedding_softmax_layer.linear(decoder_outputs)
      logits = tf.squeeze(logits, axis=[1])
      return logits, cache

    return self.fast_decode(
        encoder_output=encoder_outputs,
        encoder_decoder_attention_bias=attention_bias,
        symbols_to_logits_fn=symbols_to_logits_fn,
        max_decode_length=max_decode_length,
        beam_size=self.params.beam_size,
        top_beams=1,
        alpha=self.params.alpha,
        eos_id=tokenizer.EOS_ID)

  def fast_decode(self, encoder_output,
      encoder_decoder_attention_bias,
      symbols_to_logits_fn,
      max_decode_length,
      beam_size=1,
      top_beams=1,
      alpha=1.0,
      eos_id=tokenizer.EOS_ID,
      batch_size=None):
    """Given encoder output and a symbols to logits function, does fast decoding.

    Implements both greedy and beam search decoding, uses beam search iff
    beam_size > 1, otherwise beam search related arguments are ignored.

    Args:
      encoder_output: Output from encoder.
      encoder_decoder_attention_bias: a bias tensor for use in encoder-decoder
        attention
      symbols_to_logits_fn: Incremental decoding; function mapping triple
        `(ids, step, cache)` to symbol logits.
      hparams: run hyperparameters
      max_decode_length: an integer.  How many additional timesteps to decode.
      vocab_size: Output vocabulary size.
      beam_size: number of beams.
      top_beams: an integer. How many of the beams to return.
      alpha: Float that controls the length penalty. larger the alpha, stronger
        the preference for slonger translations.
      eos_id: End-of-sequence symbol in beam search.
      batch_size: an integer scalar - must be passed if there is no input

    Returns:
        A dict of decoding results {
            "outputs": integer `Tensor` of decoded ids of shape
                [batch_size, <= decode_length] if top_beams == 1 or
                [batch_size, top_beams, <= decode_length] otherwise
            "scores": decoding log probs from the beam search,
                None if using greedy decoding (beam_size=1)
        }

      Raises:
        NotImplementedError: If beam size > 1 with partial targets.
    """
    if encoder_output is not None:
      batch_size = tf.shape(encoder_output)[0]

    key_channels = self.params.hidden_size
    value_channels = self.params.hidden_size
    num_layers = self.params.num_hidden_layers

    cache = {
      "layer_%d" % layer: {
        "k": tf.zeros([batch_size, 0, key_channels]),
        "v": tf.zeros([batch_size, 0, value_channels]),
      }
      for layer in range(num_layers)
    }

    if encoder_output is not None:
      cache["encoder_output"] = encoder_output
      cache["encoder_decoder_attention_bias"] = encoder_decoder_attention_bias

    if beam_size > 1:  # Beam Search
      initial_ids = tf.zeros([batch_size], dtype=tf.int32)
      decoded_ids, scores = beam_search.sequence_beam_search(
          symbols_to_logits_fn=symbols_to_logits_fn,
          initial_ids=initial_ids,
          initial_cache=cache,
          vocab_size=self.params.vocab_size,
          beam_size=beam_size,
          alpha=alpha,
          max_decode_length=max_decode_length,
          eos_id=eos_id)

      if top_beams == 1:
        decoded_ids = decoded_ids[:, 0, 1:]
        scores = scores[:, 0]
      else:
        decoded_ids = decoded_ids[:, :top_beams, 1:]
        scores = scores[:, top_beams]
    else:  # Greedy

      def inner_loop(i, finished, next_id, decoded_ids, cache):
        """One step of greedy decoding."""
        logits, cache = symbols_to_logits_fn(next_id, i, cache)
        next_id = tf.argmax(logits, -1)
        finished |= tf.equal(next_id, eos_id)
        next_id = tf.expand_dims(next_id, axis=1)
        decoded_ids = tf.concat([decoded_ids, next_id], axis=1)
        return i + 1, finished, next_id, decoded_ids, cache

      def is_not_finished(i, finished, *_):
        return (i < max_decode_length) & tf.logical_not(tf.reduce_all(finished))

      decoded_ids = tf.zeros([batch_size, 0], dtype=tf.int64)
      finished = tf.fill([batch_size], False)
      next_id = tf.zeros([batch_size, 1], dtype=tf.int64)
      _, _, _, decoded_ids, _ = tf.while_loop(
          is_not_finished,
          inner_loop,
          [tf.constant(0), finished, next_id, decoded_ids, cache],
          shape_invariants=[
            tf.TensorShape([]),
            tf.TensorShape([None]),
            tf.TensorShape([None, None]),
            tf.TensorShape([None, None]),
            nest.map_structure(beam_search.get_state_shape_invariants, cache),
          ])
      scores = None

    return {"outputs": decoded_ids, "scores": scores}


class EmbeddingSharedWeights(tf.layers.Layer):
  """Calculates input embeddings and pre-softmax linear with shared weights."""

  def __init__(self, vocab_size, hidden_size):
    super(EmbeddingSharedWeights, self).__init__()
    self.vocab_size = vocab_size
    self.hidden_size = hidden_size

  def build(self, _):
    with tf.variable_scope("embedding_and_softmax", reuse=tf.AUTO_REUSE):
      # Create and initialize weights. The random normal initializer was chosen
      # randomly, and works well.
      self.shared_weights = tf.get_variable(
          "weights", [self.vocab_size, self.hidden_size],
          initializer=tf.random_normal_initializer(
              0., self.hidden_size ** -0.5))

    self.built = True

  def call(self, x):
    """Get token embeddings of x.

    Args:
      x: An int64 tensor with shape [batch_size, length]
    Returns:
      embeddings: float32 tensor with shape [batch_size, length, embedding_size]
      padding: float32 tensor with shape [batch_size, length] indicating the
        locations of the padding tokens in x.
    """
    with tf.name_scope("embedding"):
      embeddings = tf.gather(self.shared_weights, x)

      # Scale embedding by the sqrt of the hidden size
      embeddings *= self.hidden_size ** 0.5

      # Create binary array of size [batch_size, length]
      # where 1 = padding, 0 = not padding
      padding = tf.to_float(tf.equal(x, 0))

      # Set all padding embedding values to 0
      embeddings *= tf.expand_dims(1 - padding, -1)
      return embeddings, padding

  def linear(self, x):
    """Computes logits by running x through a linear layer.

    Args:
      x: A float32 tensor with shape [batch_size, length, hidden_size]
    Returns:
      float32 tensor with shape [batch_size, length, vocab_size].
    """
    with tf.name_scope("presoftmax_linear"):
      batch_size = tf.shape(x)[0]
      length = tf.shape(x)[1]

      x = tf.reshape(x, [-1, self.hidden_size])
      logits = tf.matmul(x, self.shared_weights, transpose_b=True)

      return tf.reshape(logits, [batch_size, length, self.vocab_size])


class Attention(tf.layers.Layer):
  """Multi-headed attention layer."""

  def __init__(self, hidden_size, num_heads, attention_dropout, train):
    assert hidden_size % num_heads == 0, (
        "Hidden size must be evenly divisible by the number of heads.")

    super(Attention, self).__init__()
    self.hidden_size = hidden_size
    self.num_heads = num_heads
    self.attention_dropout = attention_dropout
    self.train = train

    # Layers for linearly projecting the queries, keys, and values.
    self.q_dense_layer = tf.layers.Dense(hidden_size, use_bias=False, name="q")
    self.k_dense_layer = tf.layers.Dense(hidden_size, use_bias=False, name="k")
    self.v_dense_layer = tf.layers.Dense(hidden_size, use_bias=False, name="v")

    self.output_dense_layer = tf.layers.Dense(hidden_size, use_bias=False,
                                              name="output_transform")

  def split_heads(self, x):
    """Split x into different heads, and transpose the resulting value.

    The tensor is transposed to insure the inner dimensions hold the correct
    values during the matrix multiplication.

    Args:
      x: A tensor with shape [batch_size, length, hidden_size]

    Returns:
      A tensor with shape [batch_size, num_heads, length, hidden_size/num_heads]
    """
    with tf.name_scope("split_heads"):
      batch_size = tf.shape(x)[0]
      length = tf.shape(x)[1]

      # Calculate depth of last dimension after it has been split.
      depth = (self.hidden_size // self.num_heads)

      # Split the last dimension
      x = tf.reshape(x, [batch_size, length, self.num_heads, depth])

      # Transpose the result
      return tf.transpose(x, [0, 2, 1, 3])

  def combine_heads(self, x):
    """Combine tensor that has been split.

    Args:
      x: A tensor [batch_size, num_heads, length, hidden_size/num_heads]

    Returns:
      A tensor with shape [batch_size, length, hidden_size]
    """
    with tf.name_scope("combine_heads"):
      batch_size = tf.shape(x)[0]
      length = tf.shape(x)[2]
      x = tf.transpose(x, [0, 2, 1, 3])  # --> [batch, length, num_heads, depth]
      return tf.reshape(x, [batch_size, length, self.hidden_size])

  def call(self, x, y, bias, cache=None):
    """Apply attention mechanism to x and y.

    Args:
      x: a tensor with shape [batch_size, length_x, hidden_size]
      y: a tensor with shape [batch_size, length_y, hidden_size]
      bias: attention bias that will be added to the result of the dot product.
      cache: (Used for decoding) dictionary with Tensors containing results of
        previous attentions. The dictionary must have the following keys/values:
            {"k": tensor with shape [batch_size, 0, key_channels],
             "v": tensor with shape [batch_size, 0, value_channels]}
    Return:
      Attention layer output with shape [batch_size, length_x, hidden_size]
    """
    # Linearly project the query (q), key (k) and value (v) using different
    # learned projections. This is in preparation of splitting them into
    # multiple heads. Multi-head attention uses multiple queries, keys, and
    # values rather than regular attention (which uses a single q, k, v).
    q = self.q_dense_layer(x)
    k = self.k_dense_layer(y)
    v = self.v_dense_layer(y)

    # Save k and v to cache
    if cache is not None:
      k = cache["k"] = tf.concat([cache["k"], k], axis=1)
      v = cache["v"] = tf.concat([cache["v"], v], axis=1)

    # Split q, k, v into heads.
    q = self.split_heads(q)
    k = self.split_heads(k)
    v = self.split_heads(v)

    # Scale q to prevent the dot product between q and k from growing too large.
    depth = (self.hidden_size // self.num_heads)
    q *= depth ** -0.5

    # Calculate dot product attention
    logits = tf.matmul(q, k, transpose_b=True)
    logits += bias
    weights = tf.nn.softmax(logits, name="attention_weights")
    if self.train:
      weights = tf.nn.dropout(weights, 1.0 - self.attention_dropout)
    output = tf.matmul(weights, v)

    # Recombine heads --> [batch_size, length, hidden_size]
    output = self.combine_heads(output)

    # Run the combined outputs through another linear projection layer.
    output = self.output_dense_layer(output)
    return output


class SelfAttention(Attention):
  """Multiheaded self-attention layer."""

  def call(self, x, bias, cache=None):
    return super(SelfAttention, self).call(x, x, bias, cache)


class FeedFowardNetwork(tf.layers.Layer):
  """Fully connected feedforward network."""

  def __init__(self, hidden_size, filter_size, relu_dropout, train):
    super(FeedFowardNetwork, self).__init__()
    self.hidden_size = hidden_size
    self.filter_size = filter_size
    self.relu_dropout = relu_dropout
    self.train = train

    self.filter_dense_layer = tf.layers.Dense(
        filter_size, use_bias=True, activation=tf.nn.relu, name="filter_layer")
    self.output_dense_layer = tf.layers.Dense(
        hidden_size, use_bias=True, name="output_layer")

  def call(self, x, padding=None):
    # Retrieve dynamically known shapes
    batch_size = tf.shape(x)[0]
    length = tf.shape(x)[1]

    if padding is not None:
      with tf.name_scope("remove_padding"):
        # Flatten padding to [batch_size*length]
        pad_mask = tf.reshape(padding, [-1])

        nonpad_ids = tf.to_int32(tf.where(pad_mask < 1e-9))

        # Reshape x to [batch_size*length, hidden_size] to remove padding
        x = tf.reshape(x, [-1, self.hidden_size])
        x = tf.gather_nd(x, indices=nonpad_ids)

        # Reshape x from 2 dimensions to 3 dimensions.
        x.set_shape([None, self.hidden_size])
        x = tf.expand_dims(x, axis=0)

    output = self.filter_dense_layer(x)
    if self.train:
      output = tf.nn.dropout(output, 1.0 - self.relu_dropout)
    output = self.output_dense_layer(output)

    if padding is not None:
      with tf.name_scope("re_add_padding"):
        output = tf.squeeze(output, axis=0)
        output = tf.scatter_nd(
            indices=nonpad_ids,
            updates=output,
            shape=[batch_size * length, self.hidden_size]
        )
        output = tf.reshape(output, [batch_size, length, self.hidden_size])
    return output


class LayerNormalization(tf.layers.Layer):
  """Applies layer normalization."""

  def __init__(self, hidden_size):
    super(LayerNormalization, self).__init__()
    self.hidden_size = hidden_size

  def build(self, _):
    self.scale = tf.get_variable("layer_norm_scale", [self.hidden_size],
                                 initializer=tf.ones_initializer())
    self.bias = tf.get_variable("layer_norm_bias", [self.hidden_size],
                                initializer=tf.zeros_initializer())
    self.built = True

  def call(self, x, epsilon=1e-6):
    mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
    variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
    norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
    return norm_x * self.scale + self.bias


class PrePostProcessingWrapper(object):
  """Wrapper class that applies layer pre-processing and post-processing."""

  def __init__(self, layer, params, train):
    self.layer = layer
    self.postprocess_dropout = params.layer_postprocess_dropout
    self.train = train

    # Create normalization layer
    self.layer_norm = LayerNormalization(params.hidden_size)

  def __call__(self, x, *args, **kwargs):
    # Preprocessing: apply layer normalization
    y = self.layer_norm(x)

    # Get layer output
    y = self.layer(y, *args, **kwargs)

    # Postprocessing: apply dropout and residual connection
    if self.train:
      y = tf.nn.dropout(y, 1 - self.postprocess_dropout)
    return x + y


class Encoder(tf.layers.Layer):
  """Transformer encoder stack."""

  def __init__(self, params, train):
    super(Encoder, self).__init__()
    self.layers = []
    for _ in range(params.num_hidden_layers):
      self_attention_layer = SelfAttention(params.hidden_size, params.num_heads,
                                           params.attention_dropout, train)
      feed_forward_network = FeedFowardNetwork(
          params.hidden_size, params.filter_size, params.relu_dropout, train)

      self.layers.append([
          PrePostProcessingWrapper(self_attention_layer, params, train),
          PrePostProcessingWrapper(feed_forward_network, params, train)])

    # Create final layer normalization layer.
    self.output_normalization = LayerNormalization(params.hidden_size)

  def call(self, encoder_inputs, attention_bias, inputs_padding):
    for n, layer in enumerate(self.layers):
      # Run inputs through the sublayers.
      self_attention_layer = layer[0]
      feed_forward_network = layer[1]

      with tf.variable_scope("layer_%d" % n):
        with tf.variable_scope("self_attention"):
          encoder_inputs = self_attention_layer(encoder_inputs, attention_bias)
        with tf.variable_scope("ffn"):
          encoder_inputs = feed_forward_network(encoder_inputs, inputs_padding)

    return self.output_normalization(encoder_inputs)


class Decoder(tf.layers.Layer):
  """Transformer decoder stack."""

  def __init__(self, params, train):
    super(Decoder, self).__init__()
    self.layers = []
    for _ in range(params.num_hidden_layers):
      self_attention_layer = SelfAttention(
          params.hidden_size, params.num_heads, params.attention_dropout, train)
      enc_dec_attention_layer = Attention(
          params.hidden_size, params.num_heads, params.attention_dropout, train)
      feed_forward_network = FeedFowardNetwork(
          params.hidden_size, params.filter_size, params.relu_dropout, train)

      self.layers.append([
          PrePostProcessingWrapper(self_attention_layer, params, train),
          PrePostProcessingWrapper(enc_dec_attention_layer, params, train),
          PrePostProcessingWrapper(feed_forward_network, params, train)])

    self.output_normalization = LayerNormalization(params.hidden_size)

  def call(self, decoder_inputs, encoder_outputs, decoder_self_attention_bias,
           attention_bias, cache=None):
    for n, layer in enumerate(self.layers):
      self_attention_layer = layer[0]
      enc_dec_attention_layer = layer[1]
      feed_forward_network = layer[2]
      # Run inputs through the sublayers.
      layer_name = "layer_%d" % n
      layer_cache = cache[layer_name] if cache is not None else None
      with tf.variable_scope(layer_name):
        with tf.variable_scope("self_attention"):
          decoder_inputs = self_attention_layer(
              decoder_inputs, decoder_self_attention_bias, cache=layer_cache)
        with tf.variable_scope("encdec_attention"):
          decoder_inputs = enc_dec_attention_layer(
              decoder_inputs, encoder_outputs, attention_bias)
        with tf.variable_scope("ffn"):
          decoder_inputs = feed_forward_network(decoder_inputs)

    return self.output_normalization(decoder_inputs)


def get_position_encoding(
    length, hidden_size, min_timescale=1.0, max_timescale=1.0e4):
  """Return positional encoding.

  Calculates the position encoding as a mix of sine and cosine functions with
  geometrically increasing wavelengths.
  Defined and formulized in Attention is All You Need, section 3.5.

  Args:
    length: Sequence length.
    hidden_size: Size of the
    min_timescale: Minimum scale that will be applied at each position
    max_timescale: Maximum scale that will be applied at each position

  Returns:
    Tensor with shape [length, hidden_size]
  """
  position = tf.to_float(tf.range(length))
  num_timescales = hidden_size // 2
  log_timescale_increment = (
      math.log(float(max_timescale) / float(min_timescale)) /
      (tf.to_float(num_timescales) - 1))
  inv_timescales = min_timescale * tf.exp(
      tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
  scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
  signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
  return signal


def prepare_encoder_inputs(inputs, hidden_size, dropout, train):
  """Preprocess inputs and calculate attention bias from the input padding."""
  with tf.name_scope("add_pos_encoding"):
    length = tf.shape(inputs)[1]
    inputs += get_position_encoding(length, hidden_size)

  if train:
    inputs = tf.nn.dropout(inputs, 1 - dropout)
  return inputs


def prepare_decoder_inputs(targets, hidden_size, dropout=0, train=False):
  """Preprocess targets and calculate the decoder's self attention bias."""
  # Shift targets to the right, and remove the last element
  with tf.name_scope("shift_targets"):
    targets = tf.pad(targets, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]

  with tf.name_scope("add_pos_encoding"):
    length = tf.shape(targets)[1]
    targets += get_position_encoding(length, hidden_size)

  if train:
    targets = tf.nn.dropout(targets, 1 - dropout)
  return targets, get_decoder_self_attention_bias(tf.shape(targets)[1])


def get_decoder_self_attention_bias(length):
  """Calculate bias for decoder that maintains model's autoregressive property.

  Creates a tensor that masks out locations that correspond to illegal
  connections, so prediction at position i cannot draw information from future
  positions.

  Args:
    length: int length of sequences in batch.

  Returns:
    float tensor of shape [1, 1, length, length]
  """
  with tf.name_scope("decoder_self_attention_bias"):
    valid_locs = tf.matrix_band_part(tf.ones([length, length]), -1, 0)
    valid_locs = tf.reshape(valid_locs, [1, 1, length, length])
    decoder_bias = -1e9 * (1.0 - valid_locs)
  return decoder_bias


def get_learning_rate(params):
  """Calculate learning rate with linear warmup and rsqrt decay."""
  with tf.name_scope("learning_rate"):
    warmup_steps = tf.to_float(params.learning_rate_warmup_steps)
    step = tf.to_float(tf.train.get_or_create_global_step())
    learning_rate = 2.0 * (params.hidden_size ** -0.5)
    # Apply linear warmup
    learning_rate *= tf.minimum(1.0, step / warmup_steps)
    # Apply rsqrt decay
    learning_rate *= tf.rsqrt(tf.maximum(step, warmup_steps))

    # Save learning rate value to TensorBoard summary.
    tf.summary.scalar("learning_rate", learning_rate)

    return learning_rate


def get_train_op(loss, params):
  learning_rate = get_learning_rate(params)
  with tf.variable_scope("get_train_op"):
    optimizer = tf.contrib.opt.LazyAdamOptimizer(
        learning_rate,
        beta1=params.optimizer_adam_beta1,
        beta2=params.optimizer_adam_beta2,
        epsilon=params.optimizer_adam_epsilon)

    global_step = tf.train.get_global_step()
    tvars = tf.trainable_variables()
    gradients = optimizer.compute_gradients(
        loss, tvars, colocate_gradients_with_ops=True)
    train_op = optimizer.apply_gradients(
        gradients, global_step=global_step, name="train")

    # Save gradient norm to Tensorboard
    tf.summary.scalar("global_norm/gradient_norm",
                      tf.global_norm(list(zip(*gradients))[0]))

    return train_op


def model_fn(features, labels, mode, params):
  """Train and evaluate transformer model."""
  with tf.variable_scope("model"):
    inputs, targets = features, labels

    # Create model and get output logits.
    model = Transformer(params, mode == tf.estimator.ModeKeys.TRAIN)
    output = model(inputs, targets)

    if mode == tf.estimator.ModeKeys.PREDICT:
      return tf.estimator.EstimatorSpec(
          tf.estimator.ModeKeys.PREDICT,
          predictions=output)

    logits = output
    # Calculate model loss.
    xentropy, weights = metrics.padded_cross_entropy_loss(
        logits, targets, params.label_smoothing, params.vocab_size)
    loss = tf.reduce_sum(xentropy * weights) / tf.reduce_sum(weights)

    # Save loss to TensorBoard summary.
    tf.identity(loss, name="loss")
    tf.summary.scalar("loss", loss)

    if mode == tf.estimator.ModeKeys.EVAL:
      return tf.estimator.EstimatorSpec(
          mode=mode, loss=loss, predictions={"predictions": logits},
          eval_metric_ops=metrics.get_eval_metrics(logits, labels, params))
    else:
      train_op = get_train_op(loss, params)
      return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)


def main(_):
  # Set logging level to INFO to display training progress (logged by the
  # estimator)
  tf.logging.set_verbosity(tf.logging.INFO)

  if FLAGS.params == "base":
    params = model_params.TransformerBaseParams
  elif FLAGS.params == "big":
    params = model_params.TransformerBigParams
  else:
    raise ValueError("Invalid parameter set defined: %s."
                     "Expected 'base' or 'big.'" % FLAGS.params)

  # Add flag-defined parameters to params object
  params.data_dir = FLAGS.data_dir
  params.num_cpu_cores = FLAGS.num_cpu_cores

  estimator = tf.estimator.Estimator(
      model_fn=model_fn, model_dir=FLAGS.model_dir, params=params)

  for _ in xrange(FLAGS.train_steps // FLAGS.steps_between_eval):
    estimator.train(dataset.train_input_fn, steps=FLAGS.steps_between_eval)
    print("evaluation results:", estimator.evaluate(dataset.eval_input_fn))


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--data_dir", "-dd", type=str, default="/tmp/translate_ende",
      help="[default: %(default)s] Directory for where the "
           "translate_ende_wmt32k dataset is saved.",
      metavar="<DD>")
  parser.add_argument(
      "--model_dir", "-md", type=str, default="/tmp/transformer_model",
      help="[default: %(default)s] Directory to save Transformer model "
           "training checkpoints",
      metavar="<MD>")
  parser.add_argument(
      "--params", "-p", type=str, default="big", choices=["base", "big"],
      help="[default: %(default)s] Parameter set to use when creating and "
           "training the model.",
      metavar="<P>")
  parser.add_argument(
      "--num_cpu_cores", "-nc", type=int, default=4,
      help="[default: %(default)s] Number of CPU cores to use in the input "
           "pipeline.",
      metavar="<NC>")

  # Flags for training with epochs (default).
  parser.add_argument(
      "--train_epochs", "-te", type=int, default=10,
      help="[default: %(default)s] The number of epochs used to train.",
      metavar="<TE>")
  parser.add_argument(
      "--epochs_between_eval", "-ebe", type=int, default=1,
      help="[default: %(default)s] The number of training epochs to run "
           "between evaluations.",
      metavar="<TE>")

  # Flags for training with steps (may be used for debugging)
  parser.add_argument(
      "--train_steps", "-ts", type=int, default=None,
      help="[default: %(default)s] Total number of training steps.",
      metavar="<TS>")
  parser.add_argument(
      "--steps_between_eval", "-sbe", type=int, default=1000,
      help="[default: %(default)s] Number of training steps to run between "
           "evaluations.",
      metavar="<SBE>",)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
