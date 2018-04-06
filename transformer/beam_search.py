"""Beam search implementation from Tensor2Tensor.

Source:
https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/beam_search.py
"""

import tensorflow as tf
from tensorflow.python.util import nest

# Default value for INF
INF = 1. * 1e7

_var_names = ['alive_seq', 'alive_log_probs', 'alive_cache', 'finished_seq',
              'finished_scores', 'finished_flags']

def _create_beam_search_state_vars(batch_size, beam_size, initial_ids,
    initial_cache):
  """Create tensors for storing solution states found in beam search.

  The following state variables are created:
    alive_seq: top beam_size sequences that haven't reached the EOS token.
      int32 tensor with shape that starts out as [batch_size, beam_size, 1]
    alive_log_probs: Log probabilities of each alive sequence.
      float32 tensor with shape [batch_size, beam_size]
    alive_cache: dict of decoding states for each alive sequence. Contains the
      encoder output, attention bias, and decoding attention values from the
      previous iteration.
    finished_seq: Current top beam_size finished sequences.
      int32 tensor with shape that starts off as [batch_size, beam_size, 1]
    finished_scores: Scores for each finished sequence.
      float32 tensor with shape [batch_size, beam_size]
    finished_flags: Boolean tensor indicating which of the finished sequences
      and scores are real and which are filler. (False->Filler)
      bool tensor with shape [batch_size, beam_size]

  Args:
    batch_size: int size of batch
    beam_size: int number of states to search in parallel
    initial_ids: Ids to start off the decoding. [batch_size]
    initial_cache: initial dict storing values used when decoding.

  Returns:
    Dictionary containing state variables and their shape invariants.
  """
  alive_seq = _expand_to_beam_size(initial_ids, beam_size)
  alive_seq = tf.expand_dims(alive_seq, axis=2)  # (batch_size, beam_size, 1)
  alive_seq_shape = tf.TensorShape([None, beam_size, None])

  # Assume initial_ids are prob 1.0
  initial_log_probs = tf.constant([[0.] + [-float("inf")] * (beam_size - 1)])
  alive_log_probs = tf.tile(initial_log_probs, [batch_size, 1])
  alive_log_probs_shape = tf.TensorShape([None, beam_size])

  def _get_shape_keeep_last_dim(tensor):
    shape_list = _shape_list(tensor)
    for i in range(len(shape_list) - 1):
      shape_list[i] = None
    return tf.TensorShape(shape_list)

  alive_cache = nest.map_structure(
      lambda t: _expand_to_beam_size(t, beam_size), initial_cache)
  alive_cache_shape = nest.map_structure(
      _get_shape_keeep_last_dim, alive_cache)

  finished_seq = tf.zeros(tf.shape(alive_seq), tf.int32)
  finished_seq_shape = tf.TensorShape([None, beam_size, None])

  # Setting the scores of the initial to negative infinity.
  finished_scores = tf.ones([batch_size, beam_size]) * -INF
  finished_scores_shape = tf.TensorShape([None, beam_size])

  finished_flags = tf.zeros([batch_size, beam_size], tf.bool)
  finished_flags_shape = tf.TensorShape([None, beam_size])

  vars = [alive_seq, alive_log_probs, alive_cache, finished_seq,
          finished_scores, finished_flags]
  shapes = [alive_seq_shape, alive_log_probs_shape, alive_cache_shape,
            finished_seq_shape, finished_scores_shape, finished_flags_shape]
  return vars, shapes


def _expand_to_beam_size(tensor, beam_size):
  """Tiles a given tensor by beam_size.

  Args:
    tensor: tensor to tile [batch_size, ...]
    beam_size: How much to tile the tensor by.

  Returns:
    Tiled tensor [batch_size, beam_size, ...]
  """
  tensor = tf.expand_dims(tensor, axis=1)
  tile_dims = [1] * tensor.shape.ndims
  tile_dims[1] = beam_size

  return tf.tile(tensor, tile_dims)

def _shape_list(tensor):
  """Return a list of the tensor's shape, and ensure no None values in list."""
  # Get statically known shape (may contain None's for unknown dimensions)
  shape = tensor.get_shape().as_list()

  # Ensure that the shape values are not None
  dynamic_shape = tf.shape(tensor)
  for i in range(len(shape)):
    if shape[i] is None:
      shape[i] = dynamic_shape[i]
  return shape

def _flatten_beam_dim(tensor):
  """Reshapes first two dimensions in to single dimension.

  Args:
    tensor: Tensor to reshape of shape [A, B, ...]

  Returns:
    Reshaped tensor of shape [A*B, ...]
  """
  shape = _shape_list(tensor)
  shape[0] *= shape[1]
  shape.pop(1)  # Remove beam dim
  return tf.reshape(tensor, shape)


def _unflatten_beam_dim(tensor, batch_size, beam_size):
  """Reshapes first dimension back to [batch_size, beam_size].

  Args:
    tensor: Tensor to reshape of shape [batch_size*beam_size, ...]
    batch_size: Tensor, original batch size.
    beam_size: int, original beam size.

  Returns:
    Reshaped tensor of shape [batch_size, beam_size, ...]
  """
  shape = _shape_list(tensor)
  new_shape = [batch_size, beam_size] + shape[1:]
  return tf.reshape(tensor, new_shape)


def _log_prob_from_logits(logits):
  return logits - tf.reduce_logsumexp(logits, axis=2, keep_dims=True)


def _gather_beams(nested, beam_indices, batch_size, new_beam_size):
  """Gather beams from nested structure of tensors.

  Each tensor in nested represents a batch of beams, where beam refers to a
  single search state (beam search involves searching through multiple states
  in parallel).

  This function is used to gather the top beams, specified by
  beam_indices, from the nested tensors.

  Args:
    nested: Nested structure (tensor, list, tuple or dict) containing tensors
      with shape [batch_size, beam_size, ...].
    beam_indices: int32 tensor with shape [batch_size, new_beam_size]. Each
     value in beam_indices must be between [0, beam_size), and are not
     necessarily unique.
    batch_size: int size of batch
    new_beam_size: int number of beams to be pulled from the nested tensors.

  Returns:
    Nested structure containing tensors with shape
      [batch_size, new_beam_size, ...]
  """
  # Computes the i'th coodinate that contains the batch index for gather_nd.
  # Batch pos is a tensor like [[0,0,0,0,],[1,1,1,1],..].
  batch_pos = tf.range(batch_size * new_beam_size) // new_beam_size
  batch_pos = tf.reshape(batch_pos, [batch_size, new_beam_size])

  # Create coordinates to be passed to tf.gather_nd. Stacking creates a tensor
  # with shape [batch_size, beam_size, 2], where the last dimension contains
  # the (i, j) gathering coordinates.
  coordinates = tf.stack([batch_pos, beam_indices], axis=2)

  return nest.map_structure(
      lambda state: tf.gather_nd(state, coordinates), nested)

def _gather_topk_beams(nested, score_or_log_prob, batch_size, beam_size):
  """Gather top beams from nested structure."""
  _, topk_indexes = tf.nn.top_k(score_or_log_prob, k=beam_size)
  return _gather_beams(nested, topk_indexes, batch_size, beam_size)


def _grow_and_keep_topk_seq(symbols_to_logits_fn, alive_seq, alive_log_probs,
    alive_cache, vocab_size, batch_size, beam_size, i, k):
  """Grow alive sequences and return the top k sequences.

  Args:
    symbols_to_logits_fn: A function that takes in ids, index, and cache as
      arguments. The passed in arguments will have shape:
        ids -> [batch_size * beam_size, index]
        index -> [] (scalar)
        cache -> nested dictionary of tensors [batch_size * beam_size, ...]
      The function must return logits and new cache.
        logits -> [batch * beam_size, vocab_size]
        new cache -> same shape/structure as inputted cache
    alive_seq: Current alive sequence
      float32 tensor with shape [batch_size, beam_size, i]
    alive_log_probs: Log probabilities of alive seqs [batch_size, beam_size]
    alive_cache: Decoder state dictionary
    vocab_size: int size of vocabulary
    batch_size: int size of current batch
    beam_size: int number of beams
    i: Current loop index, which is the number of tokens that have been decoded
    k: Number of sequences to keep.

  Returns:
    Tuple of
      (Top k sequences [batch_size, beam_size, i + 1],
       Scores of returned sequences [batch_size, beam_size],
       New alive cache, which has the same shape and structure as alive_cache)
  """

  # Get logits for the next candidate IDs for the alive sequences. Get the new
  # cache values at the same time.
  flat_ids = _flatten_beam_dim(alive_seq)  # [batch_size * beam_size
  flat_cache = nest.map_structure(_flatten_beam_dim, alive_cache)
  #print('flat_cache', flat_cache, alive_cache)
  flat_ids = tf.Print(flat_ids, [flat_ids, alive_seq, tf.shape(alive_seq)], 'flat ids ', summarize=10)
  flat_logits, flat_cache = symbols_to_logits_fn(flat_ids, i, flat_cache)
  #print('flat_cache2', flat_cache, alive_cache)
  #print('flat_logits', flat_logits)
  #print('flat_ids', flat_ids, alive_seq)
  # Unflatten logits to shape [batch_size, beam_size, vocab_size]
  logits = _unflatten_beam_dim(flat_logits, batch_size, beam_size)
  new_cache = nest.map_structure(
      lambda t: _unflatten_beam_dim(t, batch_size, beam_size), flat_cache)
  #print('new_cache_1', new_cache)
  logits = tf.Print(logits, [tf.shape(logits), tf.shape(flat_logits)], 'final logits: ', summarize=10)
  # Convert logits to normalized log probs
  candidate_log_probs = _log_prob_from_logits(logits)

  # Calculate new log probabilities if each of the alive sequences were extended
  # by the the candidate IDs. Shape [batch_size, beam_size, vocab_size]
  log_probs = candidate_log_probs + tf.expand_dims(alive_log_probs, axis=2)

  # Each batch item has beam_size * vocab_size candidate sequences. For each
  # batch item, get the k candidates with the highest log probabilities.
  flat_log_probs = tf.reshape(log_probs, [-1, beam_size * vocab_size])
  topk_log_probs, topk_indices = tf.nn.top_k(flat_log_probs, k=k)

  # Use the indices to get the original beam indices and ids.
  # Both tensors have shape [batch_size, k]
  topk_beam_indices = topk_indices // vocab_size
  topk_ids = topk_indices % vocab_size

  topk_seq, new_cache = _gather_beams(
      [alive_seq, new_cache], topk_beam_indices, batch_size, k)

  # Append the most probable IDs to the topk sequences
  topk_ids = tf.expand_dims(topk_ids, axis=2)
  topk_seq = tf.concat([topk_seq, topk_ids], axis=2)
  #print('returned_cache1',new_cache)
  return topk_seq, topk_log_probs, new_cache


def _collect_top_alive_seq(new_seq, new_log_probs, new_cache, eos_id,
    batch_size, beam_size):
  """Gather the top k sequences that are still alive.

  Args:
    new_seq: New sequences generated by growing the current alive sequences
      int32 tensor with shape [batch_size, beam_size, i + 1]
    new_log_probs: Log probabilities of new sequences
      float32 tensor with shape [batch_size, beam_size]
    new_cache: Dict of decoder cached states
    eos_id: int id of EOS token
    batch_size: int size of batch
    beam_size: int number of beams

  Returns:
    Tuple of
      (Top beam_size sequences that are still alive (don't end with eos_id)
       Log probabilities of top alive sequences
       Dict cache storing decoder states for top alive sequences)
  """
  # To prevent finished sequences from being considered, set log probs to -INF
  new_finished_flags = tf.equal(new_seq[:, :, -1], eos_id)
  new_log_probs += tf.to_float(new_finished_flags) * -INF
  #print('collect_alive:', 'new_cache', new_cache)
  return _gather_topk_beams(
      [new_seq, new_log_probs, new_cache], new_log_probs, batch_size, beam_size)


def _collect_top_finished_seq(
    finished_seq, finished_scores, finished_flags, new_seq, new_log_probs,
    eos_id, batch_size, beam_size, alpha, i):
  """Combine new and old finished sequences, and gather the top k sequences.

  Args:
    finished_seq: Current finished sequences
      int32 tensor with shape [batch_size, beam_size, i]
    finished_scores: scores for each finished sequence
      float32 tensor with shape [batch_size, beam_size]
    finished_flags: booleans indicating which sequences and scores in the
      finished_seq and finished_scores are finished or filler
      bool tensor with shape [batch_szie, beam_size]
    new_seq: New sequences generated by growing the current alive sequences
      int32 tensor with shape [batch_size, beam_size, i + 1]
    new_log_probs: Log probabilities of new sequences
      float32 tensor with shape [batch_size, beam_size]
    eos_id: int id of EOS token
    batch_size: int size of batch
    beam_size: int number of beams
    alpha: float defining the strength of length normalization
    i: loop index

  Returns:
    Tuple of
      (Top beam_size finished sequences based on score,
       Scores of finished sequences,
       Finished flags of finished sequences)
  """
  # First append a column of 0'ids to finished to make the same length with
  # finished scores. New shape of finished_seq: [batch_size, beam_size, i + 1]
  finished_seq = tf.concat(
      [finished_seq, tf.zeros([batch_size, beam_size, 1], tf.int32)], axis=2)

  # Calculate new seq scores from log probabilities
  length_norm = _length_normalization(alpha, i + 1)
  #print('new_log_probs',new_log_probs)
  new_scores = new_log_probs / length_norm

  # Set the scores of the still-alive seq in new_seq to large negative values
  new_finished_flags = tf.equal(new_seq[:, :, -1], eos_id)
  new_scores += (1. - tf.to_float(new_finished_flags)) * -INF

  # Combine sequences, scores, and flags
  finished_seq = tf.concat([finished_seq, new_seq], axis=1)
  finished_scores = tf.concat([finished_scores, new_scores], axis=1)
  finished_flags = tf.concat([finished_flags, new_finished_flags], axis=1)

  return _gather_topk_beams(
      [finished_seq, finished_scores, finished_flags], finished_scores,
      batch_size, beam_size)


def _length_normalization(alpha, sequence_length):
  """Return length normalization factor."""
  return tf.pow(((5. + tf.to_float(sequence_length)) / 6.), alpha)


def sequence_beam_search(
    symbols_to_logits_fn, initial_ids, initial_cache, vocab_size, beam_size,
    alpha, max_decode_length, eos_id):
  """DErp derp.

  Args:
    symbols_to_logits_fn: A function that takes in ids, index, and cache as
      arguments. The passed in arguments will have shape:
        ids -> [batch_size * beam_size, index]
        index -> [] (scalar)
        cache -> nested dictionary of tensors [batch_size * beam_size, ...]
      The function must return logits and new cache.
        logits -> [batch * beam_size, vocab_size]
        new cache -> same shape/structure as inputted cache
    initial_ids: Starting ids for each batch item.
      int32 tensor with shape [batch_size]
    initial_cache: dict containing starting decoder variables information
    vocab_size: int size of tokens
    beam_size: int number of beams
    alpha: float defining the strength of length normalization
    max_decode_length: maximum length to decoded sequence
    eos_id: int id of eos token, used to determine when a sequence has finished

  Returns:
    Top decoded sequences [batch_size, beam_size, max_decode_length]
    sequence scores [batch_size, beam_size]
  """
  def continue_loop_condition(
      i, unused_alive_seq, alive_log_probs, unused_alive_cache,
      unused_finished_seq, finished_scores, finished_flags):
    """Return whether to continue the loop.

    The loops should terminate when
      1) when decode length has been reached, or
      2) when the worst score in the finished sequences is better than the best
         score in the alive sequences (i.e. the finished sequences are provably
         unchanging)

    Args:
      i: loop index
      alive_log_probs: Log probabilities of alive sequences.
        float32 tensor with shape [batch_size, beam_size]
      finished_scores: Scores for each finished finished sequences.
        float32 tensor with shape [batch_size, beam_size]
      finished_flags: tensor indicating which elements in the finished_scores
        are scores, and which are filler. True->score, False->filler
        (Because tensors have a fixed shape, finished_scores starts with filler
        scores that get replaced the actual scores as finished seqs are found.)
        bool tensor with shape [batch_size, beam_size]

    Returns:
      Bool tensor with value True if loop should continue, False if loop should
      terminate.
    """
    not_at_max_decode_length = tf.less(i, max_decode_length)

    # Calculate largest length penalty (the larger penalty, the better score).
    max_length_normalization = _length_normalization(alpha, max_decode_length)
    # Get the best possible scores from alive sequences.
    best_alive_scores = alive_log_probs[:, 0] / max_length_normalization

    # Compute worst score in finished sequences for each batch element
    #print('finished_scores', finished_scores)
    finished_scores *= tf.to_float(finished_flags)  # set filler scores to zero
    lowest_finished_scores = tf.reduce_min(finished_scores, axis=1)

    # If there are no finished sequences in a batch element, then set the lowest
    # finished score to -INF for that element.
    finished_batches = tf.reduce_any(finished_flags, 1)
    lowest_finished_scores += (1. - tf.to_float(finished_batches)) * -INF

    worst_finished_score_better_than_best_alive_score = tf.reduce_all(
        tf.greater(lowest_finished_scores, best_alive_scores)
    )

    return tf.logical_and(
        not_at_max_decode_length,
        tf.logical_not(worst_finished_score_better_than_best_alive_score)
    )

  def loop_body(i, alive_seq, alive_log_probs, alive_cache, finished_seq,
      finished_scores, finished_flags):
    """Beam search loop body.

    Grow alive sequences a single ID. Sequences that have reached the EOS token
    are marked as finished. The alive and finished sequences with the highest
    log probabilities and scores are returned.

    A sequence's finished score is calculating by dividing the log probability
    by the length normalization factor. Without length normalization, the
    search is more likely to return shorter sequences.

    Args:
      i: loop index
      alive_seq: top beam_size sequences that haven't reached the EOS token.
        int32 tensor with shape [batch_size, beam_size, i + 1]
      alive_log_probs: Log probabilities of each alive sequence.
        float32 tensor with shape [batch_size, beam_size]
      alive_cache: dict of decoding states for each alive sequence. Contains the
        encoder output, attention bias, and decoding attention values from the
        previous iteration.
      finished_seq: Current top beam_size finished sequences.
        int32 tensor with shape [batch_size, beam_size, i + 1]
      finished_scores: Scores for each finished sequence.
        float32 tensor with shape [batch_size, beam_size]
      finished_flags: Boolean tensor indicating which of the finished sequences
        and scores are real and which are filler. (False->Filler)
        bool tensor with shape [batch_size, beam_size]

    Returns:
      Values for the next loop iteration. Tuple of
        (Incremented loop index,
         New alive sequences,
         New alive sequence log probabilities,
         New cached states for the alive sequences,
         New finished sequences,
         Scores for new finished sequences,
         Flags for finished sequences and scores)
    """
    #print("="*100)
    # Grow alive sequences by a single ,and collect top 2*beam_size sequences
    # Collect 2*beam_size sequences because some sequences may have reached the
    # EOS token, and 2*beam_size ensures that at least beam_size sequences are
    # still alive.
    new_seq, new_log_probs, new_cache = _grow_and_keep_topk_seq(
        symbols_to_logits_fn, alive_seq, alive_log_probs, alive_cache,
        vocab_size, batch_size, beam_size, i,  k=2 * beam_size)
    #print('original_cache', alive_cache)
    #print('new_cache', new_cache)
    # Collect top beam_size alive sequences
    alive_seq, alive_log_prob, alive_cache = _collect_top_alive_seq(
        new_seq, new_log_probs, new_cache, eos_id, batch_size, beam_size)
    #print('alive_cache', alive_cache)
    # Combine newly finished sequences with existing finished sequences, and
    # collect the top k scoring sequences.
    finished_seq, finished_scores, finished_flags = _collect_top_finished_seq(
        finished_seq, finished_scores, finished_flags, new_seq, new_log_probs,
        eos_id, batch_size, beam_size, alpha, i)

    return (i + 1, alive_seq, alive_log_prob, alive_cache, finished_seq,
            finished_scores, finished_flags)

  batch_size = tf.shape(initial_ids)[0]
  state_vars, state_shapes = _create_beam_search_state_vars(
      batch_size, beam_size, initial_ids, initial_cache)


  for n,(i,j) in enumerate(zip(state_vars, state_shapes)):
    print(_var_names[n])
    print(i)
    print(j)
    print()

  #print(state_vars)
  #print(state_shapes)

  (_, alive_seq, alive_log_probs, _, finished_seq, finished_scores,
   finished_flags) = tf.while_loop(
       continue_loop_condition, loop_body,
       loop_vars=[tf.constant(0)] + state_vars,
       shape_invariants=[tf.TensorShape([])] + state_shapes,
       parallel_iterations=1,
       back_prop=False)

  # Account for corner case where there are no finished sequences for a
  # particular batch item. In that case, return alive sequences for that batch
  # item.
  finished_seq = tf.where(
      tf.reduce_any(finished_flags, 1), finished_seq, alive_seq)
  finished_scores = tf.where(
      tf.reduce_any(finished_flags, 1), finished_scores, alive_log_probs)

  return finished_seq, finished_scores
