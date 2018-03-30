from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import six

import tensorflow as tf

_ESCAPE_CHARS = set(u'\\_u;0123456789')

# min_count is the minimum number of times a subtoken must appear in the data
# before before it is added to the vocabulary. min_count directly affects the
# number of subtokens in the vocabulary. min_count is determined using binary
# search to obtain the target vocabulary size.
_MIN_MIN_COUNT = 1     # min value to use when binary searching for min_count
_MAX_MIN_COUNT = 1000  # max value to use when binary searching for min_count

_TARGET_VOCAB_SIZE = 32768
_TARGET_THRESHOLD = 327  # Allow vocab size within this threshold

VOCAB_FILE = 'vocab.ende.%d' % _TARGET_VOCAB_SIZE


class Subtokenizer(object):
  def __init__(self, vocab_file):
    pass

  @staticmethod
  def init_from_data(raw_files, vocab_file):
    """Create a Subtokenizer based on the data files, and save a vocab file."""
    pass

  def encode(self):
    pass

  def decode(self):
    pass




def get_or_generate_vocab_dict():
  """Create vocab dictionary and save a file"""
  # Read files and count each word. (TODO)
  token_counts = {}

  alphabet = generate_alphabet_dict(token_counts)
  subtoken_dict = generate_subtokens_with_target_size(token_counts, alphabet)

  return subtoken_dict


def _list_to_index_dict(lst):
  """Create dictionary mapping list items to their indices in the list."""
  return {item: n for n, item in enumerate(lst)}


def _escape_token(token, alphabet):
  """Remove characters that aren't in the alphabet and append '_' to token."""
  token = token.replace(u'\\', u'\\\\').replace(u'_', u'\\u')
  ret = [c if c in alphabet and c != u'\n' else r'\%d;' % ord(c) for c in token]
  return u''.join(ret) + '_'


def _split_token_to_subtokens(token, subtoken_dict, max_subtoken_length):
  return []  # TODO


def generate_subtokens_with_target_size(token_counts, alphabet):
  """TODO"""
  target = _TARGET_VOCAB_SIZE

  def bisect(min_val, max_val):
    """TODO"""
    cur_count = (min_val + max_val) // 2
    tf.logging.info('Binary search: trying min_count=%d' % cur_count)
    subtoken_dict = generate_subtokens(
        token_counts, alphabet, cur_count)

    val = len(subtoken_dict)
    within_threshold = abs(val - target) < _TARGET_THRESHOLD
    if within_threshold or min_val >= max_val or cur_count < 2:
      return subtoken_dict, alphabet
    if val > target:
      other_subtoken_dict = bisect(cur_count + 1, max_val)
    else:
      other_subtoken_dict = bisect(min_val, cur_count - 1)

    # Return vocabulary dictionary with the closest number of tokens.
    other_val = len(other_subtoken_dict)
    if abs(other_val - target) < abs(val - target):
      return other_subtoken_dict
    return subtoken_dict

  return bisect(_MIN_MIN_COUNT, _MAX_MIN_COUNT)


def generate_alphabet_dict(token_counts):
  """TODO"""
  # Create set of characters that can appear in any token.
  alphabet = {c for token in token_counts for c in token}
  alphabet |= _ESCAPE_CHARS  # Add escape characters to alphabet set.

  return alphabet


def generate_subtokens(token_counts, alphabet, min_count, num_iterations=4):
  """Create a dictionary of subword tokens mapping strings to ints"""

  # Build list of subword tokens

  # Use alphabet set to create initial dictionary of subword token->id
  max_subtoken_length = 1
  subtoken_dict = _list_to_index_dict(alphabet)

  # On each iteration, segment all words using the subtokens defined in
  # subtoken_dict, count how often the resulting subtokens appear, and update
  # the dictionary with subtokens w/ high enough counts.
  for i in xrange(num_iterations):
    tf.logging.info('Generating subtokens: iteration %d' % i)

    subtoken_counts = collections.defaultdict(int)
    for token, count in token_counts:
      token = _escape_token(token, alphabet)
      subtokens = _split_token_to_subtokens(
          token, subtoken_dict, max_subtoken_length)

      start = 0
      for subtoken in subtokens:
        for end in xrange(start + 1, len(token) + 1):
          new_subtoken = token[start:end]
          subtoken_counts[new_subtoken] += count
        start += len(subtoken)

    # Create list where the element at index i is a set of subtokens. Each
    # subtoken has length i.
    len_to_subtokens = []
    for subtoken, count in six.iteritems(subtoken_counts):
      if count < min_count:  # Filter out subtokens that don't appear enough
        continue
      while len(len_to_subtokens) <= len(subtoken):
        len_to_subtokens.append(set())
      len_to_subtokens[len(subtoken)].add(subtoken)

    # Get new candidate subtokens
    new_subtokens = []
    # Go through the list in reverse order to consider longer subtokens first.
    for subtoken_len in xrange(len(len_to_subtokens) - 1, 0, -1):
      for subtoken in len_to_subtokens[subtoken_len]:
        count = subtoken_counts[subtoken]

        if count < min_count:  # Possibly true if this subtoken is a prefix of
          # another subtoken.
          continue

        # Ignore alphabet tokens, which will be added manually later.
        if subtoken not in alphabet:
          new_subtokens.append((count, subtoken))
        # Decrement count of the subtoken's prefixes (if a longer subtoken is
        # added, its prefixes lose priority to be added).
        for end in xrange(1, subtoken_len):
          subtoken_counts[subtoken[:end]] -= count

    # Add alphabet subtokens (guarantees that all strings are encodable)
    new_subtokens.extend((subtoken_counts.get(a, 0), a) for a in alphabet)
    new_subtokens = [t for _, t in sorted(new_subtokens, reverse=True)]

    # Generate new subtoken->id dictionary using the new subtoken list.
    subtoken_dict = _list_to_index_dict(new_subtokens)
  return subtoken_dict, alphabet











