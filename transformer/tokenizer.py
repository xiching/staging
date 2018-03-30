"""Defines Subtokenizer class to encode and decode strings."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import re
import sys
import unicodedata

import six
import tensorflow as tf

PAD = '<pad>'
PAD_ID = 0
EOS = '<EOS>'
EOS_ID = 1
RESERVED_TOKENS = [PAD, EOS]

_ESCAPE_CHARS = set(u'\\_u;0123456789')
_UNESCAPE_REGEX = re.compile(r'\\u|\\\\|\\([0-9]+);')

# Set contains all letter and number characters.
_ALPHANUMERIC_CHAR_SET = set(
    six.unichr(i) for i in xrange(sys.maxunicode)
    if (unicodedata.category(six.unichr(i)).startswith('L') or
        unicodedata.category(six.unichr(i)).startswith('N')))

# min_count is the minimum number of times a subtoken must appear in the data
# before before it is added to the vocabulary. The value is found using binary
# search to obtain the target vocabulary size.
_MIN_MIN_COUNT = 1     # min value to use when binary searching for min_count
_MAX_MIN_COUNT = 1000  # max value to use when binary searching for min_count

_TARGET_VOCAB_SIZE = 32768
_TARGET_THRESHOLD = 327  # Accept vocabulary if size is within this threshold

VOCAB_FILE = 'vocab.ende.%d' % _TARGET_VOCAB_SIZE


class Subtokenizer(object):
  """Encodes and decodes strings to/from integer IDs."""

  def __init__(self, vocab_file):
    """Initializes class, creating a vocab file if data_files is provided."""
    tf.logging.info('Initializing Subtokenizer from file %s.' % vocab_file)

    self.subtoken_list = _load_vocab_file(vocab_file)
    self.alphabet = _generate_alphabet_dict(self.subtoken_list)
    self.subtoken_to_id_dict = _list_to_index_dict(self.subtoken_list)

    self.max_subtoken_length = 0
    for subtoken in self.subtoken_list:
      self.max_subtoken_length = max(self.max_subtoken_length, len(subtoken))

    # Create cache to speed up subtokenization
    self._cache_size = 2 ** 20
    self._cache = [(None, None)] * self._cache_size

  @staticmethod
  def init_from_files(vocab_file, files, file_byte_limit=1e6):
    """Create subtoken vocabulary based on files, and save vocab to file.

    Args:
      vocab_file: String name of vocab file to store subtoken vocabulary.
      files: List of file paths that will be used to generate vocabulary.
      file_byte_limit: (Default 1e6) Maximum number of bytes of sample text that
        will be drawn from the files.

    Returns:
      Subtokenizer object
    """

    if tf.gfile.Exists(vocab_file):
      tf.logging.info('Vocab file already exists (%s)' % vocab_file)
    else:
      tf.logging.info('Begin steps to create subtoken vocabulary...')
      token_counts = _count_tokens(files, file_byte_limit)
      alphabet = _generate_alphabet_dict(token_counts)
      subtoken_list = _generate_subtokens_with_target_vocab_size(
          token_counts, alphabet)
      tf.logging.info('Generated vocabulary with %d subtokens.' %
                      len(subtoken_list))
      _save_vocab_file(vocab_file, subtoken_list)
    return Subtokenizer(vocab_file)

  def encode(self, raw_string, add_eos=False):
    """Encodes a string into a list of int subtoken ids."""
    ret = []
    tokens = _split_string_to_tokens(_native_to_unicode(raw_string))
    for token in tokens:
      ret.extend(self._token_to_subtoken_ids(token))
    if add_eos:
      ret.append(EOS_ID)
    return ret

  def _token_to_subtoken_ids(self, token):
    """Encode a single token into a list of subtoken ids."""
    cache_location = hash(token) % self._cache_size
    cache_key, cache_value = self._cache[cache_location]
    if cache_key == token:
      return cache_value

    ret = _split_token_to_subtokens(
        _escape_token(token, self.alphabet), self.subtoken_to_id_dict,
        self.max_subtoken_length)
    ret = [self.subtoken_to_id_dict[subtoken_id] for subtoken_id in ret]

    self._cache[cache_location] = (token, ret)
    return ret

  def decode(self, subtokens):
    return _unicode_to_native(
        _join_tokens_to_string(self._subtoken_ids_to_tokens(subtokens)))

  def _subtoken_ids_to_tokens(self, subtokens):
    """Convert list of int subtoken ids to a list of string tokens."""
    escaped_tokens = ''.join([
        self.subtoken_list[s] if s <= len(self.subtoken_list) else ''
        for s in subtokens])
    escaped_tokens = escaped_tokens.split('_')

    ret = []
    for token in escaped_tokens:
      if token:
        ret.append(_unescape_token(token))
    return ret


def _save_vocab_file(vocab_file, subtoken_list):
  """Save subtokens to file."""
  with tf.gfile.Open(vocab_file, mode='w') as f:
    for subtoken in subtoken_list:
      f.write('\'%s\'\n' % _unicode_to_native(subtoken))


def _load_vocab_file(vocab_file):
  """Load vocabulary while ensuring reserved tokens are at the top."""
  subtoken_list = []
  with tf.gfile.Open(vocab_file, mode='r') as f:
    for line in f:
      subtoken = _native_to_unicode(line.strip())
      subtoken = subtoken[1:-1]  # Remove surrounding single-quotes
      if subtoken in RESERVED_TOKENS:
        continue
      subtoken_list.append(_native_to_unicode(subtoken))
  return RESERVED_TOKENS + subtoken_list


# Conversion between Unicode and UTF-8, if required (on Python2).
if six.PY2:

  def _native_to_unicode(s):
    return s if isinstance(s, unicode) else s.decode('utf-8')

  def _unicode_to_native(s):
    return s.encode('utf-8') if isinstance(s, unicode) else s

else:  # No conversion required on Python >= 3.

  def _native_to_unicode(s):
    return s

  def _unicode_to_native(s):
    return s


def _split_string_to_tokens(text):
  """Splits text to a list of string tokens."""
  if not text:
    return []
  ret = []
  token_start = 0
  # Classify each character in the input string
  is_alnum = [c in _ALPHANUMERIC_CHAR_SET for c in text]
  for pos in xrange(1, len(text)):
    if is_alnum[pos] != is_alnum[pos - 1]:
      token = text[token_start:pos]
      if token != u' ' or token_start == 0:
        ret.append(token)
      token_start = pos
  final_token = text[token_start:]
  ret.append(final_token)
  return ret


def _join_tokens_to_string(tokens):
  """Join a list of string tokens into a single string."""
  token_is_alnum = [t[0] in _ALPHANUMERIC_CHAR_SET for t in tokens]
  ret = []
  for i, token in enumerate(tokens):
    if i > 0 and token_is_alnum[i - 1] and token_is_alnum[i]:
      ret.append(u' ')
    ret.append(token)
  return ''.join(ret)


def _escape_token(token, alphabet):
  """Remove characters that aren't in the alphabet and append '_' to token."""
  token = token.replace(u'\\', u'\\\\').replace(u'_', u'\\u')
  ret = [c if c in alphabet and c != u'\n' else r'\%d;' % ord(c) for c in token]
  return u''.join(ret) + '_'


def _unescape_token(token):
  """Inverse of _escape_token()."""

  def match(m):
    if m.group(1) is None:
      return u'_' if m.group(0) == u'\\u' else u'\\'

    try:
      return six.unichr(int(m.group(1)))
    except (ValueError, OverflowError) as _:
      return u'\u3013'  # Unicode for undefined character.

  return _UNESCAPE_REGEX.sub(match, token)


def _count_tokens(files, file_byte_limit=1e6):
  """Return token counts of words in the files.

  Samples file_byte_limit bytes from each file, and counts the words that appear
  in the samples. The samples are semi-evenly distributed across the file.

  Args:
    files: List of filepaths
    file_byte_limit: Max number of bytes that will be read from each file.

  Returns:
    Dictionary mapping tokens to the number of times they appear in the sampled
    lines from the files.
  """
  token_counts = collections.defaultdict(int)

  for filepath in files:
    with tf.gfile.Open(filepath, mode='r') as reader:
      file_byte_budget = file_byte_limit
      counter = 0
      lines_to_skip = int(reader.size() / (file_byte_budget * 2))
      for line in reader:
        if counter < lines_to_skip:
          counter += 1
        else:
          if file_byte_budget < 0:
            break
          line = line.strip()
          file_byte_budget -= len(line)
          counter = 0

          # Add words to token counts
          for token in _split_string_to_tokens(_native_to_unicode(line)):
            token_counts[token] += 1
  return token_counts


def _list_to_index_dict(lst):
  """Create dictionary mapping list items to their indices in the list."""
  return {item: n for n, item in enumerate(lst)}


def _split_token_to_subtokens(token, subtoken_dict, max_subtoken_length):
  """Splits a token into subtokens defined in the subtoken dict."""
  ret = []
  start = 0
  token_len = len(token)
  while start < token_len:
    for end in xrange(min(token_len, start + max_subtoken_length), start, -1):
      subtoken = token[start:end]
      if subtoken in subtoken_dict:
        ret.append(subtoken)
        start = end
        break
    else:  # Did not break
      # If there is no possible encoding of the escaped token then one of the
      # characters in the token is not in the alphabet. This should be
      # impossible and would be indicative of a bug.
      assert False, 'Was unable to split token \"%s\" into subtokens.' % token

  return ret


def _generate_subtokens_with_target_vocab_size(token_counts, alphabet):
  """Generate subtoken vocabulary close to the target size."""
  target = _TARGET_VOCAB_SIZE
  tf.logging.info('Finding best min_count to get target size of %d' % target)

  def bisect(min_val, max_val):
    """Recursive function to binary search for subtoken vocabulary."""
    cur_count = (min_val + max_val) // 2
    tf.logging.info('Binary search: trying min_count=%d (%d %d)' %
                    (cur_count, min_val, max_val))
    subtoken_list = _generate_subtokens(token_counts, alphabet, cur_count)

    val = len(subtoken_list)
    tf.logging.info('Binary search: min_count=%d resulted in %d tokens' %
                    (cur_count, val))

    within_threshold = abs(val - target) < _TARGET_THRESHOLD
    if within_threshold or min_val >= max_val or cur_count < 2:
      return subtoken_list
    if val > target:
      other_subtoken_list = bisect(cur_count + 1, max_val)
    else:
      other_subtoken_list = bisect(min_val, cur_count - 1)

    # Return vocabulary dictionary with the closest number of tokens.
    other_val = len(other_subtoken_list)
    if abs(other_val - target) < abs(val - target):
      return other_subtoken_list
    return subtoken_list

  return bisect(_MIN_MIN_COUNT, _MAX_MIN_COUNT)


def _generate_alphabet_dict(iterable):
  """Create set of characters that can appear in any element in the iterable."""
  alphabet = {c for token in iterable for c in token}
  alphabet |= {c for token in RESERVED_TOKENS for c in token}
  alphabet |= _ESCAPE_CHARS  # Add escape characters to alphabet set.
  return alphabet


def _generate_subtokens(token_counts, alphabet, min_count, num_iterations=4):
  """Create a list of subtokens in decreasing order of frequency."""
  # Use alphabet set to create initial list of subtokens
  max_subtoken_length = 1
  subtoken_list = RESERVED_TOKENS + list(alphabet)
  subtoken_dict = _list_to_index_dict(subtoken_list)

  # On each iteration, segment all words using the subtokens defined in
  # subtoken_dict, count how often the resulting subtokens appear, and update
  # the dictionary with subtokens w/ high enough counts.
  for i in xrange(num_iterations):
    tf.logging.info('\tGenerating subtokens: iteration %d' % i)

    subtoken_counts = collections.defaultdict(int)
    for token, count in six.iteritems(token_counts):
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

    # Get new candidate subtokens and reset max_subtoken_length
    new_subtokens = []
    max_subtoken_length = 1

    # Go through the list in reverse order to consider longer subtokens first.
    for subtoken_len in xrange(len(len_to_subtokens) - 1, 0, -1):
      for subtoken in len_to_subtokens[subtoken_len]:
        count = subtoken_counts[subtoken]

        if count < min_count:  # Possibly true if this subtoken is a prefix of
          # another subtoken.
          continue

        # Ignore alphabet/reserved tokens, which will be added manually later.
        if subtoken not in alphabet and subtoken not in RESERVED_TOKENS:
          new_subtokens.append((count, subtoken))
          max_subtoken_length = max(max_subtoken_length, len(subtoken))

        # Decrement count of the subtoken's prefixes (if a longer subtoken is
        # added, its prefixes lose priority to be added).
        for end in xrange(1, subtoken_len):
          subtoken_counts[subtoken[:end]] -= count

    # Add alphabet subtokens (guarantees that all strings are encodable)
    new_subtokens.extend((subtoken_counts.get(a, 0), a) for a in alphabet)
    new_subtokens = [t for _, t in sorted(new_subtokens, reverse=True)]

    # Generate new subtoken->id dictionary using the new subtoken list.
    subtoken_list = RESERVED_TOKENS + new_subtokens
    subtoken_dict = _list_to_index_dict(subtoken_list)

    tf.logging.info('\tVocab size: %d' % len(subtoken_list))
  return subtoken_list
