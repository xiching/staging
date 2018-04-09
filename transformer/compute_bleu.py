"""Script to compute official BLEU score."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import re
import six
import sys
import unicodedata

import metrics


class UnicodeRegex(object):
  """Ad-hoc hack to recognize all punctuation and symbols."""

  def __init__(self):
    punctuation = self.property_chars('P')
    self.nondigit_punct_re = re.compile(r'([^\d])([' + punctuation + r'])')
    self.punct_nondigit_re = re.compile(r'([' + punctuation + r'])([^\d])')
    self.symbol_re = re.compile('([' + self.property_chars('S') + '])')

  def property_chars(self, prefix):
    return ''.join(six.unichr(x) for x in range(sys.maxunicode)
                   if unicodedata.category(six.unichr(x)).startswith(prefix))


uregex = UnicodeRegex()

def bleu_tokenize(string):
  r"""Tokenize a string following the official BLEU implementation.

  See https://github.com/moses-smt/mosesdecoder/'
           'blob/master/scripts/generic/mteval-v14.pl#L954-L983
  In our case, the input string is expected to be just one line
  and no HTML entities de-escaping is needed.
  So we just tokenize on punctuation and symbols,
  except when a punctuation is preceded and followed by a digit
  (e.g. a comma/dot as a thousand/decimal separator).

  Note that a numer (e.g. a year) followed by a dot at the end of sentence
  is NOT tokenized,
  i.e. the dot stays with the number because `s/(\p{P})(\P{N})/ $1 $2/g`
  does not match this case (unless we add a space after each sentence).
  However, this error is already in the original mteval-v14.pl
  and we want to be consistent with it.

  Args:
    string: the input string

  Returns:
    a list of tokens
  """
  string = uregex.nondigit_punct_re.sub(r'\1 \2 ', string)
  string = uregex.punct_nondigit_re.sub(r' \1 \2', string)
  string = uregex.symbol_re.sub(r' \1 ', string)
  return string.split()


def bleu_wrapper(ref_filename, hyp_filename, case_sensitive=False):
  """Compute BLEU for two files (reference and hypothesis translation)."""
  ref_lines = open(ref_filename).read().splitlines()
  hyp_lines = open(hyp_filename).read().splitlines()
  assert len(ref_lines) == len(hyp_lines)
  if not case_sensitive:
    ref_lines = [x.lower() for x in ref_lines]
    hyp_lines = [x.lower() for x in hyp_lines]
  ref_tokens = [bleu_tokenize(x) for x in ref_lines]
  hyp_tokens = [bleu_tokenize(x) for x in hyp_lines]
  return metrics.compute_bleu(ref_tokens, hyp_tokens)


def main(unused_argv):
  if FLAGS.bleu_variant in ("both", "uncased"):
    score = 100 * bleu_wrapper(FLAGS.reference, FLAGS.translation, False)
    print("Case-sensitive results", score)

  if FLAGS.bleu_variant in ("both", "cased"):
    score = 100 * bleu_wrapper(FLAGS.reference, FLAGS.translation, True)
    print("Case-sensitive results", score)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--model_dir', '-md', type=str, default='/tmp/transformer_model',
      help='[default: %(default)s] Directory containing Transformer model '
           'checkpoints.',
      metavar='<MD>')
  parser.add_argument(
      '--params', '-p', type=str, default='base', choices=['base', 'big'],
      help='[default: %(default)s] Parameter used for trained model.',
      metavar='<P>')
  parser.add_argument(
      '--translation', '-t', type=str, default=None, required=True,
      help='[default: %(default)s] File containing translated text.',
      metavar='<T>')
  parser.add_argument(
      '--reference', '-r', type=str, default=None, required=True,
      help='[default: %(default)s] File containing reference translation',
      metavar='<R>')
  parser.add_argument(
      '--bleu_variant', '-bv', type=str, default='both',
      help='[default: %(default)s] Whether to return case-sensitive, case-'
           'insensitive or both results. Possible values: \"cased\", '
           '\"uncased\", or \"both\".',
      metavar='<R>')

  FLAGS, unparsed = parser.parse_known_args()
  main(sys.argv)
