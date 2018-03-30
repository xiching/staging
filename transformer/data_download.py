"""Download and preprocess WMT17 ende training and evaluation datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import urllib
import os
import tarfile

import tensorflow as tf

import dataset
import tokenizer

_TRAIN_DATA_SOURCES = [
    [
        'http://data.statmt.org/wmt17/translation-task/training-parallel-nc-v12.tgz',
        ('news-commentary-v12.de-en.en', 'news-commentary-v12.de-en.de'),
    ],
    [
        'http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz',
        ('commoncrawl.de-en.en', 'commoncrawl.de-en.de')
    ],
    [
        'http://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz',
        ('europarl-v7.de-en.en', 'europarl-v7.de-en.de')
    ]
]

_EVAL_DATA_SOURCES = [
    [
        'http://data.statmt.org/wmt17/translation-task/dev.tgz',
        ('newstest2013.en', 'newstest2013.de')
    ]
]


def find_file(path, filename, max_depth=5):
  """Returns full filepath if the file is in path or a subdirectory."""
  path_list = [path, filename]
  for _ in range(max_depth + 1):
    matched_files = tf.gfile.Glob(os.path.join(*path_list))
    if len(matched_files) > 0:
      return matched_files[0]
    path_list.insert(1, '*')
  return None


###############################################################################
# Download and extraction functions
###############################################################################
def get_raw_files(raw_dir, data_source):
  """Return files from data source. Downloads/extracts if needed."""
  raw_files = []
  for url, filenames in data_source:
    raw_files.append(download_and_extract(raw_dir, url, filenames))
  return raw_files


def download_report_hook(count, block_size, total_size):
  """Report hook for download progress.

  Args:
    count: current block number
    block_size: block size
    total_size: total size
  """
  percent = int(count * block_size * 100 / total_size)
  print("\r%d%%" % percent + " completed", end="\r")


def download_and_extract(path, url, filenames):
  """Extract files from downloaded compressed archive file."""
  if not tf.gfile.Exists(path):
    tf.logging.info("Creating directory %s" % path)
    tf.gfile.MakeDirs(path)

  # Checkif extracted files already exist in path
  input_file = find_file(path, filenames[0])
  target_file = find_file(path, filenames[1])
  if input_file and target_file:
    tf.logging.info("Already downloaded and extracted %s." % url)
    return input_file, target_file

  # Download archive file if it doesn't already exist.
  compressed_file = url.split('/')[-1]
  compressed_file = os.path.join(path, compressed_file)
  if find_file(path, compressed_file, max_depth=0) is None:
    tf.logging.info("Downloading from %s to %s." % (url, compressed_file))
    inprogress_filepath = compressed_file + ".incomplete"
    inprogress_filepath, _ = urllib.urlretrieve(
        url, inprogress_filepath, reporthook=download_report_hook)
    # Print newline to clear the carriage return from the download progress.
    print()
    tf.gfile.Rename(inprogress_filepath, compressed_file)
  else:
    tf.logging.info("Already downloaded: %s (at %s)." % (url, compressed_file))

  # Extract compressed files
  tf.logging.info("Extracting %s." % compressed_file)
  with tarfile.open(compressed_file, 'r:gz') as corpus_tar:
    corpus_tar.extractall(path)

  # Return filepaths of the requested files.
  input_file = find_file(path, filenames[0])
  target_file = find_file(path, filenames[1])
  if input_file and target_file:
    return input_file, target_file
  raise OSError("Download/extraction failed for url %s to path %s" %
                (url, path))


###############################################################################
# Data preprocessing: Tokenization and conversion to Example/TFRecord format.
###############################################################################
def preprocess_files(subtokenizer, data_dir, raw_files, suffix):
  """Run preprocessing steps on the raw files."""
  pass


def create_subtokenizer(train_raw_files, data_dir):

  pass

def main(unused):
  tf.logging.set_verbosity(tf.logging.INFO)

  # Get paths of download/extracted training and evaluation files.
  train_raw_files = get_raw_files(FLAGS.raw_dir, _TRAIN_DATA_SOURCES)
  eval_raw_files = get_raw_files(FLAGS.raw_dir, _EVAL_DATA_SOURCES)

  # Create subtokenizer based on the training file.
  vocab_file = os.path.join(FLAGS.data_dir, tokenizer.VOCAB_FILE)
  subtokenizer = tokenizer.Subtokenizer.init_from_data(
      train_raw_files, vocab_file)

  # Tokenize and save data as Examples in the TFRecord format.
  preprocess_files(subtokenizer, FLAGS.data_dir, train_raw_files, 'train')
  preprocess_files(subtokenizer, FLAGS.data_dir, eval_raw_files, 'dev')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_dir',
      type=str,
      default=os.path.expanduser('~/data/translate_ende'),
      help='Directory for where the translate_ende_wmt32k dataset is saved.')
  parser.add_argument(
      '--raw_dir',
      type=str,
      default='/tmp/translate_ende_raw',
      metavar='N',
      help='Path where the raw data will be downloaded and extracted if not '
           'already there.')

  FLAGS, unparsed = parser.parse_known_args()
  main(sys.argv)
