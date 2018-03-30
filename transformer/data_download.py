"""Download and preprocess WMT17 ende training and evaluation datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import random
import sys
import tarfile
import urllib

import six
import tensorflow as tf

import tokenizer


_TRAIN_DATA_SOURCES = [
    [
        ('http://data.statmt.org/wmt17/translation-task/'
         'training-parallel-nc-v12.tgz'),
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

_PREFIX = 'wmt32k'
_TRAIN_TAG = 'train'
_EVAL_TAG = 'dev'  # Following WMT and Tensor2Tensor conventions, in which the
                   # evaluation datasets are tagged as 'dev' for development.

# Number of files to split train and evaluation data
_TRAIN_SHARDS = 100
_EVAL_SHARDS = 1


def find_file(path, filename, max_depth=5):
  """Returns full filepath if the file is in path or a subdirectory."""
  path_list = [path, filename]
  for _ in range(max_depth + 1):
    matched_files = tf.gfile.Glob(os.path.join(*path_list))
    if matched_files:
      return matched_files[0]
    path_list.insert(1, '*')
  return None


###############################################################################
# Download and extraction functions
###############################################################################
def get_raw_files(raw_dir, data_source):
  """Return raw files from source. Downloads/extracts if needed."""
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
  print('\r%d%%' % percent + ' completed', end='\r')


def download_and_extract(path, url, filenames):
  """Extract files from downloaded compressed archive file."""
  # Check if extracted files already exist in path
  input_file = find_file(path, filenames[0])
  target_file = find_file(path, filenames[1])
  if input_file and target_file:
    tf.logging.info('Already downloaded and extracted %s.' % url)
    return input_file, target_file

  # Download archive file if it doesn't already exist.
  compressed_file = url.split('/')[-1]
  compressed_file = os.path.join(path, compressed_file)
  if find_file(path, compressed_file, max_depth=0) is None:
    tf.logging.info('Downloading from %s to %s.' % (url, compressed_file))
    inprogress_filepath = compressed_file + '.incomplete'
    inprogress_filepath, _ = urllib.urlretrieve(
        url, inprogress_filepath, reporthook=download_report_hook)
    # Print newline to clear the carriage return from the download progress.
    print()
    tf.gfile.Rename(inprogress_filepath, compressed_file)
  else:
    tf.logging.info('Already downloaded: %s (at %s).' % (url, compressed_file))

  # Extract compressed files
  tf.logging.info('Extracting %s.' % compressed_file)
  with tarfile.open(compressed_file, 'r:gz') as corpus_tar:
    corpus_tar.extractall(path)

  # Return filepaths of the requested files.
  input_file = find_file(path, filenames[0])
  target_file = find_file(path, filenames[1])
  if input_file and target_file:
    return input_file, target_file
  raise OSError('Download/extraction failed for url %s to path %s' %
                (url, path))


def txt_line_iterator(path):
  """Iterate through lines of file."""
  with tf.gfile.Open(path) as f:
    for line in f:
      yield line.strip()


def compile_files(raw_dir, raw_files, tag):
  """Compile raw files into a single file for each language."""
  tf.logging.info('Compiling files with tag %s.' % tag)
  filename = '%s-%s' % (_PREFIX, tag)
  input_compiled_file = os.path.join(raw_dir, filename + '.lang1')
  target_compiled_file = os.path.join(raw_dir, filename + '.lang2')

  if (tf.gfile.Exists(input_compiled_file) and
      tf.gfile.Exists(target_compiled_file)):
    tf.logging.info('Files already compiled (%s and %s).' %
                    (input_compiled_file, target_compiled_file))
    return input_compiled_file, target_compiled_file

  with tf.gfile.Open(input_compiled_file, mode='w') as input_writer:
    with tf.gfile.Open(target_compiled_file, mode='w') as target_writer:
      for input_file, target_file in raw_files:
        tf.logging.info('Reading files %s and %s.' % (input_file, target_file))
        for input_line, target_line in zip(
            txt_line_iterator(input_file), txt_line_iterator(target_file)):
          input_writer.write(input_line)
          input_writer.write('\n')
          target_writer.write(target_line)
          target_writer.write('\n')
  return input_compiled_file, target_compiled_file


###############################################################################
# Data preprocessing
###############################################################################
def encode_and_save_files(
    subtokenizer, data_dir, raw_files, tag, num_shards, shuffle=False):
  """Save data from files as encoded Examples in TFrecord format.

  Args:
    subtokenizer: Subtokenizer object that will be used to encode the strings.
    data_dir: The directory in which to write the examples
    raw_files: A tuple of (input, target) data files. Each line in the input and
      the corresponding line in target file will be saved in a tf.Example.
    tag: String that will be added onto the file names.
    num_shards: Number of files to divide the data into.
    shuffle: If true, shuffles the records in each file.
  """
  filepaths = [
      os.path.join(
          data_dir, '%s-%s-%.5d-of-%.5d' % (_PREFIX, tag, n + 1, num_shards))
      for n in range(num_shards)]

  if all_exist(filepaths):
    tf.logging.info('Files with tag %s already exist.' % tag)
    return

  tf.logging.info('Saving files with tag %s.' % tag)

  input_file = raw_files[0]
  target_file = raw_files[1]

  if shuffle:
    filepaths = [fname + '.unshuffled' for fname in filepaths]

  tmp_filepaths = [fname + '.incomplete' for fname in filepaths]
  writers = [tf.python_io.TFRecordWriter(fname) for fname in tmp_filepaths]
  counter, shard = 0, 0
  for input_line, target_line in zip(
      txt_line_iterator(input_file), txt_line_iterator(target_file)):
    if counter > 0 and counter % 100000 == 0:
      tf.logging.info('\tSaving case %d.' % counter)
    counter += 1
    example = dict_to_example(
        {'inputs': subtokenizer.encode(input_line, add_eos=True),
         'targets': subtokenizer.encode(target_line, add_eos=True)})
    writers[shard].write(example.SerializeToString())
    shard = (shard + 1) % num_shards

  for writer in writers:
    writer.close()

  for tmp_name, final_name in zip(tmp_filepaths, filepaths):
    tf.gfile.Rename(tmp_name, final_name)

  tf.logging.info('Saved %d Examples', counter)

  if shuffle:
    out_filepaths = [fname.replace('.unshuffled', '') for fname in filepaths]
    for fname, out_fname in zip(filepaths, out_filepaths):
      shuffle_records(fname, out_fname)

def shuffle_records(fname, out_fname):
  """Shuffle records in a single file."""
  tf.logging.info('Shuffling records in file %s' % fname)

  reader = tf.python_io.tf_record_iterator(fname)
  records = []
  for record in reader:
    records.append(record)
    if len(records) % 100000 == 0:
      tf.logging.info('\tRead: %d', len(records))

  random.shuffle(records)

  with tf.python_io.TFRecordWriter(out_fname) as w:
    for count, record in enumerate(records):
      w.write(record)
      if count > 0 and count % 100000 == 0:
        tf.logging.info('\tWriting record: %d' % count)
  tf.gfile.Remove(fname)


def dict_to_example(dictionary):
  """Converts a dictionary of string->int to a tf.Example."""
  features = {}
  for k, v in six.iteritems(dictionary):
    features[k] = tf.train.Feature(int64_list=tf.train.Int64List(value=v))
  return tf.train.Example(features=tf.train.Features(feature=features))


def all_exist(filepaths):
  """Returns true if all files in the list exist."""
  for fname in filepaths:
    if not tf.gfile.Exists(fname):
      return False
  return True


def make_dir(path):
  if not tf.gfile.Exists(path):
    tf.logging.info('Creating directory %s' % path)
    tf.gfile.MakeDirs(path)


def main(unused_argv):
  """Obtain training and evaluation data for the Transformer model."""
  tf.logging.set_verbosity(tf.logging.INFO)

  make_dir(FLAGS.raw_dir)
  make_dir(FLAGS.data_dir)

  # Get paths of download/extracted training and evaluation files.
  tf.logging.info('Step 1/4: Downloading data from source')
  train_files = get_raw_files(FLAGS.raw_dir, _TRAIN_DATA_SOURCES)
  eval_files = get_raw_files(FLAGS.raw_dir, _EVAL_DATA_SOURCES)

  # Create subtokenizer based on the training files.
  tf.logging.info('Step 2/4: Creating subtokenizer and building vocabulary')
  train_files_flat = []
  for filepath_tuple in train_files:
    train_files_flat.extend(list(filepath_tuple))
  vocab_file = os.path.join(FLAGS.data_dir, tokenizer.VOCAB_FILE)
  subtokenizer = tokenizer.Subtokenizer.init_from_files(
      vocab_file, train_files_flat)

  tf.logging.info('Step 3/4: Compiling training and evaluation data')
  compiled_train_files = compile_files(FLAGS.raw_dir, train_files, _TRAIN_TAG)
  compiled_eval_files = compile_files(FLAGS.raw_dir, eval_files, _EVAL_TAG)

  # Tokenize and save data as Examples in the TFRecord format.
  tf.logging.info('Step 4/4: Preprocessing and saving data')
  encode_and_save_files(
      subtokenizer, FLAGS.data_dir, compiled_train_files, _TRAIN_TAG,
      _TRAIN_SHARDS, shuffle=True)
  encode_and_save_files(
      subtokenizer, FLAGS.data_dir, compiled_eval_files, _EVAL_TAG,
      _EVAL_SHARDS, shuffle=True)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_dir', '-dd', type=str,
      default=os.path.expanduser('~/data/translate_ende'),
      help='[default: %(default)s] Directory for where the '
           'translate_ende_wmt32k dataset is saved.',
      metavar='<DD>')
  parser.add_argument(
      '--raw_dir', '-rd', type=str, default='/tmp/translate_ende_raw',
      help='[default: %(default)s] Path where the raw data will be downloaded '
           'and extracted.',
      metavar='<RD>')

  FLAGS, unparsed = parser.parse_known_args()
  main(sys.argv)
