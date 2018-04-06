import argparse
import os
import sys

import tensorflow as tf

import model_params
import tokenizer
import transformer

_BATCH_SIZE = 32
_EXTRA_DECODE_LENGTH = 100
_BEAM_SIZE = 3
_ALPHA = 0.6

def _get_sorted_inputs(filename):
  """Read and sort lines from the file sorted by decreasing length.

  Args:
    filename: String name of file to read inputs from.
  Returns:
    Sorted list of inputs, and dictionary mapping original index->sorted index
    of each element.
  """
  with tf.gfile.Open(filename, 'r') as f:
    records = f.read().split('\n')
    inputs = [record.strip() for record in records]
    if not inputs[-1]:
      inputs.pop()

  input_lens = [(i, len(line.split())) for i, line in enumerate(inputs)]
  sorted_input_lens = sorted(input_lens, key=lambda x: x[1], reverse=True)

  sorted_inputs = []
  sorted_keys = {}
  for i, (index, _) in enumerate(sorted_input_lens):
    sorted_inputs.append(inputs[index])
    sorted_keys[index] = i
  return sorted_inputs, sorted_keys



def _encode_and_add_eos(line, subtokenizer):
  return subtokenizer.encode(line).append(tokenizer.EOS_ID)


def translate_file(subtokenizer, input_file, output_file=None):
  output_writer = None
  if output_file is not None:
    if tf.gfile.IsDirectory(output_file):
      tf.logging.error('File output is a directory, will not save outputs '
                       'to file.')
    else:
      output_writer = tf.gfile.Open(output_file, 'w')

  sorted_inputs, sorted_keys = _get_sorted_inputs(input_file)
  num_decode_batches = (len(sorted_inputs) - 1) // _BATCH_SIZE + 1

  ds = tf.data.Dataset.from_tensor_slices(sorted_inputs)
  ds = ds.map(lambda line: _encode_and_add_eos(line, subtokenizer))
  ds = ds.batch(_BATCH_SIZE)

  if output_writer is not None:
    output_writer.close()


def translate_text(estimator, subtokenizer, txt):
  subtokenizer = tokenizer.Subtokenizer(
      os.path.join(FLAGS.data_dir, tokenizer.VOCAB_FILE))
  txt = subtokenizer.encode(txt)
  txt.append(tokenizer.EOS_ID)
  def input_fn():
    ds = tf.data.Dataset.from_tensor_slices([txt,txt])
    ds = ds.batch(_BATCH_SIZE)
    return ds.make_one_shot_iterator().get_next()

  '''
  with tf.Session() as sess:
    x = input_fn()
    print(sess.run(x))
  '''
  for x in estimator.predict(input_fn=input_fn):
    print x, ':)'
    print subtokenizer.decode(x['outputs'])
    break


  print(txt)


def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.INFO)

  if FLAGS.text is None and FLAGS.file is None:
    tf.logging.info('Nothing to translate. Make sure to call this script using'
                    'flags --text or --file.')
    return

  subtokenizer = tokenizer.Subtokenizer(os.path.join(FLAGS.data_dir,
                                                     tokenizer.VOCAB_FILE))

  if FLAGS.params == 'base':
    params = model_params.TransformerBaseParams
  elif FLAGS.params == 'big':
    params = model_params.TransformerBigParams
  else:
    raise ValueError('Invalid parameter set defined: %s.'
                     'Expected "base" or "big.' % FLAGS.params)

  params.beam_size = _BEAM_SIZE
  params.alpha = _ALPHA
  params.extra_decode_length = _EXTRA_DECODE_LENGTH
  params.batch_size = _BATCH_SIZE
  estimator = tf.estimator.Estimator(
      model_fn=transformer.model_fn, model_dir=FLAGS.model_dir, params=params)

  if FLAGS.text is not None:
    tf.logging.info('Translating text: %s' % FLAGS.text)
    translate_text(estimator, subtokenizer, FLAGS.text)

  '''
  if FLAGS.file is not None:
    input_file = os.path.abspath(FLAGS.file)
    tf.logging.info('Translating file: %s' % input_file)
    if not tf.gfile.Exists(FLAGS.file):
      tf.logging.error('File does not exist: %s' % input_file)
    else:
      output_file = None
      if FLAGS.file_out is not None:
        output_file = os.path.abspath(FLAGS.file_out)
        tf.logging.info('File output specified: %s' % output_file)

      translate_file(subtokenizer, input_file, output_file)
  '''

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_dir', '-dd', type=str,
      default=os.path.expanduser('~/data/translate_ende'),
      help='[default: %(default)s] Directory for where the '
           'translate_ende_wmt32k dataset is saved.',
      metavar='<DD>')
  parser.add_argument(
      '--model_dir', '-md', type=str, default='/usr/local/google/home/kathywu/train/transformer2',#'/tmp/transformer_model',
      help='[default: %(default)s] Directory to save Transformer model '
           'training checkpoints',
      metavar='<MD>')
  parser.add_argument(
      '--params', '-p', type=str, default='base', choices=['base', 'big'],
      help='[default: %(default)s] Parameter used for trained model.',
      metavar='<P>')
  parser.add_argument(
      '--text', '-t', type=str, default="want to eat some lunch",
      help='[default: %(default)s] Text to translate. Output will be printed '
           'to console.',
      metavar='<T>')
  parser.add_argument(
      '--file', '-f', type=str, default='yo.txt',
      help='[default: %(default)s] File containing text to translate. '
           'Translation will be printed to console and, if --file_out is '
           'provided, saved to an output file.',
      metavar='<F>')
  parser.add_argument(
      '--file_out', '-fo', type=str, default='www',
      help='[default: %(default)s] If --file flag is specified, save '
           'translation to this file.',
      metavar='<FO>')




  FLAGS, unparsed = parser.parse_known_args()
  main(sys.argv)