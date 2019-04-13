import glob
import os

import argparse
import numpy as np
import pandas as pd
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.layers import GRU, Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.lib.io import file_io
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def

MODEL_NAME = 'stock.hdf5'
CHECKPOINT_FILE_PATH = 'checkpoint.{epoch:02d}.hdf5'

class ContinuousEval(Callback):
    """Continuous eval callback to evaluate the checkpoint once
       every so many epochs.
    """

    def __init__(self,
                 eval_frequency,
                 eval_files,
                 learning_rate,
                 job_dir,
                 steps=1000):
        self.eval_files = eval_files
        self.eval_frequency = eval_frequency
        self.learning_rate = learning_rate
        self.job_dir = job_dir
        self.steps = steps

    def on_epoch_begin(self, epoch, logs={}):
        """Compile and save model."""
        if epoch > 0 and epoch % self.eval_frequency == 0:
            # Unhappy hack to work around h5py not being able to write to GCS.
            # Force snapshots and saves to local filesystem, then copy them over to GCS.
            model_path_glob = 'checkpoint.*'
            if not self.job_dir.startswith('gs://'):
                model_path_glob = os.path.join(self.job_dir, model_path_glob)
            checkpoints = glob.glob(model_path_glob)
            if len(checkpoints) > 0:
                checkpoints.sort()
                census_model = load_model(checkpoints[-1])
                census_model = compile_model(census_model, self.learning_rate)
                loss, acc = census_model.evaluate_generator(model.generator_input(self.eval_files, chunk_size=CHUNK_SIZE), steps=self.steps)
                print('\nEvaluation epoch[{}] metrics[{:.2f}, {:.2f}] {}'.format(epoch, loss, acc, census_model.metrics_names))
                if self.job_dir.startswith('gs://'):
                    copy_file_to_gcs(self.job_dir, checkpoints[-1])
            else:
                print('\nEvaluation epoch[{}] (no checkpoints found)'.format(epoch))

def generator(data, lookback, delay, min_index, max_index, shuffle=False, batch_size=128, step=1):
    if max_index is None:
        max_index = len(data) - delay - 1

    i = min_index + lookback

    while 1:
        rows = []
        if shuffle:
            rows = np.random.randint(min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
                rows = np.arange(i, min(i + batch_size, max_index))
                i += len(rows)

        samples = np.zeros((len(rows), lookback // step, data.shape[-1]))
        targets = np.zeros((len(rows),))

        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]

        yield samples, targets


def train(args):
    try:
        os.makedirs(args.job_dir)
    except:
        pass

    checkpoint_path = CHECKPOINT_FILE_PATH
    # Model checkpoint callback.
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor='val_loss',
        verbose=1,
        period=args.checkpoint_epochs,
        mode='min')

    # Continuous eval callback.
    evaluation = ContinuousEval(args.eval_frequency, args.eval_files, args.learning_rate, args.job_dir)

    # Tensorboard logs callback.
    tb_log = TensorBoard(
        log_dir=os.path.join(args.job_dir, 'logs'),
        histogram_freq=0,
        write_graph=True,
        embeddings_freq=0)

    callbacks = [checkpoint, evaluation, tb_log]

    df = pd.read_csv(args.train_files[0], index_col='date', parse_dates=['date'])
    data = df.values[:, [0, 1, 4, 6, 7]]

    max_idx_size = int(round(len(data) * 0.7))
    mean = data[:max_idx_size].mean(axis=0)
    data -= mean
    data = data.astype(int)
    std = data[:max_idx_size].std(axis=0)
    data = data / std

    lookback = 10
    step = 1
    delay = 1
    batch_size = 128

    train_gen = generator(data,
                          lookback=lookback,
                          delay=delay,
                          min_index=0,
                          max_index=round(len(data) * 0.7),
                          shuffle=True,
                          step=step,
                          batch_size=batch_size)
    val_gen = generator(data,
                        lookback=lookback,
                        delay=delay,
                        min_index=round(len(data) * 0.7) + 1,
                        max_index=round(len(data) * 0.9),
                        shuffle=True,
                        step=step,
                        batch_size=batch_size)
    test_gen = generator(data,
                         lookback=lookback,
                         delay=delay,
                         min_index=round(len(data) * 0.9) + 1,
                         max_index=None,
                         shuffle=True,
                         step=step,
                         batch_size=batch_size)
    val_steps = (round(len(data) * 0.9) - round(len(data) * 0.7) + 1 - lookback)
    test_steps = (len(data) - round(len(data) * 0.9) + 1 - lookback)

    model = Sequential()
    model.add(GRU(32,
                  dropout=0.2,
                  recurrent_dropout=0.2,
                  input_shape=(None, data.shape[-1])))
    model.add(Dense(1))
    model = compile_model(model)
    history = model.fit_generator(train_gen,
                                  steps_per_epoch=500,
                                  epochs=1,
                                  validation_data=val_gen,
                                  validation_steps=val_steps,
                                  callbacks=callbacks)

    if args.job_dir.startswith('gs://'):
        model.save(MODEL_NAME)
        copy_file_to_gcs(args.job_dir, MODEL_NAME)
    else:
        model.save(MODEL_NAME)

    # Convert the Keras model to TensorFlow SavedModel.
    to_savedmodel(model, os.path.join(args.job_dir, 'export'))


def compile_model(model, learning_rate=0.01):
    model.compile(optimizer=RMSprop(), loss='mae')
    return model

# h5py workaround: copy local models over to GCS if the job_dir is GCS.
def copy_file_to_gcs(job_dir, file_path):
    with file_io.FileIO(file_path, mode='rb') as input_f:
        with file_io.FileIO(os.path.join(job_dir, file_path), mode='w+') as output_f:
            output_f.write(input_f.read())


def to_savedmodel(model, export_path):
    """Convert the Keras HDF5 model into TensorFlow SavedModel."""

    builder = saved_model_builder.SavedModelBuilder(export_path)

    signature = predict_signature_def(
        inputs={'input': model.inputs[0]}, outputs={'income': model.outputs[0]})

    with K.get_session() as sess:
        builder.add_meta_graph_and_variables(
            sess=sess,
            tags=[tag_constants.SERVING],
            signature_def_map={
                signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature
            })
        builder.save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train-files',
        nargs='+',
        help='Training file local or GCS')
    parser.add_argument(
        '--job-dir',
        type=str,
        help='GCS or local dir to write checkpoints and export model',
        default='/tmp/keras')
    parser.add_argument(
        '--eval-frequency',
        default=10,
        help='Perform one evaluation per n epochs')
    parser.add_argument(
        '--eval-files',
        nargs='+',
        help='Evaluation file local or GCS',
        default='')
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.003,
        help='Learning rate for SGD')
    parser.add_argument(
        '--checkpoint-epochs',
        type=int,
        default=5,
        help='Checkpoint per n training epochs')

    args, _ = parser.parse_known_args()
    train(args)
