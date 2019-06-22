import argparse
import json
import os
from shutil import copyfile

import keras
from keras.callbacks import CSVLogger

from model import load_from_pretrained, load_model
from teams_callback import TeamsCallback
from tokenization import FullTokenizer
from utils import read_squad_examples, str2bool, data_generator
from variables import MODEL_FILE_FORMAT, LAST_MODEL_FILE_FORMAT, TEAMS_WEBHOOK_URL


def main():
    parser = argparse.ArgumentParser()
    # path
    parser.add_argument('--train_data', required=True)
    parser.add_argument('--valid_data', required=True)
    parser.add_argument('--train_dir', required=True)
    parser.add_argument('--pretrained_dir', required=True)

    # params
    parser.add_argument('--max_seq_length', default=512, type=int)
    parser.add_argument('--doc_stride', default=128, type=int)
    parser.add_argument('--max_query_length', default=64, type=int)
    parser.add_argument("--do_lower_case", type=str2bool, nargs='?', const=False, default=False)

    # learning option
    parser.add_argument('--batch_size', default=5, type=int)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--train_steps', default=5000, type=int)
    parser.add_argument('--validation_steps', default=1000, type=int)
    parser.add_argument('--early_stop_patience', default=5, type=int)
    parser.add_argument('--optimizer_type', default='decay', const='decay', nargs='?', choices=['decay', 'warmup'])
    parser.add_argument('--warmup_proportion', default=0.1, type=float)
    parser.add_argument('--decay_rate', default=0.002, type=float)

    conf = parser.parse_args()

    os.makedirs(conf.train_dir, exist_ok=True)
    with open(os.path.join(conf.train_dir, 'config.json'), "w") as f:
        json.dump(vars(conf), f, sort_keys=True, indent=4, separators=(',', ': '))

    train_examples = []
    for train_file in conf.train_data.split(','):
        id_prefix = os.path.basename(train_file) + "_"
        train_examples.extend(read_squad_examples(train_file, id_prefix))
    valid_examples = read_squad_examples(conf.valid_data)

    copyfile(os.path.join(conf.pretrained_dir, 'vocab.txt'), os.path.join(conf.train_dir, 'vocab.txt'))
    tokenizer = FullTokenizer(os.path.join(conf.pretrained_dir, 'vocab.txt'), do_lower_case=conf.do_lower_case)
    train_generator = data_generator(tokenizer, train_examples, conf.batch_size,
                                     conf.max_seq_length, conf.doc_stride, conf.max_query_length, insert_unk=True)
    valid_generator = data_generator(tokenizer, valid_examples, conf.batch_size,
                                     conf.max_seq_length, conf.doc_stride, conf.max_query_length)

    train_steps = conf.train_steps
    if train_steps == 0:
        train_steps = len(train_examples)  # approximate
    validation_steps = conf.validation_steps
    if validation_steps == 0:
        validation_steps = len(valid_examples)

    last_state_path = os.path.join(conf.train_dir, LAST_MODEL_FILE_FORMAT)

    if os.path.exists(last_state_path):
        model, initial_epoch = load_model(conf.train_dir)
    else:
        decay_steps = train_steps * conf.epochs
        warmup_steps = int(train_steps * conf.warmup_proportion)
        model = load_from_pretrained(conf.pretrained_dir,
                                     lr=conf.lr,
                                     seq_len=conf.max_seq_length,
                                     optimizer_type=conf.optimizer_type,
                                     decay_rate=conf.decay_rate,
                                     warmup_steps=warmup_steps,
                                     decay_steps=decay_steps)
        initial_epoch = 0

    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(monitor='loss', patience=conf.early_stop_patience),
        CSVLogger(os.path.join(conf.train_dir, "history.txt"), append=True),
        keras.callbacks.TensorBoard(log_dir=os.path.join(conf.train_dir, "graph"), histogram_freq=0,
                                    write_graph=True, write_images=True),
        keras.callbacks.ModelCheckpoint(os.path.join(conf.train_dir, MODEL_FILE_FORMAT), monitor='loss',
                                        save_best_only=False),
        keras.callbacks.ModelCheckpoint(os.path.join(conf.train_dir, 'last.h5')),
        TeamsCallback(TEAMS_WEBHOOK_URL, name='korquad')
    ]

    model.fit_generator(
        generator=train_generator,
        steps_per_epoch=train_steps,
        epochs=conf.epochs,
        validation_data=valid_generator,
        validation_steps=validation_steps,
        initial_epoch=initial_epoch,
        callbacks=callbacks,
        verbose=1,
    )


if __name__ == '__main__':
    main()
