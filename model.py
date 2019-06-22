import glob
import os

import keras
from keras.initializers import TruncatedNormal
from keras.layers import Lambda
from keras.optimizers import Adam
from keras_bert import load_model_weights_from_checkpoint, build_model_from_config, get_custom_objects

from AdamWD import AdamWD
from variables import LAST_MODEL_FILE_FORMAT, MODEL_REGEX_PATTERN


def custom_loss(y_true, y_pred):
    import keras.backend as K
    return K.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)


def load_from_pretrained(pretrained_path,
                         lr,
                         seq_len,
                         optimizer_type,
                         decay_rate,
                         warmup_steps,
                         decay_steps):
    config_path = os.path.join(pretrained_path, 'bert_config.json')
    checkpoint_path = os.path.join(pretrained_path, 'bert_model.ckpt')
    model, config = build_model_from_config(
        config_path,
        training=False,
        trainable=True,
        output_layer_num=1,
        seq_len=seq_len,
    )
    load_model_weights_from_checkpoint(model, config, checkpoint_path, training=False)
    inputs = model.inputs
    outputs = model.outputs
    transformer_output = outputs[0]
    logits = keras.layers.Dense(
        units=2,
        trainable=True,
        name='logits',
        kernel_initializer=TruncatedNormal(stddev=0.02)
    )(transformer_output)
    start_logits = Lambda(lambda x: x[:, :, 0], name='start-logits')(logits)
    end_logits = Lambda(lambda x: x[:, :, 1], name='end-logits')(logits)

    model = keras.models.Model(inputs=inputs, outputs=[start_logits, end_logits])

    if optimizer_type == 'decay':
        optimizer = Adam(lr=lr,
                         amsgrad=True,
                         decay=decay_rate)
    else:
        optimizer = AdamWD(lr=lr,
                           amsgrad=True,
                           warmup_steps=warmup_steps,
                           decay_steps=decay_steps)

    model.compile(
        optimizer=optimizer,
        loss=custom_loss,
    )

    model.summary()

    return model


def get_last_epoch(model_path):
    matched = MODEL_REGEX_PATTERN.match(model_path)
    if matched:
        return int(matched.group(1))
    try:
        with open(os.path.join(os.path.dirname(model_path), 'history.txt'), 'r') as f:
            lines = f.readlines()
            return int(lines[-1].split(',')[0]) + 1
    except:
        print("warning: coudn`t extract last epoch num")
        return 0


def get_best_model_path(model_dir):
    last_model = sorted(glob.glob(os.path.join(model_dir, 'weights*.h5')))[-1]
    if MODEL_REGEX_PATTERN.match(last_model):
        return last_model
    else:
        print("warning: coudn`t find best model path")
        return model_dir


def load_model(train_dir):
    try:
        if os.path.isfile(train_dir):
            model_path = train_dir
        elif os.path.isdir(train_dir):
            model_path = os.path.join(train_dir, LAST_MODEL_FILE_FORMAT)
        else:
            raise Exception('path not exist')

        last_epoch = get_last_epoch(model_path)
        print("load from => {}".format(model_path))
        custom_objects = get_custom_objects()
        custom_objects['custom_loss'] = custom_loss
        custom_objects['AdamWD'] = AdamWD
        model = keras.models.load_model(model_path, custom_objects=custom_objects)
        return model, last_epoch

    except Exception as e:
        print(str(e))
        print("model file not found")
