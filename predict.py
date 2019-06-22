import argparse
import concurrent.futures
import json
import os
from math import ceil

from tqdm import tqdm

from model import load_model, get_best_model_path
from tokenization import FullTokenizer
from utils import read_squad_examples, data_generator, convert_examples_to_features, write_predictions, RawResult


def predict(data_list, model_dir, show_summary=True, gpuid=None):
    if gpuid is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpuid)

    model, _ = load_model(model_dir)
    #if show_summary:
    #    model.summary(line_length=200)

    results = []
    if show_summary:
        data_list = tqdm(data_list)
    for data in data_list:
        inputs, outputs = data
        batch_size, seq_len = inputs[0].shape
        predicts = model.predict(inputs[:2], batch_size=batch_size)

        for i in range(batch_size):
            results.append(
                RawResult(
                    unique_id=inputs[2][i],
                    start_logits=predicts[0][i, :].tolist(),
                    end_logits=predicts[1][i, :].tolist()))

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--model_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--gpu_num', default=1, type=int)
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--n_best_size', default=20, type=int)
    parser.add_argument('--max_answer_length', default=30, type=int)

    conf = parser.parse_args()

    if os.path.isfile(conf.model_dir):
        model_path = conf.model_dir
    else:
        model_path = get_best_model_path(conf.model_dir)
    with open(os.path.join(os.path.dirname(model_path), "config.json"), "r") as f:
        model_conf = json.load(f)
        max_seq_length = model_conf['max_seq_length']
        doc_stride = model_conf['doc_stride']
        max_query_length = model_conf['max_query_length']
        do_lower_case = model_conf['do_lower_case']

    examples = read_squad_examples(conf.input_dir)
    tokenizer = FullTokenizer(os.path.join(os.path.dirname(model_path), 'vocab.txt'), do_lower_case=do_lower_case)
    generator = data_generator(tokenizer, examples, conf.batch_size,
                               max_seq_length, doc_stride, max_query_length, for_predict=True)
    data_list = [f for f in generator]
    data_size = len(data_list)
    if conf.gpu_num == 1:
        results = predict(data_list, model_path, show_summary=True, gpuid=None)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=conf.gpu_num) as executor:
            per_data = int(ceil(data_size / conf.gpu_num))
            futures = []
            for idx in range(conf.gpu_num):
                sub_data_list = data_list[per_data * idx: min(per_data * (idx + 1), data_size)]
                future = executor.submit(predict, sub_data_list, model_path, show_summary=idx == 0, gpuid=idx)
                futures.append(future)
            results = []
            for future in futures:
                results.extend(future.result())

    features = convert_examples_to_features(examples=examples,
                                            tokenizer=tokenizer,
                                            max_seq_length=max_seq_length,
                                            doc_stride=doc_stride,
                                            max_query_length=max_query_length,
                                            insert_unk=False)

    output_prediction_file = os.path.join(conf.output_dir, "predictions.json")
    output_nbest_file = os.path.join(conf.output_dir, "nbest_predictions.json")
    output_prediction_with_answer_file = os.path.join(conf.output_dir, "ans_predictions.json")

    write_predictions(examples, features, results, conf.n_best_size, conf.max_answer_length, do_lower_case,
                      output_prediction_file, output_nbest_file, output_prediction_with_answer_file)


if __name__ == '__main__':
    main()
