from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import json
import logging
import math
import os
import pickle
import random

import numpy as np
from khaiii import KhaiiiApi
from tqdm import tqdm

logger = logging.getLogger(__name__)
'''
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
fileHandler = logging.FileHandler('./train.log')
logger.addHandler(fileHandler)
'''

RawResult = collections.namedtuple("RawResult", ["unique_id", "start_logits", "end_logits"])

khaiii = KhaiiiApi()


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def data_generator(tokenizer, raw_data, batch_size, max_seq_length, doc_stride, max_query_length, insert_unk=False,
                   for_predict=False):
    data = np.array(raw_data)
    data_size = len(data)
    while True:
        if not for_predict:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            data = data[shuffle_indices]
        inputs = [[], []]
        if for_predict:
            inputs.append([])
        outputs = [[], []]
        for e in data:
            features = convert_examples_to_features(examples=[e],
                                                    tokenizer=tokenizer,
                                                    max_seq_length=max_seq_length,
                                                    doc_stride=doc_stride,
                                                    max_query_length=max_query_length,
                                                    insert_unk=insert_unk)
            for feature in features:
                inputs[0].append(feature.input_ids)
                inputs[1].append(feature.segment_ids)
                if for_predict:
                    inputs[2].append(feature.unique_id)
                outputs[0].append(feature.start_position)
                outputs[1].append(feature.end_position)

                if len(inputs[0]) == batch_size:
                    inputs = [np.asarray(x) for x in inputs]
                    outputs = [np.asarray(x) for x in outputs]
                    yield inputs, outputs
                    inputs = [[], []]
                    if for_predict:
                        inputs.append([])
                    outputs = [[], []]
        if len(inputs[0]) > 0:
            inputs = [np.asarray(x) for x in inputs]
            outputs = [np.asarray(x) for x in outputs]
            yield inputs, outputs
        if for_predict:
            return


class SquadExample(object):
    # Copyright 2018 The Google AI Language Team Authors.
    # https://github.com/google-research/bert/blob/master/run_squad.py
    def __init__(self,
                 qas_id,
                 q_raw_text,
                 q_morp_token,
                 p_raw_text,
                 p_morp_token,
                 p_morp_position_list,
                 a_raw_text=None,
                 a_morp_token=None,
                 a_begin_morp=None,
                 a_end_morp=None):
        self.qas_id = qas_id
        self.q_raw_text = q_raw_text
        self.q_morp_token = q_morp_token
        self.p_raw_text = p_raw_text
        self.p_morp_token = p_morp_token
        self.p_morp_position_list = p_morp_position_list
        self.a_raw_text = a_raw_text
        self.a_morp_token = a_morp_token
        self.a_begin_morp = a_begin_morp
        self.a_end_morp = a_end_morp

        self.p_raw_bytes = p_raw_text.encode()
        self.p_morp_position_list.append(len(self.p_raw_bytes))

        if a_raw_text is not None and a_end_morp is not None and len(p_morp_position_list) > a_end_morp:
            begin_pos = p_morp_position_list[a_begin_morp]
            end_pos = p_morp_position_list[a_end_morp + 1]
            pred_answer = self.p_raw_bytes[begin_pos:end_pos].decode().strip()
            if self.a_raw_text != pred_answer:
                logger.info("[diff answer span] %s\t%s" % (self.a_raw_text, pred_answer))

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % self.qas_id
        s += ", question_text: %s" % self.q_raw_text
        s += ", doc_text: [%s]" % self.p_raw_text
        if self.a_begin_morp:
            s += ", start_position: %d" % self.a_begin_morp
        if self.a_end_morp:
            s += ", end_position: %d" % self.a_end_morp
        return s


class InputFeatures(object):
    # Copyright 2018 The Google AI Language Team Authors.
    # https://github.com/google-research/bert/blob/master/run_squad.py
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 start_position=None,
                 end_position=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position


def mapping_answer_korquad(rep_p, answer_text, answer_start, answer_end=-1):
    # Copyright ETRI
    # http://aiopen.etri.re.kr/
    if answer_end == -1:
        answer_end = answer_start + len(answer_text)
    if answer_text != rep_p['text'][answer_start:answer_end]:
        idx = rep_p['text'].find(answer_text)
        if idx == -1:
            logger.info(
                '[mapping_answer_korquad error]\t%s\t%s' % (answer_text, rep_p['text'][answer_start:answer_end]))
            return None
        answer_start = idx
        answer_end = answer_start + len(answer_text)

    byte_answer_start = len(rep_p['text'][:answer_start].encode())
    byte_answer_end = len(rep_p['text'][:answer_end].encode())

    begin_morp_id = end_morp_id = -1
    morp_size = len(rep_p['morp_list'])
    for morp_i, position in enumerate(rep_p['position_list']):
        lemma = rep_p['lemma_list'][morp_i]
        if position == byte_answer_start:
            begin_morp_id = morp_i
            break
        elif position < byte_answer_start and morp_i + 1 < morp_size and \
                byte_answer_start < rep_p['position_list'][morp_i + 1]:
            begin_morp_id = morp_i

            logger.warn('[begin not exact match] %s\t->\t%s' % (answer_text, lemma))
            break
        elif position > byte_answer_start:
            begin_morp_id = morp_i
            logger.error('[begin error] %s\t->\t%s' % (answer_text, lemma))
            break

    if begin_morp_id != -1 and end_morp_id == -1 and \
            byte_answer_end <= rep_p['position_list'][-1] + len(rep_p['lemma_list'][-1].encode()):
        for morp_i in range(morp_size - 1, -1, -1):
            lemma = rep_p['lemma_list'][morp_i]
            position = rep_p['position_list'][morp_i]
            if position == byte_answer_end:
                end_morp_id = morp_i - 1
                break
            elif position < byte_answer_end and byte_answer_end <= position + len(lemma.encode()):
                end_morp_id = morp_i
                break
            elif position < byte_answer_end:
                end_morp_id = morp_i
                logger.error('[end error] %s\t->\t%s' % (answer_text, lemma))
                break

    if begin_morp_id == -1 or end_morp_id == -1:
        return None

    p_text_bytes = rep_p['text'].encode()
    begin_pos = rep_p['position_list'][begin_morp_id]
    end_pos = len(p_text_bytes)
    if end_morp_id + 1 < len(rep_p['position_list']): end_pos = rep_p['position_list'][end_morp_id + 1]
    pred_text = p_text_bytes[begin_pos: end_pos].decode().strip()
    if answer_text != pred_text:
        logger.warn('[check morp index] %s\t%s' % (answer_text, pred_text))

    return {'begin_morp': begin_morp_id, 'end_morp': end_morp_id, 'text': answer_text}


def morph_analyze(text):
    morp_list = []
    lemma_list = []
    position_list = []
    for results in khaiii.analyze(text):
        for morph in results.morphs:
            lemma_list.append(morph.lex)
            if morph.tag in ['SWK', 'ZN', 'ZV', 'ZZ']:
                morp_list.append(morph.lex)
                logger.warning("[POS] {}: {}".format(morph.tag, morph.lex))
            else:
                morp_list.append(morph.lex + "/" + morph.tag)
            position_list.append(len(text[:morph.begin].encode()))
    return {'text': text, 'lemma_list': lemma_list, 'morp_list': morp_list, 'position_list': position_list}


def read_squad_examples(input_file, id_prefix=''):
    # Copyright 2018 The Google AI Language Team Authors.
    # https://github.com/google-research/bert/blob/master/run_squad.py
    if not os.path.isfile(input_file):
        raise ValueError("not exist file or folder : %s" % input_file)

    if os.path.exists(input_file + '.pickle'):
        with open(input_file + '.pickle', 'rb') as f:
            examples = pickle.load(f)
        return examples

    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)

    pqa_list = []
    for paragraphs_title in tqdm(input_data['data']):
        for pq in paragraphs_title['paragraphs']:
            passage_text = pq['context']
            rep_p = morph_analyze(passage_text)

            for qa in pq['qas']:
                qas_id = id_prefix + qa['id']
                question_text = qa['question']
                rep_q = morph_analyze(question_text)

                rep_a = mapping_answer_korquad(rep_p, qa['answers'][0]['text'].strip(),
                                               qa['answers'][0]['answer_start'])
                if rep_a:
                    pqa_list.append({'id': qas_id, 'passage': rep_p, 'question': rep_q, 'answer': rep_a})

    examples = []
    for pqa in pqa_list:
        a_raw_text = pqa['answer']['text']
        a_begin_morp = pqa['answer']['begin_morp']
        a_end_morp = pqa['answer']['end_morp']
        a_morp_token = pqa['passage']['morp_list'][a_begin_morp: a_end_morp + 1]

        example = SquadExample(
            qas_id=pqa['id'],
            q_raw_text=pqa['question']['text'],
            q_morp_token=pqa['question']['morp_list'],
            p_raw_text=pqa['passage']['text'],
            p_morp_token=pqa['passage']['morp_list'],
            p_morp_position_list=pqa['passage']['position_list'],
            a_raw_text=a_raw_text,
            a_morp_token=a_morp_token,
            a_begin_morp=a_begin_morp,
            a_end_morp=a_end_morp)
        examples.append(example)

    logger.info('len(examples) : %d' % len(examples))
    with open(input_file + '.pickle', 'wb') as f:
        pickle.dump(examples, f)
    return examples


def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 doc_stride, max_query_length, insert_unk,
                                 verbose=False):
    # Copyright 2018 The Google AI Language Team Authors.
    # https://github.com/google-research/bert/blob/master/run_squad.py
    features = []
    for (example_index, example) in enumerate(examples):
        query_tokens = []
        for q_morp in example.q_morp_token:
            query_tokens.extend(tokenizer.tokenize(q_morp))
        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example.p_morp_token):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        tok_start_position = orig_to_tok_index[example.a_begin_morp]
        if example.a_end_morp + 1 < len(orig_to_tok_index):
            tok_end_position = orig_to_tok_index[example.a_end_morp + 1] - 1
        else:
            tok_end_position = orig_to_tok_index[-1]

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []
            tokens.append("[CLS]")
            segment_ids.append(0)
            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(0)
            tokens.append("[SEP]")
            segment_ids.append(0)

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                       split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            if insert_unk:
                mask_rate = 0.05
                mask_unk_rate = 0.2
                token_input = []
                masked_input = []
                VALUE_MASK = 4
                VALUE_UNK = 1
                VALUE_PAD = 0
                VALUE_SEP = 2
                VALUE_CLS = 3
                size_token = len(tokenizer.vocab)
                for it in input_ids:
                    if it == VALUE_SEP or it == VALUE_CLS:
                        token_input.append(it)
                    elif np.random.random() < mask_rate:
                        masked_input.append(1)
                        r = np.random.random()
                        if r < mask_unk_rate:
                            token_input.append(VALUE_UNK)
                        else:
                            while True:
                                random_token = random.randrange(0, size_token)
                                if random_token is VALUE_PAD or random_token is VALUE_UNK or random_token is VALUE_MASK:
                                    pass
                                else:
                                    break
                            token_input.append(random_token)
                    else:
                        masked_input.append(0)
                        token_input.append(it)
                input_ids = token_input

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            doc_start = doc_span.start
            doc_end = doc_span.start + doc_span.length - 1
            out_of_span = False
            if not (tok_start_position >= doc_start and
                    tok_end_position <= doc_end):
                out_of_span = True
            if out_of_span:
                start_position = 0
                end_position = 0
            else:
                doc_offset = len(query_tokens) + 2
                start_position = tok_start_position - doc_start + doc_offset
                end_position = tok_end_position - doc_start + doc_offset

            unique_id = "{}_{}".format(example.qas_id, doc_span_index)
            if verbose:
                logger.info("*** Example ***")
                logger.info("unique_id: %s" % (unique_id))
                logger.info("example_index: %s" % (example_index))
                logger.info("doc_span_index: %s" % (doc_span_index))
                logger.info("tokens: %s" % " ".join(tokens))
                logger.info("token_to_orig_map: %s" % " ".join([
                    "%d:%d" % (x, y) for (x, y) in token_to_orig_map.items()]))
                logger.info("token_is_max_context: %s" % " ".join([
                    "%d:%s" % (x, y) for (x, y) in token_is_max_context.items()
                ]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info(
                    "input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                answer_text = " ".join(tokens[start_position:(end_position + 1)])
                logger.info("start_position: %d" % (start_position))
                logger.info("end_position: %d" % (end_position))
                logger.info("answer: %s" % (answer_text))
                logger.info("orig_answer: %s" % (example.a_raw_text))

            features.append(
                InputFeatures(
                    unique_id=unique_id,
                    example_index=example_index,
                    doc_span_index=doc_span_index,
                    tokens=tokens,
                    token_to_orig_map=token_to_orig_map,
                    token_is_max_context=token_is_max_context,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    start_position=start_position,
                    end_position=end_position))

    return features


def _check_is_max_context(doc_spans, cur_span_index, position):
    # Copyright 2018 The Google AI Language Team Authors.
    # https://github.com/google-research/bert/blob/master/run_squad.py
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def _get_best_indexes(logits, n_best_size):
    # Copyright 2018 The Google AI Language Team Authors.
    # https://github.com/google-research/bert/blob/master/run_squad.py
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    # Copyright 2018 The Google AI Language Team Authors.
    # https://github.com/google-research/bert/blob/master/run_squad.py
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


def write_predictions(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case, output_prediction_file,
                      output_nbest_file, output_prediction_with_answer_file, verbose_logging=True):
    # Copyright 2018 The Google AI Language Team Authors.
    # https://github.com/google-research/bert/blob/master/run_squad.py

    logger.info("Writing predictions to: %s" % output_prediction_file)
    logger.info("Writing nbest to: %s" % output_nbest_file)

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    all_ans_predictions = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]

            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index]))

        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit"])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]

            orig_doc_start = feature.token_to_orig_map[pred.start_index]
            orig_doc_end = feature.token_to_orig_map[pred.end_index]

            p_begin_morp = example.p_morp_position_list[orig_doc_start]
            p_end_morp = example.p_morp_position_list[orig_doc_end + 1]

            final_text = example.p_raw_bytes[p_begin_morp:p_end_morp].decode().strip()

            if final_text in seen_predictions:
                continue

            seen_predictions[final_text] = True
            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
                _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        assert len(nbest) >= 1

        total_scores = []
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        for cand in nbest_json:
            if '\"' in cand['text'] or '다. ' in cand['text']:
                continue
            all_predictions[example.qas_id] = cand["text"]
            all_ans_predictions[example.qas_id] = [cand["text"], example.a_raw_text]
            break
        if example.qas_id not in all_predictions:
            all_predictions[example.qas_id] = nbest_json[0]["text"]
            all_ans_predictions[example.qas_id] = [nbest_json[0]["text"], example.a_raw_text]
        all_nbest_json[example.qas_id] = nbest_json

    with open(output_prediction_file, "w") as writer:
        writer.write(
            json.dumps(all_predictions, ensure_ascii=False, indent=4, sort_keys=True, separators=(',', ': ')) + "\n")

    if output_nbest_file:
        with open(output_nbest_file, "w") as writer:
            writer.write(
                json.dumps(all_nbest_json, ensure_ascii=False, indent=4, sort_keys=True, separators=(',', ': ')) + "\n")

    if output_prediction_with_answer_file:
        with open(output_prediction_with_answer_file, "w") as writer:
            writer.write(
                json.dumps(all_ans_predictions, ensure_ascii=False, indent=4, sort_keys=True,
                           separators=(',', ': ')) + "\n")
