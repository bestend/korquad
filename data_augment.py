import argparse
import concurrent.futures
import html.parser as htmlparser
import json

from google.cloud import translate
from tqdm import trange, tqdm

parser = htmlparser.HTMLParser()

translate_client = translate.Client()


def translate_google_api(query, src, tgt):
    translation = translate_client.translate(
        query,
        source_language=src,
        target_language=tgt)
    return parser.unescape(translation['translatedText'])


def augment_text(query, func=translate_google_api):
    translated = func(query, 'ko', 'en')
    augmented = func(translated, 'en', 'ko')
    return augmented


def augment(src, idx):
    try:
        augmented = augment_text(src[idx]["question"])
        dst = src[idx].copy()
        dst["id"] = 'augmented_{}'.format(src[idx]["id"])
        dst["question"] = augmented
        src.append(dst)
        return ""
    except Exception as e:
        return src[idx]["id"]


def get_target_id_list(qas_list):
    augmented = []
    original = []
    for q in qas_list:
        if q['id'].startswith('augmented_'):
            augmented.append(q['id'].lstrip('augmented_'))
        else:
            original.append(q['id'])
    for id in augmented:
        original.remove(id)
    return original


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--worker_num', default=5, type=int)

    conf = parser.parse_args()

    with open(conf.input, "r") as f:
        data = json.loads(f.read())

    failed_count = 0
    try_count = 1

    with concurrent.futures.ThreadPoolExecutor(max_workers=conf.worker_num) as executor:
        t = trange(len(data["data"]), desc='failed {}%'.format(failed_count / try_count * 100.0))
        for i in t:
            futures = []
            for j in range(len(data["data"][i]["paragraphs"])):
                id_list = get_target_id_list(data["data"][i]["paragraphs"][j]["qas"])
                for k in range(len(data["data"][i]["paragraphs"][j]["qas"])):
                    if data["data"][i]["paragraphs"][j]["qas"][k]['id'] in id_list:
                        future = executor.submit(augment, data["data"][i]["paragraphs"][j]["qas"], k)
                        futures.append(future)
                        try_count += 1
            for future in tqdm(futures):
                failed = future.result()
                if failed:
                    failed_count += 1
                    t.set_description('failed {}%'.format(failed_count / try_count * 100.0))

    print("failed = {}".format(failed_count))

    with open(conf.output, "w") as f:
        json.dump(data, f, ensure_ascii=False, sort_keys=True, indent=4, separators=(',', ': '))


if __name__ == '__main__':
    main()
