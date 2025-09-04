from fastwarc.warc import ArchiveIterator, WarcRecordType
import random
import tqdm
import fasttext

from cs336_data import data


def load_and_run_filters():
    with open(
        "/home/liyang2029/cs336_2025/assignment4-data/cs336_data/subsampled_positive_urls.warc.warc.gz",
        "rb",
    ) as f:
        texts = []
        warc = ArchiveIterator(f)
        num_filtered = 0
        for record in tqdm.tqdm(warc):
            if record.record_type == WarcRecordType.response:
                text = data.extract_text(record.reader.read())
                if not data.gopher_quality_filters(text):
                    num_filtered += 1
                    continue
                identifier, score = data.classify_nsfw(text)
                if identifier == "nsfw" and score > 0.9:
                    num_filtered += 1
                    continue
                identifier, score = data.classify_toxic_speech(text)
                if identifier == "toxic" and score > 0.9:
                    num_filtered += 1
                    continue
                texts.append(text)
        print(f"total number of filtered records: {num_filtered}")
        print(f"total number of remaining records: {len(texts)}")
    return texts


def sample_warc(max_records: int = 200_000):
    with open(
        "/home/liyang2029/cs336_2025/assignment4-data/cs336_data/example.warc.gz", "rb"
    ) as f:
        texts = []
        warc = ArchiveIterator(f)
        for record in warc:
            if record.record_type == WarcRecordType.response:
                text = data.extract_text(record.reader.read())
                texts.append(text)
        print(f"total number of records: {len(texts)}")
    return random.sample(texts, min(len(texts), max_records))


def data_generation():
    positives = load_and_run_filters()
    negatives = sample_warc(max_records=200_000)
    positives *= len(negatives) // len(positives)
    print(f"total number of positives: {len(positives)}")
    print(f"total number of negatives: {len(negatives)}")
    positives = [f"__label__wiki {t.replace('\n', ' ')}" for t in positives]
    negatives = [f"__label__cc {t.replace('\n', ' ')}" for t in negatives]
    all_texts = positives + negatives
    random.shuffle(all_texts)
    train_texts = all_texts[: int(len(all_texts) * 0.8)]
    valid_texts = all_texts[int(len(all_texts) * 0.8) :]
    print(f"total number of train texts: {len(train_texts)}")
    print(f"total number of valid texts: {len(valid_texts)}")
    with open(
        "/home/liyang2029/cs336_2025/assignment4-data/cs336_data/quality_classifier_data.train",
        "w",
    ) as f:
        for text in train_texts:
            f.write(f"{text}\n")
    with open(
        "/home/liyang2029/cs336_2025/assignment4-data/cs336_data/quality_classifier_data.valid",
        "w",
    ) as f:
        for text in valid_texts:
            f.write(f"{text}\n")
    print("done")


def train_model():
    model = fasttext.train_supervised(
        "/home/liyang2029/cs336_2025/assignment4-data/cs336_data/quality_classifier_data.train"
    )
    model.save_model(
        "/home/liyang2029/cs336_2025/assignment4-data/cs336_data/quality_classifier_model.bin"
    )
    print("predict")
    print(model.predict("Which baking dish is best to bake a banana bread ?"))
    print("eval")
    print(
        model.test(
            "/home/liyang2029/cs336_2025/assignment4-data/cs336_data/quality_classifier_data.valid"
        )
    )


if __name__ == "__main__":
    data_generation()
    train_model()
