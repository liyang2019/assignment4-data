import collections
import hashlib
import os
import random
import re
import string
import unicodedata
from typing import Any

import fasttext
import nltk
import numpy as np
import resiliparse
import resiliparse.extract.html2text
import resiliparse.parse.encoding
from fastwarc.warc import ArchiveIterator, WarcRecordType

nltk.download("punkt_tab")

lang_model = None
nsfw_model = None
toxic_model = None
quality_model = None


def extract_text(html_bytes: bytes):
    encoding = resiliparse.parse.encoding.detect_encoding(html_bytes)
    text = html_bytes.decode(encoding, errors="replace")
    output = resiliparse.extract.html2text.extract_plain_text(text)
    return output


def run_extract_text():
    with open(
        "/home/liyang2029/cs336_2025/assignment4-data/cs336_data/example.warc.gz", "rb"
    ) as f:
        warc = ArchiveIterator(f)
        for record in warc:
            if record.record_type == WarcRecordType.response:
                return extract_text(record.reader.read())
        raise ValueError("No response record found in warc file")


def language_identification(text: str) -> tuple[Any, float]:
    global lang_model
    if lang_model is None:
        lang_model = fasttext.load_model(
            "/home/liyang2029/cs336_2025/assignment4-data/cs336_data/lid.176.bin"
        )
    identifiers, scores = lang_model.predict(text.replace("\n", " "))
    assert len(identifiers) == 1
    return identifiers[0].replace("__label__", ""), scores[0]


def run_language_identification():
    with open(
        "/home/liyang2029/cs336_2025/assignment4-data/cs336_data/example.warc.gz", "rb"
    ) as f:
        warc = ArchiveIterator(f)
        i = 0
        for record in warc:
            if record.record_type == WarcRecordType.response:
                i += 1
                text = extract_text(record.reader.read())
                lang, score = language_identification(text)
                print("====================", lang, score)
                print(text.replace("\n", " "))
                if i >= 20:
                    return


def mask_emails(text: str) -> tuple[str, int]:
    return re.subn(
        r"[a-zA-Z0-9]+\@[a-zA-Z0-9]+\.[a-zA-Z0-9]+",
        "|||EMAIL_ADDRESS|||",
        text,
    )


def mask_phone_numbers(text: str) -> tuple[str, int]:
    return re.subn(
        r"\d?[-]?[\(]?\d{3}[\)]?[-\s]?\d{3}[-\s]?\d{4}",
        "|||PHONE_NUMBER|||",
        text,
    )


def mask_ips(text: str) -> tuple[str, int]:
    p = "(25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9])"
    return re.subn(
        rf"{p}\.{p}\.{p}\.{p}",
        "|||IP_ADDRESS|||",
        text,
    )


def run_mask_pii():
    with open(
        "/home/liyang2029/cs336_2025/assignment4-data/cs336_data/example.warc.gz", "rb"
    ) as f:
        warc = ArchiveIterator(f)
        i = 0
        for record in warc:
            if record.record_type == WarcRecordType.response:
                text = extract_text(record.reader.read())
                count = 0
                text, c = mask_emails(text)
                count += c
                text, c = mask_phone_numbers(text)
                count += c
                text, c = mask_ips(text)
                count += c
                if count > 0:
                    i += 1
                    print("====================", count)
                    print(text.replace("\n", " "))
                    if i >= 20:
                        return


def classify_nsfw(text: str) -> tuple[str, float]:
    global nsfw_model
    if nsfw_model is None:
        nsfw_model = fasttext.load_model(
            "/home/liyang2029/cs336_2025/assignment4-data/cs336_data/jigsaw_fasttext_bigrams_nsfw_final.bin"
        )
    identifiers, scores = nsfw_model.predict(text.replace("\n", " "))
    assert len(identifiers) == 1
    return identifiers[0].replace("__label__", ""), scores[0]


def classify_toxic_speech(text: str) -> tuple[str, float]:
    global toxic_model
    if toxic_model is None:
        toxic_model = fasttext.load_model(
            "/home/liyang2029/cs336_2025/assignment4-data/cs336_data/jigsaw_fasttext_bigrams_hatespeech_final.bin"
        )
    identifiers, scores = toxic_model.predict(text.replace("\n", " "))
    assert len(identifiers) == 1
    return identifiers[0].replace("__label__", ""), scores[0]


def run_harmful_identification():
    with open(
        "/home/liyang2029/cs336_2025/assignment4-data/cs336_data/example.warc.gz", "rb"
    ) as f:
        warc = ArchiveIterator(f)
        i = 0
        for record in warc:
            if record.record_type == WarcRecordType.response:
                i += 1
                text = extract_text(record.reader.read())
                identifier, score = classify_nsfw(text)
                print("====================", identifier, score)
                identifier, score = classify_toxic_speech(text)
                print("====================", identifier, score)
                print(text.replace("\n", " "))
                if i >= 20:
                    break


def gopher_quality_filters(text: str) -> bool:
    words = nltk.word_tokenize(text)
    if len(words) < 50 or len(words) > 100_000:
        return False
    mean_word_length = sum(len(word) for word in words) / len(words)
    if mean_word_length < 3 or mean_word_length > 10:
        return False
    lines = text.split("\n")
    if sum(line.endswith("...") for line in lines) / len(lines) > 0.3:
        return False
    if (
        sum(re.search(r"[a-zA-Z]", word) is not None for word in words) / len(words)
        < 0.8
    ):
        return False
    return True


def run_gopher_quality_filters():
    with open(
        "/home/liyang2029/cs336_2025/assignment4-data/cs336_data/example.warc.gz", "rb"
    ) as f:
        warc = ArchiveIterator(f)
        i = 0
        for record in warc:
            if record.record_type == WarcRecordType.response:
                i += 1
                text = extract_text(record.reader.read())
                label = gopher_quality_filters(text)
                print("====================", label)
                print(text.replace("\n", " "))
                if i >= 20:
                    break


def classify_quality(text: str) -> tuple[str, float]:
    global quality_model
    if quality_model is None:
        quality_model = fasttext.load_model(
            "/home/liyang2029/cs336_2025/assignment4-data/cs336_data/quality_classifier_model.bin"
        )
    identifiers, scores = quality_model.predict(text.replace("\n", " "))
    assert len(identifiers) == 1
    return identifiers[0].replace("__label__", ""), scores[0]


def exact_line_deduplication(paths: list[os.PathLike], output_dir: os.PathLike) -> None:
    counts = collections.Counter()
    for path in paths:
        with open(path, "r") as f:
            for line in f.readlines():
                counts[hash(line)] += 1
    for path in paths:
        with open(path, "r") as f:
            lines = f.readlines()
            new_lines = []
            for line in lines:
                if counts[hash(line)] == 1:
                    new_lines.append(line)
        with open(os.path.join(output_dir, os.path.basename(path)), "w") as f:
            f.writelines(new_lines)


def strip_accents(text: str) -> str:
    # Decompose, drop combining marks, then recompose
    decomposed = unicodedata.normalize("NFKD", text)
    filtered = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return unicodedata.normalize("NFC", filtered)


def blake2_keyed(data, key, digest_size=32):
    if isinstance(data, str):
        data = data.encode("utf-8")
    if isinstance(key, str):
        key = key.encode("utf-8")
    return hashlib.blake2b(data, key=key, digest_size=digest_size).hexdigest()


def minhash_deduplication(
    paths: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    n_gram_length: int,
    jaccard_threshold: float,
    output_dir: os.PathLike,
) -> None:
    docs_by_minhash = collections.defaultdict(set)
    for i, path in enumerate(paths):
        with open(path, "r") as f:
            text = f.read()
            text = text.lower()
            text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
            text = re.sub(r"\s", " ", text)
            text = strip_accents(text)
            text = unicodedata.normalize("NFD", text)
            ngrams = nltk.ngrams(nltk.word_tokenize(text), n_gram_length)
            ngrams = ["".join(ngram) for ngram in ngrams]
            minhashes = []
            for key in range(num_hashes):
                minhashes.append(min(blake2_keyed(ngram, str(key)) for ngram in ngrams))
            bands = [tuple(b.tolist()) for b in np.array_split(minhashes, num_bands)]
            for j, band in enumerate(bands):
                docs_by_minhash[(j, band)].add(i)
    print("=========")
    print(docs_by_minhash.values())
    removed = set()
    for key, docs in docs_by_minhash.items():
        docs -= removed
        if len(docs) > 1:
            docs = list(docs)
            random.shuffle(docs)
            doc = docs[0]
            removed |= set(docs[1:])
            path = paths[doc]
            with open(path, "r") as f:
                text = f.read()
            with open(os.path.join(output_dir, os.path.basename(path)), "w") as f:
                f.write(text)


if __name__ == "__main__":
    # print(run_extract_text())
    # model = fasttext.load_model(
    #     "/home/liyang2029/cs336_2025/assignment4-data/cs336_data/lid.176.bin"
    # )
    # run_language_identification()
    # run_mask_pii()
    # run_harmful_identification()
    run_gopher_quality_filters()
