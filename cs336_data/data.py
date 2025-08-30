import re
from typing import Any

import fasttext
import resiliparse
import resiliparse.extract.html2text
import resiliparse.parse.encoding
from fastwarc.warc import ArchiveIterator, WarcRecordType

fasttext_model = fasttext.load_model(
    "/home/liyang2029/cs336_2025/assignment4-data/cs336_data/lid.176.bin"
)


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
    identifiers, scores = fasttext_model.predict(text.replace("\n", " "))
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


if __name__ == "__main__":
    # print(run_extract_text())
    # model = fasttext.load_model(
    #     "/home/liyang2029/cs336_2025/assignment4-data/cs336_data/lid.176.bin"
    # )
    # run_language_identification()
    run_mask_pii()
