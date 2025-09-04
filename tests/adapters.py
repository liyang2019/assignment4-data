from __future__ import annotations

import os
import re
from typing import Any

from cs336_data import data, train_quality_classifier


def run_extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    return data.extract_text(html_bytes)


def run_identify_language(text: str) -> tuple[Any, float]:
    return data.language_identification(text)


def run_mask_emails(text: str) -> tuple[str, int]:
    return data.mask_emails(text)


def run_mask_phone_numbers(text: str) -> tuple[str, int]:
    return data.mask_phone_numbers(text)


def run_mask_ips(text: str) -> tuple[str, int]:
    return data.mask_ips(text)


def run_classify_nsfw(text: str) -> tuple[Any, float]:
    return data.classify_nsfw(text)


def run_classify_toxic_speech(text: str) -> tuple[Any, float]:
    return data.classify_toxic_speech(text)


def run_classify_quality(text: str) -> tuple[Any, float]:
    return data.classify_quality(text)


def run_gopher_quality_filter(text: str) -> bool:
    return data.gopher_quality_filters(text)


def run_exact_line_deduplication(
    input_files: list[os.PathLike], output_directory: os.PathLike
):
    return data.exact_line_deduplication(input_files, output_directory)


def run_minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    data.minhash_deduplication(
        input_files,
        num_hashes,
        num_bands,
        ngrams,
        jaccard_threshold,
        output_directory,
    )
