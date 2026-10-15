from pathlib import Path as _BavarPath
_BAVAR_ROOT = _BavarPath(__file__).resolve().parents[3]

import os
import re
import glob
import string
from collections import Counter

import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
from sacrebleu.metrics import BLEU
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer






INPUT_DIR = str(_BAVAR_ROOT / 'results/knowledge_retrieval/cot')

OUTPUT_DIR = os.path.join(
    INPUT_DIR,
    "evaluation"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)






SBERT_MODEL = "BAAI/bge-large-en-v1.5"

HF_CACHE = str(_BAVAR_ROOT / '.cache/huggingface')

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

SBERT_BATCH_SIZE = 128






bleu_metric = BLEU(
    effective_order=True,
    smooth_method="exp"
)

rouge_metric = rouge_scorer.RougeScorer(
    [
        "rouge1",
        "rouge2",
        "rougeL"
    ],
    use_stemmer=True
)






def normalize_text(text):

    if pd.isna(text):
        return ""

    text = str(text).lower().strip()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    text = text.translate(
        str.maketrans(
            "",
            "",
            string.punctuation
        )
    )

    return text.strip()


def tokenize(text):

    text = normalize_text(text)

    return re.findall(
        r"\b\w+\b",
        text
    )






def exact_match(
    prediction,
    reference
):

    prediction = normalize_text(
        prediction
    )

    reference = normalize_text(
        reference
    )

    return int(
        prediction == reference
    )






def token_overlap_scores(
    prediction,
    reference
):

    pred_tokens = tokenize(
        prediction
    )

    ref_tokens = tokenize(
        reference
    )

    if (
        len(pred_tokens) == 0
        or len(ref_tokens) == 0
    ):
        return 0.0, 0.0, 0.0

    common = (
        Counter(pred_tokens)
        &
        Counter(ref_tokens)
    )

    overlap = sum(
        common.values()
    )

    if overlap == 0:
        return 0.0, 0.0, 0.0

    precision = (
        overlap
        /
        len(pred_tokens)
    )

    recall = (
        overlap
        /
        len(ref_tokens)
    )

    f1 = (
        2
        *
        precision
        *
        recall
        /
        (
            precision
            +
            recall
        )
    )

    return (
        precision,
        recall,
        f1
    )






def jaccard_similarity(
    prediction,
    reference
):

    pred_tokens = set(
        tokenize(
            prediction
        )
    )

    ref_tokens = set(
        tokenize(
            reference
        )
    )

    if (
        len(pred_tokens) == 0
        or len(ref_tokens) == 0
    ):
        return 0.0

    intersection = len(
        pred_tokens
        &
        ref_tokens
    )

    union = len(
        pred_tokens
        |
        ref_tokens
    )

    if union == 0:
        return 0.0

    return (
        intersection
        /
        union
    )






def sentence_bleu(
    prediction,
    reference
):

    score = bleu_metric.sentence_score(
        str(prediction),
        [
            str(reference)
        ]
    ).score

    return (
        score
        /
        100.0
    )






def calculate_rouge(
    prediction,
    reference
):

    scores = rouge_metric.score(
        str(reference),
        str(prediction)
    )

    rouge1 = scores[
        "rouge1"
    ].fmeasure

    rouge2 = scores[
        "rouge2"
    ].fmeasure

    rougeL = scores[
        "rougeL"
    ].fmeasure

    return (
        rouge1,
        rouge2,
        rougeL
    )






print()
print(
    "========================================"
)

print(
    "Loading SBERT model..."
)

print(
    f"Model: {SBERT_MODEL}"
)

print(
    f"Device: {DEVICE}"
)

print(
    "========================================"
)

print()


sbert_model = SentenceTransformer(
    SBERT_MODEL,
    device=DEVICE,
    cache_folder=HF_CACHE
)


print(
    "SBERT loaded successfully."
)

print()






csv_files = sorted(
    glob.glob(
        os.path.join(
            INPUT_DIR,
            "*.csv"
        )
    )
)


print(
    f"Found {len(csv_files)} CSV files in:"
)

print(
    INPUT_DIR
)

print()






summary_results = []






for file_path in csv_files:

    filename = os.path.basename(
        file_path
    )

    model_name = os.path.splitext(
        filename
    )[0]


    print()
    print(
        "============================================================"
    )

    print(
        f"Processing: {model_name}"
    )

    print(
        "============================================================"
    )


    try:

        df = pd.read_csv(
            file_path
        )

    except Exception as e:

        print(
            f"Could not read file: {e}"
        )

        continue


    
    
    

    required_columns = [
        "answer_generated",
        "golden_answer"
    ]

    missing_columns = [
        column
        for column
        in required_columns
        if column not in df.columns
    ]


    if missing_columns:

        print(
            "Skipped."
        )

        print(
            f"Missing columns: {missing_columns}"
        )

        continue


    
    
    

    generated = (
        df[
            "answer_generated"
        ]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    golden = (
        df[
            "golden_answer"
        ]
        .fillna("")
        .astype(str)
        .str.strip()
    )


    valid_mask = (
        generated.ne("")
        &
        golden.ne("")
        &
        ~generated.str.startswith(
            "ERROR:"
        )
    )


    n_total = len(
        df
    )

    n_scored = int(
        valid_mask.sum()
    )

    n_missing = (
        n_total
        -
        n_scored
    )


    print(
        f"Total rows: {n_total}"
    )

    print(
        f"Scored rows: {n_scored}"
    )

    print(
        f"Missing/error rows: {n_missing}"
    )


    if n_scored == 0:

        print(
            "No valid answers found."
        )

        continue


    eval_df = (
        df.loc[
            valid_mask
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


    predictions = (
        eval_df[
            "answer_generated"
        ]
        .fillna("")
        .astype(str)
        .tolist()
    )

    references = (
        eval_df[
            "golden_answer"
        ]
        .fillna("")
        .astype(str)
        .tolist()
    )


    
    
    

    print(
        "Calculating Exact Match..."
    )


    exact_scores = [
        exact_match(
            pred,
            ref
        )

        for pred, ref

        in zip(
            predictions,
            references
        )
    ]


    
    
    

    print(
        "Calculating Token Precision, Recall, F1..."
    )


    token_precision_scores = []

    token_recall_scores = []

    token_f1_scores = []


    for pred, ref in zip(
        predictions,
        references
    ):

        precision, recall, f1 = (
            token_overlap_scores(
                pred,
                ref
            )
        )

        token_precision_scores.append(
            precision
        )

        token_recall_scores.append(
            recall
        )

        token_f1_scores.append(
            f1
        )


    
    
    

    print(
        "Calculating Jaccard similarity..."
    )


    jaccard_scores = [
        jaccard_similarity(
            pred,
            ref
        )

        for pred, ref

        in zip(
            predictions,
            references
        )
    ]


    
    
    

    print(
        "Calculating BLEU..."
    )


    bleu_scores = []


    for pred, ref in tqdm(
        zip(
            predictions,
            references
        ),
        total=n_scored,
        desc="BLEU"
    ):

        score = sentence_bleu(
            pred,
            ref
        )

        bleu_scores.append(
            score
        )


    corpus_bleu = (
        bleu_metric
        .corpus_score(
            predictions,
            [
                references
            ]
        )
        .score
        /
        100.0
    )


    
    
    

    print(
        "Calculating ROUGE..."
    )


    rouge1_scores = []

    rouge2_scores = []

    rougeL_scores = []


    for pred, ref in tqdm(
        zip(
            predictions,
            references
        ),
        total=n_scored,
        desc="ROUGE"
    ):

        r1, r2, rL = (
            calculate_rouge(
                pred,
                ref
            )
        )

        rouge1_scores.append(
            r1
        )

        rouge2_scores.append(
            r2
        )

        rougeL_scores.append(
            rL
        )


    
    
    

    print(
        "Encoding generated answers with SBERT..."
    )


    pred_embeddings = (
        sbert_model.encode(
            predictions,
            batch_size=SBERT_BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
    )


    print(
        "Encoding golden answers with SBERT..."
    )


    ref_embeddings = (
        sbert_model.encode(
            references,
            batch_size=SBERT_BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
    )


    
    

    sbert_scores = np.sum(
        pred_embeddings
        *
        ref_embeddings,
        axis=1
    )


    
    
    

    generated_word_count = [
        len(
            tokenize(
                text
            )
        )
        for text
        in predictions
    ]


    golden_word_count = [
        len(
            tokenize(
                text
            )
        )
        for text
        in references
    ]


    
    
    

    eval_df[
        "exact_match"
    ] = exact_scores


    eval_df[
        "token_precision"
    ] = token_precision_scores


    eval_df[
        "token_recall"
    ] = token_recall_scores


    eval_df[
        "token_f1"
    ] = token_f1_scores


    eval_df[
        "jaccard"
    ] = jaccard_scores


    eval_df[
        "bleu"
    ] = bleu_scores


    eval_df[
        "rouge1"
    ] = rouge1_scores


    eval_df[
        "rouge2"
    ] = rouge2_scores


    eval_df[
        "rougeL"
    ] = rougeL_scores


    eval_df[
        "sbert_cosine"
    ] = sbert_scores


    eval_df[
        "generated_word_count"
    ] = generated_word_count


    eval_df[
        "golden_word_count"
    ] = golden_word_count


    
    
    

    item_output = os.path.join(
        OUTPUT_DIR,
        f"{model_name}_metrics.csv"
    )


    eval_df.to_csv(
        item_output,
        index=False,
        encoding="utf-8-sig"
    )


    
    
    

    model_summary = {

        "model": model_name,

        "n_total": n_total,

        "n_scored": n_scored,

        "n_missing_or_error": n_missing,

        "coverage": (
            n_scored
            /
            n_total
        ),

        "exact_match": float(
            np.mean(
                exact_scores
            )
        ),

        "token_precision": float(
            np.mean(
                token_precision_scores
            )
        ),

        "token_recall": float(
            np.mean(
                token_recall_scores
            )
        ),

        "token_f1": float(
            np.mean(
                token_f1_scores
            )
        ),

        "jaccard": float(
            np.mean(
                jaccard_scores
            )
        ),

        "mean_sentence_bleu": float(
            np.mean(
                bleu_scores
            )
        ),

        "corpus_bleu": float(
            corpus_bleu
        ),

        "rouge1": float(
            np.mean(
                rouge1_scores
            )
        ),

        "rouge2": float(
            np.mean(
                rouge2_scores
            )
        ),

        "rougeL": float(
            np.mean(
                rougeL_scores
            )
        ),

        "sbert_cosine": float(
            np.mean(
                sbert_scores
            )
        ),

        "avg_generated_words": float(
            np.mean(
                generated_word_count
            )
        ),

        "avg_golden_words": float(
            np.mean(
                golden_word_count
            )
        )
    }


    summary_results.append(
        model_summary
    )


    
    
    

    print()
    print(
        "Results:"
    )

    print(
        f"Exact Match       : "
        f"{model_summary['exact_match']:.4f}"
    )

    print(
        f"Token Precision   : "
        f"{model_summary['token_precision']:.4f}"
    )

    print(
        f"Token Recall      : "
        f"{model_summary['token_recall']:.4f}"
    )

    print(
        f"Token F1          : "
        f"{model_summary['token_f1']:.4f}"
    )

    print(
        f"Jaccard           : "
        f"{model_summary['jaccard']:.4f}"
    )

    print(
        f"Sentence BLEU     : "
        f"{model_summary['mean_sentence_bleu']:.4f}"
    )

    print(
        f"Corpus BLEU       : "
        f"{model_summary['corpus_bleu']:.4f}"
    )

    print(
        f"ROUGE-1           : "
        f"{model_summary['rouge1']:.4f}"
    )

    print(
        f"ROUGE-2           : "
        f"{model_summary['rouge2']:.4f}"
    )

    print(
        f"ROUGE-L           : "
        f"{model_summary['rougeL']:.4f}"
    )

    print(
        f"SBERT Cosine      : "
        f"{model_summary['sbert_cosine']:.4f}"
    )

    print(
        f"Avg Generated Len : "
        f"{model_summary['avg_generated_words']:.2f}"
    )

    print(
        f"Avg Golden Len    : "
        f"{model_summary['avg_golden_words']:.2f}"
    )

    print()

    print(
        f"Saved item-level results:"
    )

    print(
        item_output
    )






summary_df = pd.DataFrame(
    summary_results
)


if len(summary_df) > 0:

    summary_df = (
        summary_df
        .sort_values(
            by="sbert_cosine",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )


summary_output = os.path.join(
    OUTPUT_DIR,
    "model_level_metrics.csv"
)


summary_df.to_csv(
    summary_output,
    index=False,
    encoding="utf-8-sig"
)






paper_columns = [

    "model",

    "n_scored",

    "exact_match",

    "token_f1",

    "jaccard",

    "mean_sentence_bleu",

    "corpus_bleu",

    "rouge1",

    "rouge2",

    "rougeL",

    "sbert_cosine"
]


available_paper_columns = [
    column
    for column
    in paper_columns
    if column in summary_df.columns
]


paper_df = summary_df[
    available_paper_columns
].copy()


numeric_columns = (
    paper_df
    .select_dtypes(
        include=[
            np.number
        ]
    )
    .columns
)


for column in numeric_columns:

    if column != "n_scored":

        paper_df[
            column
        ] = (
            paper_df[
                column
            ]
            .round(
                4
            )
        )


paper_output = os.path.join(
    OUTPUT_DIR,
    "paper_metrics.csv"
)


paper_df.to_csv(
    paper_output,
    index=False,
    encoding="utf-8-sig"
)






print()
print(
    "============================================================"
)

print(
    "ALL MODELS COMPLETED"
)

print(
    "============================================================"
)

print()

print(
    "Model-level metrics:"
)

print(
    summary_output
)

print()

print(
    "Paper-friendly metrics:"
)

print(
    paper_output
)

print()

print(
    "Item-level files are saved in:"
)

print(
    OUTPUT_DIR
)

print()