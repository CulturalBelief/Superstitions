from pathlib import Path as _BavarPath
_BAVAR_ROOT = _BavarPath(__file__).resolve().parents[3]
def _bavar_read_table(path, *args, **kwargs):
    kwargs.pop("sheet_name", None)
    return pd.read_csv(path, *args, **kwargs)

import asyncio
import os
import re

import pandas as pd
from openai import AsyncOpenAI
from tqdm import tqdm


INPUT_FILE = str(_BAVAR_ROOT / 'data/verification/true_false.csv')
SHEET_NAME = "True_False_All"

OUTPUT_DIR = str(_BAVAR_ROOT / 'results/verification/cot')

TAG = "Gemma-4-26B-A4B-CoT"

OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    f"{TAG}_true_false.csv"
)

MODEL = "benchmark-model"
BASE_URL = "http://127.0.0.1:8000/v1"

CONCURRENCY = 48
MAX_TOKENS = 8
TEMPERATURE = 0.0
RETRIES = 5
CHECKPOINT_SIZE = 100


client = AsyncOpenAI(
    base_url=BASE_URL,
    api_key="EMPTY"
)


def parse_true_false(text):
    if text is None:
        return ""

    x = str(text).strip().lower()

    x = re.sub(
        r"^[\s\"'`*_#]+|[\s\"'`*_#]+$",
        "",
        x
    )

    if re.fullmatch(r"true[.!]?", x):
        return "True"

    if re.fullmatch(r"false[.!]?", x):
        return "False"

    matches = re.findall(
        r"\b(true|false)\b",
        x
    )

    if len(matches) == 1:
        return matches[0].capitalize()

    if matches and len(set(matches)) == 1:
        return matches[0].capitalize()

    return ""


async def ask_model(question, semaphore):
    async with semaphore:
        for attempt in range(RETRIES):
            try:
                response = await client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are answering a cultural-belief "
                                "verification benchmark. "
                                "Before answering, think carefully step by step "
                                "and reason internally about the question. "
                                "Do not reveal your reasoning or provide any explanation. "
                                "Output only True or False."
                            )
                        },
                        {
                            "role": "user",
                            "content": question
                        }
                    ],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS
                )

                answer = (
                    response
                    .choices[0]
                    .message
                    .content
                    or ""
                )

                return answer.strip()

            except Exception as e:
                if attempt == RETRIES - 1:
                    return f"ERROR: {e}"

                await asyncio.sleep(
                    2 ** attempt
                )


async def main():
    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    df = _bavar_read_table(
        INPUT_FILE,
        sheet_name=SHEET_NAME
    )

    required = [
        "Pair_ID",
        "Culture",
        "Question",
        "Gold_Label"
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    result_df = df[
        required
    ].copy()

    result_df["Model"] = TAG

    result_df["Model_Response"] = pd.Series(
        "",
        index=result_df.index,
        dtype="string"
    )

    result_df["Parsed_Response"] = pd.Series(
        "",
        index=result_df.index,
        dtype="string"
    )

    result_df["Correct"] = pd.Series(
        pd.NA,
        index=result_df.index,
        dtype="Int64"
    )

    if os.path.exists(OUTPUT_CSV):
        old = pd.read_csv(
            OUTPUT_CSV
        )

        if len(old) == len(result_df):
            if "Model_Response" in old.columns:
                result_df["Model_Response"] = (
                    old["Model_Response"]
                    .fillna("")
                    .astype("string")
                )

            if "Parsed_Response" in old.columns:
                result_df["Parsed_Response"] = (
                    old["Parsed_Response"]
                    .fillna("")
                    .astype("string")
                )

            if "Correct" in old.columns:
                result_df["Correct"] = (
                    pd.to_numeric(
                        old["Correct"],
                        errors="coerce"
                    )
                    .astype("Int64")
                )

            print(
                f"Checkpoint found: {OUTPUT_CSV}"
            )

    pending = []

    for i, row in result_df.iterrows():
        response = row["Model_Response"]

        if (
            pd.isna(response)
            or not str(response).strip()
            or str(response).startswith("ERROR:")
        ):
            pending.append(i)

    print("=" * 70)
    print(f"Model: {TAG}")
    print(f"Input: {INPUT_FILE}")
    print(f"Total questions: {len(result_df)}")
    print(f"Pending questions: {len(pending)}")
    print("=" * 70)

    semaphore = asyncio.Semaphore(
        CONCURRENCY
    )

    progress = tqdm(
        total=len(pending),
        desc=TAG
    )

    for start in range(
        0,
        len(pending),
        CHECKPOINT_SIZE
    ):
        batch_indices = pending[
            start:start + CHECKPOINT_SIZE
        ]

        tasks = [
            ask_model(
                str(
                    result_df.at[
                        i,
                        "Question"
                    ]
                ),
                semaphore
            )
            for i in batch_indices
        ]

        answers = await asyncio.gather(
            *tasks
        )

        for i, answer in zip(
            batch_indices,
            answers
        ):
            parsed = parse_true_false(
                answer
            )

            gold = str(
                result_df.at[
                    i,
                    "Gold_Label"
                ]
            ).strip().capitalize()

            result_df.at[
                i,
                "Model_Response"
            ] = answer

            result_df.at[
                i,
                "Parsed_Response"
            ] = parsed

            if parsed in [
                "True",
                "False"
            ]:
                result_df.at[
                    i,
                    "Correct"
                ] = int(
                    parsed == gold
                )
            else:
                result_df.at[
                    i,
                    "Correct"
                ] = 0

            progress.update(1)

        result_df.to_csv(
            OUTPUT_CSV,
            index=False
        )

    progress.close()

    result_df.to_csv(
        OUTPUT_CSV,
        index=False
    )

    true_df = result_df[
        result_df["Gold_Label"]
        .astype(str)
        .str.strip()
        .str.lower()
        == "true"
    ]

    false_df = result_df[
        result_df["Gold_Label"]
        .astype(str)
        .str.strip()
        .str.lower()
        == "false"
    ]

    overall_accuracy = (
        result_df["Correct"]
        .fillna(0)
        .astype(float)
        .mean()
    )

    true_accuracy = (
        true_df["Correct"]
        .fillna(0)
        .astype(float)
        .mean()
    )

    false_accuracy = (
        false_df["Correct"]
        .fillna(0)
        .astype(float)
        .mean()
    )

    valid_rate = (
        result_df["Parsed_Response"]
        .isin(
            ["True", "False"]
        )
        .mean()
    )

    n_true_response = (
        result_df["Parsed_Response"]
        .eq("True")
        .sum()
    )

    n_false_response = (
        result_df["Parsed_Response"]
        .eq("False")
        .sum()
    )

    n_invalid = (
        ~result_df["Parsed_Response"]
        .isin(
            ["True", "False"]
        )
    ).sum()

    print()
    print("=" * 70)
    print(f"Saved: {OUTPUT_CSV}")
    print(
        f"Overall accuracy: "
        f"{overall_accuracy:.4f}"
    )
    print(
        f"True accuracy: "
        f"{true_accuracy:.4f}"
    )
    print(
        f"False accuracy: "
        f"{false_accuracy:.4f}"
    )
    print(
        f"Valid parsed response rate: "
        f"{valid_rate:.4f}"
    )
    print(
        f"Model answered True: "
        f"{n_true_response}"
    )
    print(
        f"Model answered False: "
        f"{n_false_response}"
    )
    print(
        f"Invalid responses: "
        f"{n_invalid}"
    )
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(
        main()
    )
