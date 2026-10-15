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


INPUT_FILE = str(_BAVAR_ROOT / 'data/discrimination/mcq_none_of_the_above.csv')
OUTPUT_DIR = str(_BAVAR_ROOT / 'results/discrimination/none_of_the_above')

TAG = "Gemma-4-26B-A4B-NO-CULTURE-GOLD-REMOVED-NONE"

OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    f"{TAG}_mcq.csv"
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


def parse_mcq(text):
    if text is None:
        return ""

    x = str(text).strip().upper()

    x = re.sub(
        r'^[\s"\'`*_#]+|[\s"\'`*_#]+$',
        "",
        x
    )

    exact = re.fullmatch(
        r"\(?([ABCD])\)?[.!]?",
        x
    )

    if exact:
        return exact.group(1)

    patterns = [
        r"\bANSWER\s*(?:IS)?\s*[:\-]?\s*([ABCD])\b",
        r"\bOPTION\s*[:\-]?\s*([ABCD])\b",
        r"\bCHOICE\s*[:\-]?\s*([ABCD])\b",
        r"\(([ABCD])\)",
        r"\b([ABCD])\b",
    ]

    for pattern in patterns:
        found = re.findall(pattern, x)
        unique = list(dict.fromkeys(found))

        if len(unique) == 1:
            return unique[0]

    return ""


def build_prompt(row):
    
    
    
    
    trigger = str(row["Trigger_EN"]).strip()

    return (
        f"Question:\n"
        f"What is believed to happen when the following occurs: "
        f"{trigger}?\n\n"
        f"A. {row['Option_A']}\n"
        f"B. {row['Option_B']}\n"
        f"C. {row['Option_C']}\n"
        f"D. {row['Option_D']}\n\n"
        f"Answer only A, B, C, or D."
    )


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
                                "Answer the multiple-choice question. "
                                "Select the answer that corresponds to "
                                "the documented folk belief. "
                                "Answer only A, B, C, or D. "
                                "Do not explain your answer."
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

                message = response.choices[0].message

                answer = (
                    message.content
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
        INPUT_FILE
    )

    required = [
        "Question_ID",
        "Trigger_EN",
        "Option_A",
        "Option_B",
        "Option_C",
        "Option_D",
        "Gold_Label",
        "Gold_Answer_Text"
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

    optional_columns = [
        "Belief_ID",
        "Culture_EN",
        "Culture_FA",
        "Taxonomy",
        "Trigger_FA",
        "Question_EN",
        "Question_FA",
        "Correct_Answer_EN",
        "Correct_Answer_FA",
        "Gold_Verified",
        "Original_Gold_Answer_Text",
        "Condition"
    ]

    keep_columns = []

    for col in optional_columns + required:
        if col in df.columns and col not in keep_columns:
            keep_columns.append(col)

    result_df = df[
        keep_columns
    ].copy()

    
    result_df["Question_No_Culture"] = result_df.apply(
        lambda row: (
            "What is believed to happen when the following occurs: "
            f"{str(row['Trigger_EN']).strip()}?"
        ),
        axis=1
    )

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

        parsed = str(
            row["Parsed_Response"]
        ).strip().upper()

        if (
            pd.isna(response)
            or not str(response).strip()
            or str(response).startswith("ERROR:")
            or parsed not in ["A", "B", "C", "D"]
        ):
            pending.append(i)


    print("=" * 70)
    print(f"Model: {TAG}")
    print("Condition: NO CULTURE + GOLD REMOVED; gold = None of these options")
    print("Condition: NO CULTURE")
    print(f"Input: {INPUT_FILE}")
    print(f"Total questions: {len(result_df)}")
    print(f"Pending questions: {len(pending)}")
    print("=" * 70)


    print("\nExample prompt sent to model:\n")

    if len(result_df) > 0:
        print(
            build_prompt(
                result_df.iloc[0]
            )
        )

    print("\n" + "=" * 70)


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
                build_prompt(
                    result_df.loc[i]
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

            parsed = parse_mcq(
                answer
            )

            gold = str(
                result_df.at[
                    i,
                    "Gold_Label"
                ]
            ).strip().upper()

            result_df.at[
                i,
                "Model_Response"
            ] = answer

            result_df.at[
                i,
                "Parsed_Response"
            ] = parsed

            if parsed in ["A", "B", "C", "D"]:

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
            index=False,
            encoding="utf-8-sig"
        )


    progress.close()


    result_df.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig"
    )


    valid_mask = (
        result_df["Parsed_Response"]
        .isin(["A", "B", "C", "D"])
    )

    overall_accuracy = (
        result_df["Correct"]
        .fillna(0)
        .astype(float)
        .mean()
    )

    valid_rate = valid_mask.mean()

    n_invalid = int(
        (~valid_mask).sum()
    )


    print()
    print("=" * 70)

    print(
        f"Saved: {OUTPUT_CSV}"
    )

    print(
        f"Overall accuracy: "
        f"{overall_accuracy:.4f}"
    )

    print(
        f"Valid parsed response rate: "
        f"{valid_rate:.4f}"
    )

    print(
        f"Invalid responses: "
        f"{n_invalid}"
    )


    print("\nPrediction counts:")

    for label in ["A", "B", "C", "D"]:

        count = (
            result_df["Parsed_Response"]
            .eq(label)
            .sum()
        )

        print(
            f"{label}: {count}"
        )


    print("\nAccuracy by gold position:")

    for label in ["A", "B", "C", "D"]:

        subset = result_df[
            result_df["Gold_Label"]
            .astype(str)
            .str.strip()
            .str.upper()
            == label
        ]

        if len(subset) > 0:

            acc = (
                subset["Correct"]
                .fillna(0)
                .astype(float)
                .mean()
            )

            print(
                f"Gold {label}: "
                f"{acc:.4f} "
                f"(n={len(subset)})"
            )


    print("=" * 70)


if __name__ == "__main__":

    asyncio.run(
        main()
    )
