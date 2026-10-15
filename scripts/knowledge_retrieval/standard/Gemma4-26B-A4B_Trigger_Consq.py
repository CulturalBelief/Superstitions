from pathlib import Path as _BavarPath
_BAVAR_ROOT = _BavarPath(__file__).resolve().parents[3]
def _bavar_read_table(path, *args, **kwargs):
    kwargs.pop("sheet_name", None)
    return pd.read_csv(path, *args, **kwargs)

import asyncio
import os

import pandas as pd
from openai import AsyncOpenAI
from tqdm import tqdm


INPUT_FILE = str(_BAVAR_ROOT / 'data/knowledge_retrieval/benchmark.csv')
SHEET_NAME = "Question_Group_1_Bilingual"
OUTPUT_DIR = str(_BAVAR_ROOT / 'results/knowledge_retrieval/standard')

OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    "Gemma4-26B-A4B_concise.csv"
)

MODEL = "benchmark-model"

BASE_URL = "http://127.0.0.1:8000/v1"

CONCURRENCY = 48
MAX_TOKENS = 64
TEMPERATURE = 0.0
RETRIES = 5
CHECKPOINT_SIZE = 100


async def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    print("Loading dataset...")

    df = _bavar_read_table(
        INPUT_FILE,
        sheet_name=SHEET_NAME
    )

    print(
        f"Total questions: {len(df)}"
    )

    client = AsyncOpenAI(
        base_url=BASE_URL,
        api_key="EMPTY",
        timeout=180.0,
        max_retries=0
    )

    semaphore = asyncio.Semaphore(
        CONCURRENCY
    )

    async def query_model(question):

        async with semaphore:

            for attempt in range(
                RETRIES
            ):

                try:

                    prompt = (
                        f"{question}\n\n"
                        "Respond only with the believed consequence or outcome. "
                        "Do not explain your answer. "
                        "Do not provide cultural background. "
                        "Do not repeat the question. "
                        "Use one short sentence or phrase only."
                    )

                    response = await client.chat.completions.create(
                        model=MODEL,
                        messages=[
                            {
                                "role": "user",
                                "content": prompt
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
                    ).strip()

                    return answer

                except Exception as e:

                    if attempt == RETRIES - 1:

                        return (
                            f"ERROR: {e}"
                        )

                    wait_time = (
                        2 ** attempt
                    )

                    await asyncio.sleep(
                        wait_time
                    )

    result_df = pd.DataFrame(
        {
            "trigger": df["Trigger"],
            "outcome": df["Outcome"],
            "question": df["Question_EN"],
            "answer_generated": "",
            "golden_answer": df["Gold_Answer_EN"]
        }
    )

    if os.path.exists(
        OUTPUT_CSV
    ):

        print(
            "Existing checkpoint found."
        )

        old_df = pd.read_csv(
            OUTPUT_CSV
        )

        if (
            len(old_df)
            == len(result_df)
            and
            "answer_generated"
            in old_df.columns
        ):

            result_df[
                "answer_generated"
            ] = (
                old_df[
                    "answer_generated"
                ]
                .fillna("")
            )

    pending_indices = []

    for index, row in (
        result_df.iterrows()
    ):

        answer = row[
            "answer_generated"
        ]

        if (
            pd.isna(answer)
            or
            not str(answer).strip()
            or
            str(answer).startswith(
                "ERROR:"
            )
        ):

            pending_indices.append(
                index
            )

    print()
    print(
        "======================================"
    )

    print(
        "Model: Gemma4-26B-A4B"
    )

    print(
        f"Total questions: "
        f"{len(result_df)}"
    )

    print(
        f"Already completed: "
        f"{len(result_df) - len(pending_indices)}"
    )

    print(
        f"Pending: "
        f"{len(pending_indices)}"
    )

    print(
        "======================================"
    )

    print()

    progress = tqdm(
        total=len(
            pending_indices
        ),
        desc="Gemma4-26B-A4B"
    )

    for start in range(
        0,
        len(pending_indices),
        CHECKPOINT_SIZE
    ):

        indices = (
            pending_indices[
                start:
                start
                + CHECKPOINT_SIZE
            ]
        )

        tasks = []

        for index in indices:

            question = str(
                result_df.at[
                    index,
                    "question"
                ]
            )

            tasks.append(
                query_model(
                    question
                )
            )

        answers = await asyncio.gather(
            *tasks
        )

        for (
            index,
            answer
        ) in zip(
            indices,
            answers
        ):

            result_df.at[
                index,
                "answer_generated"
            ] = answer

        result_df.to_csv(
            OUTPUT_CSV,
            index=False,
            encoding="utf-8-sig"
        )

        progress.update(
            len(indices)
        )

    progress.close()

    result_df.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    successful = (
        result_df[
            "answer_generated"
        ]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
        .sum()
    )

    errors = (
        result_df[
            "answer_generated"
        ]
        .fillna("")
        .astype(str)
        .str.startswith(
            "ERROR:"
        )
        .sum()
    )

    print()

    print(
        "======================================"
    )

    print(
        "Gemma4-26B-A4B completed"
    )

    print(
        f"Successful: "
        f"{successful}/"
        f"{len(result_df)}"
    )

    print(
        f"Errors: {errors}"
    )

    print(
        "Saved to:"
    )

    print(
        OUTPUT_CSV
    )

    print(
        "======================================"
    )


if __name__ == "__main__":

    asyncio.run(
        main()
    )
