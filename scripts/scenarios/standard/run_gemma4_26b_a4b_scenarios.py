from pathlib import Path as _BavarPath
_BAVAR_ROOT = _BavarPath(__file__).resolve().parents[3]


import asyncio
import json
import os
from pathlib import Path

import pandas as pd
from openai import AsyncOpenAI
from tqdm import tqdm





BENCHMARK_PATH = str(_BAVAR_ROOT / 'data/scenarios/benchmark_final.json')
OUTPUT_DIR = str(_BAVAR_ROOT / 'results/scenarios/model_outputs')



MODEL = "benchmark-model"
MODEL_TAG = "Gemma-4-26B-A4B"


QWEN_THINKING_OFF = False

BASE_URL = "http://127.0.0.1:8000/v1"

CONCURRENCY = 48
TEMPERATURE = 0.0
MAX_TOKENS = 160
RETRIES = 5
CHECKPOINT_EVERY = 100

OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    f"{MODEL_TAG}_scenario_answers.csv",
)

client = AsyncOpenAI(
    base_url=BASE_URL,
    api_key="EMPTY",
)


def load_benchmark():
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("benchmark_final.json must contain a JSON list.")

    df = pd.DataFrame(data)

    required = [
        "Question_ID",
        "scenario",
        "question",
        "expected_cultural_knowledge",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required benchmark fields: {missing}"
        )

    if df["Question_ID"].duplicated().any():
        dupes = (
            df.loc[df["Question_ID"].duplicated(), "Question_ID"]
            .astype(str)
            .tolist()
        )
        raise ValueError(
            f"Duplicate Question_ID values found: {dupes[:20]}"
        )

    
    usable = (
        df["Question_ID"].notna()
        & df["scenario"].notna()
        & df["question"].notna()
        & df["scenario"].astype(str).str.strip().ne("")
        & df["question"].astype(str).str.strip().ne("")
    )

    skipped = df.loc[~usable].copy()
    df = df.loc[usable].reset_index(drop=True)

    return df, skipped


def build_prompt(scenario, question):
    
    
    
    return f"""Scenario:
{scenario}

Question:
{question}

Answer the question directly and concisely in one or two sentences.
Do not explain your chain of thought."""


async def ask_model(prompt, semaphore):
    async with semaphore:
        for attempt in range(RETRIES):
            try:
                kwargs = {
                    "model": MODEL,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are answering a culturally grounded "
                                "scenario benchmark. Read the scenario carefully "
                                "and answer the question directly. Give only the "
                                "final answer in one or two concise sentences. "
                                "Do not provide chain-of-thought reasoning."
                            ),
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                    "temperature": TEMPERATURE,
                    "max_tokens": MAX_TOKENS,
                }

                if QWEN_THINKING_OFF:
                    kwargs["extra_body"] = {
                        "chat_template_kwargs": {
                            "enable_thinking": False
                        }
                    }

                response = await client.chat.completions.create(
                    **kwargs
                )

                text = (
                    response.choices[0].message.content or ""
                ).strip()

                if text:
                    return text

            except Exception as e:
                if attempt == RETRIES - 1:
                    return f"ERROR: {type(e).__name__}: {e}"

            await asyncio.sleep(2 ** attempt)

        return "ERROR: empty response"


async def main():
    Path(OUTPUT_DIR).mkdir(
        parents=True,
        exist_ok=True,
    )

    df, skipped = load_benchmark()

    if len(skipped):
        skipped.to_csv(
            os.path.join(
                OUTPUT_DIR,
                "skipped_unusable_benchmark_items.csv",
            ),
            index=False,
            encoding="utf-8-sig",
        )

    
    result = df.copy()
    result["Model"] = MODEL_TAG
    result["Model_Response"] = ""

    
    
    
    if os.path.exists(OUTPUT_CSV):
        old = pd.read_csv(OUTPUT_CSV)

        if (
            "Question_ID" in old.columns
            and "Model_Response" in old.columns
        ):
            old_map = (
                old[["Question_ID", "Model_Response"]]
                .drop_duplicates("Question_ID", keep="last")
                .set_index("Question_ID")["Model_Response"]
                .fillna("")
                .astype(str)
                .to_dict()
            )

            result["Model_Response"] = (
                result["Question_ID"]
                .map(old_map)
                .fillna("")
                .astype(str)
            )

            print(
                f"Resuming from checkpoint: {OUTPUT_CSV}"
            )

    pending = []

    for i, r in result.iterrows():
        ans = str(
            r.get("Model_Response", "") or ""
        ).strip()

        if (
            not ans
            or ans.startswith("ERROR:")
        ):
            pending.append(i)

    print("=" * 72)
    print(f"MODEL TAG        : {MODEL_TAG}")
    print(f"BENCHMARK        : {BENCHMARK_PATH}")
    print(f"TOTAL ITEMS      : {len(result)}")
    print(f"PENDING          : {len(pending)}")
    print(
        f"QWEN THINKING OFF: {QWEN_THINKING_OFF}"
    )
    print("=" * 72)

    semaphore = asyncio.Semaphore(
        CONCURRENCY
    )

    bar = tqdm(
        total=len(pending),
        desc=MODEL_TAG,
    )

    for start in range(
        0,
        len(pending),
        CHECKPOINT_EVERY,
    ):
        idxs = pending[
            start:start + CHECKPOINT_EVERY
        ]

        tasks = []

        for i in idxs:
            prompt = build_prompt(
                str(result.at[i, "scenario"]),
                str(result.at[i, "question"]),
            )

            tasks.append(
                ask_model(
                    prompt,
                    semaphore,
                )
            )

        answers = await asyncio.gather(
            *tasks
        )

        for i, answer in zip(
            idxs,
            answers,
        ):
            result.at[
                i,
                "Model_Response",
            ] = answer

            bar.update(1)

        
        result.to_csv(
            OUTPUT_CSV,
            index=False,
            encoding="utf-8-sig",
        )

    bar.close()

    result["Response_Word_Count"] = (
        result["Model_Response"]
        .fillna("")
        .astype(str)
        .str.split()
        .str.len()
    )

    result.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    errors = (
        result["Model_Response"]
        .fillna("")
        .astype(str)
        .str.startswith("ERROR:")
        .sum()
    )

    empty = (
        result["Model_Response"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    print()
    print("=" * 72)
    print(f"SAVED : {OUTPUT_CSV}")
    print(f"ERRORS: {errors}")
    print(f"EMPTY : {empty}")
    print("=" * 72)


if __name__ == "__main__":
    asyncio.run(main())
