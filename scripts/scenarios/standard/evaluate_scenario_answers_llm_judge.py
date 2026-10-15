from pathlib import Path as _BavarPath
_BAVAR_ROOT = _BavarPath(__file__).resolve().parents[3]



























import argparse
import asyncio
import glob
import json
import os
import re
from pathlib import Path

import pandas as pd
from openai import AsyncOpenAI
from tqdm import tqdm


BENCHMARK_DEFAULT = (
    str(_BAVAR_ROOT / 'data/scenarios/benchmark_final.json')
)

RESPONSES_DEFAULT = (
    str(_BAVAR_ROOT / 'results/scenarios/model_outputs')
)

OUTPUT_DEFAULT = (
    str(_BAVAR_ROOT / 'results/scenarios/llm_judge')
)

BASE_URL = "http://127.0.0.1:8000/v1"
SERVED_MODEL = "benchmark-model"

CONCURRENCY = 32
TEMPERATURE = 0.0
MAX_TOKENS = 140
RETRIES = 5
CHECKPOINT_EVERY = 100


def parse_judge_output(text):
    raw = str(text or "").strip()

    try:
        obj = json.loads(raw)

        if isinstance(obj, dict):
            return {
                "label": str(
                    obj.get(
                        "label",
                        "",
                    )
                ).upper(),
                "reason": str(
                    obj.get(
                        "reason",
                        "",
                    )
                ),
            }

    except Exception:
        pass

    match = re.search(
        r"\{.*?\}",
        raw,
        flags=re.S,
    )

    if match:
        try:
            obj = json.loads(
                match.group(0)
            )

            return {
                "label": str(
                    obj.get(
                        "label",
                        "",
                    )
                ).upper(),
                "reason": str(
                    obj.get(
                        "reason",
                        "",
                    )
                ),
            }

        except Exception:
            pass

    label_match = re.search(
        r"\b(CORRECT|PARTIAL|INCORRECT)\b",
        raw.upper(),
    )

    if label_match:
        return {
            "label": label_match.group(1),
            "reason": "",
        }

    return {
        "label": "",
        "reason": "",
    }


def build_prompt(
    scenario,
    question,
    reference,
    model_response,
    answer_type,
):
    return f"""You are a strict evaluator of culturally grounded reasoning.

Evaluate the MODEL RESPONSE only against the documented REFERENCE KNOWLEDGE
and the QUESTION. Do not judge whether the cultural belief is scientifically
true. Judge whether the model correctly applies the documented belief.

SCENARIO:
{scenario}

QUESTION:
{question}

ANSWER TYPE:
{answer_type}

REFERENCE KNOWLEDGE:
{reference}

MODEL RESPONSE:
{model_response}

Use exactly one label:

CORRECT:
The response gives the culturally expected interpretation/outcome required by
the question. Paraphrases and concise answers are fully acceptable.

PARTIAL:
The response is directionally or generally correct, but misses an important
detail required by the question/reference.

For answer_type="valence_only", do not penalize a response merely for omitting
extra details if it correctly gives the positive/negative interpretation that
the question requires.

INCORRECT:
The response contradicts the reference, gives a different outcome, is
irrelevant, or is too vague to establish the required interpretation.

Return JSON only:
{{
  "label": "CORRECT|PARTIAL|INCORRECT",
  "reason": "one short sentence"
}}"""


async def judge_one(
    client,
    prompt,
    semaphore,
    qwen_judge,
):
    async with semaphore:
        for attempt in range(RETRIES):
            try:
                kwargs = {
                    "model": SERVED_MODEL,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are a strict semantic evaluator. "
                                "Return valid JSON only."
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

                if qwen_judge:
                    kwargs["extra_body"] = {
                        "chat_template_kwargs": {
                            "enable_thinking": False
                        }
                    }

                response = (
                    await client
                    .chat
                    .completions
                    .create(**kwargs)
                )

                return (
                    response
                    .choices[0]
                    .message
                    .content
                    or ""
                ).strip()

            except Exception as e:
                if attempt == RETRIES - 1:
                    return (
                        f"ERROR: "
                        f"{type(e).__name__}: {e}"
                    )

                await asyncio.sleep(
                    2 ** attempt
                )

        return "ERROR: no response"


async def evaluate_file(
    benchmark,
    response_file,
    output_dir,
    qwen_judge,
    judge_name,
):
    resp = pd.read_csv(
        response_file
    )

    required = [
        "Question_ID",
        "Model_Response",
    ]

    missing = [
        c for c in required
        if c not in resp.columns
    ]

    if missing:
        raise ValueError(
            f"{response_file}: "
            f"missing {missing}"
        )

    keep = [
        "Question_ID",
        "Model_Response",
    ]

    if "Model" in resp.columns:
        keep.append("Model")

    resp = (
        resp[keep]
        .drop_duplicates(
            "Question_ID",
            keep="last",
        )
    )

    df = benchmark.merge(
        resp,
        on="Question_ID",
        how="left",
        validate="one_to_one",
    )

    if (
        "Model" in df.columns
        and df["Model"].notna().any()
    ):
        model_name = str(
            df["Model"]
            .dropna()
            .iloc[0]
        )
    else:
        model_name = (
            Path(response_file)
            .stem
            .replace(
                "_scenario_answers",
                "",
            )
        )

    out_file = (
        output_dir
        / f"{model_name}_LLM_JUDGE.csv"
    )

    if out_file.exists():
        previous = pd.read_csv(
            out_file
        )

        prev_cols = [
            c for c in [
                "Question_ID",
                "Judge_Raw",
                "Judge_Label",
                "Judge_Reason",
            ]
            if c in previous.columns
        ]

        previous = (
            previous[prev_cols]
            .drop_duplicates(
                "Question_ID",
                keep="last",
            )
        )

        df = df.merge(
            previous,
            on="Question_ID",
            how="left",
        )

    for col in [
        "Judge_Raw",
        "Judge_Label",
        "Judge_Reason",
    ]:
        if col not in df.columns:
            df[col] = ""

    pending = []

    for i, row in df.iterrows():
        answer = (
            ""
            if pd.isna(
                row["Model_Response"]
            )
            else str(
                row["Model_Response"]
            ).strip()
        )

        label = (
            ""
            if pd.isna(
                row["Judge_Label"]
            )
            else str(
                row["Judge_Label"]
            )
            .strip()
            .upper()
        )

        if (
            answer
            and not answer.startswith("ERROR:")
            and label not in {
                "CORRECT",
                "PARTIAL",
                "INCORRECT",
            }
        ):
            pending.append(i)

    client = AsyncOpenAI(
        base_url=BASE_URL,
        api_key="EMPTY",
    )

    semaphore = asyncio.Semaphore(
        CONCURRENCY
    )

    bar = tqdm(
        total=len(pending),
        desc=f"Judge {model_name}",
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
                scenario=str(
                    df.at[i, "scenario"]
                ),
                question=str(
                    df.at[i, "question"]
                ),
                reference=str(
                    df.at[
                        i,
                        "expected_cultural_knowledge",
                    ]
                ),
                model_response=str(
                    df.at[
                        i,
                        "Model_Response",
                    ]
                ),
                answer_type=str(
                    df.at[
                        i,
                        "answer_type",
                    ]
                    if "answer_type"
                    in df.columns
                    else ""
                ),
            )

            tasks.append(
                judge_one(
                    client,
                    prompt,
                    semaphore,
                    qwen_judge,
                )
            )

        raws = await asyncio.gather(
            *tasks
        )

        for i, raw in zip(
            idxs,
            raws,
        ):
            parsed = parse_judge_output(
                raw
            )

            df.at[
                i,
                "Judge_Raw",
            ] = raw

            df.at[
                i,
                "Judge_Label",
            ] = parsed["label"]

            df.at[
                i,
                "Judge_Reason",
            ] = parsed["reason"]

            bar.update(1)

        df.to_csv(
            out_file,
            index=False,
            encoding="utf-8-sig",
        )

    bar.close()

    labels = (
        df["Judge_Label"]
        .fillna("")
        .astype(str)
        .str.upper()
    )

    df["Judge_Strict_Correct"] = (
        labels == "CORRECT"
    ).astype(int)

    soft_map = {
        "CORRECT": 1.0,
        "PARTIAL": 0.5,
        "INCORRECT": 0.0,
    }

    df["Judge_Soft_Score"] = (
        labels
        .map(soft_map)
        .fillna(0.0)
    )

    df["Judge_Model"] = judge_name

    df.to_csv(
        out_file,
        index=False,
        encoding="utf-8-sig",
    )

    return model_name, df


async def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--benchmark",
        default=BENCHMARK_DEFAULT,
    )

    ap.add_argument(
        "--responses_dir",
        default=RESPONSES_DEFAULT,
    )

    ap.add_argument(
        "--output_dir",
        default=OUTPUT_DEFAULT,
    )

    ap.add_argument(
        "--judge_name",
        default="fixed-judge",
    )

    ap.add_argument(
        "--qwen_judge",
        action="store_true",
    )

    args = ap.parse_args()

    benchmark = pd.read_json(
        args.benchmark
    )

    required = [
        "Question_ID",
        "scenario",
        "question",
        "expected_cultural_knowledge",
    ]

    missing = [
        c for c in required
        if c not in benchmark.columns
    ]

    if missing:
        raise ValueError(
            f"Benchmark missing: {missing}"
        )

    response_files = sorted(
        glob.glob(
            os.path.join(
                args.responses_dir,
                "*_scenario_answers.csv",
            )
        )
    )

    if not response_files:
        raise FileNotFoundError(
            f"No response files in "
            f"{args.responses_dir}"
        )

    out = Path(
        args.output_dir
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_rows = []
    subgroup_rows = []

    for f in response_files:
        model_name, df = (
            await evaluate_file(
                benchmark=benchmark,
                response_file=f,
                output_dir=out,
                qwen_judge=args.qwen_judge,
                judge_name=args.judge_name,
            )
        )

        strict = float(
            df[
                "Judge_Strict_Correct"
            ].mean()
        )

        soft = float(
            df[
                "Judge_Soft_Score"
            ].mean()
        )

        labels = (
            df["Judge_Label"]
            .fillna("")
            .astype(str)
            .str.upper()
        )

        model_rows.append({
            "Model": model_name,
            "Judge": args.judge_name,
            "N": len(df),
            "Strict_Accuracy": strict,
            "Soft_Score": soft,
            "Correct_Rate": float(
                (
                    labels == "CORRECT"
                ).mean()
            ),
            "Partial_Rate": float(
                (
                    labels == "PARTIAL"
                ).mean()
            ),
            "Incorrect_Rate": float(
                (
                    labels == "INCORRECT"
                ).mean()
            ),
        })

        for group_col in [
            "Macro_Cultural_Group",
            "difficulty",
            "answer_type",
            "benchmark_group",
        ]:
            if group_col not in df.columns:
                continue

            for value, g in df.groupby(
                group_col,
                dropna=False,
            ):
                glabels = (
                    g["Judge_Label"]
                    .fillna("")
                    .astype(str)
                    .str.upper()
                )

                subgroup_rows.append({
                    "Model": model_name,
                    "Judge": args.judge_name,
                    "Group_Type": group_col,
                    "Group": str(value),
                    "N": len(g),
                    "Strict_Accuracy": float(
                        g[
                            "Judge_Strict_Correct"
                        ].mean()
                    ),
                    "Soft_Score": float(
                        g[
                            "Judge_Soft_Score"
                        ].mean()
                    ),
                    "Correct_Rate": float(
                        (
                            glabels == "CORRECT"
                        ).mean()
                    ),
                    "Partial_Rate": float(
                        (
                            glabels == "PARTIAL"
                        ).mean()
                    ),
                    "Incorrect_Rate": float(
                        (
                            glabels == "INCORRECT"
                        ).mean()
                    ),
                })

        print(
            f"{model_name}: "
            f"strict={strict:.4f} "
            f"soft={soft:.4f}"
        )

    (
        pd.DataFrame(model_rows)
        .sort_values(
            "Strict_Accuracy",
            ascending=False,
        )
        .to_csv(
            out
            / "LLM_JUDGE_MODEL_SUMMARY.csv",
            index=False,
            encoding="utf-8-sig",
        )
    )

    pd.DataFrame(
        subgroup_rows
    ).to_csv(
        out
        / "LLM_JUDGE_SUBGROUP_RESULTS.csv",
        index=False,
        encoding="utf-8-sig",
    )


if __name__ == "__main__":
    asyncio.run(main())
