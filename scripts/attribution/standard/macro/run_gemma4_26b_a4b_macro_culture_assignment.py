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


INPUT_FILE = str(_BAVAR_ROOT / 'data/knowledge_retrieval/benchmark.csv')
OUTPUT_DIR = str(_BAVAR_ROOT / 'results/attribution/standard')


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


TAG = 'Gemma-4-26B-A4B-MACRO-CULTURE'
OUTPUT_CSV = os.path.join(OUTPUT_DIR, f"{TAG}_predictions.csv")
LABEL_MAP_CSV = os.path.join(OUTPUT_DIR, f"{TAG}_label_map.csv")

def load_dataset():
    belief_df = _bavar_read_table(INPUT_FILE, sheet_name="Merged_Deduplicated")
    label_df = _bavar_read_table(INPUT_FILE, sheet_name="Question_Group_1")

    required_belief = ["Merged_ID", "Belief_Text"]
    required_label = ["Belief_ID", "Culture_For_Question", "Macro_Cultural_Group"]

    missing_belief = [c for c in required_belief if c not in belief_df.columns]
    missing_label = [c for c in required_label if c not in label_df.columns]

    if missing_belief:
        raise ValueError(f"Missing columns in Merged_Deduplicated: {missing_belief}")
    if missing_label:
        raise ValueError(f"Missing columns in Question_Group_1: {missing_label}")

    keep_label_cols = [
        "Belief_ID",
        "Culture_For_Question",
        "Macro_Cultural_Group",
        "Culture_Normalized_All",
        "Culture_Raw_All",
        "Culture_Status",
        "Culture_Label_Count",
        "Taxonomy",
    ]
    keep_label_cols = [c for c in keep_label_cols if c in label_df.columns]

    df = belief_df[["Merged_ID", "Belief_Text"]].merge(
        label_df[keep_label_cols],
        left_on="Merged_ID",
        right_on="Belief_ID",
        how="inner",
        validate="one_to_one",
    )

    if len(df) != len(belief_df):
        raise ValueError(
            f"Merge produced {len(df)} rows but Merged_Deduplicated has {len(belief_df)} rows."
        )

    return df

def build_label_map(df):
    df["Macro_Cultural_Group"] = (
        df["Macro_Cultural_Group"].fillna("").astype(str).str.strip()
    )
    if (df["Macro_Cultural_Group"] == "").any():
        bad = int((df["Macro_Cultural_Group"] == "").sum())
        raise ValueError(f"Found {bad} rows with empty Macro_Cultural_Group.")

    labels = sorted(
        df["Macro_Cultural_Group"].dropna().astype(str).str.strip().unique().tolist(),
        key=str.casefold,
    )
    code_to_label = {f"M{i:02d}": label for i, label in enumerate(labels, start=1)}
    label_to_code = {label: code for code, label in code_to_label.items()}
    return code_to_label, label_to_code


def build_candidate_block(code_to_label):
    return "\n".join(f"{code}: {label}" for code, label in code_to_label.items())


def build_prompt(belief_text, candidate_block):
    belief_text = str(belief_text).strip()
    return (
        "Task: infer the most likely MACRO cultural group associated with the folk belief below.\n\n"
        "This is a CLOSED-SET classification task. Choose exactly ONE label code from the allowed list.\n"
        "Use the belief content itself to infer the macro-cultural group.\n"
        "Do not invent a new group, do not return the group name, and do not explain.\n\n"
        "Allowed macro-cultural labels:\n"
        f"{candidate_block}\n\n"
        "Belief:\n"
        f"{belief_text}\n\n"
        "Answer only the label code, for example: M03"
    )


def parse_code(text, valid_codes):
    if text is None:
        return ""
    x = str(text).strip().upper()
    x = re.sub(r'^[\s"\'`*_#]+|[\s"\'`*_#]+$', '', x)
    exact = re.fullmatch(r"(M\d{2})[.!]?", x)
    if exact and exact.group(1) in valid_codes:
        return exact.group(1)
    found = re.findall(r"\bM\d{2}\b", x)
    found = [c for c in dict.fromkeys(found) if c in valid_codes]
    return found[0] if len(found) == 1 else ""


async def ask_model(prompt, semaphore):
    async with semaphore:
        for attempt in range(RETRIES):
            try:
                response = await client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": 'You are performing closed-set macro-cultural attribution of folk beliefs. Use only the supplied candidate labels. Return only the requested label code with no explanation.',
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS
                )
                return (response.choices[0].message.content or "").strip()
            except Exception as e:
                if attempt == RETRIES - 1:
                    return f"ERROR: {e}"
                await asyncio.sleep(2 ** attempt)


async def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_dataset()
    code_to_label, label_to_code = build_label_map(df)
    valid_codes = set(code_to_label)
    candidate_block = build_candidate_block(code_to_label)

    pd.DataFrame(
        [{"Label_Code": c, "Macro_Culture_Label": l} for c, l in code_to_label.items()]
    ).to_csv(LABEL_MAP_CSV, index=False, encoding="utf-8-sig")

    result_df = df.copy()
    result_df["Gold_Macro_Culture"] = result_df["Macro_Cultural_Group"]
    result_df["Gold_Code"] = result_df["Gold_Macro_Culture"].map(label_to_code)
    result_df["Model"] = TAG
    result_df["Model_Response"] = pd.Series("", index=result_df.index, dtype="string")
    result_df["Predicted_Code"] = pd.Series("", index=result_df.index, dtype="string")
    result_df["Predicted_Macro_Culture"] = pd.Series("", index=result_df.index, dtype="string")
    result_df["Correct"] = pd.Series(pd.NA, index=result_df.index, dtype="Int64")

    if os.path.exists(OUTPUT_CSV):
        old = pd.read_csv(OUTPUT_CSV)
        if len(old) == len(result_df):
            for col in ["Model_Response", "Predicted_Code", "Predicted_Macro_Culture"]:
                if col in old.columns:
                    result_df[col] = old[col].fillna("").astype("string")
            if "Correct" in old.columns:
                result_df["Correct"] = pd.to_numeric(old["Correct"], errors="coerce").astype("Int64")
            print(f"Checkpoint found: {OUTPUT_CSV}")

    pending = []
    for i, row in result_df.iterrows():
        response = row["Model_Response"]
        code = str(row["Predicted_Code"]).strip().upper()
        if (
            pd.isna(response)
            or not str(response).strip()
            or str(response).startswith("ERROR:")
            or code not in valid_codes
        ):
            pending.append(i)

    print("=" * 70)
    print(f"Model: {TAG}")
    print("Task: MACRO CULTURE ASSIGNMENT")
    print(f"Input: {INPUT_FILE}")
    print(f"Total beliefs: {len(result_df)}")
    print(f"Allowed macro-cultural labels: {len(code_to_label)}")
    print(f"Pending beliefs: {len(pending)}")
    print("Qwen thinking: N/A for this model family")
    print("=" * 70)

    if len(result_df) > 0:
        print("\nExample prompt sent to model:\n")
        print(build_prompt(result_df.iloc[0]["Belief_Text"], candidate_block))
        print("\n" + "=" * 70)

    semaphore = asyncio.Semaphore(CONCURRENCY)
    progress = tqdm(total=len(pending), desc=TAG)

    for start in range(0, len(pending), CHECKPOINT_SIZE):
        batch_indices = pending[start:start + CHECKPOINT_SIZE]
        tasks = [
            ask_model(
                build_prompt(result_df.at[i, "Belief_Text"], candidate_block),
                semaphore,
            )
            for i in batch_indices
        ]
        answers = await asyncio.gather(*tasks)

        for i, answer in zip(batch_indices, answers):
            code = parse_code(answer, valid_codes)
            pred_label = code_to_label.get(code, "")
            gold_code = str(result_df.at[i, "Gold_Code"]).strip().upper()

            result_df.at[i, "Model_Response"] = answer
            result_df.at[i, "Predicted_Code"] = code
            result_df.at[i, "Predicted_Macro_Culture"] = pred_label
            result_df.at[i, "Correct"] = int(code == gold_code) if code else 0
            progress.update(1)

        result_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    progress.close()
    result_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    valid_mask = result_df["Predicted_Code"].isin(valid_codes)
    accuracy = result_df["Correct"].fillna(0).astype(float).mean()
    valid_rate = valid_mask.mean()
    n_invalid = int((~valid_mask).sum())

    print("\n" + "=" * 70)
    print(f"Saved predictions: {OUTPUT_CSV}")
    print(f"Saved label map: {LABEL_MAP_CSV}")
    print(f"Overall exact-match accuracy: {accuracy:.4f}")
    print(f"Valid parsed response rate: {valid_rate:.4f}")
    print(f"Invalid responses: {n_invalid}")

    print("\nTop predicted macro cultures:")
    counts = result_df["Predicted_Macro_Culture"].value_counts().head(20)
    for label, count in counts.items():
        print(f"{label}: {count}")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
