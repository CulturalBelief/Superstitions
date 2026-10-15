from pathlib import Path as _BavarPath
_BAVAR_ROOT = _BavarPath(__file__).resolve().parents[3]

import os
import glob
import pandas as pd


INPUT_DIR = str(_BAVAR_ROOT / 'results/verification/cot')

OUTPUT_FILE = os.path.join(
    INPUT_DIR,
    "true_false_pair_metrics.csv"
)


csv_files = sorted(
    glob.glob(
        os.path.join(
            INPUT_DIR,
            "*_true_false.csv"
        )
    )
)


summary = []


for file_path in csv_files:

    df = pd.read_csv(
        file_path
    )

    required = [
        "Pair_ID",
        "Gold_Label",
        "Parsed_Response",
        "Correct"
    ]

    if not all(
        col in df.columns
        for col in required
    ):
        print(
            f"Skipping {file_path}: "
            "required columns missing"
        )
        continue

    if "Model" in df.columns:
        model = str(
            df["Model"].iloc[0]
        )
    else:
        model = os.path.basename(
            file_path
        )

    df["Gold_Label"] = (
        df["Gold_Label"]
        .astype(str)
        .str.strip()
        .str.capitalize()
    )

    df["Parsed_Response"] = (
        df["Parsed_Response"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.capitalize()
    )

    true_rows = df[
        df["Gold_Label"] == "True"
    ].copy()

    false_rows = df[
        df["Gold_Label"] == "False"
    ].copy()

    true_map = (
        true_rows
        .set_index("Pair_ID")[
            "Parsed_Response"
        ]
    )

    false_map = (
        false_rows
        .set_index("Pair_ID")[
            "Parsed_Response"
        ]
    )

    pair_ids = sorted(
        set(true_map.index)
        & set(false_map.index)
    )

    correct_discrimination = 0
    true_bias = 0
    false_bias = 0
    reversed_pairs = 0
    invalid_pairs = 0

    for pair_id in pair_ids:

        true_answer = true_map.loc[
            pair_id
        ]

        false_answer = false_map.loc[
            pair_id
        ]

        if (
            true_answer not in [
                "True",
                "False"
            ]
            or false_answer not in [
                "True",
                "False"
            ]
        ):
            invalid_pairs += 1

        elif (
            true_answer == "True"
            and false_answer == "False"
        ):
            correct_discrimination += 1

        elif (
            true_answer == "True"
            and false_answer == "True"
        ):
            true_bias += 1

        elif (
            true_answer == "False"
            and false_answer == "False"
        ):
            false_bias += 1

        elif (
            true_answer == "False"
            and false_answer == "True"
        ):
            reversed_pairs += 1

    n_pairs = len(
        pair_ids
    )

    numeric_correct = pd.to_numeric(
        df["Correct"],
        errors="coerce"
    ).fillna(0)

    true_correct = pd.to_numeric(
        true_rows["Correct"],
        errors="coerce"
    ).fillna(0)

    false_correct = pd.to_numeric(
        false_rows["Correct"],
        errors="coerce"
    ).fillna(0)

    summary.append(
        {
            "model": model,
            "n_pairs": n_pairs,
            "overall_accuracy": (
                numeric_correct.mean()
            ),
            "true_accuracy": (
                true_correct.mean()
            ),
            "false_accuracy": (
                false_correct.mean()
            ),
            "correct_pair_rate": (
                correct_discrimination
                / n_pairs
                if n_pairs
                else 0
            ),
            "true_bias_rate": (
                true_bias
                / n_pairs
                if n_pairs
                else 0
            ),
            "false_bias_rate": (
                false_bias
                / n_pairs
                if n_pairs
                else 0
            ),
            "reversal_rate": (
                reversed_pairs
                / n_pairs
                if n_pairs
                else 0
            ),
            "invalid_pair_rate": (
                invalid_pairs
                / n_pairs
                if n_pairs
                else 0
            )
        }
    )


summary_df = pd.DataFrame(
    summary
)


if not summary_df.empty:

    summary_df = (
        summary_df
        .sort_values(
            "correct_pair_rate",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )


summary_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print(
    summary_df.to_string(
        index=False
    )
)

print()
print(
    f"Saved: {OUTPUT_FILE}"
)
