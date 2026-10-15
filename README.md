<div align="center">
  <img src="assets/Superstitions-image.png" alt="BAVAR: a watercolor panorama celebrating cultural traditions" width="100%" />

  <h1>🌏 BAVAR</h1>

  <p><strong>Belief-Aware Verification, Attribution, and Reasoning for Cultural Superstitions in Large Language Models</strong></p>

  <p><em>Do language models know cultural beliefs, attribute them correctly, and reason about them without assuming everyone shares them?</em></p>

  <p>
    <img alt="Benchmark" src="https://img.shields.io/badge/Benchmark-Cultural%20Reasoning-2563EB?style=flat-square" />
    <img alt="LLM Evaluation" src="https://img.shields.io/badge/Focus-LLM%20Evaluation-0F766E?style=flat-square" />
    <img alt="Code license MIT" src="https://img.shields.io/badge/Code%20License-MIT-64748B?style=flat-square" />
    <img alt="Data release in preparation" src="https://img.shields.io/badge/Datasets-Release%20in%20Preparation-D97706?style=flat-square" />
  </p>

  <p>
    <a href="#overview">Overview</a> ·
    <a href="#benchmark-tasks">Tasks</a> ·
    <a href="#datasets">Datasets</a> ·
    <a href="#key-findings">Findings</a> ·
    <a href="#evaluation">Evaluation</a> ·
    <a href="#citation">Citation</a>
  </p>
</div>

---

## Overview

**BAVAR** is a benchmark for evaluating how large language models represent, retrieve, verify, and apply culturally grounded superstitions. It separates three related but distinct abilities:

- **Know:** Recover or verify the association between a culturally documented *trigger* and its *believed consequence*.
- **Attribute:** Connect a belief to an appropriate cultural context, at both fine-grained and coarse-grained levels.
- **Calibrate:** Avoid inferring that an individual endorses a belief based on cultural background alone, especially when contextual evidence is incomplete.
**Important distinction:** BAVAR evaluates whether a model accurately represents a *documented belief*. It does **not** evaluate whether a superstition is scientifically true, nor does cultural association imply individual endorsement.

## Benchmark tasks

| Task | What the model must do | Response format | Evaluation |
|:--|:--|:--|:--|
| **01 · Knowledge Retrieval** | Infer the culturally associated consequence given a trigger | Open-ended text | LLM-as-a-judge: Correct / Partial / Incorrect / Unknown |
| **02 · Knowledge Verification** | Independently identify correct and mismatched trigger–consequence statements | True / False | Overall, True, False, and matched-pair accuracy |
| **03 · Knowledge Discrimination** | Recognize when the correct consequence is absent among plausible choices | Multiple choice with **None of the Above** | Accuracy |
| **04 · Scenario-Based Reasoning** | Apply the relevant cultural association to a naturalistic situation | Open-ended text | LLM-as-a-judge with partial credit |
| **05 · Cultural Attribution** | Identify the culture associated with a documented superstition | Culture selection | Fine-grained and coarse-grained accuracy |

## Datasets
**Taxonomy:** Omen · Luck · Protective · Ritual · Divination · Taboo · Contagion · Other.
<details>
<summary><strong>📁 Repository structure</strong></summary>

```text
Superstitions/
├── assets/
│   └── Superstitions.png
├── data/
│   ├── knowledge_retrieval/
│   ├── verification/
│   ├── discrimination/
│   ├── scenarios/
│   ├── attribution/
│   └── calibration/
├── prompts/
├── scripts/
├── results/
├── docs/
│   ├── DATASET_CARD.md
│   └── RELEASE_CHECKLIST.md
├── LICENSE                

```

</details>

## Key findings

The numbers below summarize the **manuscript's reported evaluations**, not results rerun by this GitHub starter template.

| Evaluation | Reported result | Interpretation |
|:--|:--|:--|
| Open-ended knowledge retrieval | **45.13%** best overall score (Mistral-Small-3.2-24B) | Direct retrieval of cultural consequences remains difficult |
| True/False verification | **65.42%** mean overall accuracy; **37.44%** mean pair accuracy | Recognizing a true statement does not guarantee rejection of a false association |
| None-of-the-Above discrimination | **21.46%** mean standard-prompting accuracy | Models often choose an incorrect but plausible consequence |
| Scenario-based reasoning | **67.79%** best overall score (Qwen3.5-27B) | Contextual cues can help apply belief knowledge |
| Cultural attribution | **41.20%** mean fine-grained vs. **71.05%** mean coarse-grained accuracy | Regional recognition is stronger than precise cultural attribution |

Additional experiments in the manuscript examine chain-of-thought prompting, fine-tuning, category-level performance, and agreement between human and LLM-as-a-judge assessments. See the paper for full protocols and model-level tables.

## Evaluation

BAVAR combines classification and open-ended evaluation. For open-ended retrieval and scenario reasoning, responses receive one of four labels:

| Label | Meaning | Credit |
|:--|:--|:--:|
| **Correct** | Captures the documented cultural association | 1.0 |
| **Partially Correct** | Captures meaningful but incomplete information | 0.5 |
| **Incorrect** | Conflicts with the documented association | 0.0 |
| **Unknown** | Nonresponsive or insufficient evidence | 0.0 |

The reported open-ended overall score is:

```text
Overall score = %Correct + 0.5 × %Partially Correct
```

For True/False verification, **pair accuracy** is the percentage of matched True/False pairs for which **both** statements are classified correctly. Attribution is evaluated at both specific-culture and broader-region levels.



Source files, exact dataset columns, model versions, and execution commands will be documented in their corresponding directories when released. No script is advertised as runnable until its implementation has been uploaded.

## Responsible use

Cultural superstitions are **documented associations**, not universal characteristics of communities or people. Researchers should not use a culture label to infer a person's beliefs, prescribe behavior to an individual, or treat these associations as verified causal claims. Preserve source provenance and review licensing and quotation rights before redistributing material derived from books.

## Citation

The associated manuscript is titled:

> **BAVAR: Belief-Aware Verification, Attribution, and Reasoning for Cultural Superstitions in Large Language Models**

The paper introducing BAVAR is currently under review. A public version of the paper and its BibTeX citation will be made available upon publication.

## License

The repository currently displays an **MIT license** for code. 
---

<div align="center">
  <sub>Built to study cultural knowledge without reducing individuals to cultural stereotypes.</sub>
</div>
