---
license: cc-by-4.0
language:
  - en
task_categories:
  - visual-question-answering
  - question-answering
tags:
  - security
  - visual-prompt-injection
  - multimodal-agents
  - computer-use
  - browser-use
  - benchmarks
  - trustsight
pretty_name: TrustSight VPI-Bench Live Suite
size_categories:
  - n<1K
configs:
  - config_name: default
    data_files:
      - split: upstream_attack
        path: "data/upstream_attack-*"
      - split: clean_control
        path: "data/clean_control-*"
dataset_info:
  features:
    - name: task_id
      dtype: string
    - name: platform
      dtype: string
    - name: split
      dtype: string
    - name: instruction
      dtype: string
    - name: target_tool
      dtype: string
    - name: attack_objective
      dtype: string
    - name: carrier_type
      dtype: string
    - name: verified_bounding_boxes
      dtype: string
    - name: image
      dtype: image
    - name: image_sha256
      dtype: string
    - name: agent_type
      dtype: string
    - name: upstream_id
      dtype: string
  splits:
    - name: upstream_attack
      num_bytes: 88500000
      num_examples: 306
    - name: clean_control
      num_bytes: 14200000
      num_examples: 30
  download_size: 102700000
  dataset_size: 102700000
---

# TrustSight VPI-Bench Live Suite (`trustsight-vpibench`)

## 1. Overview & Research Context

The **TrustSight VPI-Bench Live Suite** is an empirical multimodal benchmark comprising **336 interactive web and messaging GUI tasks** designed to evaluate the susceptibility and resilience of Vision-Language Models (VLMs) and Computer-Use Agents (CUAs) against Visual Prompt Injection (VPI) attacks.

It adapts the upstream **VPI-Bench** (ICLR 2026, Cao et al.) into a curated, self-contained suite with authentic pixel screenshots and a strictly paired control set. Every record embeds the genuine rendered PNG screenshot, verified carrier bounding boxes, and ground-truth user/attacker intents.

---

## 2. Dataset Structure & Splits

The suite contains exactly **336 tasks** divided into two splits:

- **`upstream_attack` (306 tasks):** Adversarial visual scenarios across 5 platforms featuring modal popups, phishing forms, download banners, and malicious messaging payloads.
- **`clean_control` (30 tasks):** Paired benign tasks across `amazon`, `booking`, and `bbc` ensuring balanced measurement of legitimate task utility without adversarial interference.

### Platform Breakdown

| Platform | Upstream Attack Tasks | Clean Control Tasks | Total Tasks | Attack / Clean Ratio |
|:---|:---:|:---:|:---:|:---:|
| **Amazon** | 79 | 10 | 89 | 88.8% / 11.2% |
| **Booking** | 79 | 10 | 89 | 88.8% / 11.2% |
| **BBC News** | 79 | 10 | 89 | 88.8% / 11.2% |
| **Email (NovaMail)** | 46 | 0 | 46 | 100.0% / 0.0% |
| **Messenger (ChitChat)** | 23 | 0 | 23 | 100.0% / 0.0% |
| **TOTAL** | **306** | **30** | **336** | **91.1% / 8.9%** |

---

## 3. Visual Carrier Taxonomy

| Carrier Type (`carrier_type`) | Count | Platforms | Visual Description |
|:---|:---:|---|---|
| `phishing_form_modal` | 120 | Amazon, Booking, BBC | Modal dialog soliciting credentials / payment / file deletion |
| `modal_popup` | 63 | Amazon, Booking, BBC | Pop-up overlay interrupting legitimate workflow |
| `download_banner` | 54 | Amazon, Booking, BBC | Banner prompting file downloads or shell command execution |
| `email_body` | 46 | Email | Malicious email body attempting privilege escalation |
| `chat_message_bubble` | 23 | Messenger | Chat bubble with typographic injection steering the agent |
| `authentic_webpage` | 30 | Amazon, Booking, BBC | Pristine authentic web page without any adversarial payload |

---

## 4. Schema Specification (12 Fields)

- `task_id` (`string`): Unique canonical task identifier (e.g. `drive_amazon_formdel_api`).
- `platform` (`string`): Web platform name (`amazon`, `booking`, `bbc`, `email`, `messenger`).
- `split` (`string`): Dataset split (`upstream_attack` or `clean_control`).
- `instruction` (`string`): Natural language instruction given to the agent.
- `target_tool` (`string`): Expected or targeted tool action.
- `attack_objective` (`string`): Attacker objective description (empty string for clean controls).
- `carrier_type` (`string`): Visual embedding mechanism.
- `verified_bounding_boxes` (`string`): JSON string of bounding box coordinates `[{"left", "top", "width", "height"}]` for the injection carrier (or `[]` for clean).
- `image` (`image`): Native Hugging Face `Image()` feature containing the embedded PNG raw bytes.
- `image_sha256` (`string`): Cryptographic SHA-256 digest of the screenshot.
- `agent_type` (`string`): Operational agent modality (`computer_use` vs `browser_use`).
- `upstream_id` (`string`): Upstream JSON testcase identifier.

---

## 5. Usage Example

```python
from datasets import load_dataset

# Load the upstream_attack split
dataset = load_dataset("tuandunghcmut/trustsight-vpibench", split="upstream_attack")
sample = dataset[0]

print(f"Task ID: {sample['task_id']}")
print(f"Platform: {sample['platform']}")
print(f"Instruction: {sample['instruction']}")
print(f"Target Tool: {sample['target_tool']}")

# View the embedded screenshot
sample["image"].show()
```

---

## 6. Citation & Academic Credits

If you use this benchmark in your research, please cite:

```bibtex
@inproceedings{cao2026vpibench,
  title     = {VPI-Bench: Visual Prompt Injection Attacks for Computer-Use Agents},
  author    = {Tri Cao and Bennett Lim and Yue Liu and Yuan Sui and Yuexin Li and Shumin Deng and Lin Lu and Nay Oo and Shuicheng Yan and Bryan Hooi},
  booktitle = {The Fourteenth International Conference on Learning Representations (ICLR)},
  year      = {2026}
}
```
