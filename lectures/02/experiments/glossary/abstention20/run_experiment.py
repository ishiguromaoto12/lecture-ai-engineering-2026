"""Colab T4 で実行する保留例追加実験。入力は既存結果とは別のディレクトリに置く。"""

import argparse
import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(work, executed_training_myself=False):
    import torch
    import transformers
    import peft
    import trl
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import SFTConfig, SFTTrainer

    work = Path(work)
    inputs = work / "inputs"
    experiment = load_module("experiment", inputs / "glossary_experiment.py")
    abstention = load_module("abstention", inputs / "glossary_abstention.py")
    glossary, original_records, questions = experiment.load_inputs(inputs)
    records, additions = abstention.build_augmented_records(original_records, glossary, questions)
    config = json.loads((inputs / "config.json").read_text())
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU が見つかりません")
    if (work / "result.json").exists():
        raise RuntimeError("結果が既にあります。別の保存先を指定してください")
    experiment.write_json(work / "run_status.json", {"stage": "loading", "pid": os.getpid()})
    (work / "train_augmented.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    experiment.write_json(work / "abstention_training_examples.json", additions)
    experiment.write_json(work / "evaluation.json", questions)
    print(f"学習例: 元の定義 {len(original_records)} + 保留 {len(additions)} = {len(records)}", flush=True)
    set_seed(config["seed"])
    tok = AutoTokenizer.from_pretrained(config["base_model"], revision=config["revision"], local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    lengths = [len(tok(tok.apply_chat_template(r["messages"], tokenize=False, enable_thinking=False),
                       add_special_tokens=False)["input_ids"]) for r in records]
    if max(lengths) > config["max_seq_len"]:
        raise ValueError("学習例が max_seq_len を超えています")
    model = AutoModelForCausalLM.from_pretrained(config["base_model"], revision=config["revision"],
                                               dtype=torch.float32, local_files_only=True).to("cuda")
    model.config.use_cache = False
    lora = LoraConfig(r=config["lora_r"], lora_alpha=config["lora_alpha"], lora_dropout=0.05,
                      bias="none", task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    args = SFTConfig(output_dir=str(work / "ckpt"), num_train_epochs=config["num_train_epochs"],
                     per_device_train_batch_size=4, gradient_accumulation_steps=2,
                     learning_rate=config["lr"], lr_scheduler_type="cosine", warmup_steps=6,
                     logging_steps=10, save_steps=100, save_total_limit=1, fp16=False,
                     max_length=config["max_seq_len"], packing=False, report_to="none", seed=config["seed"])
    trainer = SFTTrainer(model=model, args=args, train_dataset=Dataset.from_list(records),
                         peft_config=lora, processing_class=tok)
    trainer.model.print_trainable_parameters()
    experiment.write_json(work / "run_status.json", {"stage": "training", "pid": os.getpid()})
    started = time.perf_counter()
    trainer.train()
    seconds = round(time.perf_counter() - started, 1)
    history = [{"step": r["step"], "training_loss": r["loss"]} for r in trainer.state.log_history if "loss" in r]
    training = {"seconds": seconds, "epochs": trainer.state.epoch, "global_step": trainer.state.global_step,
                "last_logged_training_loss": history[-1]["training_loss"] if history else None,
                "loss_history": history, "logging_steps": 10, "max_training_tokens": max(lengths)}
    experiment.write_json(work / "training.json", training)
    print("TRAINING:", json.dumps(training), flush=True)
    model = trainer.model
    model.save_pretrained(work / "adapter")
    tok.save_pretrained(work / "adapter")
    model.config.use_cache = True
    model.eval()
    reference = experiment.format_glossary_reference(glossary)

    def generate(question, reference_text=None):
        prompt = tok.apply_chat_template(experiment.build_messages(question, reference_text),
                                         tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tok(prompt, return_tensors="pt", truncation=False).to(model.device)
        with torch.no_grad():
            output = model.generate(**ids, max_new_tokens=config["max_new_tokens"], do_sample=False,
                                    pad_token_id=tok.pad_token_id)
        return tok.decode(output[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    evaluations = {}
    for label, ref in [("with_reference", reference), ("without_reference", None)]:
        experiment.write_json(work / "run_status.json", {"stage": "evaluating_" + label, "pid": os.getpid()})
        summary, rows = experiment.evaluate_model(lambda q: generate(q, ref), questions, label, work / f"eval_{label}.json")
        rows = abstention.augment_observations(rows)
        summary["semantic_review"] = abstention.reviewed_summary(rows)
        experiment.write_json(work / f"eval_{label}.json", {"summary": summary, "rows": rows})
        evaluations[label] = summary
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs.iterdir() if p.is_file()}
    result = {"config": config, "versions": {m.__name__: m.__version__ for m in [torch, transformers, peft, trl]},
              "training": training, "original_examples": len(original_records), "abstention_examples": len(additions),
              "train_examples": len(records), "evaluation_questions": len(questions), "input_sha256": hashes,
              "training_terms_for_abstention": abstention.ABSTENTION_TERMS, "evaluations": evaluations,
              "baseline": "epochs20_with_reference の定義例だけで学習した結果を同じ質問・入力で比較",
              "review_note": "キーワードや保留語句の検出は補助。誤断定率は原文の意味を理由付きで確認して算出する。",
              "executed_training_myself": executed_training_myself,
              "executed_by": "本人がノートブックで実行" if executed_training_myself else "Codex が利用者の接続中 Colab T4 で実行"}
    experiment.write_json(work / "result.json", result)
    experiment.write_json(work / "run_status.json", {"stage": "complete", "pid": os.getpid()})
    print("COMPLETE:", work, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", required=True)
    args = parser.parse_args()
    try:
        run(args.work_dir)
    except Exception as exc:
        Path(args.work_dir).mkdir(parents=True, exist_ok=True)
        (Path(args.work_dir) / "run_status.json").write_text(json.dumps({"stage": "failed", "error": repr(exc)}))
        raise
