"""
Fine-tune IndoBERT untuk NER atribut produk (skema BIO).

Input  : data/processed/{train,val,test}.jsonl
         tiap baris: {"tokens": ["Sepatu", "Nike", ...], "ner_tags": ["B-PRODUCT", "B-BRAND", ...]}
Output : models/indobert-ner/   (model terbaik)
         outputs/test_report.txt (classification report per entitas)

Jalankan: python src/train.py
Mac M2  : Trainer otomatis memakai MPS (tanpa fp16).
"""
import argparse
import json
from pathlib import Path

import numpy as np
from datasets import Dataset
from seqeval.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, recall_score)
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          DataCollatorForTokenClassification, Trainer,
                          TrainingArguments, set_seed)

ENTITIES = ["PRODUCT", "BRAND", "COLOR", "SIZE", "MATERIAL", "PRICE", "QUALITY", "PROBLEM"]
LABELS = ["O"] + [f"{p}-{e}" for e in ENTITIES for p in ("B", "I")]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}


def read_jsonl(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    for r in rows:
        assert len(r["tokens"]) == len(r["ner_tags"]), f"Panjang token != label: {r}"
    return Dataset.from_list([{"tokens": r["tokens"], "ner_tags": r["ner_tags"]} for r in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="indobenchmark/indobert-base-p1")
    ap.add_argument("--data_dir", default="data/processed")
    ap.add_argument("--out_dir", default="models/indobert-ner")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    set_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model)
    assert tok.is_fast, "Butuh fast tokenizer untuk word_ids(); pakai AutoTokenizer, bukan BertTokenizer."

    def tokenize_and_align(batch):
        enc = tok(batch["tokens"], is_split_into_words=True, truncation=True, max_length=args.max_len)
        all_labels = []
        for i, tags in enumerate(batch["ner_tags"]):
            prev, labs = None, []
            for w in enc.word_ids(batch_index=i):
                if w is None or w == prev:      # special token / sub-word lanjutan -> diabaikan
                    labs.append(-100)
                else:
                    labs.append(LABEL2ID[tags[w]])
                prev = w
            all_labels.append(labs)
        enc["labels"] = all_labels
        return enc

    data = {s: read_jsonl(f"{args.data_dir}/{s}.jsonl") for s in ("train", "val", "test")}
    data = {s: d.map(tokenize_and_align, batched=True, remove_columns=d.column_names) for s, d in data.items()}

    model = AutoModelForTokenClassification.from_pretrained(
        args.model, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID
    )

    def decode(preds, label_ids):
        y_true, y_pred = [], []
        for p_row, l_row in zip(preds, label_ids):
            t, p = [], []
            for pi, li in zip(p_row, l_row):
                if li != -100:
                    t.append(ID2LABEL[int(li)])
                    p.append(ID2LABEL[int(pi)])
            y_true.append(t)
            y_pred.append(p)
        return y_true, y_pred

    def compute_metrics(p):
        y_true, y_pred = decode(np.argmax(p.predictions, axis=2), p.label_ids)
        return {
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
            "accuracy": accuracy_score(y_true, y_pred),
        }

    targs = TrainingArguments(
        output_dir=args.out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",   # di transformers lama: evaluation_strategy
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=20,
        seed=args.seed,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=data["train"],
        eval_dataset=data["val"],
        processing_class=tok,
        data_collator=DataCollatorForTokenClassification(tok),
        compute_metrics=compute_metrics,
    )
    trainer.train()

    # Evaluasi akhir di test set
    out = trainer.predict(data["test"])
    y_true, y_pred = decode(np.argmax(out.predictions, axis=2), out.label_ids)
    report = classification_report(y_true, y_pred, digits=4)
    print(report)
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/test_report.txt").write_text(report, encoding="utf-8")

    trainer.save_model(args.out_dir)
    tok.save_pretrained(args.out_dir)
    print(f"Model tersimpan di {args.out_dir}")


if __name__ == "__main__":
    main()
