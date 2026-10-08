#!/usr/bin/env python3
"""Independent eager HF oracle. Local checkpoint only; no Julia imports."""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(2)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    root = Path(args.model_dir).resolve()
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        root, local_files_only=True, dtype=torch.float32, attn_implementation="eager"
    ).cpu().eval()
    cases = []
    with torch.inference_mode():
        for prompt in ["Hello", "The quick brown fox", "Julia is a programming language."]:
            ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids
            logits = model(ids).logits[0, -1].tolist()
            generated = model.generate(
                ids,
                attention_mask=torch.ones_like(ids),
                max_new_tokens=8,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.eos_token_id,
            )[0].tolist()
            cases.append({"prompt": prompt, "prompt_ids": ids[0].tolist(),
                          "generated_ids": generated, "last_logits": logits})
    hashes = {}
    for path in sorted(root.iterdir()):
        if path.is_file() and (path.suffix in (".json", ".safetensors") or path.name == "merges.txt"):
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            hashes[path.name] = digest.hexdigest()
    receipt = {"schema": "gesso-hf-reference-v1", "oracle": "hf-pytorch-eager",
               "model_dir": str(root), "torch": torch.__version__,
               "transformers": transformers.__version__, "dtype": "float32",
               "device": "cpu", "atol": 1e-2, "rtol": 0,
               "max_new_tokens": 8, "checkpoint_sha256": hashes, "cases": cases}
    Path(args.output).write_text(json.dumps(receipt, allow_nan=False) + "\n")
    print(json.dumps({"output": args.output, "cases": len(cases), "dtype": "float32",
                      "torch": torch.__version__, "transformers": transformers.__version__}))


if __name__ == "__main__":
    main()
