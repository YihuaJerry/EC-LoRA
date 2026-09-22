#!/usr/bin/env python3
from __future__ import annotations

import sys


def patch_peft_adapter_dtype() -> None:
    try:
        from peft import PeftModel
    except ModuleNotFoundError:
        return

    original_from_pretrained = PeftModel.from_pretrained

    def from_pretrained_no_autocast(cls, model, model_id, *args, **kwargs):
        kwargs.setdefault("autocast_adapter_dtype", False)
        try:
            return original_from_pretrained(model, model_id, *args, **kwargs)
        except TypeError as exc:
            text = str(exc)
            if "autocast_adapter_dtype" not in text and "unexpected keyword" not in text:
                raise
            kwargs.pop("autocast_adapter_dtype", None)
            return original_from_pretrained(model, model_id, *args, **kwargs)

    PeftModel.from_pretrained = classmethod(from_pretrained_no_autocast)


def main() -> int:
    patch_peft_adapter_dtype()
    from qwen_eval.model_vqa import main as qwen_model_vqa_main

    return int(qwen_model_vqa_main())


if __name__ == "__main__":
    sys.exit(main())
