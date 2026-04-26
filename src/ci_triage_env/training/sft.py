"""SFT warmstart trainer — Qwen3-4B + LoRA on the C3 trajectory dataset.

All GPU-heavy imports (trl, torch, peft) are lazy so the module is
importable without a GPU for testing.
"""

from __future__ import annotations

MODEL_NAME = "Qwen/Qwen3-4B"
MAX_SEQ_LEN = 8192


def load_model_for_sft(
    model_name: str = MODEL_NAME,
    max_seq_length: int = MAX_SEQ_LEN,
):
    """Load Qwen3-4B in 4-bit via bitsandbytes + LoRA via PEFT. Requires GPU."""
    import torch
    from peft import LoraConfig, TaskType, get_peft_model  # type: ignore[import]
    from transformers import (  # type: ignore[import]
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.gradient_checkpointing_enable()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = max_seq_length

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer


def format_for_sft(traj: dict, tokenizer) -> dict:
    """Render a trajectory's messages into a single training-example string."""
    text = tokenizer.apply_chat_template(
        traj["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def run_sft(
    dataset_path: str,
    output_dir: str,
    num_epochs: int = 3,
    per_device_batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    model_name: str = MODEL_NAME,
) -> str:
    """Train the SFT warmstart model. Requires GPU + trl + peft + bitsandbytes.

    Args:
        dataset_path: Path to a HF Dataset saved by trajectory_gen (save_to_disk).
        output_dir: Where to write the LoRA adapter checkpoint.
        num_epochs: Training epochs over the SFT dataset.
        per_device_batch_size: Batch size per GPU.
        gradient_accumulation_steps: Steps before optimizer update.
        model_name: Base model or existing checkpoint path.

    Returns:
        output_dir path as a string.
    """
    from datasets import load_from_disk
    from trl import SFTConfig, SFTTrainer  # type: ignore[import]

    model, tokenizer = load_model_for_sft(model_name=model_name)
    raw = load_from_disk(dataset_path)
    train = raw.map(lambda t: format_for_sft(t, tokenizer))

    config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=2e-5,
        warmup_ratio=0.05,
        logging_steps=10,
        save_steps=100,
        report_to="wandb",
        max_seq_length=MAX_SEQ_LEN,
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train,
        args=config,
    )
    trainer.train()
    trainer.save_model(output_dir)
    return output_dir
