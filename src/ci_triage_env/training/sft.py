"""SFT warmstart trainer — Qwen3-4B + QLoRA via unsloth.

QLoRA setup:
  - Base model loaded in NF4 4-bit (frozen) — the "Q" in QLoRA
  - LoRA adapter matrices trained in bf16 on top — the "LoRA" part
  - Double quantization enabled by default in unsloth for ~0.4 bit extra savings
  - unsloth calls prepare_model_for_kbit_training() internally

All GPU-heavy imports (unsloth, trl, torch) are lazy so the module is
importable without a GPU for testing.
"""

from __future__ import annotations

# unsloth hosts optimised weights; the bnb-4bit variant is pre-quantised to NF4
# so it loads ~2x faster than quantising float16 on the fly.
MODEL_NAME = "unsloth/Qwen3-4B-bnb-4bit"
MAX_SEQ_LEN = 8192


def load_model_for_sft(
    model_name: str = MODEL_NAME,
    max_seq_length: int = MAX_SEQ_LEN,
):
    """Load Qwen3-4B with QLoRA (NF4 base + bf16 LoRA adapters) via unsloth."""
    from unsloth import FastLanguageModel  # type: ignore[import]

    # load_in_4bit=True → base weights frozen in NF4 4-bit (QLoRA)
    # dtype=None        → adapter compute dtype auto-selected (bf16 on Ampere+)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        dtype=None,
    )

    # QLoRA LoRA config: r=16 is standard for 4B models.
    # lora_alpha=32 (= 2×r) follows the QLoRA paper's scaling recommendation.
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",  # 30% faster than standard
        random_state=3407,
    )
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
    """Train the SFT warmstart model. Requires GPU + unsloth + trl installed.

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
        bf16=True,
        fp16=False,
        logging_steps=10,
        save_steps=100,
        report_to="wandb",
        max_seq_length=MAX_SEQ_LEN,
        dataset_text_field="text",
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
