"""SFT warmstart trainer — Qwen3-4B + LoRA via unsloth.

Follows unsloth's official Qwen3 fine-tuning guide:
  https://unsloth.ai/docs/models/qwen3.5/fine-tune

  - load_in_16bit=True, load_in_4bit=False  (bf16 LoRA, unsloth's recommendation)
  - transformers v5 required (installed via git in Dockerfile.train)
  - unsloth must be imported before trl/transformers/peft (done inside functions)
  - max_seq_length and dataset_text_field live on SFTTrainer, not SFTConfig
    (API changed in newer TRL shipped with transformers v5)

All GPU-heavy imports are lazy so the module is importable without a GPU.
"""

from __future__ import annotations

MODEL_NAME  = "unsloth/Qwen3-4B"
MAX_SEQ_LEN = 8192


def load_model_for_sft(
    model_name: str = MODEL_NAME,
    max_seq_length: int = MAX_SEQ_LEN,
):
    """Load Qwen3-4B with bf16 LoRA via unsloth (transformers v5 required)."""
    import unsloth  # noqa: F401 — must be first; patches trl/transformers on import
    from unsloth import FastLanguageModel  # type: ignore[import]

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=False,
        load_in_16bit=True,     # bf16 LoRA — unsloth's Qwen3 recommendation
        full_finetuning=False,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=32,          # 2×r per QLoRA paper scaling rule
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
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
    """Train the SFT warmstart model.

    Requires GPU + unsloth + trl (newer version shipped with transformers v5).
    max_seq_length and dataset_text_field are passed to SFTTrainer, not SFTConfig —
    this is the API as of TRL shipped alongside transformers v5.
    """
    from datasets import load_from_disk
    from trl import SFTConfig, SFTTrainer  # type: ignore[import]

    model, tokenizer = load_model_for_sft(model_name=model_name)
    raw   = load_from_disk(dataset_path)
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
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train,
        args=config,
        max_seq_length=MAX_SEQ_LEN,   # moved from SFTConfig in newer TRL
        dataset_text_field="text",     # moved from SFTConfig in newer TRL
    )
    trainer.train()
    trainer.save_model(output_dir)
    return output_dir
