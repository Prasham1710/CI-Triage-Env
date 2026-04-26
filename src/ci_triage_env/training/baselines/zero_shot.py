"""ZeroShotPolicy — any HF model in zero-shot inference mode.

All GPU-heavy imports (unsloth, torch) are lazy so the module is importable without GPU.
"""

from __future__ import annotations


class ZeroShotPolicy:
    """Wrapper around any HF model used in zero-shot inference mode."""

    def __init__(
        self,
        model_name: str,
        system_prompt: str,
        name: str | None = None,
    ) -> None:
        self.name = name or f"zero_shot_{model_name.split('/')[-1]}"
        from unsloth import FastLanguageModel  # type: ignore[import]

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=8192,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(self.model)
        self.system_prompt = system_prompt

    def act(self, obs, history: list) -> dict:
        import torch  # type: ignore[import]

        from ci_triage_env.training.trajectory_gen import TrajectoryGenerator

        messages = self._build_messages(obs, history)
        input_ids = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        ).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                input_ids,
                max_new_tokens=400,
                do_sample=False,
                temperature=1.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        response = self.tokenizer.decode(
            out[0][input_ids.shape[1]:], skip_special_tokens=True
        )
        action = TrajectoryGenerator._parse_action(response)
        if action is not None:
            return action.model_dump()
        return {
            "action_type": "submit_diagnosis",
            "diagnosis": "ambiguous",
            "confidence": 0.5,
            "secondary_actions": [],
        }

    def _build_messages(self, obs, history: list) -> list[dict]:
        messages = [{"role": "system", "content": self.system_prompt}]
        if obs.failure_summary:
            fs = obs.failure_summary
            content = (
                f"CI FAILURE ALERT\nTest: {fs.test_name}\nSuite: {fs.suite}\n"
                f"Branch: {fs.branch}\nLog excerpt:\n{fs.initial_log_excerpt}\n\n"
                "Investigate and submit your diagnosis."
            )
        else:
            content = "CI failure detected. Begin investigation."
        messages.append({"role": "user", "content": content})
        for entry in history:
            messages.append({"role": "assistant", "content": str(entry)})
        return messages
