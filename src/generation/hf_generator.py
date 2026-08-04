"""Hugging Face text-generation backend."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.generation.base_generator import BaseGenerator


class HFGenerator(BaseGenerator):
    """Generate text with a Hugging Face causal language model."""

    DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        *,
        device: str = "auto",
        torch_dtype: torch.dtype | str | None = "auto",
        trust_remote_code: bool = False,
        tokenizer: Any | None = None,
        model: Any | None = None,
    ) -> None:
        if not isinstance(model_name, str):
            raise TypeError("model_name must be a string.")

        cleaned_model_name = model_name.strip()

        if not cleaned_model_name:
            raise ValueError("model_name cannot be empty.")

        if not isinstance(device, str):
            raise TypeError("device must be a string.")

        cleaned_device = device.strip().lower()

        if not cleaned_device:
            raise ValueError("device cannot be empty.")

        if not isinstance(trust_remote_code, bool):
            raise TypeError("trust_remote_code must be a boolean.")

        self._model_name = cleaned_model_name
        self.requested_device = cleaned_device
        self.torch_dtype = torch_dtype
        self.trust_remote_code = trust_remote_code
        self.device = self._resolve_device(cleaned_device)

        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(
            self._model_name,
            trust_remote_code=self.trust_remote_code,
        )

        self.model = model or AutoModelForCausalLM.from_pretrained(
            self._model_name,
            torch_dtype=self.torch_dtype,
            trust_remote_code=self.trust_remote_code,
        )

        if hasattr(self.model, "to"):
            self.model = self.model.to(self.device)

        if hasattr(self.model, "eval"):
            self.model.eval()

        self._ensure_padding_token()

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        """Generate a response from a grounded RAG prompt."""
        cleaned_prompt = self._validate_generation_inputs(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        model_prompt = self._format_for_instruct_model(
            cleaned_prompt
        )

        encoded = self.tokenizer(
            model_prompt,
            return_tensors="pt",
        )

        if not isinstance(encoded, Mapping):
            raise RuntimeError(
                "Tokenizer must return a mapping of model inputs."
            )

        model_inputs = {
            key: self._move_to_device(value)
            for key, value in encoded.items()
        }

        input_ids = model_inputs.get("input_ids")

        if input_ids is None:
            raise RuntimeError(
                "Tokenizer output does not contain input_ids."
            )

        if not hasattr(input_ids, "shape") or len(input_ids.shape) != 2:
            raise RuntimeError(
                "input_ids must be a two-dimensional tensor."
            )

        input_length = int(input_ids.shape[1])
        do_sample = temperature > 0.0

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
        }

        if do_sample:
            generation_kwargs["temperature"] = float(temperature)

        eos_token_id = getattr(
            self.tokenizer,
            "eos_token_id",
            None,
        )

        if eos_token_id is not None:
            generation_kwargs["eos_token_id"] = eos_token_id

        with torch.inference_mode():
            generated = self.model.generate(
                **model_inputs,
                **generation_kwargs,
            )

        if not hasattr(generated, "shape") or len(generated.shape) != 2:
            raise RuntimeError(
                "Model generate() must return a two-dimensional tensor."
            )

        continuation = generated[:, input_length:]

        decoded = self.tokenizer.batch_decode(
            continuation,
            skip_special_tokens=True,
        )

        if not isinstance(decoded, list) or not decoded:
            raise RuntimeError(
                "Tokenizer batch_decode() returned no generated text."
            )

        return self._clean_answer(decoded[0])

    def _format_for_instruct_model(
        self,
        prompt: str,
    ) -> str:
        """Apply a tokenizer chat template when one is available."""
        apply_chat_template = getattr(
            self.tokenizer,
            "apply_chat_template",
            None,
        )

        if not callable(apply_chat_template):
            return prompt

        try:
            formatted = apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        except (TypeError, ValueError, RuntimeError):
            return prompt

        if not isinstance(formatted, str) or not formatted.strip():
            return prompt

        return formatted

    @staticmethod
    def _clean_answer(text: str) -> str:
        """Remove common accidental conversation continuations."""
        answer = text.strip()

        stop_markers = (
            "\nHuman:",
            "\nUser:",
            "\n### Human:",
            "\n### User:",
            "\n<|im_start|>user",
        )

        cut_positions = [
            answer.find(marker)
            for marker in stop_markers
            if answer.find(marker) >= 0
        ]

        if cut_positions:
            answer = answer[:min(cut_positions)].rstrip()

        return answer

    def _resolve_device(self, requested_device: str) -> torch.device:
        if requested_device == "auto":
            requested_device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        try:
            resolved = torch.device(requested_device)
        except (TypeError, RuntimeError) as exc:
            raise ValueError(
                f"Unsupported device: {requested_device}."
            ) from exc

        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise ValueError(
                "CUDA was requested but is not available."
            )

        return resolved

    def _ensure_padding_token(self) -> None:
        pad_token_id = getattr(
            self.tokenizer,
            "pad_token_id",
            None,
        )

        if pad_token_id is not None:
            return

        eos_token_id = getattr(
            self.tokenizer,
            "eos_token_id",
            None,
        )

        if eos_token_id is None:
            raise RuntimeError(
                "Tokenizer provides neither pad_token_id "
                "nor eos_token_id."
            )

        self.tokenizer.pad_token_id = eos_token_id

    def _move_to_device(self, value: Any) -> Any:
        if hasattr(value, "to"):
            return value.to(self.device)

        return value

    def _validate_generation_inputs(
        self,
        *,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string.")

        cleaned_prompt = prompt.strip()

        if not cleaned_prompt:
            raise ValueError("prompt cannot be empty.")

        if (
            not isinstance(max_new_tokens, int)
            or isinstance(max_new_tokens, bool)
        ):
            raise TypeError(
                "max_new_tokens must be an integer."
            )

        if max_new_tokens <= 0:
            raise ValueError(
                "max_new_tokens must be greater than zero."
            )

        if (
            not isinstance(temperature, (int, float))
            or isinstance(temperature, bool)
        ):
            raise TypeError(
                "temperature must be a number."
            )

        numeric_temperature = float(temperature)

        if numeric_temperature < 0.0:
            raise ValueError(
                "temperature cannot be negative."
            )

        return cleaned_prompt
