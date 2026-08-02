"""Tests for HFGenerator."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from src.generation.hf_generator import HFGenerator


def make_tokenizer() -> MagicMock:
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 2
    tokenizer.return_value = {
        "input_ids": torch.tensor([[10, 11, 12]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }
    tokenizer.batch_decode.return_value = [
        "Generated answer."
    ]
    return tokenizer


def make_model() -> MagicMock:
    model = MagicMock()
    model.to.return_value = model
    model.eval.return_value = None
    model.generate.return_value = torch.tensor(
        [[10, 11, 12, 20, 21]]
    )
    return model


def make_generator(
    *,
    tokenizer: MagicMock | None = None,
    model: MagicMock | None = None,
) -> HFGenerator:
    return HFGenerator(
        model_name="test/model",
        device="cpu",
        tokenizer=tokenizer or make_tokenizer(),
        model=model or make_model(),
    )


def test_constructor_preserves_model_name():
    generator = make_generator()

    assert generator.model_name == "test/model"


def test_constructor_moves_model_to_device_and_eval():
    model = make_model()

    make_generator(model=model)

    model.to.assert_called_once_with(torch.device("cpu"))
    model.eval.assert_called_once_with()


@patch("src.generation.hf_generator.AutoTokenizer.from_pretrained")
@patch("src.generation.hf_generator.AutoModelForCausalLM.from_pretrained")
def test_constructor_loads_huggingface_dependencies(
    model_loader,
    tokenizer_loader,
):
    tokenizer_loader.return_value = make_tokenizer()
    model_loader.return_value = make_model()

    HFGenerator(
        model_name="org/model",
        device="cpu",
        torch_dtype="auto",
        trust_remote_code=True,
    )

    tokenizer_loader.assert_called_once_with(
        "org/model",
        trust_remote_code=True,
    )
    model_loader.assert_called_once_with(
        "org/model",
        torch_dtype="auto",
        trust_remote_code=True,
    )


def test_constructor_uses_eos_as_padding_when_missing():
    tokenizer = make_tokenizer()
    tokenizer.pad_token_id = None
    tokenizer.eos_token_id = 7

    make_generator(tokenizer=tokenizer)

    assert tokenizer.pad_token_id == 7


def test_constructor_rejects_tokenizer_without_padding_or_eos():
    tokenizer = make_tokenizer()
    tokenizer.pad_token_id = None
    tokenizer.eos_token_id = None

    with pytest.raises(RuntimeError):
        make_generator(tokenizer=tokenizer)


def test_generate_returns_only_continuation():
    generator = make_generator()

    answer = generator.generate(
        "Prompt",
        max_new_tokens=32,
        temperature=0.0,
    )

    assert answer == "Generated answer."


def test_generate_tokenizes_cleaned_prompt():
    tokenizer = make_tokenizer()
    generator = make_generator(tokenizer=tokenizer)

    generator.generate("   Prompt text   ")

    tokenizer.assert_called_once_with(
        "Prompt text",
        return_tensors="pt",
    )


def test_generate_uses_greedy_decoding_for_zero_temperature():
    model = make_model()
    generator = make_generator(model=model)

    generator.generate(
        "Prompt",
        max_new_tokens=64,
        temperature=0.0,
    )

    kwargs = model.generate.call_args.kwargs

    assert kwargs["max_new_tokens"] == 64
    assert kwargs["do_sample"] is False
    assert "temperature" not in kwargs


def test_generate_uses_sampling_for_positive_temperature():
    model = make_model()
    generator = make_generator(model=model)

    generator.generate(
        "Prompt",
        temperature=0.7,
    )

    kwargs = model.generate.call_args.kwargs

    assert kwargs["do_sample"] is True
    assert kwargs["temperature"] == pytest.approx(0.7)


def test_generate_decodes_only_new_tokens():
    tokenizer = make_tokenizer()
    model = make_model()
    generator = make_generator(
        tokenizer=tokenizer,
        model=model,
    )

    generator.generate("Prompt")

    decoded_tensor = tokenizer.batch_decode.call_args.args[0]

    assert decoded_tensor.tolist() == [[20, 21]]
    tokenizer.batch_decode.assert_called_once_with(
        decoded_tensor,
        skip_special_tokens=True,
    )


@pytest.mark.parametrize("prompt", ["", "   ", "\n\t"])
def test_generate_rejects_empty_prompt(prompt):
    generator = make_generator()

    with pytest.raises(ValueError):
        generator.generate(prompt)


def test_generate_rejects_non_string_prompt():
    generator = make_generator()

    with pytest.raises(TypeError):
        generator.generate(123)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [0, -1])
def test_generate_rejects_non_positive_max_new_tokens(value):
    generator = make_generator()

    with pytest.raises(ValueError):
        generator.generate(
            "Prompt",
            max_new_tokens=value,
        )


@pytest.mark.parametrize("value", [1.5, "10", True])
def test_generate_rejects_non_integer_max_new_tokens(value):
    generator = make_generator()

    with pytest.raises(TypeError):
        generator.generate(
            "Prompt",
            max_new_tokens=value,  # type: ignore[arg-type]
        )


def test_generate_rejects_negative_temperature():
    generator = make_generator()

    with pytest.raises(ValueError):
        generator.generate(
            "Prompt",
            temperature=-0.1,
        )


@pytest.mark.parametrize("value", ["0.5", None, True])
def test_generate_rejects_invalid_temperature_type(value):
    generator = make_generator()

    with pytest.raises(TypeError):
        generator.generate(
            "Prompt",
            temperature=value,  # type: ignore[arg-type]
        )


def test_generate_rejects_tokenizer_output_without_input_ids():
    tokenizer = make_tokenizer()
    tokenizer.return_value = {
        "attention_mask": torch.tensor([[1, 1]])
    }
    generator = make_generator(tokenizer=tokenizer)

    with pytest.raises(RuntimeError):
        generator.generate("Prompt")


def test_generate_rejects_invalid_generated_tensor_rank():
    model = make_model()
    model.generate.return_value = torch.tensor([1, 2, 3])
    generator = make_generator(model=model)

    with pytest.raises(RuntimeError):
        generator.generate("Prompt")


def test_generate_rejects_empty_decoded_output():
    tokenizer = make_tokenizer()
    tokenizer.batch_decode.return_value = []
    generator = make_generator(tokenizer=tokenizer)

    with pytest.raises(RuntimeError):
        generator.generate("Prompt")


def test_call_forwards_to_generate():
    generator = make_generator()
    generator.generate = MagicMock(
        return_value="answer"
    )

    result = generator(
        "Prompt",
        max_new_tokens=10,
        temperature=0.2,
    )

    assert result == "answer"
    generator.generate.assert_called_once_with(
        "Prompt",
        max_new_tokens=10,
        temperature=0.2,
    )


def test_model_name_must_be_string():
    with pytest.raises(TypeError):
        HFGenerator(
            model_name=123,  # type: ignore[arg-type]
            device="cpu",
            tokenizer=make_tokenizer(),
            model=make_model(),
        )


def test_model_name_cannot_be_empty():
    with pytest.raises(ValueError):
        HFGenerator(
            model_name="   ",
            device="cpu",
            tokenizer=make_tokenizer(),
            model=make_model(),
        )


def test_device_must_be_string():
    with pytest.raises(TypeError):
        HFGenerator(
            model_name="test/model",
            device=123,  # type: ignore[arg-type]
            tokenizer=make_tokenizer(),
            model=make_model(),
        )


def test_explicit_unavailable_cuda_is_rejected():
    if torch.cuda.is_available():
        pytest.skip("CUDA is available in this environment.")

    with pytest.raises(ValueError):
        HFGenerator(
            model_name="test/model",
            device="cuda",
            tokenizer=make_tokenizer(),
            model=make_model(),
        )
