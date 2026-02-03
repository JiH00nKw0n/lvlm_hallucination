import asyncio
import os
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

import torch
from openai import OpenAI
from tabulate import tabulate
from torch import Tensor
from transformers import AutoTokenizer, PreTrainedModel


def _clean_token(token: str) -> str:
    if token.startswith("<") and token.endswith(">"):
        return ""
    return token.replace("▁", " ").replace("Ġ", " ")


def _tokens_to_str(tokens: Sequence[int], tokenizer: AutoTokenizer) -> list[str]:
    str_tokens = tokenizer.convert_ids_to_tokens(list(tokens))
    return [_clean_token(tok) for tok in str_tokens]


def tokenize_and_concat_dataset(
    tokenizer: AutoTokenizer,
    dataset: list[str],
    seq_len: int,
    add_bos: bool = True,
    max_tokens: Optional[int] = None,
) -> torch.Tensor:
    full_text = tokenizer.eos_token.join(dataset)

    # Chunked tokenization for speed.
    num_chunks = 20
    chunk_length = (len(full_text) - 1) // num_chunks + 1
    chunks = [
        full_text[i * chunk_length : (i + 1) * chunk_length] for i in range(num_chunks)
    ]

    tokens = tokenizer(chunks, return_tensors="pt", padding=True)["input_ids"].flatten()

    # Drop padding tokens.
    if tokenizer.pad_token_id is not None:
        tokens = tokens[tokens != tokenizer.pad_token_id]

    if max_tokens is not None:
        tokens = tokens[: max_tokens + seq_len + 1]

    num_tokens = len(tokens)
    num_batches = num_tokens // seq_len
    tokens = tokens[: num_batches * seq_len].reshape(num_batches, seq_len)

    if add_bos and tokenizer.bos_token_id is not None:
        tokens[:, 0] = tokenizer.bos_token_id
    return tokens


def get_dataset_list_of_strs(
    dataset: Iterable[dict],
    column_name: str,
    min_row_chars: int,
    total_chars: int,
) -> list[str]:
    total_chars_so_far = 0
    result = []

    for row in dataset:
        text = row.get(column_name, "")
        if not isinstance(text, str):
            continue
        if len(text) > min_row_chars:
            result.append(text)
            total_chars_so_far += len(text)
            if total_chars_so_far > total_chars:
                break
    return result


def load_and_tokenize_dataset(
    dataset: Iterable[dict],
    tokenizer: AutoTokenizer,
    ctx_len: int,
    num_tokens: int,
    column_name: str = "text",
    add_bos: bool = True,
) -> torch.Tensor:
    texts = get_dataset_list_of_strs(
        dataset, column_name, min_row_chars=100, total_chars=num_tokens * 5
    )
    tokens = tokenize_and_concat_dataset(
        tokenizer, texts, ctx_len, add_bos=add_bos, max_tokens=num_tokens
    )
    assert (tokens.shape[0] * tokens.shape[1]) > num_tokens
    return tokens


def get_bos_pad_eos_mask(tokens: torch.Tensor, tokenizer: AutoTokenizer) -> torch.Tensor:
    mask = torch.zeros_like(tokens, dtype=torch.bool)
    if tokenizer.pad_token_id is not None:
        mask |= tokens == tokenizer.pad_token_id
    if tokenizer.eos_token_id is not None:
        mask |= tokens == tokenizer.eos_token_id
    if tokenizer.bos_token_id is not None:
        mask |= tokens == tokenizer.bos_token_id
    return ~mask


def get_k_largest_indices(
    x: Tensor,
    k: int,
    buffer: int = 0,
    no_overlap: bool = False,
) -> Tensor:
    x = x[:, buffer:-buffer]
    indices = x.flatten().argsort(-1, descending=True)
    rows = indices // x.size(1)
    cols = indices % x.size(1) + buffer

    if no_overlap:
        unique_indices = []
        seen_positions = set()
        for row, col in zip(rows.tolist(), cols.tolist()):
            if (row, col) not in seen_positions:
                unique_indices.append((row, col))
                for offset in range(-buffer, buffer + 1):
                    seen_positions.add((row, col + offset))
            if len(unique_indices) == k:
                break
        rows, cols = torch.tensor(
            unique_indices, dtype=torch.int64, device=x.device
        ).unbind(dim=-1)

    return torch.stack((rows, cols), dim=1)[:k]


def get_iw_sample_indices(
    x: Tensor,
    k: int,
    buffer: int = 0,
    use_squared_values: bool = True,
) -> Tensor:
    x = x[:, buffer:-buffer]
    if use_squared_values:
        x = x.pow(2)

    probabilities = x.flatten() / x.sum()
    indices = torch.multinomial(probabilities, k, replacement=False)

    rows = indices // x.size(1)
    cols = indices % x.size(1) + buffer
    return torch.stack((rows, cols), dim=1)[:k]


def index_with_buffer(x: Tensor, indices: Tensor, buffer: int = 0) -> Tensor:
    rows, cols = indices.unbind(dim=-1)
    rows = rows.repeat_interleave(buffer * 2 + 1).view(-1, buffer * 2 + 1)
    cols = cols.repeat_interleave(buffer * 2 + 1).view(-1, buffer * 2 + 1) + torch.arange(
        -buffer, buffer + 1, device=cols.device
    )
    return x[rows, cols]


def _sae_dense_acts(sae: Any, hidden_states: Tensor) -> Tensor:
    acts = sae.encode(hidden_states)
    if isinstance(acts, tuple) and len(acts) == 2:
        top_acts, top_indices = acts
        latent_size = getattr(sae, "W_dec", None)
        if latent_size is None:
            raise ValueError("SAE must define W_dec to infer latent size.")
        latent_size = sae.W_dec.shape[0]
        dense = torch.zeros(
            (*top_acts.shape[:-1], latent_size),
            device=top_acts.device,
            dtype=top_acts.dtype,
        )
        dense.scatter_(dim=-1, index=top_indices, src=top_acts)
        return dense
    return acts


@torch.no_grad()
def collect_sae_activations_hidden(
    tokens: Tensor,
    model: PreTrainedModel,
    sae: Any,
    batch_size: int,
    hidden_state_index: int,
    tokenizer: AutoTokenizer,
    mask_bos_pad_eos_tokens: bool = False,
    selected_latents: Optional[list[int]] = None,
    activation_dtype: Optional[torch.dtype] = None,
) -> Tensor:
    sae_acts = []
    for i in range(0, tokens.shape[0], batch_size):
        tokens_bl = tokens[i : i + batch_size]
        attention_mask = (
            (tokens_bl != tokenizer.pad_token_id).long()
            if tokenizer.pad_token_id is not None
            else None
        )
        outputs = model(
            input_ids=tokens_bl.to(model.device),
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden_states = outputs.hidden_states[hidden_state_index]
        sae_act_blf = _sae_dense_acts(sae, hidden_states)

        if selected_latents is not None:
            sae_act_blf = sae_act_blf[:, :, selected_latents]

        if mask_bos_pad_eos_tokens:
            attn_mask_bl = get_bos_pad_eos_mask(tokens_bl, tokenizer)
        else:
            attn_mask_bl = torch.ones_like(tokens_bl, dtype=torch.bool)
        attn_mask_bl = attn_mask_bl.to(device=sae_act_blf.device)
        sae_act_blf = sae_act_blf * attn_mask_bl[:, :, None]

        if activation_dtype is not None:
            sae_act_blf = sae_act_blf.to(dtype=activation_dtype)
        sae_acts.append(sae_act_blf)

    return torch.cat(sae_acts, dim=0)


@torch.no_grad()
def get_feature_activation_sparsity_hidden(
    tokens: Tensor,
    model: PreTrainedModel,
    sae: Any,
    batch_size: int,
    hidden_state_index: int,
    tokenizer: AutoTokenizer,
    mask_bos_pad_eos_tokens: bool = False,
) -> Tensor:
    device = sae.W_dec.device if hasattr(sae, "W_dec") else model.device
    latent_size = sae.W_dec.shape[0]
    running_sum_f = torch.zeros(latent_size, dtype=torch.float32, device=device)
    total_tokens = 0

    for i in range(0, tokens.shape[0], batch_size):
        tokens_bl = tokens[i : i + batch_size]
        attention_mask = (
            (tokens_bl != tokenizer.pad_token_id).long()
            if tokenizer.pad_token_id is not None
            else None
        )
        outputs = model(
            input_ids=tokens_bl.to(model.device),
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden_states = outputs.hidden_states[hidden_state_index]
        sae_act_blf = _sae_dense_acts(sae, hidden_states)
        sae_act_blf = (sae_act_blf > 0).to(dtype=torch.float32)

        if mask_bos_pad_eos_tokens:
            attn_mask_bl = get_bos_pad_eos_mask(tokens_bl, tokenizer)
        else:
            attn_mask_bl = torch.ones_like(tokens_bl, dtype=torch.bool)
        attn_mask_bl = attn_mask_bl.to(device=sae_act_blf.device)
        sae_act_blf = sae_act_blf * attn_mask_bl[:, :, None]
        total_tokens += attn_mask_bl.sum().item()
        running_sum_f += sae_act_blf.sum(dim=(0, 1))

    return running_sum_f / max(total_tokens, 1)


@dataclass
class AutoInterpConfig:
    model_name: str
    openai_api_key: str
    openai_model: str = "gpt-4o-mini"
    n_latents: int = 1000
    override_latents: Optional[list[int]] = None
    dead_latent_threshold: float = 15
    random_seed: int = 42
    dataset_name: str = "monology/pile-uncopyrighted"
    llm_context_size: int = 128
    llm_batch_size: int = 32
    buffer: int = 10
    no_overlap: bool = True
    act_threshold_frac: float = 0.01
    total_tokens: int = 2_000_000
    scoring: bool = True
    max_tokens_in_explanation: int = 30
    use_demos_in_explanation: bool = True
    n_top_ex_for_generation: int = 10
    n_iw_sampled_ex_for_generation: int = 5
    n_top_ex_for_scoring: int = 2
    n_random_ex_for_scoring: int = 10
    n_iw_sampled_ex_for_scoring: int = 2

    def __post_init__(self) -> None:
        if self.n_latents is None:
            if self.override_latents is None:
                raise ValueError("override_latents must be set when n_latents is None")
            self.latents = self.override_latents
            self.n_latents = len(self.latents)
        else:
            self.latents = None

    @property
    def n_top_ex(self) -> int:
        return self.n_top_ex_for_generation + self.n_top_ex_for_scoring

    @property
    def n_ex_for_generation(self) -> int:
        return self.n_top_ex_for_generation + self.n_iw_sampled_ex_for_generation

    @property
    def n_ex_for_scoring(self) -> int:
        return (
            self.n_top_ex_for_scoring
            + self.n_random_ex_for_scoring
            + self.n_iw_sampled_ex_for_scoring
        )

    @property
    def n_correct_for_scoring(self) -> int:
        return self.n_top_ex_for_scoring + self.n_iw_sampled_ex_for_scoring

    @property
    def max_tokens_in_prediction(self) -> int:
        return 2 * self.n_ex_for_scoring + 5


class Example:
    def __init__(
        self,
        toks: list[int],
        acts: list[float],
        act_threshold: float,
        tokenizer: AutoTokenizer,
    ) -> None:
        self.toks = toks
        self.str_toks = _tokens_to_str(toks, tokenizer)
        self.acts = acts
        self.act_threshold = act_threshold
        self.toks_are_active = [act > act_threshold for act in self.acts]
        self.is_active = any(self.toks_are_active)

    def to_str(self, mark_toks: bool = False) -> str:
        return (
            "".join(
                f"<<{tok}>>" if (mark_toks and is_active) else tok
                for tok, is_active in zip(self.str_toks, self.toks_are_active)
            )
            .replace("�", "")
            .replace("\n", "↵")
        )


class Examples:
    def __init__(self, examples: list[Example], shuffle: bool = False) -> None:
        self.examples = examples
        if shuffle:
            random.shuffle(self.examples)
        else:
            self.examples = sorted(self.examples, key=lambda x: max(x.acts), reverse=True)

    def display(self, predictions: Optional[list[int]] = None) -> str:
        return tabulate(
            [
                (
                    [max(ex.acts), ex.to_str(mark_toks=True)]
                    if predictions is None
                    else [
                        max(ex.acts),
                        "Y" if ex.is_active else "",
                        "Y" if (i + 1 in predictions) else "",
                        ex.to_str(mark_toks=False),
                    ]
                )
                for i, ex in enumerate(self.examples)
            ],
            headers=["Top act"]
            + ([] if predictions is None else ["Active?", "Predicted?"])
            + ["Sequence"],
            tablefmt="simple_outline",
            floatfmt=".3f",
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __iter__(self):
        return iter(self.examples)

    def __getitem__(self, i: int) -> Example:
        return self.examples[i]


class AutoInterpRunner:
    def __init__(
        self,
        cfg: AutoInterpConfig,
        model: PreTrainedModel,
        sae: Any,
        tokenized_dataset: Tensor,
        sparsity: Tensor,
        tokenizer: AutoTokenizer,
        hidden_state_index: int,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.sae = sae
        self.tokenized_dataset = tokenized_dataset
        self.tokenizer = tokenizer
        self.hidden_state_index = hidden_state_index

        if cfg.latents is not None:
            self.latents = cfg.latents
        else:
            sparsity = sparsity * cfg.total_tokens
            alive_latents = (
                torch.nonzero(sparsity > cfg.dead_latent_threshold)
                .squeeze(1)
                .tolist()
            )
            if len(alive_latents) < cfg.n_latents:
                self.latents = alive_latents
            else:
                self.latents = random.sample(alive_latents, k=cfg.n_latents)
        self.n_latents = len(self.latents)

    async def run(self, explanations_override: dict[int, str] | None = None) -> dict[int, dict[str, Any]]:
        explanations_override = explanations_override or {}
        generation_examples, scoring_examples = self.gather_data()
        latents_with_data = sorted(generation_examples.keys())
        n_dead = self.n_latents - len(latents_with_data)
        if n_dead > 0:
            print(
                f"Found data for {len(latents_with_data)}/{self.n_latents} alive latents; {n_dead} dead"
            )

        with ThreadPoolExecutor(max_workers=10) as executor:
            tasks = [
                self.run_single_feature(
                    executor,
                    latent,
                    generation_examples[latent],
                    scoring_examples[latent],
                    explanations_override.get(latent, None),
                )
                for latent in latents_with_data
            ]
            results = {}
            for future in asyncio.as_completed(tasks):
                result = await future
                if result:
                    results[result["latent"]] = result
        return results

    async def run_single_feature(
        self,
        executor: ThreadPoolExecutor,
        latent: int,
        generation_examples: Examples,
        scoring_examples: Examples,
        explanation_override: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        gen_prompts = self.get_generation_prompts(generation_examples)
        (explanation_raw,), logs = await asyncio.get_event_loop().run_in_executor(
            executor,
            self.get_api_response,
            gen_prompts,
            self.cfg.max_tokens_in_explanation,
        )
        explanation = self.parse_explanation(explanation_raw)
        results: dict[str, Any] = {
            "latent": latent,
            "explanation": explanation,
            "logs": f"Generation phase\n{logs}\n{generation_examples.display()}",
        }

        if self.cfg.scoring:
            scoring_prompts = self.get_scoring_prompts(
                explanation=explanation_override or explanation,
                scoring_examples=scoring_examples,
            )
            (predictions_raw,), logs = await asyncio.get_event_loop().run_in_executor(
                executor,
                self.get_api_response,
                scoring_prompts,
                self.cfg.max_tokens_in_prediction,
            )
            predictions = self.parse_predictions(predictions_raw)
            if predictions is None:
                return None
            score = self.score_predictions(predictions, scoring_examples)
            results |= {
                "predictions": predictions,
                "correct seqs": [
                    i for i, ex in enumerate(scoring_examples, start=1) if ex.is_active
                ],
                "score": score,
                "logs": results["logs"]
                + f"\nScoring phase\n{logs}\n{scoring_examples.display(predictions)}",
            }
        return results

    def parse_explanation(self, explanation: str) -> str:
        return explanation.split("activates on")[-1].rstrip(".").strip()

    def parse_predictions(self, predictions: str) -> Optional[list[int]]:
        predictions_split = (
            predictions.strip()
            .rstrip(".")
            .replace("and", ",")
            .replace("None", "")
            .split(",")
        )
        predictions_list = [i.strip() for i in predictions_split if i.strip() != ""]
        if predictions_list == []:
            return []
        if not all(pred.strip().isdigit() for pred in predictions_list):
            return None
        return [int(pred.strip()) for pred in predictions_list]

    def score_predictions(self, predictions: list[int], scoring_examples: Examples) -> float:
        classifications = [
            i in predictions for i in range(1, len(scoring_examples) + 1)
        ]
        correct_classifications = [ex.is_active for ex in scoring_examples]
        return sum(
            c == cc for c, cc in zip(classifications, correct_classifications)
        ) / len(classifications)

    def get_api_response(
        self, messages: list[dict[str, str]], max_tokens: int, n_completions: int = 1
    ) -> tuple[list[str], str]:
        for message in messages:
            if set(message.keys()) != {"content", "role"}:
                raise ValueError("Messages must have 'role' and 'content' only.")
        client = OpenAI(api_key=self.cfg.openai_api_key)
        result = client.chat.completions.create(
            model=self.cfg.openai_model,
            messages=messages,  # type: ignore[arg-type]
            n=n_completions,
            max_tokens=max_tokens,
            stream=False,
        )
        response = [choice.message.content.strip() for choice in result.choices]
        logs = tabulate(
            [
                m.values()
                for m in messages + [{"role": "assistant", "content": response[0]}]
            ],
            tablefmt="simple_grid",
            maxcolwidths=[None, 120],
        )
        return response, logs

    def get_generation_prompts(self, generation_examples: Examples) -> list[dict[str, str]]:
        examples_as_str = "\n".join(
            [
                f"{i + 1}. {ex.to_str(mark_toks=True)}"
                for i, ex in enumerate(generation_examples)
            ]
        )

        system_prompt = (
            "We're studying neurons in a neural network. Each neuron activates on some "
            "particular word/words/substring/concept in a short document. The activating "
            "words in each document are indicated with << ... >>. We will give you a list "
            "of documents on which the neuron activates, in order from most strongly activating "
            "to least strongly activating. Look at the parts of the document the neuron activates "
            "for and summarize in a single sentence what the neuron is activating on. Try not "
            "to be overly specific in your explanation. Note that some neurons will activate "
            "only on specific words or substrings, but others will activate on most/all words in "
            "a sentence provided that sentence contains some particular concept. Your explanation "
            "should cover most or all activating words (for example, don't give an explanation "
            "which is specific to a single word if all words in a sentence cause the neuron to "
            "activate). Pay attention to things like the capitalization and punctuation of the "
            "activating words or concepts, if that seems relevant. Keep the explanation as short "
            "and simple as possible, limited to 20 words or less. Omit punctuation and formatting. "
            "You should avoid giving long lists of words."
        )
        if self.cfg.use_demos_in_explanation:
            system_prompt += (
                " Some examples: \"This neuron activates on the word 'knows' in rhetorical "
                "questions\", and \"This neuron activates on verbs related to decision-making "
                "and preferences\", and \"This neuron activates on the substring 'Ent' at the "
                "start of words\", and \"This neuron activates on text about government economic "
                "policy\"."
            )
        else:
            system_prompt += 'Your response should be in the form "This neuron activates on...".'

        user_prompt = f"The activating documents are given below:\n\n{examples_as_str}"

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def get_scoring_prompts(
        self, explanation: str, scoring_examples: Examples
    ) -> list[dict[str, str]]:
        examples_as_str = "\n".join(
            [
                f"{i + 1}. {ex.to_str(mark_toks=False)}"
                for i, ex in enumerate(scoring_examples)
            ]
        )
        example_response = sorted(
            random.sample(
                range(1, 1 + self.cfg.n_ex_for_scoring),
                k=self.cfg.n_correct_for_scoring,
            )
        )
        example_response_str = ", ".join([str(i) for i in example_response])
        system_prompt = (
            "We're studying neurons in a neural network. Each neuron activates on some particular "
            "word/words/substring/concept in a short document. You will be given a short explanation "
            f"of what this neuron activates for, and then be shown {self.cfg.n_ex_for_scoring} example "
            "sequences in random order. You will have to return a comma-separated list of the examples "
            "where you think the neuron should activate at least once, on ANY of the words or substrings "
            f"in the document. For example, your response might look like \"{example_response_str}\". Try "
            "not to be overly specific in your interpretation of the explanation. If you think there are "
            "no examples where the neuron will activate, you should just respond with \"None\". You should "
            "include nothing else in your response other than comma-separated numbers or the word \"None\" "
            "- this is important."
        )
        user_prompt = (
            f"Here is the explanation: this neuron fires on {explanation}.\n\n"
            f"Here are the examples:\n\n{examples_as_str}"
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def gather_data(self) -> tuple[dict[int, Examples], dict[int, Examples]]:
        dataset_size, seq_len = self.tokenized_dataset.shape

        acts = collect_sae_activations_hidden(
            self.tokenized_dataset,
            self.model,
            self.sae,
            self.cfg.llm_batch_size,
            self.hidden_state_index,
            self.tokenizer,
            mask_bos_pad_eos_tokens=True,
            selected_latents=self.latents,
            activation_dtype=torch.bfloat16,
        )

        generation_examples: dict[int, Examples] = {}
        scoring_examples: dict[int, Examples] = {}

        for i, latent in enumerate(self.latents):
            rand_indices = torch.stack(
                [
                    torch.randint(0, dataset_size, (self.cfg.n_random_ex_for_scoring,)),
                    torch.randint(
                        self.cfg.buffer,
                        seq_len - self.cfg.buffer,
                        (self.cfg.n_random_ex_for_scoring,),
                    ),
                ],
                dim=-1,
            )
            rand_toks = index_with_buffer(
                self.tokenized_dataset, rand_indices, buffer=self.cfg.buffer
            )

            top_indices = get_k_largest_indices(
                acts[..., i],
                k=self.cfg.n_top_ex,
                buffer=self.cfg.buffer,
                no_overlap=self.cfg.no_overlap,
            )
            top_toks = index_with_buffer(
                self.tokenized_dataset, top_indices, buffer=self.cfg.buffer
            )
            top_values = index_with_buffer(acts[..., i], top_indices, buffer=self.cfg.buffer)
            act_threshold = self.cfg.act_threshold_frac * top_values.max().item()

            threshold = top_values[:, self.cfg.buffer].min().item()
            acts_thresholded = torch.where(acts[..., i] >= threshold, 0.0, acts[..., i])
            if acts_thresholded[:, self.cfg.buffer : -self.cfg.buffer].max() < 1e-6:
                continue
            iw_indices = get_iw_sample_indices(
                acts_thresholded, k=self.cfg.n_iw_sampled_ex, buffer=self.cfg.buffer
            )
            iw_toks = index_with_buffer(
                self.tokenized_dataset, iw_indices, buffer=self.cfg.buffer
            )
            iw_values = index_with_buffer(acts[..., i], iw_indices, buffer=self.cfg.buffer)

            rand_top_ex_split_indices = torch.randperm(self.cfg.n_top_ex)
            top_gen_indices = rand_top_ex_split_indices[: self.cfg.n_top_ex_for_generation]
            top_scoring_indices = rand_top_ex_split_indices[self.cfg.n_top_ex_for_generation :]
            rand_iw_split_indices = torch.randperm(self.cfg.n_iw_sampled_ex)
            iw_gen_indices = rand_iw_split_indices[: self.cfg.n_iw_sampled_ex_for_generation]
            iw_scoring_indices = rand_iw_split_indices[self.cfg.n_iw_sampled_ex_for_generation :]

            def create_examples(all_toks: Tensor, all_acts: Optional[Tensor] = None) -> list[Example]:
                if all_acts is None:
                    all_acts = torch.zeros_like(all_toks).float()
                return [
                    Example(
                        toks=toks,
                        acts=acts.tolist(),
                        act_threshold=act_threshold,
                        tokenizer=self.tokenizer,
                    )
                    for (toks, acts) in zip(all_toks.tolist(), all_acts.tolist())
                ]

            generation_examples[latent] = Examples(
                create_examples(top_toks[top_gen_indices], top_values[top_gen_indices])
                + create_examples(iw_toks[iw_gen_indices], iw_values[iw_gen_indices]),
            )
            scoring_examples[latent] = Examples(
                create_examples(top_toks[top_scoring_indices], top_values[top_scoring_indices])
                + create_examples(iw_toks[iw_scoring_indices], iw_values[iw_scoring_indices])
                + create_examples(rand_toks),
                shuffle=True,
            )

        return generation_examples, scoring_examples


def load_tokenized_dataset_cached(
    dataset: Iterable[dict],
    tokenizer: AutoTokenizer,
    ctx_len: int,
    total_tokens: int,
    text_column: str,
    cache_path: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    if cache_path and os.path.exists(cache_path):
        tokens = torch.load(cache_path)
    else:
        tokens = load_and_tokenize_dataset(
            dataset,
            tokenizer,
            ctx_len=ctx_len,
            num_tokens=total_tokens,
            column_name=text_column,
        )
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            torch.save(tokens, cache_path)
    if device is not None:
        tokens = tokens.to(device)
    return tokens
