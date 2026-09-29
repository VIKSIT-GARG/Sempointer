"""RULER synthetic long-context benchmark (NVIDIA/RULER).

Faithful re-implementation of the RULER data generators for the tasks:
niah_single_1, niah_single_2, niah_single_3, niah_multikey_1, niah_multikey_2,
vt (variable tracking), cwe (common words extraction), fwe (frequent words
extraction).

Primary sources (fetched 2026-09-28 from NVIDIA/RULER @ main):
  - Repo:        https://github.com/NVIDIA/RULER
  - Paper:       arXiv:2404.06654 ("RULER: What's the Real Context Size of
                 Your Long-Context Language Models?")
  - Task configs: scripts/data/synthetic.yaml
  - Templates:   scripts/data/synthetic/constants.py (TASKS dict), composed as
                 ``template + answer_prefix`` in scripts/data/prepare.py for
                 model_template_type='base'.
  - Generators:  scripts/data/synthetic/{niah,variable_tracking,
                 common_words_extraction,freq_words_extraction}.py
  - Official eval: scripts/eval/synthetic/constants.py -> string_match_all /
                 string_match_part: per sample, the fraction of gold
                 references contained (case-insensitive) in the prediction;
                 task score = mean over samples x 100. The official evaluator
                 (scripts/eval/evaluate.py::postprocess_pred) only strips
                 non-printable characters -- there is NO answer-extraction
                 step. Implemented in metrics/official_metrics.py.

Prompt layout (official, base template): the model prompt is
``pre + {context} + post`` where ``post`` already ends with the official
answer_prefix (scripts/pred/call_api.py sends input + answer_prefix to the
model). In BenchExample: ``context`` is the exact official {context} fill and
``question`` = pre + post (the full instruction + question + answer prefix,
with the haystack removed). meta["prompt_before_context"] /
meta["prompt_after_context"] let a runner reconstruct the byte-exact official
prompt as before + context + after.

Documented deviations (conservative; see notes inline):
  1. Essay corpus: we cache the first 5 Paul Graham essays in the official
     sorted order (official corpus = 275 essays; ours matches its prefix).
     The official repetition logic is kept for longer requests.
  2. Word lists: `wonderwords` is not installed in this environment; its
     word-list assets (v3.0.1) are cached and read identically. NIAH keys
     sample adjective and noun independently -- distribution-identical to the
     official uniform choice over the adj-noun product set.
  3. Token counting: exact when ``tokenizer_fn`` is provided (official
     binary-search haystack sizing). Without a tokenizer, documented
     per-family approximations are used and realized lengths can deviate up
     to ~20%: prose ~0.75 BPE tokens/word (niah/vt), ~1 token per numbered
     list entry (cwe), ~2 tokens per 6-letter coded word (fwe).
  4. The official per-task jsonl bookkeeping fields (length,
     token_position_answer, ...) are not reproduced; gold answers are
     recorded at generation time from the seeded RNG instead.
"""

from __future__ import annotations

import json
import random
import re
import string
import uuid
from pathlib import Path

import numpy as np

from .common import BenchExample

RULER_GITHUB = "https://github.com/NVIDIA/RULER"
RULER_PAPER = "https://arxiv.org/abs/2404.06654"
ESSAY_SOURCE = "https://github.com/gkamradt/LLMTest_NeedleInAHaystack"

_CACHE_DIR = Path(__file__).resolve().parent / "data" / "ruler_cache"
_ESSAY_DIR = _CACHE_DIR / "essays"
_WORDLIST_DIR = _CACHE_DIR / "wordlists"
_ENGLISH_WORDS_JSON = _CACHE_DIR / "english_words.json"

# ---------------------------------------------------------------------------
# Official constants (verbatim from NVIDIA/RULER scripts)
# ---------------------------------------------------------------------------

# scripts/data/synthetic/{niah,variable_tracking}.py
NOISE_SENTENCE = (
    "The grass is green. The sky is blue. The sun is yellow. "
    "Here we go. There and back again."
)
NEEDLE_FMT = "One of the special magic {type_needle_v} for {key} is: {value}."

# scripts/data/synthetic/constants.py (verbatim)
_OFFICIAL_TASKS = {
    "niah": {
        "tokens_to_generate": 128,
        "template": (
            "Some special magic {type_needle_v} are hidden within the "
            "following text. Make sure to memorize it. I will quiz you about "
            "the {type_needle_v} afterwards.\n{context}\nWhat are all the "
            "special magic {type_needle_v} for {query} mentioned in the "
            "provided text?"
        ),
        "answer_prefix": (
            " The special magic {type_needle_v} for {query} mentioned in the "
            "provided text are"
        ),
    },
    "variable_tracking": {
        "tokens_to_generate": 30,
        "template": (
            "Memorize and track the chain(s) of variable assignment hidden in "
            "the following text.\n\n{context}\nQuestion: Find all variables "
            "that are assigned the value {query} in the text above."
        ),
        "answer_prefix": (
            " Answer: According to the chain(s) of variable assignment in the "
            "text above, {num_v} variables are assigned the value {query}, "
            "they are: "
        ),
    },
    "common_words_extraction": {
        "tokens_to_generate": 120,
        "template": (
            "Below is a numbered list of words. In these words, some appear "
            "more often than others. Memorize the ones that appear most "
            "often.\n{context}\nQuestion: What are the 10 most common words "
            "in the above list?"
        ),
        "answer_prefix": (
            " Answer: The top 10 words that appear most often in the list are:"
        ),
    },
    "freq_words_extraction": {
        "tokens_to_generate": 50,
        "template": (
            "Read the following coded text and track the frequency of each "
            "coded word. Find the three most frequently appeared coded words. "
            "{context}\nQuestion: Do not provide any explanation. Please "
            "ignore the dots '....'. What are the three most frequently "
            "appeared words in the above coded text?"
        ),
        "answer_prefix": (
            " Answer: According to the coded text above, the three most "
            "frequently appeared words are:"
        ),
    },
}

# scripts/data/synthetic/synthetic.yaml (verbatim args for the 8 tasks)
TASK_ARGS = {
    "niah_single_1": dict(task="niah", type_haystack="noise", type_needle_k="words",
                          type_needle_v="numbers", num_needle_k=1, num_needle_v=1,
                          num_needle_q=1),
    "niah_single_2": dict(task="niah", type_haystack="essay", type_needle_k="words",
                          type_needle_v="numbers", num_needle_k=1, num_needle_v=1,
                          num_needle_q=1),
    "niah_single_3": dict(task="niah", type_haystack="essay", type_needle_k="words",
                          type_needle_v="uuids", num_needle_k=1, num_needle_v=1,
                          num_needle_q=1),
    "niah_multikey_1": dict(task="niah", type_haystack="essay", type_needle_k="words",
                            type_needle_v="numbers", num_needle_k=4, num_needle_v=1,
                            num_needle_q=1),
    "niah_multikey_2": dict(task="niah", type_haystack="needle", type_needle_k="words",
                            type_needle_v="numbers", num_needle_k=1, num_needle_v=1,
                            num_needle_q=1),
    "vt": dict(task="variable_tracking", type_haystack="noise", num_chains=1,
               num_hops=4),
    "cwe": dict(task="common_words_extraction", freq_cw=30, freq_ucw=3, num_cw=10),
    "fwe": dict(task="freq_words_extraction", alpha=2.0),
}

RULER_TASKS = sorted(TASK_ARGS)

# scripts/data/synthetic/niah.py: 40 evenly spaced depths in [0, 100]
DEPTHS = [int(d) for d in np.round(np.linspace(0, 100, num=40, endpoint=True))]

# Official haystack sizing increments (generate_samples / generators)
_INCREMENTAL = {"essay": 500, "noise": 25, "needle": 25}

# Documented word-based token approximation (deviation 3)
_WORDS_PER_TOKEN_PROSE = 0.75   # ~0.75 BPE tokens per English prose word


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------

def _load_wordlists() -> dict:
    """wonderwords 3.0.1 word lists, read exactly like
    wonderwords.random_word._get_words_from_text_file (readlines + rstrip)."""
    out = {}
    for name in ("nounlist.txt", "adjectivelist.txt", "verblist.txt"):
        path = _WORDLIST_DIR / name
        if not path.exists():
            raise FileNotFoundError(
                f"Missing cached word list {path} (wonderwords 3.0.1 assets; "
                "cache them under benchmarks/data/ruler_cache/wordlists/ -- "
                "see the module docstring for provenance)."
            )
        out[name] = [w.rstrip() for w in path.read_text(encoding="utf-8").splitlines()
                     if w.strip()]
    return out


def _load_essay_words() -> list:
    """Official flat haystack word list.

    Mirrors scripts/data/synthetic/niah.py:
        haystack = re.sub(r'\\s+', " ", essay_text).split(" ")
    where essay_text concatenates the cached Paul Graham essays in sorted
    filename order (the official corpus order: gkamradt repo .txt files).
    """
    paths = sorted(_ESSAY_DIR.glob("*.txt")) if _ESSAY_DIR.exists() else []
    if not paths:
        raise FileNotFoundError(
            f"No cached Paul Graham essays under {_ESSAY_DIR} (fetched at "
            f"build time from {ESSAY_SOURCE})."
        )
    text = "".join(p.read_text(encoding="utf-8") for p in paths)
    return re.sub(r"\s+", " ", text).split(" ")


def _randle_words() -> list:
    """RULER's english_words.json fallback pool (cwe, only when the numbered
    list exceeds the wonderwords vocabulary)."""
    if not _ENGLISH_WORDS_JSON.exists():
        raise FileNotFoundError(f"Missing {_ENGLISH_WORDS_JSON}")
    with open(_ENGLISH_WORDS_JSON, "r", encoding="utf-8") as f:
        return list(json.load(f).values())


def _sent_tokenize(text: str) -> list:
    import nltk  # official generators use nltk.tokenize.sent_tokenize
    return nltk.tokenize.sent_tokenize(text)


# ---------------------------------------------------------------------------
# Token accounting / haystack sizing (official binary search)
# ---------------------------------------------------------------------------

def _prose_token_count(tokenizer_fn):
    """niah/vt counter: exact with tokenizer_fn, else word-based estimate."""
    if tokenizer_fn is not None:
        return tokenizer_fn
    return lambda text: int(len(text.split()) / _WORDS_PER_TOKEN_PROSE)


def _format_sides(template_full: str, **vals):
    """Split the official ``template + answer_prefix`` at {context} and fill
    every remaining placeholder on each side. Returns (pre, post) such that
    pre + context + post == the fully formatted official model prompt."""
    pre, post = template_full.split("{context}", 1)
    return pre.format(**vals), post.format(**vals)


def _binary_search_haystack(incremental: int, count_fn, tokens_to_generate: int,
                            context_length: int, build_text_fn, sample_units: int,
                            extra_tokens: int = 0):
    """Official sizing search: estimate tokens/unit from a sample haystack,
    3x slack upper bound, then binary-search the largest haystack size whose
    input + tokens_to_generate (+ extra_tokens, e.g. the vt ICL example)
    fits the context length. build_text_fn(units) returns (text, <ignored>)."""
    sample_text, _ = build_text_fn(sample_units)
    tokens_per_unit = max(count_fn(sample_text), 1) / sample_units
    estimated_max = int((context_length / tokens_per_unit) * 3)

    lo, hi = incremental, max(estimated_max, incremental * 2)
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        text, _ = build_text_fn(mid)
        if count_fn(text) + tokens_to_generate + extra_tokens <= context_length:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best if best is not None else incremental


# ---------------------------------------------------------------------------
# NIAH  (scripts/data/synthetic/niah.py)
# ---------------------------------------------------------------------------

def _random_key(type_needle_k, wordlists, rng=random):
    """Official generate_random for keys. For 'words', official samples from
    the sorted set of all adjective-noun compounds; independent uniform
    adjective + noun sampling is distribution-identical (deviation 2)."""
    if type_needle_k == "numbers":
        return str(rng.randint(10**6, 10**7 - 1))
    if type_needle_k == "words":
        return f"{rng.choice(wordlists['adjectivelist.txt'])}-" \
               f"{rng.choice(wordlists['nounlist.txt'])}"
    if type_needle_k == "uuids":
        return str(uuid.UUID(int=rng.getrandbits(128), version=4))
    raise NotImplementedError(type_needle_k)


def _random_value(type_needle_v, rng=random):
    if type_needle_v == "numbers":
        return str(rng.randint(10**6, 10**7 - 1))
    if type_needle_v == "uuids":
        return str(uuid.UUID(int=rng.getrandbits(128), version=4))
    raise NotImplementedError(type_needle_v)


def _singularize_niah_template(template: str, type_needle_v: str):
    """Official niah.py singularization (num_needle_q * num_needle_v == 1):
    'Some'->'A', 'are all'->'is', 'are'->'is', 'answers'->'answer', and the
    needle-type word loses its trailing 's'."""
    template = template.replace("Some", "A")
    template = template.replace("are all", "is")
    template = template.replace("are", "is")
    template = template.replace("answers", "answer")
    return template, type_needle_v[:-1]


def build_niah(task: str, args: dict, context_length: int, n_samples: int,
               seed: int, count_fn) -> list:
    random.seed(seed)
    np.random.seed(seed)

    tokens_to_generate = _OFFICIAL_TASKS["niah"]["tokens_to_generate"]
    template = (_OFFICIAL_TASKS["niah"]["template"]
                + _OFFICIAL_TASKS["niah"]["answer_prefix"])

    type_haystack = args["type_haystack"]
    type_needle_k, type_needle_v = args["type_needle_k"], args["type_needle_v"]
    num_needle_k = max(args["num_needle_k"], args["num_needle_q"])
    num_needle_v, num_needle_q = args["num_needle_v"], args["num_needle_q"]

    if num_needle_q * num_needle_v == 1:
        template, type_needle_v_fmt = _singularize_niah_template(template, type_needle_v)
    else:
        type_needle_v_fmt = type_needle_v

    wordlists = _load_wordlists() if type_needle_k == "words" else None
    essay_words = _load_essay_words() if type_haystack == "essay" else None
    incremental = _INCREMENTAL[type_haystack]

    def essay_text(num_haystack):
        if num_haystack <= len(essay_words):
            return " ".join(essay_words[:num_haystack])
        repeats = (num_haystack + len(essay_words) - 1) // len(essay_words)
        return " ".join((essay_words * repeats)[:num_haystack])

    def generate(num_haystack):
        """Mirrors niah.py generate_input_output. Returns
        (input_text, context, answers, needle_depths, query)."""
        keys, values, needles = [], [], []
        for _ in range(num_needle_k):
            keys.append(_random_key(type_needle_k, wordlists, random))
            value = []
            for _ in range(num_needle_v):
                value.append(_random_value(type_needle_v, random))
                needles.append(NEEDLE_FMT.format(type_needle_v=type_needle_v,
                                                 key=keys[-1], value=value[-1]))
            values.append(value)

        random.Random(seed).shuffle(needles)  # official needle order shuffle

        needle_depths = []
        if type_haystack == "essay":
            document_sents = _sent_tokenize(essay_text(num_haystack).strip())
            insertion_positions = [0] + sorted(
                [int(len(document_sents) * (depth / 100))
                 for depth in random.sample(DEPTHS, len(needles))]) \
                + [len(document_sents)]
            parts = []
            for i in range(1, len(insertion_positions)):
                last_pos, next_pos = insertion_positions[i - 1], insertion_positions[i]
                parts.append(" ".join(document_sents[last_pos:next_pos]))
                if i - 1 < len(needles):
                    parts.append(needles[i - 1])
                    needle_depths.append((needles[i - 1], round(
                        insertion_positions[i] / max(len(document_sents), 1), 4)))
            context = " ".join(parts)
        else:
            if type_haystack == "noise":
                sentences = [NOISE_SENTENCE] * num_haystack
            else:  # 'needle': haystack lines are needle-like distractors
                sentences = [
                    NEEDLE_FMT.format(type_needle_v=type_needle_v,
                                      key=_random_key(type_needle_k, wordlists, random),
                                      value=_random_value(type_needle_v, random))
                    for _ in range(num_haystack)]
            indexes = sorted(random.sample(range(num_haystack), len(needles)),
                             reverse=True)
            for index, element in zip(indexes, needles):
                sentences.insert(index, element)
                needle_depths.append((element, round(index / max(num_haystack, 1), 4)))
            context = "\n".join(sentences)

        indices = random.sample(range(num_needle_k), num_needle_q)
        queries = [keys[i] for i in indices]
        answers = [a for i in indices for a in values[i]]
        query = (", ".join(queries[:-1]) + ", and " + queries[-1]
                 if len(queries) > 1 else queries[0])
        input_text = template.format(type_needle_v=type_needle_v_fmt,
                                     context=context, query=query)
        return input_text, context, answers, needle_depths, query

    def build_text(n):
        input_text, _, _, _, _ = generate(n)
        return input_text, None  # full formatted prompt, like official sizing

    num_haystack = _binary_search_haystack(
        incremental, count_fn, tokens_to_generate, context_length, build_text,
        incremental)

    examples = []
    for idx in range(n_samples):
        used = num_haystack
        while True:
            input_text, context, answers, needle_depths, query = generate(used)
            if count_fn(input_text) + tokens_to_generate <= context_length \
                    or used <= incremental:
                break
            used -= incremental

        pre, post = _format_sides(template, type_needle_v=type_needle_v_fmt,
                                  query=query)
        depth_fraction = None
        for needle_text, d in needle_depths:
            if all(a in needle_text for a in answers):
                depth_fraction = d
                break

        examples.append(BenchExample(
            id=f"{task}-{context_length}-{idx}",
            context=context,
            question=pre + post,
            gold=list(answers),
            metric="ruler_containment",
            meta={
                "task": task,
                "benchmark": "ruler",
                "source_urls": [RULER_GITHUB, RULER_PAPER],
                "needle_depth_fraction": depth_fraction,
                "all_needle_depths": [d for _, d in needle_depths],
                "haystack_type": type_haystack,
                "num_haystack_units": used,
                "type_needle_k": type_needle_k,
                "type_needle_v": type_needle_v,
                "query_key": query,
                "prompt_before_context": pre,
                "prompt_after_context": post,
                "answer_prefix": _OFFICIAL_TASKS["niah"]["answer_prefix"].format(
                    type_needle_v=type_needle_v_fmt, query=query),
                "tokens_to_generate": tokens_to_generate,
                "context_length_requested": context_length,
            },
        ))
    return examples


# ---------------------------------------------------------------------------
# Variable tracking  (scripts/data/synthetic/variable_tracking.py)
# ---------------------------------------------------------------------------

def _generate_chains(num_chains, num_hops, is_icl=False):
    """Official variable_tracking.py generate_chains."""
    k = 5 if not is_icl else 3
    num_hops_eff = num_hops if not is_icl else min(10, num_hops)
    vars_all = ["".join(random.choices(string.ascii_uppercase, k=k)).upper()
                for _ in range((num_hops_eff + 1) * num_chains)]
    while len(set(vars_all)) < num_chains * (num_hops_eff + 1):
        vars_all.append("".join(random.choices(string.ascii_uppercase, k=k)).upper())

    vars_ret, chains_ret = [], []
    for i in range(0, len(vars_all), num_hops_eff + 1):
        this_vars = vars_all[i:i + num_hops_eff + 1]
        vars_ret.append(this_vars)
        if is_icl:
            this_chain = [f"VAR {this_vars[0]} = 12345"]
        else:
            this_chain = [f"VAR {this_vars[0]} = {int(np.random.randint(10000, 99999))}"]
        for j in range(num_hops_eff):
            this_chain.append(f"VAR {this_vars[j + 1]} = VAR {this_vars[j]} ")
        chains_ret.append(this_chain)
    return vars_ret, chains_ret


def _randomize_icl(icl_example: str, num_hops: int) -> str:
    """Official variable_tracking.py randomize_icl."""
    icl_tgt = icl_example.strip().split()[-num_hops - 1:]
    for item in icl_tgt:
        new_item = "".join(random.choices(string.ascii_uppercase, k=len(item))).upper()
        icl_example = icl_example.replace(item, new_item)
    new_value = str(int(np.random.randint(10000, 99999)))
    icl_example = icl_example.replace("12345", new_value)
    return icl_example


def build_vt(task: str, args: dict, context_length: int, n_samples: int,
             seed: int, count_fn) -> list:
    random.seed(seed)
    np.random.seed(seed)

    spec = _OFFICIAL_TASKS["variable_tracking"]
    tokens_to_generate = spec["tokens_to_generate"]
    template = spec["template"] + spec["answer_prefix"]
    num_chains, num_hops = args["num_chains"], args["num_hops"]
    incremental = 10  # official vt noise-haystack increment

    def generate(num_noises, is_icl=False):
        """Official generate_input_output (noise haystack branch)."""
        vars_ret, chains = _generate_chains(num_chains, num_hops, is_icl=is_icl)
        value = chains[0][0].split("=")[-1].strip()

        sentences = [NOISE_SENTENCE] * num_noises
        for chain in chains:
            positions = sorted(random.sample(range(len(sentences)), len(chain)))
            for insert_pi, j in zip(positions, range(len(chain))):
                sentences.insert(insert_pi + j, chain[j])
        context = "\n".join(sentences).replace(". \n", ".\n")  # official replace

        input_text = template.format(context=context, query=value,
                                     num_v=num_hops + 1)
        return context, input_text, vars_ret[0], value

    # Official main(): a 1-shot ICL example (~500-token budget) is generated
    # with is_icl=True, randomized per sample, and prepended to each input
    # (with the base model template, the official cutoff is 0).
    def icl_build(n):
        _, text, _, _ = generate(n, is_icl=True)
        return text, None

    icl_num_noises = _binary_search_haystack(5, count_fn, 0, 500, icl_build, 5)
    _, icl_input, icl_outputs, _ = generate(icl_num_noises, is_icl=True)
    icl_example_base = icl_input + " " + " ".join(icl_outputs) + "\n"
    # Official: the ICL example's tokens are counted in the haystack sizing
    # (example_tokens) and in the per-sample length check (the insertion
    # happens before the official length assert).
    example_tokens = count_fn(icl_example_base)

    def build_text(n):
        _, text, _, _ = generate(n)
        return text, None

    num_noises = _binary_search_haystack(
        incremental, count_fn, tokens_to_generate, context_length, build_text,
        incremental, extra_tokens=example_tokens)

    examples = []
    for idx in range(n_samples):
        used = num_noises
        while True:
            context, input_text, answers, query = generate(used)
            icl_example = _randomize_icl(icl_example_base, num_hops)
            full_input = icl_example + "\n" + input_text
            if count_fn(full_input) + tokens_to_generate <= context_length \
                    or used <= incremental:
                break
            used -= incremental

        input_text = icl_example + "\n" + input_text

        pre, post = _format_sides(template, query=query, num_v=num_hops + 1)
        examples.append(BenchExample(
            id=f"{task}-{context_length}-{idx}",
            context=context,
            question=icl_example + "\n" + pre + post,
            gold=list(answers),
            metric="ruler_containment",
            meta={
                "task": task,
                "benchmark": "ruler",
                "source_urls": [RULER_GITHUB, RULER_PAPER],
                "needle_depth_fraction": None,
                "haystack_type": "noise",
                "num_haystack_units": used,
                "num_chains": num_chains,
                "num_hops": num_hops,
                "query_value": query,
                "icl_example": icl_example,
                "prompt_before_context": icl_example + "\n" + pre,
                "prompt_after_context": post,
                "answer_prefix": spec["answer_prefix"].format(
                    query=query, num_v=num_hops + 1),
                "tokens_to_generate": tokens_to_generate,
                "context_length_requested": context_length,
            },
        ))
    return examples


# ---------------------------------------------------------------------------
# Common words extraction (scripts/data/synthetic/common_words_extraction.py)
# ---------------------------------------------------------------------------

def build_cwe(task: str, args: dict, context_length: int, n_samples: int,
              seed: int, tokenizer_fn) -> list:
    random.seed(seed)

    spec = _OFFICIAL_TASKS["common_words_extraction"]
    tokens_to_generate = spec["tokens_to_generate"]
    template = spec["template"] + spec["answer_prefix"]
    freq_cw, freq_ucw, num_cw = args["freq_cw"], args["freq_ucw"], args["num_cw"]
    num_fewshot = 1  # official default

    if tokenizer_fn is not None:
        count_fn = tokenizer_fn
    else:
        # ~1 BPE token per numbered-list entry word ("123. word" -> ~2 tokens
        # for 2 whitespace words); documented approximation.
        count_fn = lambda text: len(text.split())

    wl = _load_wordlists()
    words = sorted(set(wl["nounlist.txt"] + wl["adjectivelist.txt"]
                       + wl["verblist.txt"]))
    random.Random(seed).shuffle(words)  # official module-level shuffle
    randle_words = _randle_words()

    def get_example(num_words, common_repeats, uncommon_repeats):
        """Official get_example."""
        if num_words <= len(words):
            word_list_full = random.sample(words, num_words)
        else:
            word_list_full = random.sample(randle_words, num_words)
        common, uncommon = word_list_full[:num_cw], word_list_full[num_cw:]
        word_list = common * int(common_repeats) + uncommon * int(uncommon_repeats)
        random.Random(seed).shuffle(word_list)
        context = " ".join(f"{i + 1}. {word}" for i, word in enumerate(word_list))
        return context, common

    def build_text(num_words):
        """Official generate_input_output (>=4096 branch: 1 few-shot of 40
        words at 10/3 repeats; main list at freq_cw/freq_ucw)."""
        few_shots = []
        for _ in range(num_fewshot):
            ctx_ex, ans_ex = get_example(40, 10, 3)
            few_shots.append(
                template.format(num_cw=num_cw, context=ctx_ex, query="")
                + " " + " ".join(f"{i + 1}. {w}" for i, w in enumerate(ans_ex)))
        few_shot_block = "\n".join(few_shots)
        context, answer = get_example(num_words, freq_cw, freq_ucw)
        main_text = template.format(num_cw=num_cw, context=context, query="")
        pre_main, post = template.split("{context}", 1)
        pre_main = pre_main.format(num_cw=num_cw, query="")
        post = post.format(num_cw=num_cw, query="")
        return few_shot_block, context, answer, pre_main, post

    def build_text_only(n):
        few_shot_block, context, _, pre_main, post = build_text(n)
        # count the FULL official text (few-shot block + pre + context + post),
        # like the official sizing which tokenizes the complete input
        return few_shot_block + "\n" + pre_main + context + post, None

    # Official sizing: tokens-per-word from a 4096-word sample, 2x slack,
    # then binary search over the number of words.
    sample_text, _ = build_text_only(4096)
    tokens_per_word = max(count_fn(sample_text), 1) / 4096
    estimated_max_words = int(context_length // tokens_per_word) * 2

    lo, hi = 10, max(estimated_max_words, 20)
    num_words = None
    while lo <= hi:
        mid = (lo + hi) // 2
        text, _ = build_text_only(mid)
        if count_fn(text) + tokens_to_generate <= context_length:
            num_words = mid
            lo = mid + 1
        else:
            hi = mid - 1
    num_words = num_words if num_words is not None else 10

    examples = []
    for idx in range(n_samples):
        used = num_words
        while True:
            few_shot_block, context, answer, pre_main, post = build_text(used)
            full_len = count_fn(few_shot_block + "\n" + pre_main + context + post) \
                + tokens_to_generate
            if full_len <= context_length or used <= 10:
                break
            used -= 10

        pre = few_shot_block + "\n" + pre_main
        examples.append(BenchExample(
            id=f"{task}-{context_length}-{idx}",
            context=context,
            question=pre + post,
            gold=[str(w) for w in answer],
            metric="ruler_containment",
            meta={
                "task": task,
                "benchmark": "ruler",
                "source_urls": [RULER_GITHUB, RULER_PAPER],
                "needle_depth_fraction": None,
                "num_words": used,
                "freq_cw": freq_cw,
                "freq_ucw": freq_ucw,
                "num_cw": num_cw,
                "prompt_before_context": pre,
                "prompt_after_context": post,
                "answer_prefix": spec["answer_prefix"],
                "tokens_to_generate": tokens_to_generate,
                "context_length_requested": context_length,
            },
        ))
    return examples


# ---------------------------------------------------------------------------
# Frequent words extraction (scripts/data/synthetic/freq_words_extraction.py)
# ---------------------------------------------------------------------------

def _zeta(alpha: float) -> float:
    try:
        from scipy.special import zeta as scipy_zeta
        return float(scipy_zeta(alpha))
    except Exception:  # pragma: no cover - scipy is installed; math fallback
        return sum(k ** (-alpha) for k in range(1, 1000000))


def build_fwe(task: str, args: dict, context_length: int, n_samples: int,
              seed: int, tokenizer_fn) -> list:
    random.seed(seed)
    np.random.seed(seed)

    spec = _OFFICIAL_TASKS["freq_words_extraction"]
    tokens_to_generate = spec["tokens_to_generate"]
    template = spec["template"] + spec["answer_prefix"]
    alpha = args["alpha"]
    coded_wordlen = 6  # official default

    if tokenizer_fn is not None:
        count_fn = tokenizer_fn
    else:
        # 6-letter coded words average ~2 BPE tokens incl. separator;
        # documented approximation.
        count_fn = lambda text: int(len(text.split()) * 2.0)

    max_seq_length = context_length - tokens_to_generate  # official subtraction
    vocab_size = max_seq_length // 50  # official default (vocab_size == -1)

    # Official vocab construction
    vocab = ["".join(random.choices(string.ascii_lowercase, k=coded_wordlen))
             for _ in range(vocab_size)]
    while len(set(vocab)) < vocab_size:
        vocab.append("".join(random.choices(string.ascii_lowercase, k=coded_wordlen)))
    vocab = sorted(set(vocab))
    random.Random(seed).shuffle(vocab)
    vocab[0] = "..."  # official: top-ranked word treated as noise

    def gen_text(num_words):
        """Official gen_text: Zipf(zeta-alpha) counts by rank."""
        k = np.arange(1, len(vocab) + 1)
        sampled_cnt = num_words * (k ** -alpha) / _zeta(alpha)
        sampled_words = [[w] * zi for w, zi in zip(vocab, sampled_cnt.astype(int))]
        sampled_words = [x for wlst in sampled_words for x in wlst]
        random.Random(seed).shuffle(sampled_words)  # official fixed-seed shuffle
        text = template.format(context=" ".join(sampled_words), query="")
        return text, vocab[1:4]

    incremental = max_seq_length // 32  # official fwe incremental

    # Official: determine num_example_words once by growing until the budget
    # fills, then stepping back one increment.
    num_words = max_seq_length // coded_wordlen
    text, _ = gen_text(num_words)
    while count_fn(text) < max_seq_length:
        num_words += incremental
        text, _ = gen_text(num_words)
    num_words -= incremental
    num_example_words = num_words

    pre, post = template.split("{context}", 1)
    pre, post = pre.format(query=""), post.format(query="")

    examples = []
    for idx in range(n_samples):
        used = num_example_words
        input_text, answer = gen_text(used)
        while count_fn(input_text) > max_seq_length and used > incremental:
            used -= incremental
            input_text, answer = gen_text(used)

        context = input_text[len(pre):len(input_text) - len(post)]
        examples.append(BenchExample(
            id=f"{task}-{context_length}-{idx}",
            context=context,
            question=pre + post,
            gold=[str(w) for w in answer],
            metric="ruler_containment",
            meta={
                "task": task,
                "benchmark": "ruler",
                "source_urls": [RULER_GITHUB, RULER_PAPER],
                "needle_depth_fraction": None,
                "num_words": used,
                "vocab_size": vocab_size,
                "alpha": alpha,
                "coded_wordlen": coded_wordlen,
                "prompt_before_context": pre,
                "prompt_after_context": post,
                "answer_prefix": spec["answer_prefix"],
                "tokens_to_generate": tokens_to_generate,
                "context_length_requested": context_length,
            },
        ))
    return examples


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_ruler(task: str, context_length: int, n_samples: int = 10, seed: int = 42,
                tokenizer_fn=None) -> list:
    """Build RULER examples for one task at one context length.

    Args:
        task: one of RULER_TASKS (niah_single_1/2/3, niah_multikey_1/2, vt,
            cwe, fwe).
        context_length: total context budget in tokens, INCLUDING the official
            tokens_to_generate (niah 128, vt 30, cwe 120, fwe 50) -- mirrors
            official max_seq_length semantics. Hardware-feasible lengths for
            the 8 GB GPU: 4096, 8192, 16384.
        n_samples: number of samples (official NUM_SAMPLES=500; we default to
            10 like the RULER quickstart).
        seed: RNG seed. Gold values are recorded at generation time from this
            seeded RNG and are never regenerated differently later.
        tokenizer_fn: optional callable text -> int token count. When given,
            haystack sizing is exact (official binary search). When omitted, a
            documented per-family word-based approximation is used (see module
            docstring, deviation 3) and realized lengths can deviate ~20%.

    Returns:
        list[BenchExample] with metric="ruler_containment" and needle
        placement info in meta.
    """
    if task not in TASK_ARGS:
        raise ValueError(f"Unknown RULER task {task!r}; expected one of {RULER_TASKS}")
    args = TASK_ARGS[task]
    family = args["task"]
    if family == "niah":
        return build_niah(task, args, context_length, n_samples, seed,
                          _prose_token_count(tokenizer_fn))
    if family == "variable_tracking":
        return build_vt(task, args, context_length, n_samples, seed,
                        _prose_token_count(tokenizer_fn))
    if family == "common_words_extraction":
        return build_cwe(task, args, context_length, n_samples, seed, tokenizer_fn)
    if family == "freq_words_extraction":
        return build_fwe(task, args, context_length, n_samples, seed, tokenizer_fn)
    raise NotImplementedError(family)
