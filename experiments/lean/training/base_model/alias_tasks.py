"""Deterministic, paired proposition-alias tasks, separate from the nine-task bank.

Only definitionally equal proposition aliases and the four existing atomic
proof constructors are used. Validation below is a small symbolic checker,
not a Lean-kernel invocation. No model, tokenizer or global RNG is needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import itertools
import json
import random
import re
from typing import Any, Mapping, Sequence

from .logic_tasks import TrulyRandomNameGenerator


FORMAT = "s5_proposition_alias_pair_v1"
SAMPLE_FORMAT = "s5_proposition_alias_sample_v1"
FAMILIES = ("alias_id", "alias_and", "alias_or_l", "alias_or_r")
PAIRS_PER_BLOCK = 32
PURE_DISTRACTORS = 2
_LEAF_STATUS = {
    "alias_id": (True,),
    "alias_and": (True, True),
    "alias_or_l": (True, False),
    "alias_or_r": (False, True),
}
# Reserve Lean syntax and tactic names even when the old random-name pool can
# produce them. All three name classes share the same collision registry.
_RESERVED = frozenset(
    "abbrev alias and apply as at attribute axiom axioms begin break by calc case "
    "cases catch class constant constants def deriving do else end example exists "
    "exact export false finally for forall from fun have if import in include "
    "inductive infix infixl infixr instance intro let local macro match mut mutual "
    "namespace noncomputable notation omit "
    "opaque open or partial prefix private protected return scoped section "
    "set_option show simp sorry structure syntax termination_by then theorem "
    "true try universe universes unsafe unless until variable variables where "
    "while with Type Prop Sort And Or True False rfl unfold change of is to "
    "if_pos if_neg".split()
)
_IDENTIFIER = re.compile(r"[a-z][a-z0-9]*\Z", flags=re.ASCII)
_SPLIT = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,63}\Z", flags=re.ASCII)


def semantic_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        .encode("utf-8")
    ).hexdigest()


def _integer(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def validate_split_name(split: str) -> None:
    if not isinstance(split, str) or _SPLIT.fullmatch(split) is None:
        raise ValueError("split must be a simple ASCII name, not a path")


def _check_family(family: str) -> None:
    if not isinstance(family, str) or family not in FAMILIES:
        raise ValueError(f"unknown alias family: {family!r}")


def _check_name(name: str) -> None:
    if (
        not isinstance(name, str)
        or _IDENTIFIER.fullmatch(name) is None
        or name in _RESERVED
    ):
        raise ValueError(f"invalid or reserved identifier: {name!r}")


def _derived_seed(*parts: Any) -> int:
    return int(semantic_sha256([FORMAT, *parts]), 16)


class _Names:
    def __init__(self, seed: int) -> None:
        self.generator = TrulyRandomNameGenerator(seed)

    def new(self, *, hypothesis: bool = False) -> str:
        for _ in range(1000):
            value = (
                self.generator.random_hyp_name()
                if hypothesis else self.generator.random_var_name()
            )
            if _IDENTIFIER.fullmatch(value) and value not in _RESERVED:
                return value
        raise RuntimeError("could not allocate a legal unused name")


@dataclass(frozen=True)
class AliasDefinition:
    name: str
    family: str
    operands: tuple[str, ...]

    @property
    def expression(self) -> str:
        if self.family == "alias_id":
            return self.operands[0]
        operator = " ∧ " if self.family == "alias_and" else " ∨ "
        return operator.join(self.operands)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "family": self.family, "operands": list(self.operands)}


def canonical_tactic(
    definition: AliasDefinition, premises: Sequence[tuple[str, str]]
) -> str:
    """Check provability against ALL raw premises and produce the atomic proof."""
    _check_family(definition.family)
    if len(definition.operands) != len(_LEAF_STATUS[definition.family]):
        raise ValueError("definition arity does not match its atomic family")
    if len(set(definition.operands)) != len(definition.operands):
        raise ValueError("binary operands must be distinct")
    by_prop: dict[str, str] = {}
    for hyp, prop in premises:
        if prop in by_prop:
            raise ValueError("one canonical hypothesis is required per supported proposition")
        by_prop[prop] = hyp
    actual = tuple(prop in by_prop for prop in definition.operands)
    if actual != _LEAF_STATUS[definition.family]:
        raise ValueError("missing support or ambiguous/wrong OR branch in global premises")
    if definition.family == "alias_id":
        return f"exact {by_prop[definition.operands[0]]}"
    if definition.family == "alias_and":
        return (
            f"exact And.intro {by_prop[definition.operands[0]]} "
            f"{by_prop[definition.operands[1]]}"
        )
    index = 0 if definition.family == "alias_or_l" else 1
    constructor = "Or.inl" if index == 0 else "Or.inr"
    return f"exact {constructor} {by_prop[definition.operands[index]]}"


@dataclass(frozen=True)
class AliasPair:
    seed: int
    split: str
    pair_index: int
    attempt: int
    propositions: tuple[str, ...]
    definitions: tuple[AliasDefinition, AliasDefinition]
    premises: tuple[tuple[str, str], ...]
    pure_distractor_propositions: tuple[str, str]
    shared: bool

    @property
    def pair_id(self) -> str:
        return semantic_sha256({
            "format": FORMAT, "seed": self.seed, "split": self.split,
            "pair_index": self.pair_index, "attempt": self.attempt,
        })

    @property
    def context_text(self) -> str:
        lines = ["state_0:", " ".join(self.propositions) + " : Prop"]
        lines.extend(f"{d.name} : Prop := {d.expression}" for d in self.definitions)
        lines.extend(f"{hyp} : {prop}" for hyp, prop in self.premises)
        return "\n".join(lines)

    @property
    def context_sha256(self) -> str:
        return hashlib.sha256(self.context_text.encode("utf-8")).hexdigest()

    def to_rows(self) -> tuple[dict[str, Any], dict[str, Any]]:
        validate_pair(self)
        rows = []
        for view_id, definition in enumerate(self.definitions):
            tactic = canonical_tactic(definition, self.premises)
            input_text = f"{self.context_text}\n⊢ {definition.name}"
            output_text = (
                f"state_0_tactic_0:\n{tactic}\nstate_1:\nno goals\nproof complete"
            )
            rows.append({
                "format": SAMPLE_FORMAT,
                "sample_id": semantic_sha256([self.pair_id, view_id]),
                "pair_id": self.pair_id,
                "seed": self.seed,
                "split": self.split,
                "pair_index": self.pair_index,
                "attempt": self.attempt,
                "view_id": view_id,
                "task_type": definition.family,
                "distractor_task_type": self.definitions[1 - view_id].family,
                "shared_proposition_count": int(self.shared),
                "goal_alias": definition.name,
                "input_text": input_text,
                "output_text": output_text,
                "expected_tactic": tactic,
                "context_sha256": self.context_sha256,
                "input_sha256": hashlib.sha256(input_text.encode("utf-8")).hexdigest(),
                "context": {
                    "propositions": list(self.propositions),
                    "definitions": [d.to_dict() for d in self.definitions],
                    "premises": [
                        {"name": hyp, "proposition": prop} for hyp, prop in self.premises
                    ],
                    "pure_distractor_propositions": list(self.pure_distractor_propositions),
                },
            })
        return rows[0], rows[1]


def validate_pair(pair: AliasPair) -> None:
    """Independently check the closed v1 grammar, proof support and pair contract."""
    for key in ("seed", "pair_index", "attempt"):
        _integer(getattr(pair, key), key)
    validate_split_name(pair.split)
    if type(pair.shared) is not bool:
        raise ValueError("shared must be a bool")
    if len(pair.definitions) != 2:
        raise ValueError("exactly two alias definitions are required")
    if len(pair.pure_distractor_propositions) != PURE_DISTRACTORS:
        raise ValueError("exactly two pure distractors are required")
    prop_set = set(pair.propositions)
    noise_set = set(pair.pure_distractor_propositions)
    alias_names = [d.name for d in pair.definitions]
    hyp_names = [hyp for hyp, _ in pair.premises]
    all_names = [*pair.propositions, *alias_names, *hyp_names]
    for name in all_names:
        _check_name(name)
    if len(set(all_names)) != len(all_names):
        raise ValueError("declaration, alias and hypothesis names must be unique")
    if len(noise_set) != PURE_DISTRACTORS:
        raise ValueError("pure distractor propositions must be distinct")
    referenced: set[str] = set()
    required: set[str] = set()
    for definition in pair.definitions:
        _check_family(definition.family)
        if len(definition.operands) != len(_LEAF_STATUS[definition.family]):
            raise ValueError("invalid alias arity")
        if len(set(definition.operands)) != len(definition.operands):
            raise ValueError("binary operands must be different original propositions")
        if not set(definition.operands) <= prop_set:
            raise ValueError("alias operands must reference declared raw propositions, not aliases")
        referenced.update(definition.operands)
        required.update(
            prop for prop, present in zip(definition.operands, _LEAF_STATUS[definition.family])
            if present
        )
    if noise_set & referenced or prop_set != referenced | noise_set:
        raise ValueError("pure distractors must be fresh and declarations must be exact")
    supported = [prop for _, prop in pair.premises]
    if len(supported) != len(set(supported)):
        raise ValueError("duplicate proofs of the same proposition are not allowed")
    if set(supported) != required | noise_set:
        raise ValueError("premises must be precisely both alias supports plus two distractors")
    intersection = set(pair.definitions[0].operands) & set(pair.definitions[1].operands)
    if len(intersection) != int(pair.shared):
        raise ValueError("alias overlap must be exactly zero or one as declared")
    if pair.shared and all(d.family == "alias_id" for d in pair.definitions):
        raise ValueError("two identity aliases may not share")
    tactics = [canonical_tactic(d, pair.premises) for d in pair.definitions]
    if tactics[0] == tactics[1]:
        raise ValueError("goal-switch pair must have different canonical answers")


@lru_cache(maxsize=128)
def _schedule_block(seed: int, split: str, block_index: int) -> tuple[tuple[str, str, bool], ...]:
    slots = [
        (left, right, shared and (left, right) != ("alias_id", "alias_id"))
        for left, right in itertools.product(FAMILIES, repeat=2)
        for shared in (False, True)
    ]
    random.Random(_derived_seed("schedule", seed, split, block_index)).shuffle(slots)
    return tuple(slots)


def scheduled_spec(*, seed: int, split: str, pair_index: int) -> tuple[str, str, bool]:
    """32 contexts -> 64 rows with exact family/pairing/conditional-overlap counts."""
    _integer(seed, "seed")
    _integer(pair_index, "pair_index")
    validate_split_name(split)
    block, slot = divmod(pair_index, PAIRS_PER_BLOCK)
    return _schedule_block(seed, split, block)[slot]


def generate_pair(
    *,
    seed: int = 42,
    split: str = "train",
    pair_index: int = 0,
    attempt: int = 0,
    families: tuple[str, str] | None = None,
    shared: bool | None = None,
) -> AliasPair:
    """Generate a scheduled pair, or an explicit family case for tests/previews.

    ``attempt`` changes naming and instances, never the scheduled family cell.
    Explicit cases require both ``families`` and ``shared``; corpus exporters
    should leave these unset to retain the frozen sampling distribution.
    """
    for key, value in (("seed", seed), ("pair_index", pair_index), ("attempt", attempt)):
        _integer(value, key)
    validate_split_name(split)
    if (families is None) != (shared is None):
        raise ValueError("families and shared must be provided together")
    if families is None:
        left, right, shared = scheduled_spec(seed=seed, split=split, pair_index=pair_index)
        families = (left, right)
    if len(families) != 2 or type(shared) is not bool:
        raise ValueError("expected two families and a bool shared flag")
    for family in families:
        _check_family(family)
    if shared and families == ("alias_id", "alias_id"):
        raise ValueError("id-id sharing would make the two goals identical")

    instance_key = (seed, split, pair_index, attempt, list(families), shared)
    logic_rng = random.Random(_derived_seed("logic", *instance_key))
    names = _Names(_derived_seed("names", *instance_key))
    order_rng = random.Random(_derived_seed("order", *instance_key))
    status0, status1 = (_LEAF_STATUS[family] for family in families)
    leaves0 = list(range(len(status0)))
    leaves1 = list(range(len(status0), len(status0) + len(status1)))
    if shared:
        choices = [
            (i, j)
            for i, present0 in enumerate(status0)
            for j, present1 in enumerate(status1)
            if present0 == present1
            and not (
                present0 and families[0] == families[1]
                and families[0] in ("alias_or_l", "alias_or_r")
            )
        ]
        if not choices:
            raise ValueError("family pairing has no valid shared proposition")
        i, j = logic_rng.choice(choices)
        leaves1[j] = leaves0[i]

    leaf_ids = sorted(set(leaves0 + leaves1))
    named_leaf = {leaf: names.new() for leaf in leaf_ids}
    definitions = [
        AliasDefinition(names.new(), family, tuple(named_leaf[x] for x in leaves))
        for family, leaves in zip(families, (leaves0, leaves1))
    ]
    noise = (names.new(), names.new())
    props = [*(named_leaf[leaf] for leaf in leaf_ids), *noise]
    needed = {
        named_leaf[leaf]
        for leaves, statuses in ((leaves0, status0), (leaves1, status1))
        for leaf, present in zip(leaves, statuses)
        if present
    }
    # Allocation follows stable proposition order; the set is membership-only.
    premises = [(names.new(hypothesis=True), prop) for prop in props if prop in needed or prop in noise]
    order_rng.shuffle(props)
    order_rng.shuffle(definitions)
    order_rng.shuffle(premises)
    pair = AliasPair(
        seed=seed, split=split, pair_index=pair_index, attempt=attempt,
        propositions=tuple(props), definitions=tuple(definitions),
        premises=tuple(premises), pure_distractor_propositions=noise, shared=shared,
    )
    validate_pair(pair)
    return pair


def validate_rows(rows: Sequence[Mapping[str, Any]]) -> AliasPair:
    """Recheck serialized structure, proofs, rendered text and all row metadata.

    Rejects extra fields as well as alterations: the condition-facing strings
    must be exactly the renderings of the audited symbolic context.
    """
    try:
        if len(rows) != 2 or any(not isinstance(row, Mapping) for row in rows):
            raise ValueError("exactly two row objects are required")
        if any(type(row["view_id"]) is not int for row in rows):
            raise ValueError("view_id must be an integer")
        ordered = sorted(rows, key=lambda row: row["view_id"])
        if [row["view_id"] for row in ordered] != [0, 1]:
            raise ValueError("pair must contain one of each view")
        first = ordered[0]
        ctx = first["context"]
        if set(ctx) != {"propositions", "definitions", "premises", "pure_distractor_propositions"}:
            raise ValueError("unexpected context fields")
        defs = []
        for definition in ctx["definitions"]:
            if set(definition) != {"name", "family", "operands"}:
                raise ValueError("unexpected definition fields")
            if not isinstance(definition["operands"], list):
                raise ValueError("operands must be a JSON list")
            defs.append(AliasDefinition(
                definition["name"], definition["family"], tuple(definition["operands"])
            ))
        premises = []
        for premise in ctx["premises"]:
            if set(premise) != {"name", "proposition"}:
                raise ValueError("unexpected premise fields")
            premises.append((premise["name"], premise["proposition"]))
        count = first["shared_proposition_count"]
        if type(count) is not int or count not in (0, 1):
            raise ValueError("invalid shared count")
        pair = AliasPair(
            seed=first["seed"], split=first["split"], pair_index=first["pair_index"],
            attempt=first["attempt"], propositions=tuple(ctx["propositions"]),
            definitions=tuple(defs), premises=tuple(premises),
            pure_distractor_propositions=tuple(ctx["pure_distractor_propositions"]),
            shared=bool(count),
        )
        validate_pair(pair)
        regenerated = pair.to_rows()
        for row, expected in zip(ordered, regenerated):
            # JSON serialization also distinguishes bool/int and list/tuple
            # mismatches that Python dict equality alone can silently accept.
            if semantic_sha256(row) != semantic_sha256(expected):
                raise ValueError("row does not match its canonical context, proof or identity")
        return pair
    except (KeyError, TypeError, IndexError, AttributeError) as exc:
        raise ValueError("malformed alias pair rows") from exc
