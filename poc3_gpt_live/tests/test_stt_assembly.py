"""Reassembly of PoC3's text, over literal fragments from the campaign.

## What it protects

PoC1's rule, `" ".join(f.text.strip() for f in fragments)`, is correct there
because Deepgram sends **finalized segments**. PoC3's fragments are not
segments: they are deltas that follow the cadence of speech, and the model sends
the separator **inside** the fragment when one is needed. With that rule **every
fragment boundary becomes a space**, which splits words and detaches
punctuation: `"die ciocho mil"` is not an extractable amount. Measurements in
`docs/hallazgo_ensamblado_stt_poc3.md`.

## The three layers

1. `TestTheTwentyLiteralExamples` — the 20 fragment triples taken from the
   censuses, exactly as they arrived, with their expected output.
2. `TestTheTwoSidesCannotDiverge` — `assemble_stt` and `assemble_agent` have
   to give the same result on the same fragments, and the source code cannot
   contain a `" ".join` over stripped texts again.
3. `TestTheFlagRule` — the rule does re-insert a separator when the fragment
   declares it does not carry one, which is PoC1's case and the one that
   prevents the inverse defect (`"teestán"`).
"""

from __future__ import annotations

import ast
import inspect

from poc3_gpt_live.app.media_streams import (
    WITH_SPACES,
    ObservedFromPoC3,
    join_fragments_of_alpha,
)
from shared.carrier_simulator.observed import TextFragment

# --------------------------------------------------------------------------
# The fragments, exactly as they arrived
# --------------------------------------------------------------------------

# `(raw texts, what has to come out)`. The texts are copied from
# `input_transcript.added` of the `campana_poc3` campaign, with their exact
# spaces. They are not edited: if any of them gets "fixed" by hand, the test
# stops testing what really happened.
SPLIT_WORD: list[tuple[list[str], str]] = [
    ([" veint", "idós de", " carga"], "veintidós de carga"),
    ([" cas", "etas", " se"], "casetas se"),
    ([". Dí", "game", " Va"], ". Dígame Va"),
    (["io", "cho", " mil"], "iocho mil"),
    (["int", "iocho", " mil"], "intiocho mil"),
    ([" die", "ciocho", " mil"], "dieciocho mil"),
    ([" veintid", "ós de", " carga"], "veintidós de carga"),
    ([" cator", "ce", " toneladas"], "catorce toneladas"),
    ([" veinticin", "co", " mil"], "veinticinco mil"),
    ([" déj", "elo", ". Contáct"], "déjelo. Contáct"),
]

GLUED_PUNCTUATION: list[tuple[list[str], str]] = [
    (["... No", ". Die", "ciocho"], "... No. Dieciocho"),
    ([" sale", ". Ahí", " déjelo"], "sale. Ahí déjelo"),
    ([" cosa", ", la", " carta"], "cosa, la carta"),
    ([" apuntar", ", traigo", " caja"], "apuntar, traigo caja"),
    ([". No", ". Diecio", "cho"], ". No. Dieciocho"),
    ([" sí", ". Qued", "amos en"], "sí. Quedamos en"),
    ([" órale", ", con", " eso"], "órale, con eso"),
]

SPLIT_SCRIPT: list[tuple[list[str], str]] = [
    (["inte", "-ocho", " mil"], "inte-ocho mil"),
    ([", diec", "- diecio", "cho"], ", diec- dieciocho"),
    ([". Veinte", "-o", "cho"], ". Veinte-ocho"),
]

THE_TWENTY = SPLIT_WORD + GLUED_PUNCTUATION + SPLIT_SCRIPT

# The complete turns the report came from, with the script line.
REAL_TURNS: list[tuple[list[str], str]] = [
    ([" Sí", ", soy Carlos", ", dí", "game"], "Sí, soy Carlos, dígame"),
    (
        [" Súbale tantito", ", yo and", "o pensando en", " die", "ciocho", " mil"],
        "Súbale tantito, yo ando pensando en dieciocho mil",
    ),
    (
        [" No", ", está bajo para", " Monterrey", "- L", "aredo"],
        "No, está bajo para Monterrey- Laredo",
    ),
    ([" M", "mm", "... Está un poquito bajo"], "Mmm... Está un poquito bajo"),
]


def _fragments(texts: list[str], *, has_spaces: bool = True) -> list[TextFragment]:
    """The fragments as `media_streams` builds them when reading the hook."""
    return [
        TextFragment(
            t=float(i),
            sender="carrier",
            text=text,
            segment=WITH_SPACES if has_spaces else "",
            final=False,
        )
        for i, text in enumerate(texts)
    ]


class TestTheTwentyLiteralExamples:
    """The campaign's 20 triples, and what the old rule would have done."""

    def test_each_example_is_joined_as_it_should(self) -> None:
        observed = ObservedFromPoC3()
        for texts, expected in THE_TWENTY:
            assert observed.assemble_stt(_fragments(texts)) == expected, texts

    def test_the_complete_turns_are_joined_as_they_should(self) -> None:
        observed = ObservedFromPoC3()
        for texts, expected in REAL_TURNS:
            assert observed.assemble_stt(_fragments(texts)) == expected, texts

    def test_the_old_rule_would_have_split_all_twenty(self) -> None:
        """None of the 20 survived `" ".join(strip)`.

        That is what makes the list worthwhile: if any example came out the same
        under both rules, it would not be testing anything.
        """
        for texts, expected in THE_TWENTY:
            old_one = " ".join(t.strip() for t in texts if t.strip())
            assert old_one != expected, texts

    def test_no_amount_survives_the_old_rule(self) -> None:
        """`"die ciocho mil"` is not an amount; `"dieciocho mil"` is.

        It is the mechanism behind the `error_de_montos` of 0.8474.
        """
        from shared.metrics.text import extract_amounts

        texts, expected = SPLIT_WORD[5]  # " die" + "ciocho" + " mil"
        old_one = " ".join(t.strip() for t in texts if t.strip())
        assert extract_amounts(expected) == [18000.0]
        assert extract_amounts(old_one) != [18000.0]


class TestTheTwoSidesCannotDiverge:
    """The defect was that two assemblers of the same format diverged."""

    def test_stt_and_agent_give_the_same(self) -> None:
        observed = ObservedFromPoC3()
        for texts, _ in THE_TWENTY + REAL_TURNS:
            fragments = _fragments(texts)
            assert observed.assemble_stt(fragments) == observed.assemble_agent(
                fragments
            ), texts

    def test_both_call_the_same_function(self) -> None:
        """Not "equivalent": the same body, delegating to the same place."""
        for method in (ObservedFromPoC3.assemble_stt, ObservedFromPoC3.assemble_agent):
            cuerpo = inspect.getsource(method)
            assert "join_fragments_of_alpha(fragments)" in cuerpo, method.__name__

    def test_the_space_join_does_not_come_back(self) -> None:
        """Regression detector: nobody may put `" ".join` over stripped texts back.

        It fails on purpose if someone writes PoC1's rule again in
        `media_streams.py`. If one day it is really needed, this test has to be
        deleted by hand and the reason explained in the commit — which is
        exactly the conversation that did not happen the first time.
        """
        import poc3_gpt_live.app.media_streams as modulo

        # The AST is inspected and not the text: this module's docstrings quote
        # the old rule on purpose, and a `grep` would mistake them for code.
        tree = ast.parse(inspect.getsource(modulo))
        culprits = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "join"
            and isinstance(node.func.value, ast.Constant)
            and node.func.value.value == " "
        ]
        assert not culprits, f"`\" \".join(...)` vuelve a estar en las líneas {culprits}"


class TestTheFlagRule:
    """The rule re-inserts a separator when the fragment declares it does NOT carry one."""

    def test_without_flag_the_separator_is_restored(self) -> None:
        """PoC1's case: one frame per word, without spaces.

        Plain concatenation would give `"teestán"`. The rule is the same; what
        changes is the flag.
        """
        fragments = _fragments(["te", "están", "oyendo"], has_spaces=False)
        assert join_fragments_of_alpha(fragments) == "te están oyendo"

    def test_with_flag_nothing_is_restored(self) -> None:
        fragments = _fragments([" die", "ciocho", " mil"], has_spaces=True)
        assert join_fragments_of_alpha(fragments) == "dieciocho mil"

    def test_without_flag_does_not_duplicate_an_existing_separator(self) -> None:
        """Even if the flag says it does not carry one, if there already is a space none is added."""
        fragments = _fragments([" Hola,", " buen día"], has_spaces=False)
        assert join_fragments_of_alpha(fragments) == "Hola, buen día"

    def test_without_fragments_empty_string(self) -> None:
        assert join_fragments_of_alpha([]) == ""

    def test_the_empties_do_not_add_spaces(self) -> None:
        fragments = _fragments([" die", "", "ciocho", "", " mil"])
        assert join_fragments_of_alpha(fragments) == "dieciocho mil"
