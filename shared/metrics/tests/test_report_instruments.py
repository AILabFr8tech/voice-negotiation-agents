"""Tests for the report's figures for Instruments, Populations and Appendices."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from shared.metrics import report_instruments as ir


def _write(ruta: Path, rows: list[dict]) -> Path:
    with ruta.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return ruta


def _row(campana, indice, poc, scen, rep, estado, in_the_pool="0", **criteria):
    base = {c: "0" for c in ir.CRITERIA}
    base.update({k: str(v) for k, v in criteria.items()})
    return {
        "campana": campana, "indice": str(indice), "clave": f"{indice:04d}-poc{poc}",
        "poc": str(poc), "scenario": str(scen), "repetition": str(rep), "estado": estado,
        **base, "failed_poc3_reviews": "", "in_the_pool": in_the_pool,
    }


def test_populations_counts_matching_criteria_and_leaves_out_what_is_after_180(tmp_path):
    # `en_la_poza` is decided by `inventario_local` with the three-PoC rule.
    rows = [
        _row("campana200", 1, 1, 1, 1, "ok", "1"), _row("campana200", 2, 2, 1, 1, "ok", "1"),
        # PoC2 fails two criteria: PoC1 drops out by the three-PoC rule.
        _row("campana200", 3, 1, 1, 2, "ok"),
        _row("campana200", 4, 2, 1, 2, "roja", insufficient_turns=1, response_without_audio=1),
        # After call 180: outside the population.
        _row("campana200", 181, 1, 1, 3, "roja", insufficient_turns=1),
        _row("campana_poc3_pcmu", 1, 3, 1, 1, "ok", "1"),
        _row("campana_poc3_pcmu", 2, 3, 1, 2, "roja", stt_missed_some_clip=1),
    ]
    tex, r = ir.analyze_populations(_write(tmp_path / "p.csv", rows))
    assert r["1"] == {**r["1"], "runs": 2, "failed_calls": 0, "matched_pair": 1, "analyzed": 1}
    assert r["2"]["failed_calls"] == 1 and r["2"]["several"] == 1 and r["2"]["analyzed"] == 1
    assert r["2"]["per_criterion"]["insufficient_turns"] == 1
    assert r["3"]["runs"] == 2 and r["3"]["analyzed"] == 1
    assert "\\newcommand{\\RExcluidasA}{1}" in tex


def test_the_playable_lines_are_47_and_each_branch_names_its_line():
    data = yaml.safe_load(ir.SCRIPT.read_text(encoding="utf-8"))
    text_lines = ir.reproducible_ones(data)
    ids = [c for _, _, c, _ in text_lines]
    assert len(ids) == 47 and len(set(ids)) == 47
    assert "e5_t6_alt1" not in ids and "e5_t7_alt1" not in ids


def test_the_condition_is_written_in_words():
    assert "18{,}000 o más en algún momento" in ir.condition_in_words({"reached_amount": 18000})
    assert "rama predeterminada" in ir.condition_in_words({"by_default": True})
    assert "en su última respuesta" in ir.condition_in_words({"compensation_in_turn": True})


def test_the_script_tables_come_out_with_notes(tmp_path):
    tex, r = ir.script_tables(tmp_path)
    # One table per scenario, with three columns: turn, when it plays, and the line.
    for e in range(1, 6):
        rows = (tmp_path / f"script_e{e}.tex").read_text(encoding="utf-8").splitlines()[1:]
        assert rows and all(f.count(" & ") == 2 for f in rows)
    # The corrected line carries the audio's text, an asterisk and a note.
    e1 = (tmp_path / "script_e1.tex").read_text(encoding="utf-8")
    assert "la necesita en Laredo?\\textsuperscript{*}\\\\" in e1
    assert "e1\\_t3" in (tmp_path / "script_notes.tex").read_text(encoding="utf-8")
    assert r["text_lines"] == 47
    m = {linea.split("{\\", 1)[1].split("}", 1)[0]: linea.rsplit("{", 1)[1].rstrip("}") for linea in tex}
    assert int(m["RGuionCorregidasRecorte"]) + int(m["RGuionCorregidasDesvio"]) == int(m["RGuionCorregidas"])
    assert m["RGuionCorregidasDesvio"] == "1"
    # The branches table gives what the audio says, not the script.
    branches = (tmp_path / "branches.tex").read_text(encoding="utf-8")
    alt = next(f for f in branches.splitlines() if "e2\\_t5\\_alt1}:" in f)
    assert alt.endswith("dieciocho.\\textsuperscript{*}\\\\")


def test_the_english_translation_covers_each_audio_and_none_is_left_over():
    # It is for the English version of the report: the 47 script lines and the 12
    # from the interruption scenario, not one more.
    script = ir.reproducible_ones(yaml.safe_load(ir.SCRIPT.read_text(encoding="utf-8")))
    interruptions = yaml.safe_load(ir.INTERRUPTIONS.read_text(encoding="utf-8"))["clips"]
    expected_items = {c for _, _, c, _ in script} | {c["id"] for c in interruptions}
    ruta = ir.SCRIPT.with_name("translation_en.yaml")
    assert set(yaml.safe_load(ruta.read_text(encoding="utf-8"))["clips"]) == expected_items
    assert len(expected_items) == 47 + 12


def test_the_per_scenario_tables_follow_the_branch_order(tmp_path):
    ir.script_tables(tmp_path)
    intr = (tmp_path / "interruption_script.tex").read_text(encoding="utf-8").splitlines()[1:]
    assert len(intr) == 12 and intr[0].startswith("Interrupción & ")
    assert all(f.count(" & ") == 1 for f in intr)
    # A turn with branches yields one line per branch, in the order they are checked.
    e2 = (tmp_path / "script_e2.tex").read_text(encoding="utf-8").splitlines()[1:]
    t5 = e2[e2.index(next(f for f in e2 if f.startswith("T5 & "))):][:3]
    assert t5[0].startswith("T5 & si el agente ya dijo 18{,}000 o más & Órale")
    assert t5[2].startswith(" & en cualquier otro caso & Ya casi.")


def test_the_english_tables_carry_the_audio_and_its_translation(tmp_path):
    ir.script_tables(tmp_path)
    for e in range(1, 6):
        es = (tmp_path / f"script_e{e}.tex").read_text(encoding="utf-8").splitlines()[1:]
        en = (tmp_path / f"script_en_e{e}.tex").read_text(encoding="utf-8").splitlines()[1:]
        # The same rows as the Spanish table, with one more column.
        assert len(en) == len(es) and all(f.count(" & ") == 3 for f in en)
    e2 = (tmp_path / "script_en_e2.tex").read_text(encoding="utf-8").splitlines()[1:]
    t5 = e2[e2.index(next(f for f in e2 if f.startswith("T5 & "))):][:3]
    assert t5[0].startswith("T5 & if the agent has already said 18{,}000 or more & Órale")
    assert t5[0].endswith("& All right, now that works for me.\\\\")
    assert t5[2].startswith(" & in any other case & Ya casi.")
    # The corrected line keeps the audio's asterisk.
    e1 = (tmp_path / "script_en_e1.tex").read_text(encoding="utf-8")
    assert "la necesita en Laredo?\\textsuperscript{*} & Uh-huh." in e1
    intr = (tmp_path / "script_en_interruptions.tex").read_text(encoding="utf-8").splitlines()[1:]
    assert len(intr) == 12 and all(f.count(" & ") == 2 for f in intr)
    assert intr[0].startswith("Interruption & ") and intr[6].startswith("Backchannel & Ajá. & Uh-huh.")


def test_the_validation_summarizes_bias_and_max_error(tmp_path):
    rows = [
        {"condition": "retardo_300", "error_ms": e, "end_of_speech_error_ms": "5", "response_error_ms": "-10",
         "error_contra_frase_ms": e, "reported_latency_ms": "1", "clip_id": "e1_t1", "scenario": "1"}
        for e in ("-10", "-30")
    ] + [{"condition": "retardo_300", "error_ms": "", "end_of_speech_error_ms": "", "response_error_ms": "",
          "error_contra_frase_ms": "", "reported_latency_ms": "", "clip_id": "e5_t4", "scenario": "5"}]
    tex, r = ir.analyze_validation(_write(tmp_path / "v.csv", rows), tmp_path / "does_not_exist.txt")
    assert r["retardo_300"]["bias"] == pytest.approx(-20.0)
    assert r["retardo_300"]["max_abs"] == pytest.approx(30.0)
    assert r["retardo_300"]["lost"] == 1
    assert "\\newcommand{\\RValPerdidosClips}{e5\\_t4}" in tex
    assert any("PENDIENTE: voz del agente falso" in t for t in tex)


def test_without_recordings_the_distance_to_full_speech_is_left_pending(tmp_path):
    tex, r = ir.analyze_onset(tmp_path / "does_not_exist.csv")
    assert "\\newcommand{\\RHayArranque}{0}" in tex and r == {}


def _interruption_clips(root: Path, content: bytes, *, with_audio: bool) -> None:
    folder = root / "shared" / "test_scenarios" / "interruption_clips"
    folder.mkdir(parents=True)
    clip = {"id": "int_01", "sha256_ulaw": hashlib.sha256(content).hexdigest(),
            "duration_s": 1.0, "end_of_speech_s": 0.9}
    (folder / "manifest.json").write_text(json.dumps({"clips": [clip]}), encoding="utf-8")
    if with_audio:
        (folder / "int_01.ulaw").write_bytes(content)


def test_the_fingerprint_of_the_interruption_clips_is_redone_over_the_audio(tmp_path, monkeypatch):
    _interruption_clips(tmp_path, b"voz", with_audio=True)
    monkeypatch.setattr(ir, "RAIZ", tmp_path)
    tex, r = ir.interruption_clip_table(tmp_path / "generated")
    assert "\\newcommand{\\RClipsIntShaOk}{1}" in tex and r == {"n": 1, "sha_ok": 1}


def test_without_the_audios_the_fingerprint_of_the_interruption_clips_is_left_pending(tmp_path, monkeypatch):
    # The `.ulaw` files are not shipped in the repo: in a clone the analysis has to keep going.
    _interruption_clips(tmp_path, b"voz", with_audio=False)
    monkeypatch.setattr(ir, "RAIZ", tmp_path)
    tex, r = ir.interruption_clip_table(tmp_path / "generated")
    assert "\\newcommand{\\RClipsIntShaOk}{---}" in tex and r == {"n": 1, "sha_ok": None}
    assert (tmp_path / "generated" / "interruption_clips.tex").is_file()
