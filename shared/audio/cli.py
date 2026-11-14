"""Línea de comandos de la preparación de clips. No gasta nada: es todo local.

    python -m shared.audio.cli inventario    # qué grabar, con qué nombre
    python -m shared.audio.cli preparar      # degrada y escribe el manifiesto
    python -m shared.audio.cli verificar     # ¿siguen siendo los mismos clips?
    python -m shared.audio.cli demo          # prueba el flujo entero sin grabar
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from shared.audio.clips import EmptyClip, write_artifacts, prepare_clip
from shared.audio.manifest import (
    MANIFEST_NAME,
    ManifestEntry,
    Manifest,
    load_manifest,
    expected_inventory,
    verify,
)
from shared.audio.level import CALIBRATION_PATH, Calibration, load_calibration
from shared.audio.resampling import TELEPHONY_SAMPLE_RATE, engine_description
from shared.audio.wav import AudioNotSupported, write_wav
from shared.metrics.scenarios import load_script

# Dónde viven los clips por default. Los WAV y los .ulaw están gitignoreados
# (pesan y son la voz de una persona — LFPDPPP); el manifiesto no, y por eso
# vive junto a ellos y no en otra carpeta.
BASE_PATH = Path(__file__).resolve().parents[1] / "test_scenarios"
DEFAULT_RAW_PATH = BASE_PATH / "raw_clips"
RUTA_CLIPS_DEFAULT = BASE_PATH / "clips"


def _force_utf8() -> None:
    """En Windows, redirigir la salida a un archivo la tira con cp1252."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure_stream = getattr(stream, "reconfigure", None)
        if reconfigure_stream is not None:
            try:
                reconfigure_stream(encoding="utf-8", errors="replace")
            except (ValueError, OSError):  # pragma: no cover
                pass


# --------------------------------------------------------------------------
# inventario
# --------------------------------------------------------------------------


def inventory_command(args: argparse.Namespace) -> int:
    """Qué hay que grabar. Es la lista de trabajo, no un reporte."""
    expected_items = expected_inventory()
    raw_values = Path(args.origin)

    print(f"{len(expected_items)} clips que pide el guion\n")
    current_scenario = None
    missing_items = 0
    for clip in expected_items:
        if clip.scenario != current_scenario:
            current_scenario = clip.scenario
            print(f"  --- escenario {current_scenario} ---")
        already_present = (raw_values / clip.wav_name).exists()
        if not already_present:
            missing_items += 1
        mark = "[ok]" if already_present else "[  ]"
        label = f"{clip.id}.wav"
        print(f"  {mark} {label:<18} {clip.text}")

    print(f"\n  grabados {len(expected_items) - missing_items}/{len(expected_items)} en {raw_values}")
    if missing_items:
        print(
            "\n  Graba cada línea en su propio archivo, con ese nombre exacto.\n"
            "  Formato: WAV PCM de 16 bits, mono o estéreo, 16 kHz o más\n"
            "  (48 kHz es lo ideal). Di la línea completa y para; el silencio\n"
            "  de los extremos se recorta solo."
        )
    return 0


# --------------------------------------------------------------------------
# preparar
# --------------------------------------------------------------------------


def prepare_folder(
    origin: Path,
    destination: Path,
    *,
    calibration: Calibration | None = None,
    require_complete: bool = True,
) -> tuple[Manifest, list[str]]:
    """Degrada todos los clips que encuentre. Devuelve `(manifiesto, avisos)`."""
    calibration = calibration or load_calibration()
    script = load_script()
    manifest = Manifest(
        generated=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        script_version=script.version,
    )
    warning_list: list[str] = []

    for expected in expected_inventory(script):
        ruta = origin / expected.wav_name
        if not ruta.exists():
            if require_complete:
                warning_list.append(f"{expected.id}: falta {ruta.name}")
            continue
        try:
            clip = prepare_clip(ruta, calibration=calibration)
        except (AudioNotSupported, EmptyClip, ValueError) as error:
            warning_list.append(f"{expected.id}: {error}")
            continue

        write_artifacts(clip, destination / expected.id)
        own_ones = clip.warning_list()
        warning_list.extend(f"{expected.id}: {a}" for a in own_ones)

        manifest.entries.append(
            ManifestEntry(
                id=expected.id,
                scenario=expected.scenario,
                turn=expected.turn,
                variant=expected.variant,
                text=expected.text,
                source=ruta.name,
                source_sha256=hashlib.sha256(ruta.read_bytes()).hexdigest(),
                origin_sample_rate=clip.origin_sample_rate,
                sha256_pcm8k=clip.sha256_pcm8k,
                sha256_ulaw=clip.sha256_ulaw,
                duration_s=round(clip.duration_s, 3),
                samples=int(clip.pcm8k.size),
                pico_dbfs=round(clip.pico_dbfs, 1),
                rms_dbfs=round(clip.rms_dbfs, 1),
                resampling_engine=clip.resampling_engine,
                start_cut_ms=clip.start_cut_ms,
                end_cut_ms=clip.end_cut_ms,
                trim_threshold_dbfs=clip.trim_threshold_dbfs,
                end_of_speech_s=clip.end_of_speech_s,
                cola_seg=clip.cola_seg,
                speech_drop_db=clip.speech_drop_db,
                speech_threshold_dbfs=clip.speech_threshold_dbfs,
                target_dbfs=clip.target_dbfs,
                drop_db=clip.drop_db,
                speech_rms_dbfs=clip.speech_rms_dbfs,
                source_speech_rms_dbfs=clip.source_speech_rms_dbfs,
                gain_db=clip.gain_db,
                provisional_calibration=clip.provisional_calibration,
                calibration_source=clip.calibration_source,
                warning_list=tuple(own_ones),
            )
        )

    return manifest, warning_list


def _print_calibration(calibration: Calibration) -> None:
    print(
        f"  objetivo de nivel    {calibration.target_dbfs:.1f} dBFS RMS de voz "
        f"(caída {calibration.drop_db:.0f} dB bajo el pico)"
    )
    print(f"  medido sobre         {calibration.source}")


def _caution_if_provisional(calibration: Calibration) -> None:
    """Lo que hay que decir cada vez, para que no se cite un número que no es."""
    if not calibration.provisional:
        return
    print(
        f"\n  [!] {calibration.caution}.\n"
        "      El objetivo se midió sobre una grabación de otro sistema por la\n"
        "      misma línea de Twilio. Re-confírmalo contra las grabaciones de\n"
        "      2 canales de PoC1 y vuelve a preparar los clips."
    )


def prepare_command(args: argparse.Namespace) -> int:
    origin, destination = Path(args.origin), Path(args.destination)
    if not origin.is_dir():
        print(f"[falta] no existe la carpeta de clips crudos: {origin}")
        print("        Grábalos primero: python -m shared.audio.cli inventario")
        return 1

    calibration = load_calibration().with_target(args.target_dbfs)
    print(f"  motor de remuestreo  {engine_description()}")
    _print_calibration(calibration)
    manifest, warning_list = prepare_folder(
        origin, destination, calibration=calibration, require_complete=not args.partial
    )

    if not manifest.entries:
        print(f"[falta] no se preparó ningún clip desde {origin}")
        return 1

    ruta = manifest.write(destination / MANIFEST_NAME)
    total = len(expected_inventory())
    duration = sum(e.duration_s for e in manifest.entries)
    deviations = [abs(e.deviation_db) for e in manifest.entries]
    tails = [e.cola_seg * 1000.0 for e in manifest.entries]
    print(f"  clips preparados     {len(manifest.entries)}/{total}")
    print(f"  duración total       {duration:.1f} s")
    print(f"  desviación de nivel  máx {max(deviations):.2f} dB")
    # La cola es lo que corre el cero de la latencia en las 3 PoC.
    print(
        f"  cola tras la voz     mediana {sorted(tails)[len(tails) // 2]:.0f} ms, "
        f"máx {max(tails):.0f} ms"
    )
    print(f"  artefactos en        {destination}")
    print(f"  manifiesto en        {ruta}")

    if warning_list:
        print(f"\n  avisos ({len(warning_list)})")
        for aviso in warning_list:
            print(f"    {aviso}")

    print(
        "\n  El manifiesto SÍ se commitea; los .wav y .ulaw no (están "
        "gitignoreados).\n  Es la única evidencia de que las 3 PoC reprodujeron "
        "el mismo audio."
    )
    _caution_if_provisional(calibration)
    return 0


# --------------------------------------------------------------------------
# verificar
# --------------------------------------------------------------------------


def verify_command(args: argparse.Namespace) -> int:
    folder = Path(args.destination)
    ruta = folder / MANIFEST_NAME
    if not ruta.is_file():
        print(f"[falta] no hay manifiesto en {ruta}")
        return 1

    manifest = load_manifest(ruta)
    problems = verify(
        manifest, folder, require_complete=not args.partial
    )

    print(f"  manifiesto      {ruta}")
    print(f"  generado        {manifest.generated}")
    print(f"  guion           versión {manifest.script_version}")
    print(f"  clips           {len(manifest.entries)}/{len(expected_inventory())}")
    targets = {e.target_dbfs for e in manifest.entries}
    print(
        "  nivel           objetivo "
        f"{', '.join(f'{o:.1f}' for o in sorted(targets)) or '(ninguno)'} dBFS"
        f" · vigente {load_calibration().target_dbfs:.1f} dBFS"
    )
    engines = {e.resampling_engine for e in manifest.entries}
    print(f"  preparados con  {', '.join(sorted(engines)) or '(ninguno)'}")
    if engines and engine_description() not in engines:
        print(
            f"                  (este entorno usaría {engine_description()}; "
            "no importa mientras no se vuelvan a preparar)"
        )

    if not problems:
        print("\n  Todo cuadra: manifiesto, archivos en disco, guion y calibración.")
        if any(e.provisional_calibration for e in manifest.entries):
            _caution_if_provisional(load_calibration())
        return 0

    print(f"\n  {len(problems)} problema(s)")
    for problem in problems:
        print(f"    {problem}")
    return 1


# --------------------------------------------------------------------------
# demo
# --------------------------------------------------------------------------


def _synthetic_clip(seconds: float, seed: int, sample_rate: int = 48_000) -> np.ndarray:
    """Algo con forma de voz: dos formantes con envolvente, y silencio alrededor.

    No pretende sonar a voz — pretende ejercitar el recorte, el remuestreo y el
    códec con una señal que no sea ni un tono puro ni ruido blanco.
    """
    generator = np.random.default_rng(seed)
    n = int(seconds * sample_rate)
    t = np.arange(n) / sample_rate
    sig = 0.5 * np.sin(2 * np.pi * 180 * t) + 0.3 * np.sin(2 * np.pi * 850 * t)
    # Envolvente de sílabas: ~4 por segundo.
    sig *= 0.5 + 0.5 * np.sin(2 * np.pi * 4 * t) ** 2
    sig += generator.normal(0, 0.01, n)
    silence = np.zeros(int(0.4 * sample_rate))
    full = np.concatenate([silence, sig * 0.6, silence])
    return np.clip(np.rint(full * 32767), -32768, 32767).astype(np.int16)


def demo_command(args: argparse.Namespace) -> int:
    """El flujo entero con audio sintético: prueba el código sin grabar nada."""
    expected_items = expected_inventory()
    calibration = load_calibration().with_target(getattr(args, "target_dbfs", None))
    print(f"  guion            {len(expected_items)} clips esperados")
    print(f"  motor            {engine_description()}")
    print(f"  objetivo         {calibration.target_dbfs:.1f} dBFS RMS de voz")

    with tempfile.TemporaryDirectory(prefix="demo_clips_") as tmp:
        raw_values = Path(tmp) / "raw_values"
        prepared = Path(tmp) / "prepared"
        raw_values.mkdir(parents=True)

        for numero, clip in enumerate(expected_items):
            # La duración cuelga del largo de la línea: ~13 caracteres por
            # segundo, que es un ritmo de habla plausible en español.
            seconds = max(0.8, min(12.0, len(clip.text) / 13.0))
            write_wav(
                raw_values / clip.wav_name, _synthetic_clip(seconds, numero), 48_000
            )

        manifest, warning_list = prepare_folder(
            raw_values, prepared, calibration=calibration
        )
        manifest.write(prepared / MANIFEST_NAME)
        problems = verify(manifest, prepared, calibration=calibration)

        duration = sum(e.duration_s for e in manifest.entries)
        trimmed = sum(
            e.start_cut_ms + e.end_cut_ms for e in manifest.entries
        )
        deviations = [abs(e.deviation_db) for e in manifest.entries]
        print(f"  preparados       {len(manifest.entries)}")
        print(f"  duración total   {duration:.1f} s")
        print(f"  silencio quitado {trimmed / 1000:.1f} s")
        print(f"  desviación nivel máx {max(deviations or [0.0]):.2f} dB")
        print(f"  avisos           {len(warning_list)}")
        print(f"  problemas        {len(problems)}")

        for problem in problems:
            print(f"    {problem}")

        if problems:
            return 1

    print(
        "\n  El flujo completo corre: nivel, degradación, artefactos, manifiesto\n"
        "  y verificación. Lo único que falta son tus grabaciones."
    )
    _caution_if_provisional(calibration)
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    _force_utf8()
    arg_parser = argparse.ArgumentParser(
        prog="shared.audio.cli",
        description=(
            "Prepara los clips del arnés: los degrada a banda telefónica una "
            "sola vez y deja un manifiesto con el hash de cada uno."
        ),
    )
    subs = arg_parser.add_subparsers(dest="command", required=True)

    def with_target(parser: argparse.ArgumentParser) -> None:
        """El objetivo de nivel se puede pisar sin editar el JSON, para probar."""
        parser.add_argument(
            "--target-dbfs",
            type=float,
            default=None,
            dest="target_dbfs",
            help=(
                "RMS de voz al que se normaliza cada clip, en dBFS. Por default el "
                f"de {CALIBRATION_PATH.name} "
                f"({load_calibration().target_dbfs:.1f}); para cambiarlo de "
                "forma permanente, edita ese archivo y vuelve a preparar"
            ),
        )

    p_inv = subs.add_parser("inventario", help="qué clips pide el guion")
    p_inv.add_argument("--origin", default=str(DEFAULT_RAW_PATH))
    p_inv.set_defaults(function=inventory_command)

    p_prep = subs.add_parser("prepare", help="degrada y escribe el manifiesto")
    p_prep.add_argument("--origin", default=str(DEFAULT_RAW_PATH))
    p_prep.add_argument("--destination", default=str(RUTA_CLIPS_DEFAULT))
    p_prep.add_argument(
        "--partial", action="store_true", help="no reclamar los clips que falten"
    )
    with_target(p_prep)
    p_prep.set_defaults(function=prepare_command)

    p_ver = subs.add_parser("verify", help="re-hashea y compara contra el guion")
    p_ver.add_argument("--destination", default=str(RUTA_CLIPS_DEFAULT))
    p_ver.add_argument("--partial", action="store_true")
    p_ver.set_defaults(function=verify_command)

    p_demo = subs.add_parser("demo", help="el flujo entero con audio sintético")
    with_target(p_demo)
    p_demo.set_defaults(function=demo_command)

    args = arg_parser.parse_args(argv)
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
