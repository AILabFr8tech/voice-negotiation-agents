"""Parte un WAV de auditoría de dos canales en dos archivos mono, para oírlos.

Existe porque los dos canales suenan a la vez y encimados es casi imposible
distinguir quién se calló primero, que es justo lo que uno va a buscar. No usa
ffmpeg ni sox a propósito: `wave` y `numpy` ya están en el entorno de PoC3, y
pedir una instalación para oír un archivo es una barrera de más.

    python -m poc3_gpt_live.diagnostico.split_audit <ruta del wav>

Deja `<nombre>_canal0_transportista.wav` y `<nombre>_canal1_agente.wav` al lado
del original, que se abren con cualquier reproductor de Windows.
"""

from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path

import numpy as np

# El orden es el mismo en las dos grabadoras —la del puente y la del arnés— y
# está declarado en el JSON hermano de cada WAV.
NAMES = ("channel0_carrier", "channel1_agent")

# Llave que solo publica la grabadora del PUENTE. Es la forma de distinguir las
# dos sin depender del nombre de la carpeta, que quien mueve un archivo cambia.
BRIDGE_KEY = "playhead"

BRIDGE_WARNING = """
  ⚠ ESTE ES EL WAV DEL PUENTE Y SUS DOS CANALES NO ESTÁN ALINEADOS.
    El canal del agente queda ~1.8 s ANTES de donde ocurrió, así que cada
    hueco entre el transportista y su respuesta se oye ~1.8 s más corto de lo
    que fue. Lo produce el cabezal monótono y no se puede corregir aquí.
    NO uses este archivo para juzgar de oído el tiempo de respuesta.
    Para eso: diagnostics/harness/auditoria_<sala>.wav, cuyo canal 0 sale del
    deadline del reproductor. Detalle en docs/hallazgos_20260922_poc3.md
"""


def is_from_bridge(ruta: Path) -> bool:
    """¿El WAV lo escribió la grabadora del puente? Se pregunta al JSON hermano.

    Sin JSON no se adivina: se devuelve `False` y el archivo se parte sin aviso,
    que es lo mismo que hacía antes de existir esta función.
    """
    side = ruta.with_suffix(".json")
    if not side.is_file():
        return False
    try:
        return BRIDGE_KEY in json.loads(side.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False


def split(ruta: Path) -> list[Path]:
    with wave.open(str(ruta), "rb") as entry:
        channel_list = entry.getnchannels()
        if channel_list != 2:
            raise SystemExit(f"{ruta.name} tiene {channel_list} canal(es), no 2")
        sample_rate, width = entry.getframerate(), entry.getsampwidth()
        data = np.frombuffer(
            entry.readframes(entry.getnframes()), dtype="<i2"
        ).reshape(-1, 2)

    outputs = []
    for indice, name in enumerate(NAMES):
        destination = ruta.with_name(f"{ruta.stem}_{name}.wav")
        with wave.open(str(destination), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(width)
            output.setframerate(sample_rate)
            output.writeframes(np.ascontiguousarray(data[:, indice]).tobytes())
        outputs.append(destination)
        print(f"  {destination}")
    return outputs


def main(argv: list[str] | None = None) -> int:
    arg_parser = argparse.ArgumentParser(
        prog="poc3_gpt_live.diagnostics.split_audit",
        description="Separa un WAV de auditoría de 2 canales en dos mono.",
    )
    arg_parser.add_argument("wav", type=Path, help="el WAV de dos canales")
    args = arg_parser.parse_args(argv)

    if not args.wav.is_file():
        raise SystemExit(f"no existe: {args.wav}")
    print(f"partiendo {args.wav.name}:")
    split(args.wav)
    if is_from_bridge(args.wav):
        # Al final y no al principio: es lo último que queda en pantalla antes
        # de que alguien abra los dos mono en el reproductor.
        print(BRIDGE_WARNING)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
