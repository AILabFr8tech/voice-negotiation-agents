"""Política de transporte del LLM: **idéntica en las 3 PoC, por contrato**.

Aquí no vive nada de negociación ni de voz. Vive una sola cosa: cuánto se
espera a que OpenAI conteste y cuántas veces se reintenta cuando no contesta.
Está en `shared/` por la misma razón que el system prompt y el motor — si cada
PoC elige sus propios timeouts, la comparativa deja de medir el framework de voz
y empieza a medir quién reintenta mejor.

## Por qué existe este archivo

Una petición al LLM se puede atorar: la API la acepta y no manda un solo byte.
Con los defaults de LiveKit (3 reintentos con 2 s de espera entre intentos) el
transportista se puede quedar medio minuto en silencio, hasta que un turno
nuevo cancela la petición atorada (el caso medido está en `CLAUDE.md`).

Eso no es una propiedad de LiveKit ni de pipecat: es el proveedor del LLM
atorándose, y es exactamente el tipo de ruido que no debe entrar a una
comparativa de tecnologías de voz. No se puede evitar que pase — sí se puede
hacer que cueste segundos en vez de medio minuto, y que cueste **lo mismo** en
las 3 PoC.

## Los defaults que se corrigen

Ninguno de los dos frameworks trae una política razonable de fábrica, y son
distintas entre sí, que es lo peor de los dos mundos:

- **PoC1 (pipecat)**: `retry_on_timeout=False`, así que no hay reintento y el
  único tope es el del SDK de OpenAI — `httpx.Timeout(timeout=600, connect=5)`.
  Una petición atorada podía colgar la llamada **hasta 10 minutos**.
- **PoC2 (LiveKit)**: 4 intentos con `retry_interval=2.0`, o sea ~4 s de espera
  muerta repartidos entre intentos, sobre un tope de lectura de 5 s por intento.

## De dónde salen los números

`FIRST_BYTE_TIMEOUT_S = 5 s` no es una cifra redonda elegida al gusto: el
TTFT más alto **medido** en las dos llamadas reales de PoC2 del 11 de agosto de
2026 fue de 2,410 ms (14 turnos entre las dos). Cinco segundos es poco más del
doble de ese peor caso. Bajarlo más empezaría a matar peticiones sanas y
**fabricaría** los errores que este módulo existe para reducir; subirlo alarga
cada intento fallido sin ganar nada.

`WAIT_BETWEEN_ATTEMPTS_S = 0` porque la espera exponencial es para no tumbar
un servicio que devuelve 429 o 5xx. Aquí el fallo es un atorón sin respuesta, y
cada intento nuevo estrena conexión: esperar solo agrega silencio en una llamada
donde alguien está escuchando. Si el problema fuera de cuota, el error vendría
como `APIStatusError` y ese sí llega marcado como no reintentable.

Los mecanismos de reintento **no son iguales** entre frameworks y no se pueden
igualar: pipecat hace un intento acotado y luego uno paciente; LiveKit hace N
intentos acotados. Lo que se iguala es lo que sí se controla — el tope del
primer intento, que haya reintento en vez de turno perdido, y que no haya tiempo
muerto entre intentos. La diferencia que queda se declara en
`docs/METRICS_CONTRACT.md`.
"""

from __future__ import annotations

from typing import Final

# Sin un solo byte del API en este tiempo, la petición se da por atorada.
# Referencia dura: el TTFT máximo medido es 2,410 ms (PoC2, 11 de agosto de
# 2026). Ver el docstring antes de tocarlo.
FIRST_BYTE_TIMEOUT_S: Final[float] = 5.0

# Handshake TCP + TLS. El default del SDK de OpenAI ya es 5 s; LiveKit lo sube a
# 15 s, que es medio minuto de silencio en tres intentos si la red está caída.
CONNECTION_TIMEOUT_S: Final[float] = 5.0

# Intentos totales, el primero incluido. Cuatro intentos acotados a 5 s cuestan
# menos que el default de LiveKit y dan cuatro conexiones distintas.
ATTEMPTS: Final[int] = 4

# Cero: ver el docstring. No es un descuido.
WAIT_BETWEEN_ATTEMPTS_S: Final[float] = 0.0

# El peor TTFT medido, en milisegundos. Se guarda como dato, no como perilla:
# es lo que justifica `TIMEOUT_PRIMER_BYTE_SEG` y lo que hay que volver a mirar
# si alguien lo quiere bajar.
TTFT_MAX_MEASURED_MS: Final[float] = 2410.0


def retries_after_first() -> int:
    """Los reintentos que espera LiveKit (`max_retry`), que no cuenta el primero."""
    return max(ATTEMPTS - 1, 0)


def summary() -> str:
    """Una línea para los chequeos de humo de cada PoC."""
    return (
        f"LLM: {FIRST_BYTE_TIMEOUT_S:.0f}s al primer byte, "
        f"{CONNECTION_TIMEOUT_S:.0f}s de conexión, {ATTEMPTS} intentos, "
        f"{WAIT_BETWEEN_ATTEMPTS_S:.0f}s entre intentos"
    )
