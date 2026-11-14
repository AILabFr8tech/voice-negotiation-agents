"""La política de transporte del LLM tiene que seguir siendo defendible.

No son pruebas de aritmética: son las condiciones que hacen que estos números
se puedan justificar en el reporte. Si alguien las rompe, es porque cambió una
decisión — y entonces hay que cambiar también el docstring del módulo.
"""

from __future__ import annotations

from shared import llm_transport as lt


def test_timeout_leaves_margin_over_worst_measured_ttft() -> None:
    """Bajar el tope por debajo del peor TTFT real fabricaría los errores.

    Esta es LA condición del módulo: el timeout existe para cortar peticiones
    atoradas, no peticiones lentas. Con margen menor a 1.5x, un turno sano en un
    mal momento se convierte en un reintento — y el reintento cuesta el doble.
    """
    margin = lt.FIRST_BYTE_TIMEOUT_S * 1000.0 / lt.TTFT_MAX_MEASURED_MS
    assert margin >= 1.5, (
        f"{lt.FIRST_BYTE_TIMEOUT_S}s deja solo {margin:.2f}x sobre el peor "
        f"TTFT medido ({lt.TTFT_MAX_MEASURED_MS} ms): demasiado apretado"
    )


def test_there_is_a_retry() -> None:
    """Un intento solo significa perder el turno completo ante cualquier atorón."""
    assert lt.ATTEMPTS >= 2
    assert lt.retries_after_first() == lt.ATTEMPTS - 1


def test_the_worst_case_fits_in_one_conversation() -> None:
    """Todos los intentos juntos no pueden durar más que la paciencia de nadie.

    Cota superior: cada intento puede gastar su tope de conexión más el de
    primer byte, más la espera entre intentos. Si eso pasa de 30 s, la política
    es peor que el problema que arregla — el fallo medido fueron 40 s.
    """
    worst = lt.ATTEMPTS * (
        lt.CONNECTION_TIMEOUT_S + lt.FIRST_BYTE_TIMEOUT_S
    ) + lt.WAIT_BETWEEN_ATTEMPTS_S * lt.retries_after_first()
    assert worst <= 45.0, f"el peor caso son {worst:.0f}s"


def test_without_dead_wait_between_attempts() -> None:
    """El fallo es un atorón sin respuesta, no una cuota: esperar no ayuda."""
    assert lt.WAIT_BETWEEN_ATTEMPTS_S == 0.0


def test_summary_mentions_the_numbers() -> None:
    """El humo de cada PoC imprime esta línea; tiene que decir algo."""
    text = lt.summary()
    assert str(int(lt.FIRST_BYTE_TIMEOUT_S)) in text
    assert str(lt.ATTEMPTS) in text
