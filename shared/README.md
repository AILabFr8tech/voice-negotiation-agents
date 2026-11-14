# shared/ — código común a las 3 PoC

Una sola fuente de verdad para todo lo que debe ser **idéntico** entre las tres
arquitecturas de voz. Si algo vive aquí, ninguna PoC lo duplica: lo importa. Así
la comparativa mide diferencias de tecnología de voz, no diferencias
accidentales de implementación.

| Carpeta | Qué contiene | Estado |
|---|---|---|
| [negotiation_engine/](negotiation_engine/) | Lógica de negociación pura: validación de precios, concesiones decrecientes, compensaciones, contrato de cierre. Más los datos del viaje (`trip.py`) y el estado por llamada (`session.py`) | v0 implementado |
| [carrier_simulator/](carrier_simulator/) | Transportista simulado: actúa los 5 escenarios contra el agente con los clips ya grabados. La mitad del arnés que no depende del framework — reproductor por turno, ruido de confort, regla de los turnos medibles, bitácora, reporte y costo | en uso por los arneses de las 3 PoC |
| [lote/](lote/) | Campañas desatendidas del arnés: alterna las PoC, se reanuda y respeta un tope de gasto | en uso |
| [system_prompt/](system_prompt/) | `v0.md`, el prompt versionado que las 3 PoC referencian, y `prompt.py`, que lo interpola y verifica que no filtre el precio máximo | v0 implementado |
| [metrics/](metrics/) | Latencia (p50/p95/p99), WER/CER, turn-taking, confiabilidad; los exportadores de la campaña y los módulos que escriben las cifras del reporte | en uso |
| [llm_transport/](llm_transport/) | Timeouts y reintentos del LLM. Cuánto se espera a OpenAI y cuántas veces se insiste: si cada PoC elige lo suyo, la comparativa mide quién reintenta mejor | implementado, cableado en PoC1 y PoC2 |
| [test_scenarios/](test_scenarios/) | Los 5 escenarios fijos, el escenario de interrupciones y los manifiestos de sus clips | escritos |
| [audio/](audio/) | Preparación de los clips del arnés: calibración de nivel, degradación a banda telefónica (8 kHz + μ-law), recorte, ruido de confort y el manifiesto con el hash de cada clip | implementado |

El **resultado de la negociación no es una métrica** de la comparativa (decisión
del 5 de agosto de 2026, CLAUDE.md). El motor sigue corriendo porque es el
vehículo de la conversación —le da vocabulario de dominio al WER y ejercita el
tool calling—, pero no aparece en ningún entregable.

## Por qué el repo git vive en la raíz

Si un `pocN_*/` termina con su propio `.git`, `shared/` queda fuera del
control de versiones de ese PoC y deja de ser una sola fuente de verdad.
Vigilar esto al crear cada PoC nuevo — `uv init` puede inicializar un repo si
no se le pasa `--vcs none`.

## Correr las pruebas

Desde la **raíz** del repo (ahí está `pytest.ini`, que pone la raíz en el
`pythonpath` para que `shared` sea importable):

```bash
uv run --project poc1_pipecat pytest poc1_pipecat shared validation
uv run --project poc2_livekit pytest poc2_livekit
uv run --project poc3_gpt_live pytest poc3_gpt_live
```

Cada PoC necesita su propio entorno para sus pruebas: las de PoC1 importan
pipecat y las de PoC2, livekit. Las de `shared/` **no importan ninguno de los
dos** —es la comprobación de que lo que vive aquí es de verdad agnóstico del
framework—, pero las de `metrics/` que calculan las cifras del reporte usan
scipy, que solo trae el entorno de PoC1.

## Convención de nombres

Los identificadores de dominio (`proponer_precio`, `monto`,
`historial_ofertas`, `compensaciones`) están en **español** a propósito: son
parte del contrato con el system prompt y con los schemas de tool-calling que
el LLM ve, y traducirlos abriría un hueco de desincronización. Los helpers
puramente técnicos y los nombres de módulo/carpeta van en inglés, según
CLAUDE.md.
