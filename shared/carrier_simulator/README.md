# carrier_simulator/ — el transportista simulado, sin su transporte

Actúa los 5 escenarios de `shared/test_scenarios/` contra el agente de cada PoC:
reproduce los clips ya preparados (`shared/audio/`) turno por turno sobre un
piso de ruido de confort continuo, observa la voz y el texto que devuelve el
agente, decide si la corrida sirve, y deja escrito qué se reprodujo, qué pasó y
cuánto costó.

Nació como `poc2_livekit/app/arnes.py`. Lo que quedó aquí es la mitad que no
depende de LiveKit, extraída el 16 de septiembre de 2026 para que PoC1 no la
vuelva a escribir.

## Por qué vive en `shared/` y no en cada PoC

Porque casi todo lo que hay aquí entra en cómo se leen las métricas, y tres
copias divergirían:

- **El cero del turno.** El fin del clip se fecha con el deadline de la trama,
  no con el regreso de la llamada al transporte. Ese instante es el origen de
  toda la latencia reportada; si cada arnés lo calculara a su manera, los p50 de
  las 3 PoC dejarían de ser comparables.
- **La regla de los turnos medibles** (`evaluar_corrida`, ≥ 5). Decide si una
  llamada se repite. Con mínimos distintos, cada PoC poblaría su percentil con
  llamadas de largo distinto.
- **La bitácora.** Es lo que corrige la referencia del WER cuando la corrida se
  ramificó. Tres formatos serían tres procedimientos manuales.
- **El ruido de confort.** Su nivel sale de `calibracion_nivel.json`, y mandar
  ceros en su lugar le rompe el endpointing al agente — el mismo error en las
  tres.

## Lo que pone cada PoC

| Hueco | Cómo se llena | PoC2 |
|---|---|---|
| Publicar la trama | heredar `Reproductor` y escribir `publicar` | `AudioSource.capture_frame` |
| Leer la transcripción | llamar `Observado.anotar_texto` | streams de `lk.transcription` |
| Reensamblar el STT | sobreescribir `Observado.ensamblar_stt` | `texto_de_reemplazos` (no-delta) |
| Publicarle el cero al worker | pasar un `emisor` a `reproducir_guion` | `app/marcas.py`, canal de datos |
| Nombres y tarifas del reporte | `ContextoDelReporte`, `ContextoDelCosto` | sala, Deepgram, `livekit_extra` |

El default de `ensamblar_stt` es delta. LiveKit manda el STT del usuario como
**reemplazos** —el último fragmento trae todo— y por eso PoC2 lo sobreescribe;
confundir los dos rompe el WER en las dos direcciones posibles, y ya pasó una
vez en cada sentido (PoC3, WER falso de 84.5%; PoC2, texto triplicado).

Cada transporte además es dueño del **dominio de su trama**. PoC2 publica PCM16
a la `AudioSource`; PoC1 y PoC3 mandan por Media Streams los bytes μ-law del
manifiesto sin convertirlos, y para eso sobreescriben `tramas_del_clip`,
`trama_de_relleno` y `como_pcm16`. `PasoDelGuion` trae las dos representaciones:
`muestras` (PCM16) y `ulaw` (los bytes tal como están en disco).

## El estímulo idéntico, y por qué comparar bytes no alcanza para probarlo

Que las 3 PoC reproduzcan **el mismo audio** es la condición de que sus WER se
puedan comparar. La tentación es verificarlo comparando los bytes que salen al
cable contra el `.ulaw` del manifiesto. Esa comparación se hace y pasa, pero
**prueba mucho menos de lo que parece, y conviene saberlo antes de apoyarse en
ella.**

Medido el 16 de septiembre de 2026 sobre los 256 códigos μ-law: **255 sobreviven
intactos** al viaje μ-law → PCM16 → μ-law. El único que no es `0x7F`, y los
clips de campaña no lo contienen — `e1_t1.ulaw` vuelve con **cero** bytes
distintos. No es un defecto del códec: la tabla de decodificación cae sobre los
mismos 256 niveles, así que la ida y vuelta es exacta por construcción (ver
`shared/audio/codec.py`).

La consecuencia es incómoda: **un cliente que decodifica y recodifica pasaría la
comparación de bytes igual que uno que no toca nada.** Lo que sí distingue es
meter una ganancia, un remuestreo o un códec distinto en el camino.

Por eso la garantía no es la comparación sino la **negativa estructural**: el
reproductor de Media Streams **no tiene** un camino desde PCM16 —
`tramas_del_clip` levanta `FallaDelArnes` en vez de convertir, y un paso sin
bytes μ-law no se reproduce— así que no hay dónde colar una conversión. La
comparación de bytes se conserva como red de seguridad barata, no como la
prueba. Está escrito en
`poc1_pipecat/tests/test_media_streams.py::test_el_viaje_por_pcm_es_exacto_y_por_eso_los_bytes_no_alcanzan`,
que además avisa si algún día el códec cambia y la comparación recupera su filo.

## Módulos

| Archivo | Qué hay |
|---|---|
| `audio.py` | ruido de confort, RMS, troceo en tramas de 20 ms |
| `guion.py` | `PasoDelGuion`, carga de clips y escenarios, estado del árbol |
| `corrida.py` | `evaluar_corrida`, el mínimo de turnos, los motivos de fin |
| `observado.py` | `Observado`, `TurnoObservado`, fragmentos de texto |
| `reproductor.py` | el reproductor por turno y las dos esperas del turn-taking |
| `apoyo_verbal.py` | clasificador de backchannel es-MX, para decidir el avance |
| `reporte.py` | bitácora, reporte en pantalla, aviso de costo |
| `auditoria.py` | grabación de dos canales para **oír** la corrida (opcional) |

## Cuándo avanza el turno

El arnés no suelta el clip siguiente hasta que el agente **contestó y terminó**.
"Contestó" no es "se oyó algo": una racha de voz cuenta como respuesta si dura
más de `TOPE_DE_APOYO_VERBAL_SEG`, o si pasa `MINIMO_DE_VOZ_SUSTANTIVA_SEG` y su
transcripción no es un apoyo verbal (`apoyo_verbal.py`).

Lo que se descarta no se tira: queda en `fragmentos_descartados` del turno y en
`turnos_con_descarte` de la bitácora. **Un turno que avanza sobre audio que no
era la respuesta queda corrido y no se nota en los números** — en la corrida del
16 de septiembre de 2026 su acuse salió en 3.42 s, dentro del rango sano de esa
misma llamada. Sin el registro solo se detecta oyendo el WAV.

Los dos criterios no son redundantes y ninguno sobra:

- la **duración** atrapa el caso que disparó todo esto — 160 ms de audio a casi
  fondo de escala que **no produjeron transcripción**, así que no había texto que
  clasificar;
- el **clasificador** atrapa el "ajá" de medio segundo, que sobrevive al mínimo.

Verificado contra un agente vivo el 16 de septiembre de 2026
(`spots-poc2-arnes-20260916-162221`): **7 turnos, un solo descarte y cero falsos
positivos**. El descarte volvió a ser el mismo patrón —el agente soltó
`'Perfecto,'` en 291 ms, calló 1.38 s y recién entonces contestó—, así que el
criterio viejo habría avanzado en 14.36 s contra una respuesta que empezaba en
14.94 s y duraba 9.8 s. El clasificador de texto **todavía no se ha ejercitado en
vivo**: en esa corrida ninguna racha cayó en la banda de 0.6 a 2.0 s.

## La grabación de auditoría (`--grabar-auditoria`)

Sirve para verificar con el oído lo que las pruebas verifican con números.
**Apagada por default**, y cuando se enciende escribe un WAV de dos canales en
la carpeta de diagnóstico de la PoC, junto a un `.json` que dice qué es:

- **canal 0**, lo que el arnés publicó: los clips y el ruido de confort entre
  ellos, sin distinción;
- **canal 1**, el audio del agente, tal como llegó.

**Separados y no mezclados**, porque lo que se quiere oír es si el agente
arrancó encima del clip y quién habló primero — en una mezcla, dos voces
simultáneas suenan como una voz con ruido.

Tres cosas que la mantienen fuera del camino medido:

- **No entra a ninguna métrica**: ni `latencias`, ni percentiles, ni WER, ni la
  poza de las 15 llamadas. El nombre del archivo y el JSON de al lado lo dicen,
  porque un `.wav` suelto dentro de seis meses no se defiende solo.
- **El audio sale del arnés, no del servicio.** No se activa Egress ni se toca
  `record`: el ambiente de la comparativa está congelado, y grabar por el
  framework cambiaría lo que se mide además de sumar un renglón a la factura.
- **No escribe a disco en vivo.** `anotar_*` mete una tupla en una lista (0.2 µs
  contra un ciclo de 20 ms) y el archivo se arma al colgar. Una corrida de 3
  minutos ronda los 12 MB en memoria.

El canal 0 se sube de banda telefónica a la tasa del agente para que quepan en
un archivo; sigue sonando a teléfono, que es como suena de verdad. Si las dos
tasas no se pueden juntar, el arnés **falla antes de conectarse**: enterarse
después de haber gastado la llamada sería el peor momento posible.

La alineación sirve para oír quién habló primero, **no** para medir
milisegundos: el canal 0 sale exacto porque el reproductor conoce su deadline,
pero el canal 1 se fecha cuando la trama llega, con el jitter del transporte
dentro. Para los milisegundos están las marcas del cero del turno.

## Lo que este paquete NO hace

- **No conoce ningún framework de voz.** Una prueba lo verifica sobre el código
  de cada módulo: si aparece `livekit`, `pipecat`, `twilio` o `websockets`, la
  suite falla. Es lo único que la suite de una sola PoC no podría detectar.
- **No toca el contrato de métricas.** Lo que observa es diagnóstico; las
  latencias que se reportan las escribe el worker de cada PoC.
- **No negocia.** El transportista actúa un guion grabado y elige rama con
  `shared/test_scenarios/decision.py`. Un simulador por LLM sigue abierto, y
  sería otra cosa: más realista y menos repetible.
