# Poblaciones y pre-registros — campaña `campana200` y campaña de PoC3

> **22 de septiembre de 2026.** Este documento existe porque varias `n` de la
> comparativa se fijaron **en conversación** y nunca se escribieron. Un
> pre-registro que solo vive en un chat no sirve: no se puede auditar, no se
> puede reproducir, y cuando alguien lo recuerda mal no hay contra qué
> contrastarlo. Aquí quedan los criterios, con el filtro exacto que los produce
> y el resultado de aplicarlo a los artefactos que hay en disco hoy.
>
> **Dos categorías, y no se mezclan:**
>
> - **Pre-registro** — criterio fijado **antes** de ver el dato que decide. Se
>   escribe una vez y no se mueve. Si se mueve, se dice cuándo y por qué, y el
>   análisis anterior no se borra.
> - **Población heredada** — filtro que ya estaba en el código y que un análisis
>   adoptó sin declararlo. No es ilegítimo, pero hay que nombrarlo, porque casi
>   siempre se diseñó para otra métrica.
>
> Nada de este documento cambia una cifra publicada. Lo único que hace es decir
> de dónde sale cada una.

## 1. Las poblaciones de `campana200` — heredadas

Todas se derivan de `diagnostico/lote/campana200/progreso.jsonl` (200 filas) y
de los resúmenes que apunta. Reproducidas el 22/09/2026.

| n | filtro exacto | reproducible | dónde se cita |
|---:|---|---|---|
| **200** | todas las filas del lote | sí | tamaño de la campaña |
| **173** | `progreso.estado == "ok"` | sí | `metrics/cli.py:137` |
| **176** | `resumen.fin.valida == true` | sí | — |
| **179** | corridas con ≥1 muestra de latencia | sí | ver §4 |
| **190** | corridas con ≥1 muestra de `llm_ttft_ms` | sí | — |
| **84** | **pares** donde las **dos** corridas cumplen `estado == "ok"` y tienen artefactos en disco | sí (+5 sueltas) | `CONTRATO_METRICAS.md:1997, 2197, 2232` |
| **86** | **pares** donde las dos cumplen `fin.valida` | sí (+4 sueltas) | — |
| **1,358** | turnos de las 200 bitácoras tras `clasificar_cola` | sí | poza del WER |
| **1,249** | entradas de `por_intervencion` con `latencia_de_intervencion_ms` no nulo | sí | `hallazgos_20260922_poc3.md:30` |

### 1.1 El 84 y el 86: el portón del WER es de confiabilidad, no de WER

`estado == "ok"` en `progreso.jsonl` equivale a `criterios_fallidos == 0`: es el
semáforo de **confiabilidad** del lote. `fin.valida` dice otra cosa — si la
corrida es analizable. Los 2 pares de diferencia son:

| par | PoC1 | PoC2 |
|---|---|---|
| e5 r16 | `0159-poc1-e5-r16` · `ok` · válida · 8/8 turnos · 0 criterios fallidos | `0160-poc2-e5-r16` · **`roja`** · **válida** · 6/6 · **1** criterio fallido |
| e5 r17 | `0169-poc1-e5-r17` · `ok` · válida · 8/8 · 0 | `0170-poc2-e5-r17` · **`roja`** · **válida** · 6/6 · **1** |

El criterio que falla en las dos de PoC2 es el mismo:
`turnos_sin_audio_del_agente: ["T6"]` — el agente emitió texto sin audio en un
turno. **Es un defecto de la salida del agente, no del estímulo**: los clips del
transportista sonaron completos en las dos corridas, y el WER mide la
transcripción de esos clips. La poza del WER heredó un portón diseñado para la
latencia y la confiabilidad.

Qué aportarían esos 2 pares a la poza del WER:

```
84 pares → 1,138 turnos, 12,560 palabras de referencia
86 pares → 1,162 turnos, 12,988 palabras
delta    →   +24 turnos,   +428 palabras = 3.41% de la referencia
```

Las 428 palabras son **todas del escenario 5**, el de estrés de reconocimiento.
Añadirlas subiría el WER de las dos PoC.

**Decisión del 22/09/2026: 84 se queda para todo lo publicado.** Es la población
que `cli referencia-de-clips` procesó y de la que salen las cifras del contrato.
Cuando se recalcule el WER contra la referencia derivada del audio se publicarán
**las dos cifras con su criterio al lado**, porque en ese momento la población
deja de ser heredada y pasa a ser una elección.

*(Las 12,560 palabras se cuentan con `str.split()`; el contrato publica 6,149
por PoC porque el WER tokeniza distinto. Los dos cuentan lo mismo, pero no se
mezclan en la misma tabla.)*

### 1.2 El 84 de la campaña de PoC3 está derivado de éste

`docs/costos_poc3.md` usa **84** como tamaño de la campaña de PoC3 (líneas 8,
80, 88, 95, 103, 120, 160, 177, 180, 210, 263, 357), y
`hallazgos_20260922_poc3.md:52` lo usa para contar cuántas veces suena `e3_t6`.
Si la población pasara a 86, la campaña de PoC3 serían 86 llamadas y la tabla de
costos se movería (~$10.94 → ~$11.2). **No se mueve por omisión: se decide
cuando la población esté cerrada.**

## 2. Pre-registro — sensibilidad por contaminación de ElevenLabs

### 2.1 La definición, fijada antes de ver latencias

> **Llamada contaminada** = al menos una línea de error de ElevenLabs
> (cualquier status) en el log de su PoC, dentro de su ventana de reloj
> `[inicio, fin]`, con `cuando = fin`.
>
> **Sensibilidad** = excluir el **par completo** si cualquiera de las dos
> corridas está contaminada. El análisis principal siguen siendo los **84**
> pares declarados en §1.

La definición se mantiene: es correcta y es la que se fijó. Lo que sigue es el
resultado de aplicarla.

### 2.2 Al aplicarla hoy: 0 pares contaminados de los 84

Patrón estricto —línea de severidad `ERROR`/`WARNING`/`CRITICAL` que además
nombra a ElevenLabs— sobre los 200 logs de
`diagnostico/lote/campana200/llamadas/`:

- **10 corridas con error real de ElevenLabs**, todas de **PoC1**, todas de las
  repeticiones **19 y 20**: `0181, 0183, 0185, 0187, 0189, 0191, 0193, 0195,
  0197, 0199`.
- El fallo es `1008 (policy violation)` en el WebSocket del TTS, 350
  apariciones.
- **Las 20 corridas de r19 y r20 —las 10 de PoC1 y las 10 de PoC2— están en
  `estado = roja`**, así que ninguna entra a los 84 pares.

```
pares contaminados dentro de los 84:  0
población de sensibilidad:           84   (idéntica al análisis principal)
```

### 2.3 Lo que circuló como «81 pares» no se reproduce

La versión que venía circulando decía **3 pares contaminados (e3r11, e3r12,
e3r14), 81 limpios, los tres por `recv loop error status_code=1006` durante el
desmontaje**. Ninguna de las tres partes sobrevive a la verificación:

1. **`1006` no aparece en ningún log de la campaña.** Cero apariciones de esa
   cadena en los 200 archivos. El código real es `1008`.
2. **Los pares no son ésos.** Un patrón laxo —cualquier línea que mencione
   «eleven» y contenga la palabra «error»— sí devuelve 3 pares, pero son
   **(5,3), (5,12) y (5,13)**, no `e3r11/12/14`. Y son **falsos positivos**: las
   líneas que engancha son `DEBUG … Generating TTS [Vamos a asegurarnos de que
   esté bien hecha y sin errores.]`. Es **el agente diciendo la palabra
   "errores" en voz alta**, no un fallo del servicio.
3. **El 81 es coincidencia aritmética.** 84 − 3 falsos positivos = 81. El número
   era reproducible solo con el patrón que se equivocaba.

**Consecuencia: el análisis de sensibilidad por contaminación no excluye nada, y
eso va escrito como resultado**, no callado por aburrido. La campaña sí tuvo un
incidente grande de ElevenLabs —se llevó las 20 últimas corridas— pero el
semáforo del lote ya lo había atrapado.

### 2.4 Defecto del criterio: es ciego a PoC2

El criterio dice «en el log de su PoC». Medido: **la cadena «eleven» aparece en
0 de los 100 logs de PoC2**, y sus logs tampoco nombran `tts`, `voice` ni
`synth`. 21 de las 100 corridas de PoC2 sí tienen líneas `ERROR`/`CRITICAL`,
pero ninguna identifica al servicio.

O sea: **el criterio solo puede detectar contaminación de PoC1.** Es asimétrico
por construcción, y un resultado de «0 pares contaminados» no significa que PoC2
no haya tenido fallos de TTS — significa que su log no los nombra. Queda
anotado; arreglarlo pide un gancho en el worker de PoC2, no un cambio de
criterio.

## 3. Pre-registro — el discriminante de dos etapas de PoC3

Separa **full-duplex** (el agente arranca encima de habla real del
transportista) de **colgajo del endpointer** (arranca en el silencio que el
detector todavía no declaró). Fijado antes de correr las 84 de PoC3.

### 3.1 Denominadores

> **N1** = turnos de las 84 llamadas de campaña de PoC3 que cumplen las cuatro:
>
> 1. el arnés emitió marca para ese turno,
> 2. la muestra tiene `t_acuse_pared` no nulo,
> 3. `reloj.interpretable == true`,
> 4. `eje_de_pared.falla == null`.
>
> **N2** = subconjunto de N1 con `d > +50 ms` y con `end_ms` colocable en el eje
> de pared.

`d` es el desfase entre el arranque del agente y el fin de voz medido del
transportista: negativo = el agente arrancó **antes** de que el clip terminara.

### 3.2 Umbrales

> **Etapa 1 — ¿hay full-duplex?**
>
> - **Se sostiene** con **≥25% de N1** en `d ≤ −50 ms`.
> - **Se refuta** con **≥80% de N1** en `d ≥ +50 ms`.
>
> **Etapa 2 — ¿es colgajo del endpointer?**
>
> - **Confirma colgajo** con **≥80% de N2** con arranque **<** `end_ms`.
> - **Pipeline ordinario** con **≥30% de N2** con arranque **>** `end_ms`.
>
> **Cobertura mínima de la etapa 2**, las dos a la vez:
>
> - N2 ≥ **60%** de los turnos con `d > +50 ms`, y
> - N2 ≥ **100** en absoluto.

### 3.3 Prohibiciones

Son parte del pre-registro, no comentario al margen:

1. **Si el resultado cae entre los dos umbrales de una etapa, no se concluye
   ninguna de las dos cosas.** Se reporta como indeciso, con su porcentaje y su
   n, y se dice qué haría falta para decidirlo.
2. **La frase «firma del full-duplex» no se escribe** —ni en el reporte, ni en
   `comparativo_resultados.md`, ni en `CLAUDE.md`— salvo que la etapa 1 la
   sostenga con su umbral. Hoy esa frase está en `CLAUDE.md` y está **sin
   respaldo**: se queda hasta que la campaña la confirme o la tumbe.
3. **Si la etapa 2 no alcanza su cobertura mínima, no se reporta la etapa 2.**
   Ni como tendencia, ni como «apunta a», ni en una nota al pie.
4. **Los umbrales no se mueven después de ver los datos.** Si se mueven, se
   fecha el cambio, se dice qué se vio antes de moverlos, y se publica el
   resultado con los dos criterios.
5. **No se parte la muestra por escenario, por turno ni por llamada para
   alcanzar un umbral.** Los estratos se publican si aportan, pero la decisión
   se toma sobre N1 y N2 completos.
6. **N1 y N2 se publican siempre**, salga lo que salga, junto con cuántos turnos
   se cayeron por cada una de las cuatro condiciones.

## 4. El cuarto n — «180 llamadas limpias» no es reproducible

`CONTRATO_METRICAS.md:1786` y `comparativo_resultados.md:357` fechan en el
21/09/2026 el hallazgo de que `llm_ttft_ms` no es comparable entre PoC1 y PoC2,
y lo sitúan «sobre las **180** llamadas limpias de la campaña».

**Ningún filtro reproduce 180.** Se probaron los 11 predicados naturales del
lote y todas sus combinaciones de hasta tres:

```
173   estado == ok              179   >=1 muestra de latencia
176   fin.valida                179   >=1 turno medible
176   sin reintentar            186   sin turnos_sin_audio
178   criterios_fallidos <= 1   190   >=1 muestra de llm_ttft_ms
```

**179 es un atractor fuerte**: veinte combinaciones distintas caen en él, porque
exactamente **una** corrida de las 200 no tiene ninguna muestra de latencia. La
lectura más probable es que 180 sea un 179 mal contado, o un recuento sobre una
instantánea anterior del lote.

**Lo defendible hoy es 179**, con el filtro *«corridas con al menos una muestra
de latencia»*. El hallazgo de §2.7.1 no depende de la cifra —lo que prueba es
que las dos llaves miden tramos distintos, y eso se ve en cualquier
subconjunto—, pero la `n` citada hay que corregirla.

## 4.bis Pre-registro — la pasada estadística de la latencia

> **23 de septiembre de 2026.** Cierra el pendiente #5 de
> `campana_poc3_estado_20260923.md`.

### 4.bis.0 ⚠ Esto NO es un pre-registro completo, y hay que decirlo

Las tres medianas **ya se calcularon** antes de escribir esta sección, el mismo
día: PoC1 2,609.5 ms, PoC2 3,081.0 ms, PoC3 2,464.2 ms. O sea que el criterio de
§3 —«fijado **antes** de ver el dato que decide»— **no se cumple aquí**, y llamar
a esto pre-registro sin más sería exactamente el vicio que este documento existe
para evitar.

Lo que sí se puede afirmar, y es lo único que se afirma:

1. **Ninguna prueba se ha corrido todavía.** No hay estadístico de contraste, ni
   valor p, ni intervalo de la diferencia calculado. Lo que hay son tres
   medianas y sus IC95 bootstrap, que son descriptivos.
2. **El margen de equivalencia no sale de los datos.** Sale de literatura
   externa (§4.bis.3) y se puede verificar que no fue elegido para que la
   comparación saliera de un lado: la brecha observada más chica (PoC1↔PoC3,
   145 ms) **cae dentro** del margen propuesto (200 ms) y la más grande
   (PoC2↔PoC3, 617 ms) **cae fuera**. Un margen elegido a conveniencia habría
   caído entre las dos.
3. **El procedimiento se fija ahora y no se mueve después.** Si se mueve, se
   fecha el cambio y se publican los dos resultados, igual que en §3.3.4.

**Consecuencia para el reporte: la sección de latencia se escribe como análisis
confirmatorio del tamaño del efecto y exploratorio de la significancia.** No se
escribe «se encontró una diferencia significativa» sin esta advertencia al lado.

### 4.bis.1 Unidad, población y pareo

> **Unidad de análisis: la intervención**, no el turno del pipeline (§2.2.5 del
> contrato). Una intervención es una línea del guion soltada al cable.
>
> **Métrica: `latencia_de_intervencion_ms`** — del fin de voz del arnés al
> primer audio del agente, las dos puntas fechadas por el arnés. Para PoC3 es la
> reconstrucción externa de `docs/instrumento_de_latencia_poc3.md`, **no**
> `turno_completo_ms` del puente.
>
> **Población:** los **84 pares** de `campana200` para PoC1 y PoC2 (§1), y las
> **90** llamadas de `campana_poc3` para PoC3.
>
> **Pareo:** PoC1 contra PoC2 se comparan **pareados por `(escenario,
> repetición)`** — es lo que la campaña alterna a propósito (§2.8 del contrato),
> y descarta la deriva de red y de proveedor entre una corrida y la siguiente.
> **PoC3 no tiene pareja**: corrió otra campaña, otro día. Sus comparaciones
> contra PoC1 y PoC2 son **no pareadas**, y eso se declara cada vez.

### 4.bis.2 El estadístico y el remuestreo

> **Estadístico: diferencia de medianas** de `latencia_de_intervencion_ms`.
> Mediana y no media: §2.9 del contrato documenta que tres de las
> distribuciones medidas son bimodales y de cola pesada, donde la media no
> describe ningún turno.
>
> **Remuestreo: bootstrap por conglomerados, con la LLAMADA como conglomerado**,
> 10,000 réplicas, semilla fija `20260923`. No por turno: los turnos de una
> misma llamada comparten worker, sesión de TTS y estado de red, así que
> tratarlos como independientes estrecharía los intervalos de mentira.
>
> - **PoC1 vs PoC2:** se remuestrean los **84 pares** (el par entero, las dos
>   corridas juntas), y dentro de cada corrida sus intervenciones.
> - **PoC3 vs cada una:** se remuestrean las llamadas de cada campaña por
>   separado.
>
> **Intervalo:** percentil 2.5 y 97.5 de la distribución bootstrap de la
> diferencia. El valor p, si se reporta, es el bootstrap de dos colas.

### 4.bis.3 El margen de equivalencia: **Δ = 200 ms**, y de dónde sale

> **Dos arquitecturas cuya mediana difiera en menos de 200 ms se declaran
> prácticamente equivalentes**, aunque la diferencia sea estadísticamente
> distinguible.

Las dos fuentes, las dos externas al proyecto:

- **Stivers et al., «Universals and cultural variation in turn-taking in
  conversation», PNAS 106(26):10587–10592, 2009.** Sobre diez lenguas, el desfase
  modal entre el fin de un turno y el inicio del siguiente es de **≈0 a 200 ms**,
  con una mediana alrededor de los 100–300 ms según la lengua. Es la escala a la
  que el turn-taking humano opera, y por lo tanto la escala por debajo de la cual
  una diferencia deja de ser perceptible como «tardó más en contestar».
- **ITU-T G.114, «One-way transmission time» (2003).** Hasta **150 ms** de
  retardo en un sentido la mayoría de las aplicaciones no percibe impacto;
  el rango aceptable llega a 400 ms. Es la referencia clásica de telefonía y la
  que justifica que 150–200 ms sea el orden del «no se nota».

Δ = 200 ms es el valor más alto que las dos fuentes sostienen a la vez. Se elige
el más alto **a propósito**, porque es el más exigente para declarar una
diferencia: cuanto mayor el margen, más difícil que una brecha cuente como real.

> **Cómo se lee el resultado**, con IC95 de la diferencia e `[−Δ, +Δ]`:
>
> | El IC95 … | Conclusión |
> |---|---|
> | cae entero dentro de `[−200, +200]` | **equivalentes en la práctica** |
> | cae entero fuera y no cruza el 0 | **diferencia real y relevante** |
> | no cruza el 0 pero se solapa con el margen | diferencia real, **relevancia indecisa** |
> | cruza el 0 | **no se distingue**; se reporta el intervalo, no «son iguales» |

### 4.bis.4 Comparaciones múltiples

> **Tres comparaciones planeadas** (1↔2, 1↔3, 2↔3), **corrección de Holm** sobre
> las tres. Holm y no Bonferroni: domina a Bonferroni uniformemente y no exige
> independencia, que aquí no se cumple porque las tres comparten el mismo
> estímulo.
>
> **Los estratos por escenario son descriptivos y NO entran a la corrección.**
> Se publican con su n y su intervalo, y ninguna conclusión del reporte cuelga
> de uno solo. Es la prohibición 5 de §3.3, aplicada aquí.

### 4.bis.5 Potencia — y una consecuencia que hay que declarar

> **⚠ Calcular la potencia produjo los intervalos.** El semiancho del IC95 del
> protocolo es ≈1.96·SE, y SE sale del **mismo** bootstrap que da el intervalo.
> O sea que no había forma de estimar la potencia sin conocer el resultado.
> Se publica en vez de esconderse, y refuerza lo que dice §4.bis.0: la sección
> de latencia del reporte es **confirmatoria del tamaño del efecto y
> exploratoria de la significancia**.

**Potencia estimada sobre la dispersión observada**, con el bootstrap por
conglomerados de §4.bis.2 (4,000 réplicas, semilla `20260923`):

| efecto verdadero | PoC1 vs PoC2 | PoC1 vs PoC3 | PoC2 vs PoC3 |
|---:|---:|---:|---:|
| 50 ms | 0.21 | 0.12 | 0.11 |
| 100 ms | 0.64 | 0.35 | 0.28 |
| 150 ms | 0.94 | 0.65 | 0.54 |
| **200 ms (= Δ)** | **1.00** | **0.88** | **0.78** |
| 300 ms | 1.00 | 1.00 | 0.98 |

**Efecto mínimo detectable al 80%:** 121 ms (1↔2), 178 ms (1↔3), **204 ms**
(2↔3).

Dos lecturas:

1. **El estudio está bien dimensionado para el margen que declara.** Al tamaño
   que importa —Δ = 200 ms— la potencia es 1.00, 0.88 y 0.78.
2. **La comparación PoC2↔PoC3 se queda justo por debajo del 0.80 convencional**,
   y su efecto mínimo detectable (204 ms) es prácticamente el margen. Eso no
   invalida nada —su diferencia observada es de 617 ms, muy por encima— pero hay
   que decirlo: si la diferencia real hubiera sido del tamaño del margen, este
   estudio no la habría detectado de forma fiable. La causa es que PoC3 no está
   pareado y arrastra la varianza de dos campañas en vez de una.

**Los intervalos que salieron de ese mismo cálculo**, con el signo
`primera − segunda`:

| comparación | diferencia | IC95 | SE | veredicto según §4.bis.3 |
|---|---:|---|---:|---|
| PoC1 − PoC2 *(pareado)* | **−471.5** | [−577.0, −408.0] | 43.1 | **diferencia real y relevante** |
| PoC1 − PoC3 | **+145.2** | [+22.8, +265.0] | 63.6 | **real, relevancia indecisa** |
| PoC2 − PoC3 | **+616.8** | [+484.7, +764.4] | 73.0 | **diferencia real y relevante** |

Las tres sobreviven a Holm: los valores p implicados son ≈0, ≈0 y ≈0.022, contra
umbrales de 0.0167, 0.025 y 0.05.

**La fila del medio es la que hay que escribir con cuidado.** El intervalo de
PoC1↔PoC3 no cruza el cero —la diferencia es real— pero **se solapa con el
margen de equivalencia**: su extremo inferior (+22.8 ms) está muy dentro de lo
que las dos fuentes de §4.bis.3 consideran imperceptible y su extremo superior
(+265 ms) está fuera. **No se escribe que PoC3 sea más rápida que PoC1 en un
sentido que importe, ni que sean equivalentes.** Se escribe que la diferencia es
real y que este estudio no puede decidir si es perceptible.

### 4.bis.6 Prohibiciones

1. **No se cambia la métrica después de ver el resultado.** Si `latencia_de_intervencion_ms`
   diera un resultado incómodo, no se sustituye por la latencia corregida de
   §2.2.8 ni por el desglose por etapa.
2. **La latencia corregida (observada − transporte) se publica siempre al lado
   de la observada**, nunca en su lugar (§2.2.8).
3. **La comparación principal se hace sobre TODOS los turnos.** La partición por
   round trip de herramienta (`docs/hallazgo_round_trip.md`) es un análisis
   secundario declarado, se publica entera, y no sustituye a la principal.
4. **Si una comparación sale indecisa, se reporta indecisa.** No se sube el n
   juntando campañas ni se baja el margen.

## 5. Pendiente

- [ ] Decidir 84 contra 86 al recalcular el WER contra la referencia derivada
      del audio, y publicar las dos.
- [ ] Corregir `180` → `179` en `CONTRATO_METRICAS.md:1786` y
      `comparativo_resultados.md:357`, con el filtro nombrado.
- [ ] Anotar en `CONTRATO_METRICAS.md` §2.10 que la población del WER hereda el
      portón de confiabilidad del lote.
- [ ] Gancho en el worker de PoC2 que nombre el servicio de TTS en sus errores,
      para que el criterio de §2 deje de ser ciego a una de las dos PoC.

---

## Artefactos de los que sale todo esto

```
diagnostico/lote/campana200/progreso.jsonl            (200 filas)
diagnostico/lote/campana200/resumenes/*.json          (200)
diagnostico/lote/campana200/llamadas/*.log            (200)
poc{1,2}_*/diagnostico/arnes/bitacora_*.json
shared/metrics/cli.py:137                             (el filtro `estado == ok`)
shared/metrics/referencia_de_clips.py                 (clasificar_cola, clips_comunes)
```
