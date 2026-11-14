# test_scenarios/ — el guion de pruebas

El guion de pruebas. **Los mismos 5 escenarios, en el mismo orden, con las
mismas repeticiones, en las 3 PoC** — es lo que hace que la comparativa sea
una comparativa.

## Dónde vive cada cosa

El guion existe en dos formatos, con dueños distintos:

- **`escenarios.yaml`** (aquí) — las líneas habladas, como dato. Es lo que lee
  `shared/metrics/` para pre-rellenar el archivo de referencia del WER después de
  cada llamada.
- **[`guion_escenarios.md`](../negotiation_engine/tests/guion_escenarios.md)** —
  el guion para la persona: qué mide cada escenario, cómo actuar, la rúbrica 1-5,
  el orden de ejecución y la plantilla de registro.

Son dos copias del mismo texto, así que hay una prueba que verifica que no se
separen: `shared/metrics/tests/test_escenarios.py::TestSinDeriva`. Si cambias una
línea en uno, cámbiala en el otro o la prueba falla.

- **`decision.py`** (aquí) — el actor automático. Los escenarios 2 y 4 no son
  lineales, y cuando el arnés actúa el guion tiene que elegir la rama igual que
  la elegiría una persona leyendo el `.md`.

## El árbol de decisión (escenarios 2 y 4)

Los escenarios 2 y 4 ramifican, y **eso es lo que miden**. El 2 mide si las
concesiones decrecientes dejan al agente llegar a 18k, y la pregunta solo
significa algo si Carlos reacciona distinto cuando llega y cuando no. El 4 mide
si ofrece una compensación antes de subir el precio; si Carlos dice "órale, con
eso ya cambia la cosa" cuando el agente nunca ofreció nada, el escenario dejó de
medir lo que dice medir. Los escenarios 1, 3 y 5 son lineales **a propósito**.

Las ramas viven en `escenarios.yaml`, en la llave `ramas` de cada turno, al lado
de las `alternativas` que lee la persona. La `linea` de cada rama tiene que ser
idéntica a la base o a una de las alternativas: esa igualdad amarra la rama al
clip grabado (`eN_tM` la base, `eN_tM_altk` la alternativa k) y es lo que impide
que el árbol y el guion en prosa se separen. Un test lo verifica.

Tres cosas que vale la pena saber antes de leer un resultado:

- **Decide por lo que el agente DIJO, no por lo que registró.** Un defecto ya
  visto es pronunciar una compensación sin llamar a
  `ofrecer_compensacion_no_monetaria`. El árbol rama con lo que se oyó —que es
  lo que haría un transportista— y el desacuerdo contra
  `compensaciones_ofrecidas` queda en el JSON como el hallazgo que es.
- **Las frases con las que se reconoce cada compensación viven en
  [`compensaciones.yaml`](../negotiation_engine/compensaciones.yaml)**, no aquí:
  si el negocio cambia el catálogo, cambian con él. Se eligen para no dar falso
  positivo aunque cueste recall, y hay una regla de negación para que "no
  tenemos flexibilidad en la fecha" no cuente como ofrecerla.
- **Cada decisión queda con su evidencia** (qué frase casó, qué montos se
  oyeron, qué ramas se descartaron y por qué). Ninguna heurística sobre texto
  libre es exacta; una rama equivocada tiene que poder verse después.

⚠ **Una corrida ramificada no dijo las líneas base.** La referencia del WER se
siembra con las líneas del guion, así que en los escenarios 2 y 4 hay que
corregirla con lo que se reprodujo de verdad. El arnés lo deja escrito en
`poc2_livekit/diagnostico/arnes/bitacora_<sala>.json`.

## Los 5 escenarios

1. **Cooperativo** — acepta cerca de la primera oferta. Camino feliz.
2. **Regateador moderado** — dos o tres contraofertas razonables. Ejercita las
   concesiones decrecientes.
3. **Agresivo / ancla alta** — abre muy por encima del techo. Ejercita el
   límite financiero: aquí es donde `proponer_precio` tiene que aguantar.
4. **Objeción no monetaria** — pide tiempo de espera pagado o similar.
   Ejercita `ofrecer_compensacion_no_monetaria` antes de mover precio.
5. **Estrés de reconocimiento / WER** — Carlos cierra trato, pero su habla va
   cargada de lo que tortura a un STT en 8 kHz: montos con letra y con dígito, una
   autocorrección a media frase, casi homófonos del español MX, topónimos de la
   ruta y términos del gremio. Es donde Deepgram, ElevenLabs Scribe y el
   reconocimiento nativo del Realtime deben separarse. Llamada normal, sin
   silencios ni cortes.

## Repeticiones

3 por escenario (15 llamadas por PoC) como objetivo. 2 como mínimo aceptable
si el tiempo aprieta — con 1 no se distingue una diferencia real de la
varianza de una llamada.

## Evidencia

El JSON por llamada (transcripción del STT + latencias) y el archivo de
referencia corregido contra **el guion actuado**. No hay grabación de audio de
por medio: la referencia del WER la escribe quien actuó, corrigiendo las líneas
`yo:` donde se desvió del guion. El método es idéntico en las 3 arquitecturas,
que es lo que hace comparables los números. Ambos archivos están gitignored.

## Pendiente

- [x] Escribir el guion turno por turno de cada escenario
- [x] Definir la configuración de negociación de cada uno (objetivo, máximo,
      ruta, tipo de carga) — el máximo va en la config de la corrida, nunca
      en el prompt
- [x] Plantilla de registro de corrida — la parte numérica la llena
      `shared/metrics/` sola; en el guion en prosa queda la plantilla de lo
      manual (resultado, cumplimiento semántico, notas 1-5)

