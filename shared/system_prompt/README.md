# system_prompt/ — el prompt versionado y su interpolación

`v0.md` es el system prompt del agente negociador. Las 3 PoC lo **referencian,
no lo duplican**: si cada una llevara su copia, la comparativa mediría
diferencias de prompt en vez de diferencias de tecnología de voz.

`prompt.py` es el código que lo carga, sustituye las 7 variables y verifica en
cada render que el precio máximo no se haya filtrado. Vive aquí, junto a la
plantilla que renderiza, por la misma razón: una copia por PoC son tres
versiones del guardarraíl que pueden divergir sin que nadie se entere.

Versionado por archivo (`v0.md`, `v1.md`, …). Un cambio de prompt invalida las
métricas de las corridas anteriores, así que la versión usada se anota en cada
corrida de pruebas.

## Los 7 placeholders

Son los **únicos** del prompt:

`{EMPRESA}`, `{ORIGEN}`, `{DESTINO}`, `{TIPO_CARGA}`, `{PRECIO_OBJETIVO}`,
`{COMPENSACIONES_DISPONIBLES}`, `{NOMBRE_TRANSPORTISTA}`.

`{COMPENSACIONES_DISPONIBLES}` se rellena con
`shared.negotiation_engine.renderizar_compensaciones()`, para que el prompt y
los valores válidos de `ofrecer_compensacion_no_monetaria(tipo)` salgan del
mismo YAML.

**No usar `str.format()` sobre el texto completo.** Cualquier llave literal
que aparezca en una revisión futura del prompt (un ejemplo, un fragmento de
JSON) lo rompería. `prompt.py` usa reemplazo explícito por clave, con un regex
que solo reconoce los 7 nombres conocidos: cualquier otra llave del archivo es
invisible para el sustituidor, por construcción.

## Lo que el prompt NO contiene, y no debe contener

`precio_maximo`. Vive únicamente en
[`shared/negotiation_engine/pricing.py`](../negotiation_engine/pricing.py),
detrás de `proponer_precio`. El prompt le dice al modelo que existe una
función que valida cada monto, nunca cuál es el techo.

## Funciones que el prompt asume

```
proponer_precio(monto)                    -> "aceptado" | "rechazado"
ofrecer_compensacion_no_monetaria(tipo)
registrar_acuerdo(monto_final, condiciones)
finalizar_sin_acuerdo(motivo)
```

Los schemas JSON exactos los define cada PoC según su framework; la
implementación de la lógica es compartida y vive en `negotiation_engine`.
