"""PoC1 layers 1-5: configuration, prompt, tool calling, pipeline and server.

The negotiation logic does **not** live here: it is in
`shared/negotiation_engine/`, already implemented and tested. This package only
wraps it in Pipecat's tool-calling schemas and connects audio to it.
"""
