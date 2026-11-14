"""PoC2 layers 1-5: configuration, prompt, tool calling, voice session and dialer.

The negotiation logic does **not** live here: it is in `shared/negotiation_engine/`,
just as in PoC1. This package only wraps it in LiveKit Agents' tool-calling
contract and connects audio to it over the SIP trunk.

Today only `config.py` exists. The rest of the package arrives in the
functional increment; which file does what is listed in the PoC's README.
"""
