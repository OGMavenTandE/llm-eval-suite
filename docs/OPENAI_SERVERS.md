# OpenAI-compatible servers

vLLM and llama.cpp server have their own connection types. Both speak `/v1/chat/completions`. TensorRT-LLM and NVIDIA NIM do too, and they use the existing OpenAI-compatible type.

## vLLM

Type `vllm`. Default base URL `http://127.0.0.1:8000/v1`. Health is `GET /health` on the server origin, not under `/v1`. Model listing is `GET /v1/models`.

The connector uses chat completions, so vLLM applies the model's chat template. It sends `max_tokens: 512` unless the connection sets another cap. vLLM's own default is 16 when the field is omitted. Streaming is on. A connection test records time to first token.

```yaml
models:
  - name: local-model
    provider: vllm
    params:
      base_url: http://127.0.0.1:8000/v1
```

## llama.cpp server

Type `llama_cpp`. Default base URL `http://127.0.0.1:8080/v1`. Health is `GET /health`. Model listing is `GET /v1/models`.

Chat completions use the chat template in the GGUF. Start `llama-server` with that template, or pass `--chat-template`, when the file does not carry one. Completions mode skips the template. The token cap and streaming match vLLM: 512 tokens, and time to first token on the connection test.

```yaml
models:
  - name: local.gguf
    provider: llama_cpp
    params:
      base_url: http://127.0.0.1:8080/v1
```

## TensorRT-LLM and NVIDIA NIM

Use type `openai` (YAML provider `openai`). Point the base URL at the local OpenAI port, usually `http://127.0.0.1:8000/v1`. Leave the API key blank unless the container was started with one.

NIM readiness is `GET /v1/health/ready` on that host. Model listing is `GET /v1/models`. There is no separate NIM or TensorRT-LLM connector. The OpenAI-compatible client already posts chat completions to whatever base URL you save.

## URL and TLS

A configured base URL must use http or https. Link-local and metadata hosts are rejected, including the range 169.254.0.0/16 and IPv6 link-local addresses (fe80::/10). localhost, 127.0.0.1, and ::1 stay allowed, so Ollama on this machine still connects. A private-network address such as 10.0.0.0/8 or 192.168.0.0/16 is allowed. The check reads the host in the URL. It does not resolve DNS.

OpenAI-compatible requests verify the server certificate. There is no switch to turn verification off. On a closed network with a private CA, set `ca_bundle` in the model params, or set `LLM_EVAL_CA_BUNDLE` to the PEM file. That path is the trust store for those requests.
