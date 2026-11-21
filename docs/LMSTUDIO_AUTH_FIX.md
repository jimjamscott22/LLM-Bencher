# LM Studio API Token Support + Run Status Fix

## Background

LM Studio's local server (port 1234) can require an API token. When it does, `/v1/chat/completions` returns `401 Unauthorized`. LLM Bencher doesn't send a token today. The run is also still saved as **SUCCEEDED**, which skews history and analytics.

## Goals

1. Send an optional bearer token to LM Studio.
2. Mark failed provider calls as FAILED, not SUCCEEDED.
3. Don't retry auth errors (4xx).

---

## Steps

### 1. Add the env var

In `.env`:

```env
LLM_BENCHER_LM_STUDIO_API_KEY=your-token-here
```

In `.env.example` (leave it blank):

```env
LLM_BENCHER_LM_STUDIO_API_KEY=
```

Confirm `.env` is listed in `.gitignore`.

### 2. Add the setting

In the settings class (pydantic-settings, `env_prefix="LLM_BENCHER_"`):

```python
lm_studio_api_key: str | None = None
```

### 3. Send the header from the LM Studio provider

Where the provider builds its httpx client:

```python
headers = {}
if settings.lm_studio_api_key:
    headers["Authorization"] = f"Bearer {settings.lm_studio_api_key}"

client = httpx.AsyncClient(
    base_url=settings.lm_studio_url,
    headers=headers,
    timeout=settings.timeout,
)
```

- Setting the header on the client covers both `/models` and `/chat/completions`.
- If the key is unset, no header is sent, so this works whether LM Studio auth is on or off.
- Never log the header or the key.

### 4. Raise on HTTP errors

After every provider request:

```python
response = await client.post("/chat/completions", json=payload)
response.raise_for_status()
```

### 5. Save the correct run status

In the run handler, a provider exception must set the run to FAILED and store the error message:

```python
try:
    result = await provider.complete(...)
    run.status = "SUCCEEDED"
    run.output = result
except Exception as exc:
    run.status = "FAILED"
    run.error = str(exc)
finally:
    save(run)
```

Check that the run page and `/analytics` leave FAILED runs out of latency and score stats.

### 6. Don't retry 4xx errors

In the retry logic (`LLM_BENCHER_RETRY_ATTEMPTS` / `RETRY_DELAY`), retry only on timeouts, connection errors, and 5xx:

```python
def should_retry(exc: Exception) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500
    return isinstance(exc, (httpx.TimeoutException, httpx.ConnectError))
```

### 7. Show a helpful message for 401

Optional but nice: catch a 401 from LM Studio and show
"LM Studio requires an API token. Set LLM_BENCHER_LM_STUDIO_API_KEY in .env."

### 8. Add a test (or test by hand)

- **Auth on, no key:** run shows FAILED with a 401 message and no retries.
- **Auth on, key set:** run SUCCEEDS.
- **Auth off, no key:** run SUCCEEDS.

To check the token outside the app:

```bash
curl http://127.0.0.1:1234/v1/chat/completions -H "Authorization: Bearer YOUR_TOKEN" -H "Content-Type: application/json" -d '{"model":"qwen/qwen3.5-9b","messages":[{"role":"user","content":"hi"}]}'
```

### 9. Update the README

Add `LLM_BENCHER_LM_STUDIO_API_KEY` to the configuration section, with a note that it's only needed when LM Studio's server authentication is on.
