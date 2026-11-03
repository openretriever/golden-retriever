"""A Gemini client for the study's model boundary, over Vertex AI.

This is the one class `study_model` was built waiting for. With it, the study's
Direct, Agentic and Orchestrated forms stop being plumbing with an offline
stand-in and become the experiment the Astranomy study actually runs: a
frontier model asked for low-level continuous control.

**Why Vertex rather than an API key.** Vertex authenticates with the
credentials `gcloud` already holds, so no key is read, stored or logged by this
code, and nothing has to be added to the environment. `AGENTS.md` requires the
default test path to need neither network nor keys, which is why every test for
this module either builds the client without calling it or stubs the transport.

**What it reports and what it does not.** Token counts come from the provider's
own `usage_metadata`. Cost does not: prices change and are not discoverable from
the API, so `usd` is `None` unless a caller supplies rates. Reporting a cost
derived from a hardcoded guess would be worse than reporting none, because the
study publishes USD per episode as a result.

**Failures are statuses, not exceptions.** A timeout, an oversize answer, a
safety refusal and a transport error each end the call with a recorded status,
which is what lets the harness count refusals as a policy's behaviour instead of
crashing the sweep. The one thing that would be dishonest is to let a failure
look like an empty answer, so every status carries its provider detail.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .study_model import ModelCallStatus, ModelConfig, ModelResponse

#: Models reachable on this project, strongest first. `gemini-robotics-er-2`
#: would be the apt choice — it is Google's embodied-reasoning model — but it is
#: listed in the Vertex catalog and returns 404 without an allowlist, so it is
#: recorded here as the access to request rather than silently unavailable.
PREFERRED_MODELS = ("gemini-robotics-er-2-preview", "gemini-2.5-pro", "gemini-2.5-flash")
#: Listed in the catalog but not callable without allowlisting. Kept so a run
#: can report what it would have used.
GATED_MODELS = ("gemini-robotics-er-2-preview", "gemini-3-pro-preview",
                "gemini-3-flash-preview", "gemini-4-argon")
DEFAULT_LOCATION = "us-central1"
#: Models that are only served from the multi-region endpoint. Probed rather
#: than assumed: `gemini-robotics-er-2-preview` returns 429 "Quota exceeded for
#: generate_content_requests_per_minute_per_project_per_base_model" in
#: us-central1 — persistently, across 20-second gaps, which is a zero quota
#: rather than a rate limit — and answers normally from `global`. Reading that
#: 429 as "no access" would have been wrong, the same way reading the 404s below
#: as "model does not exist" would be: every Gemini 3.x and 4 name the model
#: garden lists is 404 on this project in us-central1, global and us-east4, so
#: they are unavailable here while plainly existing elsewhere.
GLOBAL_ONLY_MODELS = ("gemini-robotics-er-2-preview",)
#: Gemini Robotics-ER 2 is an embodied-reasoning model: spatial grounding and
#: robot control rather than general chat, which is the slot the Astranomy study
#: fills with a frontier general model. It is first in PREFERRED_MODELS because
#: on this task that specialisation matters more than a newer general model's
#: headline capability — and because the newer general models are not reachable.
#: Retries for a rate-limited or briefly unavailable call, and the base of the
#: exponential backoff between them.
#:
#: This exists because of how the two failure kinds differ. A refusal is the
#: policy's behaviour and belongs in the results; a 429 is the provider saying
#: "later" and belongs nowhere near them. Without a retry, running several
#: episodes at once turns quota pressure into silent no-motion actions, and the
#: run measures the quota rather than the model.
MAX_RETRIES = 4
BACKOFF_BASE_SECONDS = 2.0
#: JPEG quality for the frame sent with each call. The observation is already
#: small (a 256 px tile per view), so this trades almost nothing for a payload
#: that does not dominate the request.
JPEG_QUALITY = 90


class GeminiUnavailable(RuntimeError):
    """The provider library or project access is missing."""


def encode_frame(image, *, quality: int = JPEG_QUALITY) -> bytes:
    """JPEG-encode one RGB frame for transport."""
    import cv2

    frame = np.ascontiguousarray(np.asarray(image, dtype=np.uint8))
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("a frame must be (H, W, 3) uint8 RGB")
    if type(quality) is not int or not 1 <= quality <= 100:
        raise ValueError("quality must be an int from 1 through 100")
    ok, buffer = cv2.imencode(".jpg", frame[:, :, ::-1],
                              [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise ValueError("frame could not be encoded")
    return bytes(buffer)


@dataclass
class GeminiClient:
    """`study_model.ModelClient` over Vertex AI.

    `usd_per_1m_input` / `usd_per_1m_output` are optional. Supplied, the client
    reports a cost per call; omitted, it reports token counts and leaves cost
    absent rather than inventing a rate.
    """

    model_id: str = PREFERRED_MODELS[0]
    project: str = ""
    location: str = DEFAULT_LOCATION
    #: Constrain the provider to JSON output. See `_request_config`.
    json_mode: bool = True
    usd_per_1m_input: float | None = None
    usd_per_1m_output: float | None = None
    max_retries: int = MAX_RETRIES
    backoff_base: float = BACKOFF_BASE_SECONDS
    client: object = None
    calls: int = 0
    refusals: int = 0
    #: Retries spent on rate limits, so a run can show whether it was throttled.
    retries: int = 0
    rate_limited: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    #: Per-call statuses, so a run can show the refusal pattern rather than a
    #: single rate.
    statuses: list = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError("model_id must be a non-empty string")
        for name in ("usd_per_1m_input", "usd_per_1m_output"):
            rate = getattr(self, name)
            if rate is not None and (type(rate) is not float or rate < 0.0):
                raise ValueError(f"{name} must be None or a non-negative float")
        if type(self.max_retries) is not int or not 0 <= self.max_retries <= 10:
            raise ValueError("max_retries must be an int from 0 through 10")
        if type(self.backoff_base) is not float or not 1.0 <= self.backoff_base <= 10.0:
            raise ValueError("backoff_base must be a float from 1.0 through 10.0")

    # --- transport -----------------------------------------------------

    def _connect(self):
        if self.client is not None:
            return self.client
        try:
            from google import genai
        except ImportError as error:  # pragma: no cover - environment-dependent
            raise GeminiUnavailable(f"google-genai is not installed: {error}") from error
        project = self.project or self._default_project()
        if not project:
            raise GeminiUnavailable(
                "no GCP project; set `project=` or run `gcloud config set project`")
        location = "global" if self.model_id in GLOBAL_ONLY_MODELS else self.location
        if location != self.location:
            self.location = location
        self.client = genai.Client(vertexai=True, project=project, location=location)
        return self.client

    @staticmethod
    def _default_project() -> str:
        import os
        import subprocess

        for key in ("GOOGLE_CLOUD_PROJECT", "GCLOUD_PROJECT"):
            value = os.environ.get(key, "").strip()
            if value:
                return value
        try:
            done = subprocess.run(["gcloud", "config", "get-value", "project"],
                                  capture_output=True, text=True, timeout=20, check=False)
        except (OSError, subprocess.SubprocessError):
            return ""
        value = (done.stdout or "").strip()
        return "" if value in ("", "(unset)") else value

    # --- the ModelClient contract --------------------------------------

    @staticmethod
    def _retryable(error: Exception) -> bool:
        """Whether the provider is saying "later" rather than "no"."""
        text = str(error).lower()
        return ("429" in text or "resource_exhausted" in text or "rate limit" in text
                or "503" in text or "unavailable" in text)

    def complete(self, prompt: str, config: ModelConfig, *, images=()) -> ModelResponse:
        started = time.perf_counter()
        try:
            parts = self._parts(prompt, images)
            client = self._connect()
        except GeminiUnavailable as error:
            return self._record(ModelResponse(
                status=ModelCallStatus.TRANSPORT_ERROR, model_id=self.model_id,
                detail=str(error)[:300], latency_s=time.perf_counter() - started))
        except Exception as error:  # noqa: BLE001 - a bad frame is this call's outcome
            return self._record(self._classify(error, started))
        last = None
        for attempt in range(self.max_retries + 1):
            try:
                response = client.models.generate_content(
                    model=self.model_id, contents=parts,
                    config=self._request_config(config))
            except Exception as error:  # noqa: BLE001 - every failure is a status
                last = error
                if attempt < self.max_retries and self._retryable(error):
                    self.retries += 1
                    time.sleep(self.backoff_base ** attempt)
                    continue
                if self._retryable(error):
                    # Exhausted: say so explicitly, because a throttled run that
                    # reports plain transport errors looks like a weak policy.
                    self.rate_limited += 1
                    return self._record(ModelResponse(
                        status=ModelCallStatus.TRANSPORT_ERROR, model_id=self.model_id,
                        latency_s=time.perf_counter() - started,
                        detail=f"rate limited after {self.max_retries} retries: "
                               f"{type(error).__name__}: {error}"[:300]))
                return self._record(self._classify(error, started))
            return self._record(self._unpack(response, config, started))
        return self._record(self._classify(last or RuntimeError("no attempt ran"), started))

    def _parts(self, prompt: str, images) -> list:
        from google.genai import types

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        parts = [types.Part.from_text(text=prompt)]
        for frame in images:
            parts.append(types.Part.from_bytes(data=encode_frame(frame),
                                               mime_type="image/jpeg"))
        return parts

    def _request_config(self, config: ModelConfig):
        from google.genai import types

        request = types.GenerateContentConfig(
            temperature=config.temperature,
            max_output_tokens=config.max_output_tokens,
            # The provider's own deadline, so a hung call ends as a timeout
            # status rather than stalling an episode.
            http_options=types.HttpOptions(timeout=int(config.timeout_seconds * 1000)),
        )
        # Ask the provider to emit JSON rather than hoping the prompt is obeyed.
        # Without it roughly a third of answers arrived wrapped in prose or a
        # markdown fence and parsed as "response contains no JSON array" — and a
        # refusal holds the previous action, so a third of the policy's
        # decisions were "do again whatever you last did". That is a harness
        # property being measured as a model property.
        if self.json_mode:
            request.response_mime_type = "application/json"
        return request

    def _classify(self, error: Exception, started: float) -> ModelResponse:
        text = str(error)
        lowered = text.lower()
        if "deadline" in lowered or "timeout" in lowered or "504" in lowered:
            status = ModelCallStatus.TIMEOUT
        elif "safety" in lowered or "blocked" in lowered or "permission" in lowered:
            status = ModelCallStatus.REFUSED
        else:
            status = ModelCallStatus.TRANSPORT_ERROR
        return ModelResponse(status=status, model_id=self.model_id,
                             detail=f"{type(error).__name__}: {text}"[:300],
                             latency_s=time.perf_counter() - started)

    def _unpack(self, response, config: ModelConfig, started: float) -> ModelResponse:
        latency = time.perf_counter() - started
        usage = getattr(response, "usage_metadata", None)
        prompt_tokens = getattr(usage, "prompt_token_count", None) if usage else None
        completion_tokens = getattr(usage, "candidates_token_count", None) if usage else None
        text = ""
        try:
            text = response.text or ""
        except Exception:  # noqa: BLE001 - a blocked answer has no text
            text = ""
        if not text.strip():
            # An empty answer is a refusal, not an unparseable chunk: saying so
            # keeps "the model would not answer" apart from "the model answered
            # something invalid", which are different findings.
            return ModelResponse(
                status=ModelCallStatus.REFUSED, model_id=self.model_id,
                prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                usd=self._usd(prompt_tokens, completion_tokens), latency_s=latency,
                detail=f"empty answer; finish_reason={self._finish_reason(response)}")
        if len(text.encode("utf-8")) > config.max_response_bytes:
            return ModelResponse(
                status=ModelCallStatus.OVERSIZE, model_id=self.model_id,
                prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                usd=self._usd(prompt_tokens, completion_tokens), latency_s=latency,
                detail=f"{len(text.encode('utf-8'))} bytes exceeds the configured bound")
        return ModelResponse(
            text=text, status=ModelCallStatus.OK, model_id=self.model_id,
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
            usd=self._usd(prompt_tokens, completion_tokens), latency_s=latency)

    @staticmethod
    def _finish_reason(response) -> str:
        candidates = getattr(response, "candidates", None) or ()
        for candidate in candidates:
            reason = getattr(candidate, "finish_reason", None)
            if reason is not None:
                return str(reason)
        return "unknown"

    def _usd(self, prompt_tokens, completion_tokens) -> float | None:
        """Cost, only when rates were supplied. See the module docstring."""
        if self.usd_per_1m_input is None or self.usd_per_1m_output is None:
            return None
        if prompt_tokens is None or completion_tokens is None:
            return None
        return round(prompt_tokens / 1e6 * self.usd_per_1m_input
                     + completion_tokens / 1e6 * self.usd_per_1m_output, 8)

    def _record(self, response: ModelResponse) -> ModelResponse:
        self.calls += 1
        self.statuses.append(response.status.value)
        if response.status is not ModelCallStatus.OK:
            self.refusals += 1
        self.tokens_in += int(response.prompt_tokens or 0)
        self.tokens_out += int(response.completion_tokens or 0)
        return response

    def describe(self) -> dict:
        return {"client": "gemini-vertex", "model_id": self.model_id,
                "location": self.location, "calls": self.calls,
                "refusals": self.refusals, "tokens_in": self.tokens_in,
                "tokens_out": self.tokens_out,
                "priced": self.usd_per_1m_input is not None,
                "retries": self.retries, "rate_limited": self.rate_limited,
                "gated_alternatives": list(GATED_MODELS)}
