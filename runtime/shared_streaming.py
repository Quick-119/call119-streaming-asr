"""Concurrent call sessions using one frozen SimulStreaming Whisper model.

Operations are serialized on one model. Each call retains its own decoder,
audio, and Silero VAD state; callers may submit chunks in any order.
"""
import contextlib
import hashlib
import json
import sys
import threading
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch


STATE_FIELDS = (
    "segments", "tokens", "context", "last_attend_frame", "detected_language",
    "log_segments", "logdir_i",
)
SUPPORTED_CHUNK_SECONDS = (0.5, 1.0, 2.0)


class SharedStreamingASR:
    def __init__(self, bundle, max_sessions=20, decoder_self_sdpa=False,
                 chunk_sec=None, audio_max_len=None, fp16_weights=False,
                 compile_encoder=False, compile_decoder=False, beam=None,
                 frame_threshold=None,
                 skip_debug_token_formatting=False, guard_debug_logs=False,
                 sparse_alignment_hook=False, cross_sdpa_alignment=False,
                 strict_chunk_cadence=False, quantization="none", quant_scope="all",
                 vendor_override=None, no_repeat_ngram_size=None):
        self.bundle = Path(bundle).resolve()
        manifest = json.loads((self.bundle / "BUNDLE.json").read_text())
        if manifest["status"] != "frozen":
            raise ValueError("Expected a frozen bundle")
        cfg = json.loads((self.bundle / "STREAMING_CONFIG.json").read_text())
        for name, expected_sha256 in manifest["files_sha256"].items():
            path = (self.bundle / name).resolve()
            if not path.is_relative_to(self.bundle):
                raise ValueError("Bundle path escapes its root")
            h = hashlib.sha256()
            with path.open("rb") as f:
                for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
                    h.update(block)
            if h.hexdigest() != expected_sha256:
                raise ValueError("Frozen bundle hash mismatch: " + name)
        if chunk_sec is not None:
            if chunk_sec <= 0 or round(chunk_sec * 16000) != chunk_sec * 16000:
                raise ValueError("chunk_sec must be positive and sample aligned")
            cfg["chunk_sec"] = chunk_sec
        if audio_max_len is not None:
            if audio_max_len <= 0 or audio_max_len > 30:
                raise ValueError("audio_max_len must be in (0, 30]")
            cfg["audio_max_len"] = audio_max_len
        if beam is not None:
            if type(beam) is not int or beam < 1 or beam > 5:
                raise ValueError("beam must be an integer in [1, 5]")
            cfg["beam"] = beam
        if frame_threshold is not None:
            if type(frame_threshold) is not int or frame_threshold < 1:
                raise ValueError("frame_threshold must be a positive integer")
            cfg["frame_threshold"] = frame_threshold
        if no_repeat_ngram_size is not None:
            if type(no_repeat_ngram_size) is not int or no_repeat_ngram_size < 0:
                raise ValueError("no_repeat_ngram_size must be a nonnegative integer")
            cfg["no_repeat_ngram_size"] = no_repeat_ngram_size
        self.cfg = cfg
        self.strict_chunk_cadence = strict_chunk_cadence
        self.max_sessions = max_sessions
        self.lock = threading.RLock()
        self.sessions = {}
        vendor = Path(vendor_override).resolve() if vendor_override else self.bundle / cfg["vendor"]
        if not (vendor / "simulstreaming_whisper.py").is_file():
            raise ValueError("Invalid SimulStreaming vendor override")
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(vendor))
        sys.path.insert(0, str(self.bundle / "scripts"))
        from simulstreaming_whisper import SimulWhisperASR, SimulWhisperOnline
        from simulstreaming.whisper.whisper_streaming.vac_online_processor import VACOnlineASRProcessor
        from streaming_alignment_heads import apply_alignment_heads
        self.online_class = SimulWhisperOnline
        self.vac_class = VACOnlineASRProcessor
        self.vad_path = self.bundle / cfg["vad_model"]
        torch.set_num_threads(cfg.get("cpu_threads", 4))
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.asr = SimulWhisperASR(
            language="ja", model_path=str(self.bundle / cfg["checkpoint"]), cif_ckpt_path=None,
            frame_threshold=cfg["frame_threshold"], audio_max_len=cfg["audio_max_len"],
            audio_min_len=cfg["audio_min_len"], segment_length=cfg["chunk_sec"],
            beams=cfg["beam"], task="transcribe", decoder_type="greedy" if cfg["beam"] == 1 else "beam",
            never_fire=cfg["never_fire"], init_prompt=None, static_init_prompt=None,
            max_context_tokens=cfg["max_context_tokens"], logdir=None,
        )
        self.asr.model.cfg.no_repeat_ngram_size = cfg.get("no_repeat_ngram_size", 0)
        self.head_audit = apply_alignment_heads(self.asr.model, cfg.get("alignment_heads_override"))
        if sparse_alignment_hook and cross_sdpa_alignment:
            raise ValueError("Select either sparse hook or cross SDPA alignment")
        if sparse_alignment_hook:
            from sparse_alignment_hook import install
            self.sparse_hook_audit = install(self.asr.model)
        if cross_sdpa_alignment:
            from cross_sdpa_alignment import install
            self.cross_sdpa_audit = install(self.asr.model)
        if guard_debug_logs:
            from guard_debug_logs import install
            self.guarded_debug_calls = install(self.asr.model)
        if skip_debug_token_formatting:
            # This method only feeds disabled DEBUG logs, but its original
            # implementation copies GPU tokens to CPU on every decode step.
            self.asr.model.debug_print_tokens = lambda tokens: None
        if decoder_self_sdpa:
            # AlignAtt reads decoder cross-attention weights only. Self-attention
            # does not expose those weights, so it can use fused SDPA.
            for block in self.asr.model.model.decoder.blocks:
                block.attn.encoder_only_sdpa = True
        if fp16_weights:
            self.asr.model.model.half()
        if quantization != "none":
            from torchao.quantization import (quantize_, Int8WeightOnlyConfig,
                                              Int8DynamicActivationInt8WeightConfig,
                                              Float8WeightOnlyConfig)
            import torch.nn as nn
            configs = {
                "int8_weight_only": Int8WeightOnlyConfig(),
                "int8_dynamic": Int8DynamicActivationInt8WeightConfig(),
                "float8_weight_only": Float8WeightOnlyConfig(),
            }
            if quantization not in configs or quant_scope not in ("all", "encoder", "decoder"):
                raise ValueError("Unsupported quantization or scope")
            def selected(module, fqn):
                return isinstance(module, nn.Linear) and (quant_scope == "all" or fqn.startswith(quant_scope + "."))
            quantize_(self.asr.model.model, configs[quantization], filter_fn=selected)
        self.quantization = quantization
        self.quant_scope = quant_scope
        if compile_encoder:
            self.asr.model.model.encoder = torch.compile(
                self.asr.model.model.encoder, mode="reduce-overhead", fullgraph=False)
        if compile_decoder:
            decoder = self.asr.model.model.decoder
            decoder.forward = torch.compile(
                decoder.forward, mode="default", fullgraph=False)
        self.asr.model.model.eval()
        dtype = {"float32": None, "autocast_float16": torch.float16,
                 "autocast_bfloat16": torch.bfloat16}[cfg["dtype"]]
        self.compute_context = lambda: (torch.autocast("cuda", dtype=dtype) if dtype else contextlib.nullcontext())
        with torch.inference_mode(), self.compute_context():
            self.asr.warmup(np.zeros(16000, dtype=np.float32))

    def _snapshot(self):
        m = self.asr.model
        return {key: getattr(m, key) for key in STATE_FIELDS}

    def _restore(self, state):
        m = self.asr.model
        for key, value in state.items():
            setattr(m, key, value)
        # Decoder caches are only valid inside one infer() call. Its normal
        # completion clears them; a fresh dict also prevents idle sessions
        # from aliasing a cache populated by another session.
        m.dec_attns = []
        m.kv_cache = {}
        if m.decoder_type == "beam":
            m.inference.kv_cache = m.kv_cache
            m.token_decoder.reset()

    def _load_vad(self, *args, **kwargs):
        if kwargs.get("repo_or_dir") != "snakers4/silero-vad" or kwargs.get("model") != "silero_vad":
            raise ValueError("Unexpected VAD request")
        return torch.jit.load(str(self.vad_path), map_location="cpu").eval(), None

    def open(self, call_id, chunk_sec=None):
        with self.lock:
            if call_id in self.sessions or len(self.sessions) >= self.max_sessions:
                raise ValueError("Duplicate call or session limit reached")
            if chunk_sec is None:
                chunk_sec = self.cfg["chunk_sec"]
            if chunk_sec not in SUPPORTED_CHUNK_SECONDS:
                raise ValueError("chunk_sec must be 0.5, 1.0, or 2.0")
            # The online wrapper calls refresh_segment(); save its private state
            # before opening another call on the same model.
            with patch("torch.hub.load", side_effect=self._load_vad):
                vac = self.vac_class(chunk_sec, self.online_class(self.asr))
            if self.strict_chunk_cadence:
                # The frozen VAC checks buffer_size > 16000 * online_chunk_size.
                # One-sample epsilon makes a complete nominal chunk eligible.
                vac.online_chunk_size = chunk_sec - 1 / 16000
            vac.vac.threshold = self.cfg["vac_threshold"]
            vac.vac.min_silence_samples = 16000 * self.cfg["vac_silence_ms"] / 1000
            vac.vac.speech_pad_samples = 16000 * self.cfg["vac_pad_ms"] / 1000
            self.sessions[call_id] = {"vac": vac, "model": self._snapshot(),
                                      "chunk_sec": chunk_sec, "closed": False}

    def set_chunk_sec(self, call_id, chunk_sec):
        if chunk_sec not in SUPPORTED_CHUNK_SECONDS:
            raise ValueError("chunk_sec must be 0.5, 1.0, or 2.0")
        with self.lock:
            session = self.sessions[call_id]
            session["vac"].online_chunk_size = (chunk_sec - 1 / 16000
                                                 if self.strict_chunk_cadence else chunk_sec)
            session["chunk_sec"] = chunk_sec

    def _step(self, call_id, audio=None, final=False):
        with self.lock:
            session = self.sessions[call_id]
            if session["closed"]:
                raise ValueError("Call already closed")
            self._restore(session["model"])
            try:
                with torch.inference_mode(), self.compute_context():
                    if audio is not None:
                        audio = np.asarray(audio, dtype=np.float32)
                        if audio.ndim != 1 or not np.isfinite(audio).all():
                            raise ValueError("Expected finite mono 16 kHz PCM")
                        session["vac"].insert_audio_chunk(audio)
                    result = session["vac"].finish() if final else session["vac"].process_iter()
            except Exception:
                # VAC may already have consumed this chunk. Do not allow a
                # retry to silently process an inconsistent call state.
                del self.sessions[call_id]
                raise
            session["model"] = self._snapshot()
            if final:
                session["closed"] = True
                del self.sessions[call_id]
            return result

    def push(self, call_id, pcm16k):
        return self._step(call_id, audio=pcm16k)

    def finish(self, call_id):
        return self._step(call_id, final=True)

    def abort(self, call_id):
        with self.lock:
            del self.sessions[call_id]
