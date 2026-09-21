"""Deterministic 3D-Speaker diarization extracted from the original service."""

from __future__ import annotations

from dataclasses import asdict
import gc
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np

from .config import AppConfig
from .errors import ModelError, TranscriptionError
from .schemas import SpeakerSegment


def seeded_cluster(
    cluster: Callable[..., Any],
    embeddings: np.ndarray,
    speaker_num: int | None,
    seed: int,
) -> np.ndarray:
    np.random.seed(seed)
    return np.asarray(
        cluster(embeddings, speaker_num=speaker_num),
        dtype="int64",
    )


def relabel_by_first_appearance(labels: list[int] | np.ndarray) -> list[int]:
    mapping: dict[int, int] = {}
    output: list[int] = []
    for raw_label in labels:
        label = int(raw_label)
        if label not in mapping:
            mapping[label] = len(mapping) + 1
        output.append(mapping[label])
    return output


def _normalized_embeddings(embeddings: np.ndarray) -> np.ndarray:
    values = np.asarray(embeddings, dtype="float32")
    return values / (np.linalg.norm(values, axis=1, keepdims=True) + 1e-8)


def merge_oversegmented_clusters(
    embeddings: np.ndarray,
    cluster_labels: np.ndarray | list[int],
    threshold: float,
    min_fraction: float = 0.6,
    relative_ratio: float = 0.55,
    major_cluster_min_fraction: float = 0.10,
    major_cluster_merge_threshold: float = 0.88,
    min_major_items: int = 16,
) -> list[int]:
    """Merge over-split clusters whose similarity is high AND stable.

    A bare mean threshold is fragile: recordings with two similar speakers
    sit just above the line, while genuine over-segmentation shows broad
    support across all cross-cluster pairs.  A candidate pair therefore
    must additionally show that at least ``min_fraction`` of its cross
    pairs reach ``threshold`` and that the cross mean is at least
    ``relative_ratio`` of the clusters' own within-cluster means.
    """

    normalized = _normalized_embeddings(embeddings)
    labels = [int(label) for label in cluster_labels]
    members: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        members.setdefault(label, []).append(index)
    groups = {label: list(indexes) for label, indexes in members.items()}
    total_items = max(1, len(labels))
    while len(groups) > 1:
        ids = list(groups)
        best_pair: tuple[int, int] | None = None
        best_value = -1.0
        for left_index, left in enumerate(ids):
            for right in ids[left_index + 1:]:
                len_left = len(groups[left])
                len_right = len(groups[right])
                is_major_pair = (
                    len_left >= min_major_items
                    and len_right >= min_major_items
                    and (len_left / total_items) >= major_cluster_min_fraction
                    and (len_right / total_items) >= major_cluster_min_fraction
                )
                if is_major_pair:
                    centroid_left = normalized[groups[left]].mean(axis=0)
                    centroid_left = centroid_left / (np.linalg.norm(centroid_left) + 1e-8)
                    centroid_right = normalized[groups[right]].mean(axis=0)
                    centroid_right = centroid_right / (np.linalg.norm(centroid_right) + 1e-8)
                    centroid_sim = float(centroid_left @ centroid_right)
                    if centroid_sim < major_cluster_merge_threshold:
                        continue

                cross = normalized[groups[left]] @ normalized[groups[right]].T
                value = float(cross.mean())
                if value < threshold:
                    continue
                if float((cross >= threshold).mean()) < min_fraction:
                    continue
                within = []
                for group in (groups[left], groups[right]):
                    if len(group) > 1:
                        matrix = normalized[group] @ normalized[group].T
                        within.append(
                            float(matrix[np.triu_indices(len(group), 1)].mean())
                        )
                if within and value < relative_ratio * min(within):
                    continue
                if value > best_value:
                    best_value = value
                    best_pair = (left, right)
        if best_pair is None:
            break
        keep, drop = best_pair
        groups[keep].extend(groups.pop(drop))

    by_embedding: dict[int, int] = {}
    ordered_groups = sorted(groups.values(), key=min)
    for new_label, indexes in enumerate(ordered_groups):
        for index in indexes:
            by_embedding[index] = new_label
    return [by_embedding[index] for index in range(len(labels))]


def _merge_adjacent(
    segments: list[tuple[float, float, int]],
    max_gap_seconds: float,
) -> list[tuple[float, float, int]]:
    merged: list[list[float | int]] = []
    for start, end, speaker in sorted(segments):
        if end <= start:
            continue
        if (
            merged
            and int(merged[-1][2]) == int(speaker)
            and start - float(merged[-1][1]) <= max_gap_seconds
        ):
            merged[-1][1] = max(float(merged[-1][1]), end)
        else:
            merged.append([start, end, int(speaker)])
    return [
        (float(start), float(end), int(speaker))
        for start, end, speaker in merged
    ]


def smooth_segments(
    segments: list[tuple[float, float, int]],
    max_middle_seconds: float = 0.8,
    max_middle_ratio: float = 0.25,
    max_gap_seconds: float = 0.2,
) -> list[tuple[float, float, int]]:
    normalized = _merge_adjacent(segments, 0.0)
    changed = True
    while changed:
        changed = False
        output: list[tuple[float, float, int]] = []
        index = 0
        while index < len(normalized):
            if index + 2 >= len(normalized):
                output.append(normalized[index])
                index += 1
                continue
            left, middle, right = normalized[index:index + 3]
            middle_seconds = middle[1] - middle[0]
            window_seconds = max(right[1] - left[0], 1e-8)
            if (
                left[2] == right[2]
                and middle[2] != left[2]
                and middle_seconds <= max_middle_seconds
                and middle_seconds / window_seconds <= max_middle_ratio
                and max(0.0, middle[0] - left[1]) <= max_gap_seconds
                and max(0.0, right[0] - middle[1]) <= max_gap_seconds
            ):
                output.append((left[0], right[1], left[2]))
                index += 3
                changed = True
            else:
                output.append(left)
                index += 1
        normalized = _merge_adjacent(output, max_gap_seconds)
    return normalized


def _compress_overlapping_chunks(
    segments: list[tuple[float, float, int]],
) -> list[tuple[float, float, int]]:
    compressed: list[list[float | int]] = []
    for raw_start, raw_end, raw_speaker in segments:
        start, end, speaker = float(raw_start), float(raw_end), int(raw_speaker)
        if not compressed:
            compressed.append([start, end, speaker])
            continue
        previous = compressed[-1]
        if speaker == int(previous[2]) and start <= float(previous[1]):
            previous[1] = max(float(previous[1]), end)
            continue
        if start < float(previous[1]):
            boundary = (float(previous[1]) + start) / 2
            previous[1] = boundary
            start = boundary
        compressed.append([start, end, speaker])
    return [
        (float(start), float(end), int(speaker))
        for start, end, speaker in compressed
        if float(end) > float(start)
    ]


class SpeakerDiarizer:
    """3D-Speaker adapter with no dependency on the legacy application."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.repo_dir = (
            config.paths.model_dir / "3d-speaker" / "3D-Speaker"
        ).resolve()
        if not self.repo_dir.is_dir():
            raise ModelError(f"3D-Speaker repository not found: {self.repo_dir}")
        if str(self.repo_dir) not in sys.path:
            sys.path.insert(0, str(self.repo_dir))
        self.model_cache_dir = (
            config.paths.model_dir / "modelscope"
        ).resolve()
        self._load_runtime()

    @classmethod
    def from_config(cls, config: AppConfig) -> "SpeakerDiarizer":
        return cls(config)

    def _load_runtime(self) -> None:
        try:
            import soundfile as sf
            import torch
            import torchaudio.functional as audio_functional
            from modelscope.pipelines import pipeline
            from modelscope.utils.constant import Tasks
            from speakerlab.utils.builder import build
            from speakerlab.utils.config import Config
            from speakerlab.utils.utils import (
                circle_pad,
                silent_print,
            )
        except Exception as exc:
            raise ModelError(f"unable to import 3D-Speaker runtime: {exc}") from exc

        self.torch = torch
        self.sf = sf
        self.audio_functional = audio_functional
        self.pipeline_factory = pipeline
        self.Tasks = Tasks
        self.build = build
        self.Config = Config
        self.circle_pad = circle_pad
        self.silent_print = silent_print
        self.device = torch.device(self.config.general.device)
        self.embedding_model, self.feature_extractor = self._load_embedding_model()
        self.vad_model = self._load_vad_model()
        self.cluster = self._load_cluster()
        self.sample_rate = self.feature_extractor.sample_rate
        self.batch_size = 64

    def _load_embedding_model(self):
        specification = {
            "model_id": "iic/speech_campplus_sv_zh_en_16k-common_advanced",
            "revision": "v1.0.0",
            "model_ckpt": "campplus_cn_en_common.pt",
            "embedding_model": {
                "obj": "speakerlab.models.campplus.DTDNN.CAMPPlus",
                "args": {"feat_dim": 80, "embedding_size": 192},
            },
            "feature_extractor": {
                "obj": "speakerlab.process.processor.FBank",
                "args": {
                    "n_mels": 80,
                    "sample_rate": 16000,
                    "mean_nor": True,
                },
            },
        }
        cache_dir = (
            self.model_cache_dir
            / "iic"
            / "speech_campplus_sv_zh_en_16k-common_advanced"
        )
        if not cache_dir.is_dir():
            raise ModelError(
                f"diarization embedding model not found: {cache_dir}"
            )
        config = self.Config(specification)
        feature_extractor = self.build("feature_extractor", config)
        embedding_model = self.build("embedding_model", config)
        state = self.torch.load(
            str(cache_dir / specification["model_ckpt"]),
            map_location="cpu",
        )
        embedding_model.load_state_dict(state)
        embedding_model.eval().to(self.device)
        return embedding_model, feature_extractor

    def _load_vad_model(self):
        cache_dir = (
            self.model_cache_dir
            / "iic"
            / "speech_fsmn_vad_zh-cn-16k-common-pytorch"
        )
        if not cache_dir.is_dir():
            raise ModelError(f"diarization VAD model not found: {cache_dir}")
        device = str(self.device)
        with self.silent_print():
            return self.pipeline_factory(
                task=self.Tasks.voice_activity_detection,
                model=str(cache_dir),
                device=device,
                **asdict(self.config.vad),
                disable_pbar=True,
                disable_update=True,
            )

    def _load_cluster(self):
        diarization = self.config.diarization
        specification = {
            "cluster": {
                "obj": "speakerlab.process.cluster.CommonClustering",
                "args": {
                    "cluster_type": "spectral",
                    "mer_cos": None,
                    "min_num_spks": diarization.min_speakers,
                    "max_num_spks": diarization.max_speakers,
                    "min_cluster_size": 4,
                    "oracle_num": diarization.speaker_count,
                    "pval": 0.012,
                },
            }
        }
        return self.build("cluster", self.Config(specification))

    def _load_audio(self, wav_path: Path):
        data, sample_rate = self.sf.read(
            str(wav_path),
            dtype="float32",
            always_2d=True,
        )
        wav = self.torch.from_numpy(data.T).mean(dim=0, keepdim=True)
        target_rate = getattr(self, "sample_rate", 16000)
        if sample_rate != target_rate:
            wav = self.audio_functional.resample(
                wav,
                orig_freq=sample_rate,
                new_freq=target_rate,
            )
        return wav

    def _vad(self, wav) -> list[tuple[float, float]]:
        kwargs = asdict(self.config.vad)
        results = self.vad_model(wav[0], **kwargs)[0]
        return [
            (float(start) / 1000, float(end) / 1000)
            for start, end in results.get("value", [])
            if end > start
        ]

    def _chunks(self, start: float, end: float) -> list[tuple[float, float]]:
        duration = self.config.diarization.chunk_seconds
        step = self.config.diarization.chunk_step_seconds
        chunks: list[tuple[float, float]] = []
        current = start
        while current + duration < end + step:
            chunks.append((current, min(current + duration, end)))
            current += step
        return chunks

    def _extract_embeddings(self, chunks, wav) -> np.ndarray:
        waveforms = [
            wav[0, int(start * self.sample_rate):int(end * self.sample_rate)]
            for start, end in chunks
        ]
        max_length = max(item.shape[0] for item in waveforms)
        batches = [
            self.circle_pad(item, max_length)
            for item in waveforms
        ]
        stacked = self.torch.stack(batches).unsqueeze(1)
        embeddings = []
        with self.torch.no_grad():
            for start in range(0, len(chunks), self.batch_size):
                waveform_batch = stacked[start:start + self.batch_size].to(
                    self.device
                )
                features = self.torch.vmap(self.feature_extractor)(waveform_batch)
                embeddings.append(self.embedding_model(features).cpu())
        return self.torch.cat(embeddings, dim=0).numpy()

    def diarize(self, wav_path: Path) -> tuple[SpeakerSegment, ...]:
        try:
            wav = self._load_audio(wav_path)
            vad_segments = self._vad(wav)
            chunks = [
                chunk
                for start, end in vad_segments
                for chunk in self._chunks(start, end)
            ]
            if not chunks:
                return ()
            embeddings = self._extract_embeddings(chunks, wav)
            labels = seeded_cluster(
                self.cluster,
                embeddings,
                self.config.diarization.speaker_count,
                self.config.general.random_seed,
            )
            labels = merge_oversegmented_clusters(
                embeddings,
                labels,
                self.config.diarization.merge_cosine,
                self.config.diarization.merge_min_fraction,
                self.config.diarization.merge_relative_ratio,
                major_cluster_min_fraction=self.config.diarization.major_cluster_min_fraction,
                major_cluster_merge_threshold=self.config.diarization.major_cluster_merge_threshold,
            )
            labels = relabel_by_first_appearance(labels)
            raw_segments = [
                (start, end, speaker)
                for (start, end), speaker in zip(chunks, labels)
            ]
            smoothed = smooth_segments(_compress_overlapping_chunks(raw_segments))
            return tuple(
                SpeakerSegment(
                    speaker_id=speaker,
                    start_ms=int(round(start * 1000)),
                    end_ms=int(round(end * 1000)),
                )
                for start, end, speaker in smoothed
                if end > start
            )
        except (ModelError, TranscriptionError):
            raise
        except Exception as exc:
            raise TranscriptionError(f"speaker diarization failed: {exc}") from exc

    def speaker_centroids(
        self,
        wav_path: Path,
        segments: tuple[SpeakerSegment, ...],
    ) -> dict[int, np.ndarray]:
        wav = self._load_audio(wav_path)
        output: dict[int, np.ndarray] = {}
        for speaker in sorted({item.speaker_id for item in segments}):
            chunks = [
                (item.start_ms / 1000, item.end_ms / 1000)
                for item in segments
                if item.speaker_id == speaker
            ]
            if not chunks:
                continue
            embeddings = _normalized_embeddings(
                self._extract_embeddings(chunks, wav)
            )
            centroid = embeddings.mean(axis=0)
            output[speaker] = centroid / (np.linalg.norm(centroid) + 1e-8)
        return output

    def embedding_for_audio(self, wav_path: Path) -> np.ndarray:
        wav = self._load_audio(wav_path)
        duration = wav.shape[-1] / self.sample_rate
        if duration <= 0:
            raise TranscriptionError(f"empty separated track: {wav_path}")
        embedding = self._extract_embeddings([(0.0, duration)], wav)[0]
        return embedding / (np.linalg.norm(embedding) + 1e-8)

    def resource_inventory(self) -> dict[str, object]:
        """Describe inspectable 3D-Speaker modules without running inference."""

        from .resource_profile import module_inventory

        embedding = module_inventory("embedding", self.embedding_model)
        vad_candidate = getattr(self.vad_model, "model", None)
        if (
            vad_candidate is None
            or not hasattr(vad_candidate, "parameters")
            or not hasattr(vad_candidate, "buffers")
        ):
            vad: dict[str, object] = {"available": False}
        else:
            vad = module_inventory("vad", vad_candidate)
        return {"embedding": embedding, "vad": vad}

    def diarize_with_resource_metrics(
        self,
        wav_path: Path,
    ) -> tuple[tuple[SpeakerSegment, ...], dict[str, int | None]]:
        """Run diarization and capture PyTorch memory only for opt-in profiling."""

        torch = self.torch
        if not torch.cuda.is_available():
            return self.diarize(wav_path), {
                "allocated_before_bytes": None,
                "reserved_before_bytes": None,
                "allocated_after_bytes": None,
                "reserved_after_bytes": None,
                "peak_allocated_bytes": None,
                "peak_reserved_bytes": None,
            }
        torch.cuda.synchronize()
        before_allocated = int(torch.cuda.memory_allocated())
        before_reserved = int(torch.cuda.memory_reserved())
        torch.cuda.reset_peak_memory_stats()
        segments = self.diarize(wav_path)
        torch.cuda.synchronize()
        return segments, {
            "allocated_before_bytes": before_allocated,
            "reserved_before_bytes": before_reserved,
            "allocated_after_bytes": int(torch.cuda.memory_allocated()),
            "reserved_after_bytes": int(torch.cuda.memory_reserved()),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }

    def close(self) -> None:
        if getattr(self, "_closed", False):
            return
        self._closed = True
        for name in (
            "embedding_model",
            "vad_model",
            "feature_extractor",
            "cluster",
        ):
            if hasattr(self, name):
                setattr(self, name, None)
        gc.collect()
        torch = getattr(self, "torch", None)
        device = getattr(self, "device", None)
        if (
            torch is not None
            and getattr(device, "type", None) == "cuda"
        ):
            torch.cuda.empty_cache()

    def __enter__(self) -> "SpeakerDiarizer":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
