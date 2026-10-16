import numpy as np
import torch

from funasr_transcriber.speaker_diarization import (
    SpeakerDiarizer,
    compute_adaptive_eigengaps,
    merge_oversegmented_clusters,
    relabel_by_first_appearance,
    seeded_cluster,
    smooth_segments,
)


def test_seeded_cluster_reseeds_each_call():
    seen = []

    def cluster(embeddings, speaker_num):
        seen.append(np.random.random())
        return np.array([1, 0, 1])

    embeddings = np.eye(3, dtype="float32")
    first = seeded_cluster(cluster, embeddings, speaker_num=None, seed=0)
    second = seeded_cluster(cluster, embeddings, speaker_num=None, seed=0)

    assert seen[0] == seen[1]
    assert first.tolist() == second.tolist()


def test_labels_follow_first_speech_time():
    assert relabel_by_first_appearance([7, 4, 7, 9]) == [1, 2, 1, 3]


def test_merge_combines_similar_oversplit_clusters():
    embeddings = np.array(
        [[1.0, 0.0], [0.98, 0.02], [0.99, -0.01], [0.97, 0.03]],
        dtype="float32",
    )
    labels = np.array([0, 0, 1, 1])

    merged = merge_oversegmented_clusters(embeddings, labels, threshold=0.95)

    assert len(set(merged)) == 1


def test_merge_keeps_distinct_clusters():
    embeddings = np.array(
        [[1.0, 0.0], [0.98, 0.02], [0.0, 1.0], [0.02, 0.98]],
        dtype="float32",
    )
    labels = np.array([0, 0, 1, 1])

    merged = merge_oversegmented_clusters(embeddings, labels, threshold=0.50)

    assert len(set(merged)) == 2


def test_merge_rejects_unstable_cross_similarity():
    # Half the cross pairs are near-orthogonal: the mean passes 0.23
    # but only 50% of pairs reach it, below the 0.6 stability floor.
    embeddings = np.array(
        [[1.0, 0.0], [0.99, 0.141], [1.0, 0.0], [0.0, 1.0]],
        dtype="float32",
    )
    labels = np.array([0, 0, 1, 1])

    blocked = merge_oversegmented_clusters(embeddings, labels, threshold=0.23)
    assert len(set(blocked)) == 2

    allowed = merge_oversegmented_clusters(
        embeddings, labels, threshold=0.23, min_fraction=0.4
    )
    assert len(set(allowed)) == 1


def test_merge_rejects_similarity_below_within_cluster_ratio():
    # Cross pairs consistently reach 0.31 (fraction 1.0) but each cluster
    # is internally near-identical, so 0.31 is far below 0.55 x within.
    embeddings = np.array(
        [[1.0, 0.0], [1.0, 0.0], [0.309, 0.951], [0.309, 0.951]],
        dtype="float32",
    )
    labels = np.array([0, 0, 1, 1])

    blocked = merge_oversegmented_clusters(embeddings, labels, threshold=0.23)
    assert len(set(blocked)) == 2

    allowed = merge_oversegmented_clusters(
        embeddings, labels, threshold=0.23, relative_ratio=0.3
    )
    assert len(set(allowed)) == 1


def test_smoothing_collapses_short_aba_flicker():
    result = smooth_segments(
        [(0.0, 2.0, 1), (2.0, 2.5, 2), (2.5, 4.0, 1)],
        max_middle_seconds=0.8,
        max_middle_ratio=0.25,
        max_gap_seconds=0.2,
    )

    assert result == [(0.0, 4.0, 1)]


def test_close_releases_owned_cuda_models_and_is_idempotent():
    calls = []

    class Cuda:
        @staticmethod
        def empty_cache():
            calls.append("empty_cache")

    class Torch:
        cuda = Cuda()

    class Device:
        type = "cuda"

    diarizer = object.__new__(SpeakerDiarizer)
    diarizer.torch = Torch()
    diarizer.device = Device()
    diarizer.embedding_model = object()
    diarizer.vad_model = object()
    diarizer.feature_extractor = object()
    diarizer.cluster = object()

    diarizer.close()
    diarizer.close()

    assert diarizer.embedding_model is None
    assert diarizer.vad_model is None
    assert calls == ["empty_cache"]


def test_speaker_inventory_reports_embedding_and_uninspectable_vad():
    diarizer = object.__new__(SpeakerDiarizer)
    diarizer.embedding_model = torch.nn.Linear(2, 1)
    diarizer.vad_model = object()

    inventory = diarizer.resource_inventory()

    assert inventory["embedding"]["parameter_count"] == 3
    assert inventory["vad"] == {"available": False}


def test_merge_protects_major_clusters_from_aggressive_merging():
    # Two large clusters (20 items each) with centroid cosine ~ 0.82.
    # Without major cluster guard, a liberal threshold/ratio would merge them.
    # Major cluster guard blocks merge because 0.82 < 0.88.
    rng = np.random.RandomState(42)
    c0 = np.tile([1.0, 0.0], (20, 1)) + rng.normal(0, 0.02, (20, 2))
    c1 = np.tile([0.82, 0.57], (20, 1)) + rng.normal(0, 0.02, (20, 2))
    embeddings = np.vstack([c0, c1]).astype("float32")
    labels = np.array([0] * 20 + [1] * 20)

    merged = merge_oversegmented_clusters(
        embeddings,
        labels,
        threshold=0.23,
        relative_ratio=0.55,
        major_cluster_min_fraction=0.10,
        major_cluster_merge_threshold=0.88,
        min_major_items=16,
    )
    assert len(set(merged)) == 2


def test_merge_allows_major_clusters_if_truly_identical():
    # Two large clusters with near-identical centroids (cosine ~ 0.99 >= 0.88).
    rng = np.random.RandomState(42)
    c0 = np.tile([1.0, 0.0], (20, 1)) + rng.normal(0, 0.01, (20, 2))
    c1 = np.tile([0.999, 0.01], (20, 1)) + rng.normal(0, 0.01, (20, 2))
    embeddings = np.vstack([c0, c1]).astype("float32")
    labels = np.array([0] * 20 + [1] * 20)

    merged = merge_oversegmented_clusters(
        embeddings,
        labels,
        threshold=0.23,
        relative_ratio=0.55,
        major_cluster_min_fraction=0.10,
        major_cluster_merge_threshold=0.88,
        min_major_items=16,
    )
    assert len(set(merged)) == 1


def test_merge_absorbs_minor_oversegmented_cluster():
    # Cluster 0 is major (30 items), cluster 1 is minor fragment (4 items).
    # Even though centroid cosine is 0.82 < 0.88, cluster 1 is minor, so it is absorbed.
    rng = np.random.RandomState(42)
    c0 = np.tile([1.0, 0.0], (30, 1)) + rng.normal(0, 0.02, (30, 2))
    c1 = np.tile([0.82, 0.57], (4, 1)) + rng.normal(0, 0.02, (4, 2))
    embeddings = np.vstack([c0, c1]).astype("float32")
    labels = np.array([0] * 30 + [1] * 4)

    merged = merge_oversegmented_clusters(
        embeddings,
        labels,
        threshold=0.23,
        relative_ratio=0.55,
        major_cluster_min_fraction=0.10,
        major_cluster_merge_threshold=0.88,
        min_major_items=16,
    )
    assert len(set(merged)) == 1



def test_adaptive_eigengaps_selects_expected_speaker_counts():
    # Multi-speaker cases with scale drift
    lambdas_2spk = np.array([0.0, 0.451, 1.155, 1.669, 2.420, 2.709, 2.762])
    gaps_2spk = compute_adaptive_eigengaps(lambdas_2spk, min_num_spks=1)
    assert int(np.argmax(gaps_2spk)) + 1 == 2

    lambdas_3spk = np.array([0.0, 0.858, 1.122, 1.910, 3.242, 4.033, 4.286])
    gaps_3spk = compute_adaptive_eigengaps(lambdas_3spk, min_num_spks=1)
    assert int(np.argmax(gaps_3spk)) + 1 == 3

    # Single-speaker case
    lambdas_1spk = np.array([0.0, 1.445, 1.609, 1.863, 2.050, 2.100])
    gaps_1spk = compute_adaptive_eigengaps(lambdas_1spk, min_num_spks=1)
    assert int(np.argmax(gaps_1spk)) + 1 == 1


def test_adaptive_eigengaps_handles_short_arrays():
    assert compute_adaptive_eigengaps([]) == [0.0]
    assert compute_adaptive_eigengaps([0.0]) == [0.0]
