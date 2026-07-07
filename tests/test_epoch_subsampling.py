from kd_sensing.engine.epoch_subsampling import EpochSubsampleSampler


def test_locality_order_keeps_random_subset_and_is_reproducible():
    locality_keys = [("source_b", idx) if idx % 2 else ("source_a", idx) for idx in range(12)]
    random_sampler = EpochSubsampleSampler(
        dataset_length=12,
        effective_num_samples=6,
        seed=99,
        rotate_each_epoch=False,
        shuffle=True,
    )
    locality_sampler = EpochSubsampleSampler(
        dataset_length=12,
        effective_num_samples=6,
        seed=99,
        rotate_each_epoch=False,
        shuffle=True,
        order="locality",
        locality_keys=locality_keys,
    )
    resumed = EpochSubsampleSampler(
        dataset_length=12,
        effective_num_samples=6,
        seed=99,
        rotate_each_epoch=False,
        shuffle=True,
        order="locality",
        locality_keys=locality_keys,
    )

    random_indices = list(random_sampler)
    locality_indices = list(locality_sampler)

    assert set(locality_indices) == set(random_indices)
    assert locality_indices == sorted(random_indices, key=lambda index: locality_keys[index])
    assert locality_indices == list(resumed)


def test_shuffle_false_maps_to_sorted_order_metadata():
    sampler = EpochSubsampleSampler(
        dataset_length=8,
        effective_num_samples=4,
        seed=5,
        rotate_each_epoch=False,
        shuffle=False,
    )

    assert sampler.metadata()["order"] == "sorted"
    assert list(sampler) == sorted(list(sampler))
