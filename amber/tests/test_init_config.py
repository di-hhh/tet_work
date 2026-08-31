from src.initialization.init_config import shortener


def test_shortener_preserves_commas_inside_hydra_list_values():
    value = "task.sample_id_filter=[sample_a,sample_b],trainer.max_epochs=1"

    assert shortener(value) == "sam_id_fil=[sample_a,sample_b],max_epo=1"


def test_shortener_splits_only_the_first_equals_in_an_override():
    value = "logger.tag='source=a,b',seed=0"

    assert shortener(value) == "tag='source=a,b',see=0"
