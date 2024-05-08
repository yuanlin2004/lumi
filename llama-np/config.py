from dataclasses import dataclass

@dataclass
class ExperimentArgs:
    # attention
    no_kv_cache: bool = False
    use_in_place_kv_cache: bool = False # to be splipped

    # generation
    one_a_time: bool = False

    # report
    report_mem: bool = False

    # use cupy
    use_cupy: bool = False

    # save sample history
    sample_history: str = None
