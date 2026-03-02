import logging
import math
import os

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def get_start_stop_index(dataset_length: int) -> tuple[int, int]:
    try:
        task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
        array_max = int(os.environ["SLURM_ARRAY_TASK_MAX"])
        items_per_job = math.ceil(dataset_length / (array_max + 1))

        start = task_id * items_per_job
        stop = min(start + items_per_job, dataset_length)
        logger.info(
            f"Processing dataset from {start} to {stop - 1}. Total items: {stop - start} over {dataset_length}"
        )
        return start, stop
    except KeyError:
        return 0, dataset_length


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    # emulate SLURM_ARRAY_TASK_ID and SLURM_ARRAY_TASK_MAX
    os.environ["SLURM_ARRAY_TASK_ID"] = "3"
    os.environ["SLURM_ARRAY_TASK_MAX"] = "3"
    import fire

    fire.Fire(get_start_stop_index)
