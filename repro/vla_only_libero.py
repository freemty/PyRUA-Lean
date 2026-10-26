"""VLA-only control for LIBERO-PRO: no language model, Pi0.5 runs the benchmark's own instruction.

Repeats `pi0_doubled(robo.task)` (the Pi0.5 contact skill, which runs the prompt until the task predicate fires
or its chunks run out) until the task is done, the episode is truncated or 8 calls of 50 chunks were spent.
"""


def run(robo):
    for _ in range(8):
        result = robo.pi0_doubled(robo.task, max_chunks=50)
        print(f"pi0_doubled: chunks={result.chunks_used} terminated={result.terminated} truncated={result.truncated}")
        if robo.done or result.terminated or result.truncated:
            break
