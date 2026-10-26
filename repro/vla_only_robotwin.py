"""VLA-only control for RoboTwin: no language model, LingBot-VLA runs the benchmark's own instruction.

Repeats `lingbot_act(chunks=4)` (LingBot always conditions on the task instruction) until the task is done, the
episode is truncated or 30 calls were spent.
"""


def run(robo):
    for _ in range(30):
        result = robo.lingbot_act(chunks=4)
        print(f"lingbot_act: terminated={result.terminated} truncated={result.truncated}")
        if robo.done or result.terminated or result.truncated:
            break
