"""Env factory resolved INSIDE spawn'd worker processes (tests only).

Importable as a top-level module: pytest (prepend import mode) puts this
tests/ directory on sys.path, and multiprocessing spawn children inherit
the parent's sys.path. Monkeypatching cannot cross the process boundary,
which is exactly why the server exposes the env_factory config seam.

Failure modes are scripted by game_file substring (uppercase markers so a
pytest tmp_path carrying a test name can never trigger them by accident):
GAME-HANG blocks far past any test timeout, GAME-DIE kills the worker
process outright, GAME-BOOM raises (worker survives).
"""

import os
import time


class FakeWorkerEnv:
    def __init__(self, game_file: str):
        self.game_file = game_file
        self.steps_taken = 0

    def reset(self):
        self.steps_taken = 0
        return f"fake obs for {self.game_file}", {"admissible_commands": ["look", "win"]}

    def step(self, command: str):
        if command == "explode":
            raise RuntimeError("scripted step failure")
        self.steps_taken += 1
        done = command == "win"
        info = {"won": done, "lost": False, "admissible_commands": ["look", "win"]}
        return f"echo:{command}:{self.steps_taken}", int(done), done, info

    def close(self):
        pass


def make_env(game_file: str, want_admissible: bool):
    if "GAME-HANG" in game_file:
        time.sleep(60)
    if "GAME-DIE" in game_file:
        os._exit(1)
    if "GAME-BOOM" in game_file:
        raise RuntimeError("scripted load failure")
    env = FakeWorkerEnv(game_file)
    obs, info = env.reset()
    return env, obs, info
