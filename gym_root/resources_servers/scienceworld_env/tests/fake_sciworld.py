"""Fake ScienceWorldEnv factory, resolved INSIDE spawn'd worker processes (tests only).

Importable as a top-level module: pytest (prepend import mode) puts this tests/
directory on sys.path, and multiprocessing spawn children inherit the parent's
sys.path. Monkeypatching cannot cross the process boundary, which is exactly
why the server exposes the env_factory config seam.

Mirrors the slice of the real API the worker touches: load / reset / step /
get_task_description / get_possible_actions / get_possible_objects /
server.getCompleted / close. Failure modes are scripted by VARIATION number
(task names must pass task_splits validation): BOOM_VAR raises at load (worker
survives), HANG_VAR blocks far past any test timeout, DIE_VAR kills the worker
process outright. Step commands: "half" -> score 50, "win" -> score 100 +
completed, "bogus" -> the simulator's no-match sentence, "explode" -> raises,
anything else -> a neutral move.
"""

import os
import time

BOOM_VAR = 13
HANG_VAR = 14
DIE_VAR = 15

TEMPLATES = ["look around", "help", "focus on OBJ", "open OBJ", "go to OBJ"]
OBJECTS = ["air", "door to kitchen", "hallway"]
INITIAL_OBS = "This room is called the hallway. In it, you see: the agent, a substance called air."


class _Server:
    def __init__(self, env):
        self._env = env

    def getCompleted(self):  # noqa: N802  (py4j Java method name)
        return "true" if self._env.completed else "false"


class FakeSciEnv:
    def __init__(self, jar_path: str, env_step_limit: int):
        self.jar_path = jar_path
        self.env_step_limit = env_step_limit
        self.server = _Server(self)
        self.task = None
        self.score = 0.0
        self.completed = False
        self.steps = 0
        self.loads = 0
        self.closed = False

    def load(self, task_name, var_num, simplification, generateGoldPath=False):  # noqa: N803
        if var_num == HANG_VAR:
            time.sleep(60)
        if var_num == DIE_VAR:
            os._exit(1)
        if var_num == BOOM_VAR:
            raise RuntimeError("scripted load failure")
        self.task = (task_name, var_num, simplification)
        self.loads += 1
        self.score = 0.0
        self.completed = False
        self.steps = 0

    def reset(self):
        self.score = 0.0
        self.completed = False
        self.steps = 0
        return INITIAL_OBS, {"score": 0, "moves": 0}

    def get_task_description(self):
        return f"Your task is to do {self.task[0]} variation {self.task[1]}."

    def get_possible_actions(self):
        return list(TEMPLATES)

    def get_possible_objects(self):
        return list(OBJECTS) + (["kitchen"] if self.steps else [])

    def step(self, command: str):
        if command == "explode":
            raise RuntimeError("scripted step failure")
        self.steps += 1
        if command == "half":
            self.score = 50.0
            obs = "You focus on the thermometer."
        elif command == "win":
            self.score = 100.0
            self.completed = True
            obs = "The task is complete."
        elif command == "drop":
            self.score = 0.0  # simulator scores can go DOWN; the wrapper keeps the running max
            obs = "You dropped it."
        elif command == "bogus":
            obs = "No known action matches that input."
        else:
            obs = f"echo:{command}:{self.steps}"
        done = self.completed or self.steps >= self.env_step_limit
        return obs, 0.0, done, {"score": self.score, "moves": self.steps}

    def close(self):
        self.closed = True


def make_env(jar_path: str, env_step_limit: int):
    return FakeSciEnv(jar_path, env_step_limit)
