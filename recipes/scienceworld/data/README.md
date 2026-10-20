# Frozen ScienceWorld eval taskset (TCOD's): rebuilt locally, never committed

`scienceworld_test1308.jsonl` is the eval set the TCOD numbers were measured on
(TCOD, arXiv 2604.24005; `TCOD_examples/scienceworld/get_sciworld_data.py` in
github.com/kokolerk/TCOD): the 13 non-train task types x their first 50%
variations = 1308 tasks, in the order one run of that script shuffled them
(`random.seed(42)` over a set difference whose iteration order is not fixed, so
the ORDER is that run's; the SET is `task_splits.synthesized_test()`). Every
dataset here is its upstream's and no derived file is committed (docs/data.md):

```
bash scripts/fetch_data.sh scienceworld --from <that run's test.jsonl>   # the paper's order
bash scripts/fetch_data.sh scienceworld --synthesize                      # same 1308 tasks, sorted order
```

`--from` re-serialises the campaign file (raw `{"task_desc": ...}` rows; the
paper's file has sha256 `9632fe4f…ebf8`) into 1308 rows of `{"task_name",
"var_num"}` (`json.dumps` default separators, one per line) at
`${SCIENCEWORLD_TEST_LIST:-$MILES_AGENT_DATA/scienceworld/scienceworld_test1308.jsonl}`
and refuses any result whose sha256 is not
`b2e1024f82a09e63011f37e4f423036c94b175b49b79c8d131eba7994cba2739`.
`--synthesize` writes the same tasks in sorted `(task_name, var_num)` order,
sha256 `f98541246a3145ce0ae2f97831a5f7dc7cd19e083a7b9e782f52dda4a1a8c11c`:
the full-set evaluation (`scienceworld_test1308.jsonl` output of
`prepare_data.py`) scores the paper's task SET either way; the 134-row
`valid_unseen` slice is the file's FIRST 134 rows, so with the synthesized
order the `scienceworld_unseen` numbers are a different subset and are not
comparable with the paper's.

`recipes/scienceworld/prepare_data.py` reads the list in file order
(`--test-jsonl`, default `SCIENCEWORLD_TEST_LIST`, else the data-root path
above): the FIRST 134 rows are the `valid_unseen` block of
`scienceworld_eval274.jsonl` (the `scienceworld_unseen` eval slice of
`run.py`), the whole file is `scienceworld_test1308.jsonl`. The train and
train_heldout splits are enumerated, not stored
(`gym_root/resources_servers/scienceworld_env/task_splits.py`).

Never hand-edit the list: `task_splits.load_test` refuses a file with other
than 1308 rows, duplicate (task, variation) pairs or a train/test overlap, and
`tests/test_scienceworld_prepare_data.py` pins the sha256 above whenever the
frozen file is present (`SCIENCEWORLD_TEST_LIST`).
