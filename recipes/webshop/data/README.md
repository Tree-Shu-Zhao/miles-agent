# Frozen WebShop goals (the dataset's source of truth): rebuilt locally, never committed

`webshop_goals.jsonl` is the goal list of the WebShop engine, exported from a
running engine by `env_servers/webshop/export_goals.py`. It is not in this
repository: the goal texts are built from the WebShop authors' product
catalog, and every dataset here is its upstream's (docs/data.md). With the
small catalog in place (`scripts/fetch_data.sh webshop --from <dir>`, then
`env_servers/webshop/setup.sh` and `launch_engine.sh`):

```
bash scripts/fetch_data.sh webshop --regenerate-goals [--engine-url http://127.0.0.1:6200]
```

rebuilds it bit-for-bit at `${WEBSHOP_GOALS_FILE:-recipes/webshop/data/webshop_goals.jsonl}`
(gitignored) and refuses any result whose `goals_digest` is not
`455c9eaa96191605` (`WEBSHOP_GOALS_DIGEST`). `recipes/webshop/prepare_data.py`
reads it in file order (`--goals`, default `WEBSHOP_GOALS_FILE`, else this
directory) to build `webshop_train.jsonl` / `webshop_eval.jsonl`; every emitted
row names its goal by `goal_id` (content id, `gym_root/agent_utils/webshop_goals.py`)
and the launcher refuses to start unless the engine's goal-set digest equals
the file's (`docs/webshop_integration.md`, "Data & task feeding"). Never
hand-edit the file: `prepare_data.py` recomputes each row's id from its
content and rejects a mismatch.

## Files (written by the exporter; gitignored here)

| file | content |
|---|---|
| `webshop_goals.jsonl` | one goal per line, in the engine's list order at export (`goal_idx` = that position) |
| `webshop_goals.meta.json` | `{"goals", "goals_digest", "protocol", "exported_at", "engine_url"}` as reported by the exporting engine |

Row shape (all `GOAL_FIELDS` of the shared module plus the two bookkeeping keys):

```
{"goal_id": "<16 hex>", "goal_idx": <int>,
 "instruction_text": "...", "asin": "...", "name": "...", "category": "...",
 "query": "...", "product_category": "...",
 "attributes": ["..."], "goal_options": {"size": "..."} , "price_upper": <float>}
```

`goal_id = sha1(canonical JSON of the GOAL_FIELDS)[:16]`; `goal_idx` and the
sampling `weight` are not part of the identity. The file's set digest is
`sha1("\n".join(sorted(set(goal_ids))))[:16]` (order-free, repeats ignored)
and must equal `/health.goals_digest` of any engine the data is used against.

## Provenance of the paper's export (what every regeneration must reproduce)

| field | value |
|---|---|
| catalog | small: `items_shuffle_1000.json` (4,467,013 B, sha256 `30a4765c3a327af7…`) + `items_ins_v2_1000.json` (147,099 B, sha256 `f88a36314a397b53…`) (`WEBSHOP_FULL=0`), obtained from the WebShop authors; `scripts/fetch_data.sh webshop --from <dir>` copies and verifies them (`env_servers/webshop/data_md5.txt`) |
| goal source | synthetic (`WEBSHOP_HUMAN_GOALS=0`; `items_human_ins.json` sha256 `cf78667548a71786…` is loaded but unused) |
| export order | engine goal list under `WEBSHOP_GOAL_SEED=233` (original WebShop / TCOD constant); file position *i* = TCOD `task_id` *i* on this catalog and goal code |
| observation mode at export | `text_rich` (does not enter the identity) |
| goals | 6910 rows = 6901 distinct questions; 9 rows repeat an earlier row's content (positions 6766→98, 2407→1083, 1367→1316, 4208→1372, 4777→1569, 4942→2701, 5073→3090, 4239→3869, 6304→4200: a product listing the same option value twice). `prepare_data.py` keeps the first occurrence, so the canonical test set is exactly engine positions 0..499 |
| goals_digest | `455c9eaa96191605` (order-free SET digest; `webshop_goals.meta.json` == `protocol.json.goals_digest` == engine `/health.goals_digest`) |
| exported_at | 2026-09-04T18:52:57+00:00 (identical rows to a first export at 18:30:15 the same day; re-run only to record the set-digest semantics) |
| engine code | the vendored WebShop text environment `env_servers/webshop/webshop` (verbatim) and the goal code of `env_servers/webshop` as of that export; the digest pin is how a later regeneration proves it still matches |
| catalog files sha256 | see the catalog row (first 16 hex shown; full sums via `sha256sum $WEBSHOP_DATA_DIR/items_*_1000.json`) |
| exported with | one engine worker on a loopback port, `env_servers/webshop/export_goals.py` |

## Re-exporting

Only after a deliberate change of catalog, vendored goal code or goal seed (any
of them changes goal content — the seed also draws the price caps — and thus
the digest), or to build a separate goals file for another arm:

```
# engine running with the intended knobs (env_servers/webshop/launch_engine.sh)
python3 env_servers/webshop/export_goals.py \
  --engine-url http://127.0.0.1:6200 \
  --output <path>          # then WEBSHOP_GOALS_FILE=<path> WEBSHOP_GOALS_DIGEST=<its digest> for that arm
```

The exporter reads `/health` (must carry `protocol` + `goals_digest`) and
`GET /goals`, writes the rows in `goal_idx` order, recomputes the set digest
and refuses to write if it differs from `/health`, then writes the `.meta.json`
sidecar. Update the provenance table above for a new paper export; from then on
every run scores the new set and says so (`WEBSHOP_ENGINE_UP ...
goals_digest=...` in the log, `goals_digest` in `protocol.json`).
