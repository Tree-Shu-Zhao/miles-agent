# Data

Every dataset belongs to its upstream project: this repo ships processing code and digests only.
`scripts/fetch_data.sh <env>` downloads or verifies the upstream files under `MILES_AGENT_DATA` (default
`$REPO/data`), runs this repo's converters and checks every derived file against its pinned digest
(`FETCH_DATA_OK env=...` last; `--dry-run` prints the commands and runs none). The rows a run trains on are built
per run by `recipes/<env>/prepare_data.py` under `$MILES_AGENT_DATA/runs/<tag>/`: they carry the policy model's
tool-call markup, so they are never shared between models. Every dataset is its upstream project's and is
downloaded by you; nothing derived from upstream data is committed. Derived files are rebuilt by
`scripts/fetch_data.sh` and verified against the digests below. Each license line states what the upstream
release says at the time of writing; check it at the source before redistributing anything you build from it.

| environment | `scripts/fetch_data.sh` | runtime dependencies | license, per the upstream repository |
|---|---|---|---|
| alfworld | `alfworld`: `alfworld-download --data-dir $ALFWORLD_DATA` (default `$MILES_AGENT_DATA/alfworld`), then `json_2.1.1/` must exist | none beyond the gym venv | ALFWorld: MIT (`alfworld.yaml`) |
| scienceworld | `scienceworld --from <TCOD campaign test.jsonl>` (the paper's order) or `scienceworld --synthesize` (the same 1308 tasks, sorted): the JDK and the jar are checked, the eval list is rebuilt and its sha256 verified | a JDK, 11 or newer (`JAVA_HOME`, else the `java` on PATH); the `scienceworld==1.2.3` wheel, the simulator jar inside it | ScienceWorld: Apache-2.0 (`scienceworld.yaml`); the TCOD task split: Apache-2.0 |
| webshop | `webshop --from <dir> [--full]`: the authors' catalog files, md5-checked; `webshop --regenerate-goals [--engine-url U]`: the goal export from a running engine, set digest `455c9eaa96191605` | Java 11 (the Lucene index), the engine's python 3.10 environment (`env_servers/webshop/setup.sh`) | the engine: MIT (vendored under `env_servers/webshop/webshop/`, `LICENSE.md` there); the catalog and the goals derived from it: from the WebShop authors, under their terms |
| search_r1 | `search_r1 [--splits <split>...]`: FlashRAG's processed QA sets from the Hub, converted and linked into the gym config's paths | a FlashRAG-compatible retriever (`RETRIEVER_URL`; one GPU) | the FlashRAG datasets collection; the sources per their releases: NQ CC BY-SA 3.0, TriviaQA Apache-2.0, PopQA MIT, HotpotQA CC BY-SA 4.0, 2WikiMultihopQA Apache-2.0, MuSiQue CC BY 4.0, Bamboogle MIT (`wiki_search.yaml` carries the single enum value `CC BY-SA 4.0`) |
| tmax | nothing: the coding and terminal tasks are Harbor datasets, fetched by Harbor (`HARBOR_DATASET`) | docker, for the task containers | Harbor's and each dataset's own |

## ALFWorld

`scripts/fetch_data.sh alfworld` runs `alfworld-download --data-dir $ALFWORLD_DATA`: the binary of the
alfworld_env gym venv when `GYM_VENV_ROOT` is set
(`$GYM_VENV_ROOT/gym_venvs/resources_servers/alfworld_env/.venv/bin/alfworld-download`), else the one on PATH
(none: a `FATAL:` printing the command to run inside the alfworld_env venv). The download is the ALFWorld
project's own (the TextWorld game files, `json_2.1.1/` under the data dir); nothing of it is in this repo.
`recipes/alfworld/prepare_data.py` builds the train rows and the 274-task eval list per run from those files.

## ScienceWorld

The simulator is a jar inside the `scienceworld==1.2.3` wheel the gym venv installs (nothing to download;
`SCIENCEWORLD_JAR` pins another jar). It runs on a JDK, 11 or newer: `JAVA_HOME` (`local.env`) or the `java`
on PATH; `scripts/fetch_data.sh scienceworld` checks `java -version`. Each worker is a warm JVM
(`SCIENCEWORLD_ENV_WORKERS`, default 128, about 350 MB each: lower it on a small machine).

The eval list `recipes/scienceworld/prepare_data.py` reads is derived data and is not in the repository:
`scienceworld_test1308.jsonl`, 1308 rows of `{"task_name", "var_num"}` (the 13 non-train task types with their
first 50% of variations; the SET is `task_splits.synthesized_test()`, the paper's ORDER is the one shuffle of the
original TCOD campaign). Two ways to rebuild it, both to
`${SCIENCEWORLD_TEST_LIST:-$MILES_AGENT_DATA/scienceworld/scienceworld_test1308.jsonl}`:

- `scripts/fetch_data.sh scienceworld --from <the campaign's test.jsonl>` (the paper's order; the source file has
  sha256 `9632fe4f…ebf8`) re-serialises that file and fails unless the result's sha256 is
  `b2e1024f82a09e63011f37e4f423036c94b175b49b79c8d131eba7994cba2739` (`SCIENCEWORLD_TEST_SHA256` pins another
  campaign's);
- `scripts/fetch_data.sh scienceworld --synthesize` writes the same 1308 tasks in sorted order, sha256
  `f98541246a3145ce0ae2f97831a5f7dc7cd19e083a7b9e782f52dda4a1a8c11c`.

The first 134 rows of the list are the `valid_unseen` block of the 274-row eval list, so the full 1308-task
evaluation scores the paper's set either way while, with the synthesized order, `scienceworld_unseen` is another
subset and not comparable with the paper's; the whole file is the 1308-row test set; the train and
train_heldout splits are enumerated by `gym_root/resources_servers/scienceworld_env/task_splits.py`, not stored.
Without `--from` or `--synthesize` an existing list is classified by its digest (`order=frozen|synthesized`) and
the upstream pointer printed (`DATA_UPSTREAM`); none present: the manual step (`DATA_MANUAL_STEP`) and exit 2.
`prepare_data.py` reads `--test-jsonl`, default `SCIENCEWORLD_TEST_LIST`, else the data-root path, and fails
with the command above when the list is missing. `recipes/scienceworld/data/README.md` has the provenance;
`task_splits.load_test` refuses a file with other than 1308 rows, a duplicate pair or a train/test overlap.

## WebShop

### The catalog

The engine's catalog files come from the WebShop authors (their public download links stopped answering in 2026;
any copy of the files works). `scripts/fetch_data.sh webshop --from <dir>` copies `items_shuffle_1000.json`,
`items_ins_v2_1000.json` and `items_human_ins.json` (`--full` adds `items_shuffle.json` and `items_ins_v2.json`,
about 6 GB) from `<dir>` into the engine's data dir, `$WEBSHOP_HOME/webshop/data` (`WEBSHOP_HOME` default
`$MILES_AGENT_DATA/webshop`), after checking each md5 against `env_servers/webshop/data_md5.txt`:

| file | md5 (prefix) |
|---|---|
| `items_shuffle_1000.json` | `df2b2610` |
| `items_ins_v2_1000.json` | `a3346e8b` |
| `items_human_ins.json` | `738c99a2` |

An entry may be a prefix of the md5 (the repo carries 8-hex prefixes), a full 32-hex entry must match whole;
a mismatch is a `FATAL:`, a file without an entry is copied with a `WARN:`; `WEBSHOP_MD5_TABLE` points at another
table. Without `--from` the script prints the manual step (`DATA_MANUAL_STEP`) and exits 2. The recipes use the
1000-item catalog (`WEBSHOP_FULL=0`) with synthetic goals; `items_human_ins.json` is loaded by the engine but
unused by the goal set.

### The runtime

`WEBSHOP_HOME=$MILES_AGENT_DATA/webshop env_servers/webshop/setup.sh [--full]` builds the engine's environment
(micromamba / mamba / conda: `python=3.10 openjdk=11 faiss-cpu` from conda-forge; without one, a venv on a
`python3.10` and a JDK 11+ found on PATH) and the Lucene index over the staged catalog; the server's `--small` /
`--full` mode must match the mode last built. `env_servers/webshop/launch_engine.sh` starts the engine on
`WEBSHOP_SERVER_PORT` (6200); `docs/webshop_integration.md` describes the engine, the observation rendering and
the reward.

### The goal export

`webshop_goals.jsonl` is the engine's goal list, exported from a running engine and read in file order by
`recipes/webshop/prepare_data.py`: 6910 rows, 6901 distinct goals (9 rows repeat an earlier row's content;
`prepare_data.py` keeps the first occurrence, so the canonical test set is exactly engine positions 0..499),
goal seed 233, order-free set digest `455c9eaa96191605`, which every run checks against the engine's
`/health.goals_digest` before it starts (`WEBSHOP_GOALS_DIGEST`). Each row's `goal_id` is a content hash;
`prepare_data.py` recomputes it and rejects a hand-edited row. The file is derived from the WebShop authors'
catalog and is not in the repository: `scripts/fetch_data.sh webshop --regenerate-goals [--engine-url U]`
rebuilds it bit-for-bit (to `WEBSHOP_GOALS_FILE`, default `webshop_goals.jsonl` under `recipes/webshop/data/`,
gitignored there) from a running engine (`U`, else
`${WEBSHOP_ENGINE_BASE_URL:-http://127.0.0.1:$WEBSHOP_SERVER_PORT}`) through
`env_servers/webshop/export_goals.py`, and fails unless the exporter's digest is `455c9eaa96191605` (the
exported file stays in place for inspection); on success the exporter's sidecar is copied to
`webshop_goals.meta.json` beside it. `prepare_data.py` reads `--goals`, default `WEBSHOP_GOALS_FILE`, else that
path, and fails with the command when the file is missing. A different catalog, goal code or seed changes the
digest, and with it every number; `recipes/webshop/data/README.md` has the provenance and the re-export rules.

## search_r1 (wiki_search)

`scripts/fetch_data.sh search_r1 [--splits <split>...]` runs
`gym_root/resources_servers/wiki_search/data/prepare.py --output-dir $WIKI_SEARCH_DATA` (default
`$MILES_AGENT_DATA/wiki_search`; network and the `huggingface_hub` package: the files come from the Hub
repository `RUC-NLPIR/FlashRAG_datasets`), then links `train.jsonl` and `validation.jsonl` into
`gym_root/resources_servers/wiki_search/data/`, the paths the gym config names (a stale link is replaced, a
placeholder copy of `example.jsonl` left by the venv build's boot probe is moved aside, any other regular file
is a `FATAL:`). The splits:

- `train`: FlashRAG's `nq/train` + `hotpotqa/train`, merged and shuffled (`--seed`, default 42);
- `validation`: 893 rows = 128 seeded-random rows from each of nq, triviaqa, popqa (test), hotpotqa,
  2wikimultihopqa, musique (dev) + all 125 rows of bamboogle (test);
- `test`: every row of those seven files.

The runtime needs a FlashRAG-compatible retriever (`RETRIEVER_URL`, e.g. `http://127.0.0.1:8000`; the recipe
probes it before the run); a dense retriever over the Wikipedia dump wants one GPU of its own. Licensing: the
FlashRAG collection re-distributes the sources' processed files; per the upstream releases the sources are NQ
(CC BY-SA 3.0), TriviaQA (Apache-2.0), PopQA (MIT), HotpotQA (CC BY-SA 4.0), 2WikiMultihopQA (Apache-2.0),
MuSiQue (CC BY 4.0) and Bamboogle (MIT), and the train/validation files you build from them are share-alike
derivatives. The single `license:` value per dataset in
`gym_root/resources_servers/wiki_search/configs/wiki_search.yaml` is `CC BY-SA 4.0` for train and validation
because nemo_gym's enum has no 3.0 and no mixture (the example fixture's is `Apache 2.0`: synthetic rows authored
in this repository).

## Fixtures

`gym_root/resources_servers/*/data/example.jsonl` are synthetic rows (webshop_env, wiki_search) or our own
prompt + task identifiers (scienceworld_env); they exist so the gym configs resolve and the boot probe runs;
they are not samples of any dataset.

## tmax (coding and terminal tasks)

Nothing to fetch here: the tasks are Harbor datasets (`HARBOR_DATASET`: `<name>@<version>`, `<org>/<name>[@<ref>]`
or a directory of task dirs), pulled by Harbor itself into its cache, each task running in its own container
(docker). Their licenses are the datasets' own, stated by Harbor's registry.

The recipe's row builder needs the list of images the tasks run in (`recipes/tmax/prepare.py rows --dataset <dir>
--images <list>`, one image reference a line; the flag stays required). For a dataset checked out as a directory
of task dirs the list is one line:

    grep -h docker_image <dataset>/*/task.toml | cut -d'"' -f2 | sort -u

The two lists the author's runs used are not shipped; a list regenerated from the same dataset release
reproduces them: `swebenchpro.txt`, 734 lines, sha256
`7dc173f71a697847af05a86258d4ce53afda6b76ba4585b5fb7fc096693a9093`; `swebench_multilingual.txt`, 302 lines, sha256
`0f91010d24f40b68e69740996869ea621ac5bacb8b6b26b2aa50d33fda044c61`.

Harness packs: the trial server's harnesses install from a pack under `HARBOR_PACK_ROOT`
(`gym_root/miles_harbor/packs/harbor_packs.sh`; `docs/local_setup.md`) that every trial container must see. With
Harbor's docker environment the public tree has no mount mechanism for it yet: build the pack, then mount that
directory into every trial container at the same path yourself (the author's private sandbox backend handed it
over as a harness-directory mount).
