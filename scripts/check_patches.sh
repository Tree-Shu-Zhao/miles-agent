#!/bin/bash
# scripts/check_patches.sh — the miles patch gate. The 3rdparty/miles gitlink pins the UPSTREAM base; the working tree
# must be that base plus every patch under patches/miles/ (`git am`, in the order of patches/miles/series). The patched
# commits exist only in working trees, and a bare `git submodule update` silently drops them: a job built from such a
# tree fails late or quietly (no env/* metrics, a crash on the first unscored sample under --reward-key, no save before
# a preemption). This is the check the cluster uploader ran before every stage, as a script of its own: each patch's
# `Subject:` must be among the last 20 commit subjects of the miles checkout (a folded subject counts by its first
# line), and series must list every patches/miles/miles-*.patch exactly once and nothing else.
#
#   scripts/check_patches.sh [--dry-run] [--apply]
#     (no flag)   check: every patch applied -> `PATCHES_OK applied=<n>/<total> newly_applied=0` as the last line;
#                 a missing one -> `FATAL: 3rdparty/miles lacks patch <name> (<subject>); apply with ...`, exit 1
#     --apply     `git am <patch>` for every series entry whose subject is missing, in series order (idempotent: an
#                 applied entry is skipped), then the check above and the marker; the first failing am ends the run
#                 with a FATAL and the `git am --abort` advice (printed, never run)
#     --dry-run   (also DRY_RUN=1) prints the argv instead: `DRY_RUN git_log: ...` (the commit log is still read, the
#                 only read there is) and, with --apply, one `DRY_RUN git_am: ...` per missing entry in series order;
#                 a missing patch or checkout is a WARN; exit 0 before any side effect, without the marker
#   Knobs (the environment, then local.env): MILES_SRC — the miles checkout, default $REPO/3rdparty/miles (the
#   submodule; its .git must exist: an empty submodule dir would resolve to the checkout around it); MILES_PATCHES_DIR
#   — the patch dir holding series, default $REPO/patches/miles; GIT — the git command, default git.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

usage() { sed -n '2,/^set -euo pipefail/p' "$0" | sed -e '$d' -e 's/^# \{0,1\}//'; }

. "$HERE/lib/common.sh"

APPLY=0
while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --dry-run) DRY_RUN=1 ;;
    --apply) APPLY=1 ;;
    *) fatal "unknown argument '$1': scripts/check_patches.sh [--dry-run] [--apply]" ;;
  esac
  shift
done
[ -z "${DRY_RUN:-}" ] || export DRY_RUN

GIT=${GIT:-git}
MILES_SRC=${MILES_SRC:-$REPO/3rdparty/miles}
PATCHES_DIR=${MILES_PATCHES_DIR:-$REPO/patches/miles}
SERIES=$PATCHES_DIR/series
ADVICE="apply with scripts/check_patches.sh --apply (patches/miles/README.md)"

# a check under --dry-run is a WARN; otherwise a FATAL
gate() { if is_dry_run; then echo "WARN: $*"; else fatal "$*"; fi; }

# <patch>: its Subject line's text after the [PATCH] tag — the first line of a folded subject, a prefix of the full
# subject `git log --format=%s` prints (the uploader's extraction)
patch_subject() { sed -n '/^Subject: \[PATCH[^]]*\] /{s///p;q;}' "$1"; }

# ---- series vs directory: every miles-*.patch exactly once, nothing else ----
[ -f "$SERIES" ] || fatal "no $SERIES (the patch order file; patches/miles/README.md)"
SERIES_NAMES=""
while IFS= read -r line || [ -n "$line" ]; do
  line=${line%$'\r'}
  line=${line%%[[:space:]]*}     # the first word: blank lines and `# ...` comment lines are skipped
  case "$line" in ''|'#'*) continue ;; esac
  case " $SERIES_NAMES " in *" $line "*) fatal "$SERIES lists $line twice" ;; esac
  [ -f "$PATCHES_DIR/$line" ] || fatal "$SERIES names $line, which is not in $PATCHES_DIR/"
  SERIES_NAMES="$SERIES_NAMES $line"
done < "$SERIES"
SERIES_NAMES=${SERIES_NAMES# }
[ -n "$SERIES_NAMES" ] || fatal "$SERIES lists no patch"
TOTAL=0
for p in "$PATCHES_DIR"/miles-*.patch; do
  [ -f "$p" ] || fatal "no $PATCHES_DIR/miles-*.patch"
  name=$(basename "$p")
  case " $SERIES_NAMES " in
    *" $name "*) ;;
    *) fatal "$SERIES does not list $name (every $PATCHES_DIR/miles-*.patch must be in series)" ;;
  esac
  TOTAL=$((TOTAL + 1))
done

# ---- the checkout's recent subjects (the one read) ----
MILES_SUBJECTS=""
CHECKOUT_OK=1
if [ ! -e "$MILES_SRC/.git" ]; then
  CHECKOUT_OK=0
  gate "no git checkout at $MILES_SRC (MILES_SRC): git submodule update --init 3rdparty/miles, then scripts/check_patches.sh --apply"
fi
read_subjects() {  # the last 20 subjects into MILES_SUBJECTS
  if is_dry_run; then dry_run_print git_log "$GIT" -C "$MILES_SRC" log --format=%s -20; fi
  MILES_SUBJECTS=$("$GIT" -C "$MILES_SRC" log --format=%s -20) || fatal "git log failed in $MILES_SRC (rc=$?)"
}
has_subject() { grep -qF -- "$1" <<<"$MILES_SUBJECTS"; }  # <subject>: among the checkout's subjects?
compute_missing() {  # MISSING = the series entries whose subject is absent, in series order; PRESENT = the others' count
  local name subj
  MISSING=""
  PRESENT=0
  for name in $SERIES_NAMES; do
    subj=$(patch_subject "$PATCHES_DIR/$name")
    [ -n "$subj" ] || fatal "$PATCHES_DIR/$name has no Subject: line (not a git format-patch file)"
    if [ "$CHECKOUT_OK" = 1 ] && has_subject "$subj"; then PRESENT=$((PRESENT + 1)); else MISSING="$MISSING $name"; fi
  done
  MISSING=${MISSING# }
}
[ "$CHECKOUT_OK" = 0 ] || read_subjects
compute_missing

# ---- --apply: the missing entries, in series order ----
NEWLY=0
if [ "$APPLY" = 1 ] && [ -n "$MISSING" ]; then
  for name in $MISSING; do
    if is_dry_run; then
      dry_run_print git_am "$GIT" -C "$MILES_SRC" am "$PATCHES_DIR/$name"
      continue
    fi
    echo "check_patches: git am $name"
    "$GIT" -C "$MILES_SRC" am "$PATCHES_DIR/$name" \
      || fatal "git am $name failed in $MILES_SRC (rc=$?); the entries before it stayed applied. Resolve the conflict there, or return to the tree before this patch with \`git -C $MILES_SRC am --abort\` (not run by this script); see patches/miles/README.md"
    NEWLY=$((NEWLY + 1))
  done
  if ! is_dry_run; then
    read_subjects
    compute_missing
  fi
fi

# ---- the verdict ----
echo "check_patches: $MILES_SRC carries $PRESENT/$TOTAL of the patches in $PATCHES_DIR"
FIRST_MISSING=""
for name in $MISSING; do
  [ -n "$FIRST_MISSING" ] || FIRST_MISSING=$name
  if is_dry_run; then
    echo "WARN: 3rdparty/miles lacks patch $name ($(patch_subject "$PATCHES_DIR/$name"))"
  else
    echo "check_patches: missing $name"
  fi
done
if is_dry_run; then
  echo "check_patches: dry run - $PRESENT/$TOTAL applied, nothing changed"
  exit 0
fi
[ -z "$FIRST_MISSING" ] || fatal "3rdparty/miles lacks patch $FIRST_MISSING ($(patch_subject "$PATCHES_DIR/$FIRST_MISSING")); $ADVICE"
echo "PATCHES_OK applied=$PRESENT/$TOTAL newly_applied=$NEWLY"
