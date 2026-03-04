#!/usr/bin/env bash
# Install the pre-commit hook into all testAnt git repos.
# Run this once after cloning, and again if new sub-repos are added.
#
# Usage: bash scripts/hooks/install.sh   (from refinery/rig)

set -euo pipefail

HOOK_SRC="$(cd "$(dirname "$0")" && pwd)/pre-commit"
# Script lives at refinery/rig/scripts/hooks/ — go up 4 levels to testAnt root
REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"

if [[ ! -f "$HOOK_SRC" ]]; then
    echo "ERROR: $HOOK_SRC not found"
    exit 1
fi

install_hook() {
    local hooks_dir="$1"
    local label="$2"
    mkdir -p "$hooks_dir"
    cp "$HOOK_SRC" "$hooks_dir/pre-commit"
    chmod +x "$hooks_dir/pre-commit"
    echo "  installed → $hooks_dir/pre-commit  ($label)"
}

echo "Installing testAnt pre-commit hook..."
echo "  Source: $HOOK_SRC"
echo ""

# Shared worktree repo (covers refinery/rig and polecats/obsidian/testAnt)
SHARED_GIT="$REPO_ROOT/.repo.git"
if [[ -d "$SHARED_GIT/hooks" ]]; then
    install_hook "$SHARED_GIT/hooks" "shared: refinery/rig + polecats/obsidian/testAnt"
fi

# crew/bob
CREW_HOOKS="$REPO_ROOT/crew/bob/.git/hooks"
if [[ -d "$REPO_ROOT/crew/bob/.git" ]]; then
    install_hook "$CREW_HOOKS" "crew/bob"
fi

# mayor/rig
MAYOR_HOOKS="$REPO_ROOT/mayor/rig/.git/hooks"
if [[ -d "$REPO_ROOT/mayor/rig/.git" ]]; then
    install_hook "$MAYOR_HOOKS" "mayor/rig"
fi

echo ""
echo "Done. All testAnt commits will now be scanned for coordinates and secrets."
