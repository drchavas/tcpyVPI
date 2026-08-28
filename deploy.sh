#!/usr/bin/env bash
#
# deploy.sh - push a new version to GitHub. Mirrors the manual workflow:
#
#     1. bump the version in setup.py AND <package>/__init__.py
#     2. git add .
#     3. git commit -m "..."
#     4. git push origin main
#
# ...with one guard in front of it: it refuses to push unless both version
# strings were actually bumped and don't already exist on PyPI. That is the
# failure this is meant to catch, because a stale version number makes the
# GitHub Action fail at the PyPI upload step, after the push looks fine.
#
# It does NOT tag and does NOT publish. You create the Release in the GitHub
# UI, which creates the tag and triggers .github/workflows/publish.yml.
#
# Usage:
#   ./deploy.sh "commit message"          # the normal path
#   ./deploy.sh --check "msg"             # also build + run the regression test
#   ./deploy.sh --dry-run "msg"           # show what would happen, change nothing
#
set -euo pipefail

DRY_RUN=0; FULL=0
while [[ "${1:-}" == --* ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --check|--full) FULL=1 ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac
  shift
done
MSG="${1:-}"

cd "$(dirname "${BASH_SOURCE[0]}")"

bold=$(tput bold 2>/dev/null || true); dim=$(tput dim 2>/dev/null || true)
red=$(tput setaf 1 2>/dev/null || true); grn=$(tput setaf 2 2>/dev/null || true)
ylw=$(tput setaf 3 2>/dev/null || true); rst=$(tput sgr0 2>/dev/null || true)
ok()   { echo "  ${grn}ok${rst}   $*"; }
warn() { echo "  ${ylw}warn${rst} $*"; }
die()  { echo "  ${red}FAIL${rst} $*" >&2; exit 1; }

# ---------------------------------------------------------------- stale lock
# A crashed git process (or a Drive sync hiccup) leaves .git/index.lock behind
# and every write operation then fails. Safe to clear when no git is running.
if [[ -f .git/index.lock ]] && ! pgrep -x git >/dev/null 2>&1; then
  warn "clearing stale .git/index.lock"
  [[ $DRY_RUN == 0 ]] && rm -f .git/index.lock
fi

# ---------------------------------------------------------------- version guard
V_INIT=$(sed -n 's/^__version__ *= *"\(.*\)"/\1/p' tcpyVPI/__init__.py | head -1)
V_SETUP=$(sed -n "s/^ *version='\(.*\)',/\1/p" setup.py | head -1)
[[ -n "$V_INIT" && -n "$V_SETUP" ]] || die "could not read the version from setup.py / tcpyVPI/__init__.py"
[[ "$V_INIT" == "$V_SETUP" ]] || die "version mismatch - setup.py=$V_SETUP but __init__.py=$V_INIT (bump BOTH)"
VERSION="$V_INIT"
ok "version $VERSION (setup.py and __init__.py agree)"

ALREADY_RELEASED=0
if command -v curl >/dev/null 2>&1; then
  if curl -fsS --max-time 10 "https://pypi.org/pypi/tcpyVPI/${VERSION}/json" >/dev/null 2>&1; then
    ALREADY_RELEASED=1
    warn "version $VERSION is ALREADY on PyPI."
    echo "         Fine if you are just pushing commits to main. But if you meant to"
    echo "         cut a new release, bump the version in setup.py AND"
    echo "         tcpyVPI/__init__.py first, or the Action will fail at upload."
  else
    ok "version $VERSION is not yet on PyPI - ready to release"
  fi
fi

[[ -z "$MSG" ]] && MSG=$([[ $ALREADY_RELEASED == 1 ]] && echo "Update" || echo "Release $VERSION")

# ---------------------------------------------------------------- optional checks
if [[ $FULL == 1 ]]; then
  if [[ -f tests/check_unit_fixes.py ]]; then
    python3 tests/check_unit_fixes.py || die "regression test failed"
    ok "regression test passed"
  fi
  if python3 -c "import build, twine" >/dev/null 2>&1; then
    rm -rf dist build ./*.egg-info
    python3 -m build >/tmp/tcpyvpi_build.log 2>&1 || { tail -20 /tmp/tcpyvpi_build.log; die "build failed"; }
    python3 -m twine check dist/* >/dev/null || die "twine check failed"
    ok "build + twine check passed ($(ls dist | tr '\n' ' '))"
  else
    warn "build/twine not installed - skipping the build check"
  fi
fi

# ---------------------------------------------------------------- the 4 steps
echo
echo "${bold}Changes:${rst}"
git status --short
if [[ -z "$(git status --porcelain)" ]]; then
  warn "working tree is clean - nothing to commit or push"
  exit 0
fi
echo
echo "  commit message: ${bold}${MSG}${rst}"
echo "  push to:        $(git remote get-url origin) (main)"

if [[ $DRY_RUN == 1 ]]; then
  echo; echo "${ylw}Dry run - nothing committed or pushed.${rst}"; exit 0
fi

echo
read -r -p "Push to GitHub? [y/N] " REPLY
[[ "$REPLY" =~ ^[Yy]$ ]] || { echo "Aborted."; exit 1; }

git add .
git commit -m "$MSG"
git push origin main

# ---------------------------------------------------------------- next steps
SLUG=$(git remote get-url origin | sed -E 's#.*github\.com[:/]##; s#\.git$##')
if [[ $ALREADY_RELEASED == 1 ]]; then
  cat <<EOF

${grn}${bold}Pushed to main.${rst} No release cut: $VERSION is already on PyPI.
To publish a new version, bump setup.py + tcpyVPI/__init__.py and run this again.
EOF
else
  cat <<EOF

${grn}${bold}Pushed.${rst} PyPI is untouched until you create the Release.

  1. https://github.com/${SLUG}/releases/new?tag=v${VERSION}
     (leave "Create new tag on publish" selected - the UI makes the tag)
  2. Publish release -> triggers the Action -> PyPI
  3. Watch: https://github.com/${SLUG}/actions
EOF
fi
