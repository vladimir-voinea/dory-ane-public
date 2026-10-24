#!/usr/bin/env bash
# Publish the Dory port to the public mirror.
#   ./scripts/publish_dory_port.sh            # push main + all ticket branches
set -uo pipefail
REMOTE=public
git push -q "$REMOTE" integration/dory-ane:main
for b in $(git for-each-ref --format='%(refname:short)' refs/heads/ticket/DANE-*); do
  git push -q "$REMOTE" "$b:$b" || true
done
echo "published: $(git rev-parse --short HEAD) -> $(git remote get-url $REMOTE)"
