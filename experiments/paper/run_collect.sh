#!/bin/zsh
# Collect everything the paper's experiments read.  Train seeds 1000+, test seeds 50000+, transformed tests 70000+.
cd "$(dirname $0)/../.."
FAMS=(bookshelf cabinet table fruit_bowl pencil_cup book_stack bottle jar pen tube spray cup blocks lipstick hammer screwdriver vase goblet chair mug desk_lamp)
{
  for f in $FAMS; do echo "train $f 40 1000"; done
  for f in $FAMS; do echo "probe $f 25 50000 none"; done
  for k in rot90 rot180 scale0.7 recolour dark noise; do for f in $FAMS; do echo "probe $f 12 70000 $k"; done; done
} | xargs -P 8 -L 1 sh -c 'PYTHONPATH=. .venv/bin/python -W ignore experiments/paper/collect.py "$@"' _
