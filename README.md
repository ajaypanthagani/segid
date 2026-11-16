# objectlearn: objects from observed separation, with an online interface learner

Implements `Object_Learning.docx`: Part I (the formal definition of an object) and Part II (a learned layer that
remembers which seams tend to separate).

```
objectlearn/formal.py           Part I, exactly: A_t, N_t, I_t, O_t from patch positions + visibility
objectlearn/learned.py          sample memory, relative flat-top kernel experts, Beta bounds, exponential weights
objectlearn/agent.py            glue: labels flow formal -> learned; only pi flows back
objectlearn/descriptor.py       phi(a,b) from 3-D patches (simulator)
objectlearn/descriptor_image.py phi(a,b) from raw local image structure (paint screen)
objectlearn/session.py          a stream of frames -> objects, learning events, predicted seams
objectlearn/paint.py            the paint world: pieces, patches, cut, move, render
objectlearn/sim.py              3-D boxes on a floor (used for the learning-curve experiments)
web/                            the paint page and its local server
tests/                          formal-layer cases and invariants, learned-layer contracts, paint world
experiments/                    run_online.py (3-D learning curve), paint_learning.py, summarize.py, plot.py
```

Setup and tests: `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/python -m pytest -q tests`

## The paint page

```
PYTHONPATH=. .venv/bin/python web/server.py            # then open http://localhost:8765
PYTHONPATH=. .venv/bin/python web/server.py --memory mem.npz     # keep what it learns between runs
```

Paint objects (or bring in a picture with **Add image…**; transparent pixels stay paper, a photo is one rectangle), cut them into parts with the **Cut** tool, then **Move** a part away from what it touches and hold
still for a second.  The model is given what a perfect tracker would give it for the things that are drawn, and nothing
else: small sensing patches that keep their identity as things move, whether each is currently visible, and the rendered
image.  The blank paper has no patches, so it is never an object and the model is never told what is background.

Naming: **patches** are the sampling points (the "sensed parts" of the definition's section 1), a grid like pixels; they
mean nothing until something splits.  **Parts** are what an object divides into: the distinct objects that exist after a
split, as parts of the original.  The model is not told what parts there will be or how you cut.

- Blank paper: no objects.  A painted block: one object.  Things that touch are one object (as the definition says);
  things that never touched are separate objects.  When an object divides it ceases to exist: every part is a new object with a new id and a record of the object it split
  from (the old id is never reused).  A part becomes its own object when the patch pairs joining it to the rest
  have been seen to move apart.
- **Formal** view: the objects of Part I.  **Predicted** view: the same, plus what it has learned about which seams tend
  to separate.  Orange ticks mark seams it expects to separate before anything moves; the Predicted view splits an
  object only when a whole cut line is marked.
- Dragging a part over another thing hides what it covers, so drag it clear.
- Each kind of change (paint, cut, move) is sent as its own frame.  If a part is painted and moved before the model has
  seen it, it is "apart when first determined" and, by the definition, never had an interface.

How fast it learns (`experiments/paint_learning.py`: random blocks, each painted, cut and half dragged away, then probed
on a fresh painted-and-cut block that is never moved): nothing flagged for the first 4 examples; from the 5th it marks
real cut seams, 55% of them by the 16th, with 0 false seams up to example 11 and 3 of 24 flagged at example 16.  It never
yet completes a whole cut line, so the Predicted view does not yet split a fresh block before it is moved; the orange ticks
are where learning shows.

## Results on the 3-D simulator (3 seeds, 60 episodes of 3 random interventions, memory starts empty)

Frozen probes on fresh scenes; the "bottle" scenes (tall thin bodies with caps) are never trained on.

| episodes | train precision / recall / ARI | bottle precision / recall / ARI |
|---|---|---|
| 10 | 0.96 / 0.59 / 0.14 | 0.98 / 0.61 / 0.16 |
| 30 | 0.95 / 0.75 / 0.65 | 0.98 / 0.74 / 0.74 |
| 60 | 0.94 / 0.79 / 0.81 | 0.97 / 0.77 / 0.84 |

Known weakness: body-on-body seams that are geometrically flush (a cap the same width as the bottle) are mostly
called coherent or unknown until one is seen to open.  Nothing in the local geometry separates them from a continuous
surface.

## Deviations from the paper (to fold back into it)
- Kernel: flat out to the k_nn-th nearest sample, decaying beyond, with support falling off for queries far from all
  samples.  The additive Gaussian kernel was too flat in 363 dimensions.
- Event budget: total weight per outcome class per intervention is capped (3 in the simulator, 16 on the paint page,
  where each example is clean and learning has to be quick) and rows are subsampled (64).  The paint page also uses a
  smaller neighbourhood (k_nn = 24 instead of 96), because each example yields only a few separating pairs.
- Descriptors: the paper describes a pair when it is first formed.  On the paint page that is before a cut is drawn, so
  the seam would never be seen.  The model instead describes pairs from the last still frame before a move, and re-states
  its predictions from the current picture every few frames.
- pi: blends 1 - p_hat with a default continuity term by support (normal alignment for 3-D patches, colour similarity
  for images).  Without a default, cold-start admission is arbitrary.
- Statements about interfaces are re-made from the current memory after each learning event (`Agent.refresh`), so
  pairs formed before anything was learned are not stuck at "unknown".
- Every object is reported, including the one the whole scene starts as (the definition: "the whole scene may begin
  as one object; objects split off as they are observed to move apart").  The paint page therefore shows one object
  at rest and a new one each time a part detaches.  Every component with at least one visible patch is drawn (patches sit on a 10 px grid, so a part smaller than about
  20 px has only a few).
- An event with too few outcome pairs is discarded as noise.
- eps must exceed the largest within-surface neighbour distance by several noise widths.
- A short push does not divide a body from what it covered: leading-edge pairs are hidden, stay undetermined, and
  never open.
- Thresholds theta_d = 0.65 and theta_c = 0.35 were tuned on three training-kind simulator scenes.

## Dropped: camera input
A webcam front end (tracked image points) was built and removed.  Tracking mistakes near object edges produced about
40% wrong "opened" labels, and a single bridging link keeps two objects fused, so the formal layer rarely divided
anything on real video.  The paint page gives the model exact patch correspondence, which is what the definition assumes.

## Learning from a picture (`experiments/image_learning.py`)
A bottle photo (white background removed on import) was added, its cap cut off and dragged away, over and over, with a
fresh uncut copy probed after each round (patch spacing 5 px, `--patch 5`; the cut is not drawn onto the picture).
It is one object when added and two once the cap moves, every round. The model marked the cap seam from round 7 (2 ticks), and by
round 16 the Predicted view split the cap off an uncut, unmoved copy. By round 25 two stray ticks elsewhere also split
extra pieces (4 predicted objects). The same picture each time, so this shows memory of one picture's seam, not yet generalisation.
At the default 10 px spacing a cap this small has too few patches to connect to the neck.

## The maths panel
**Inspect** tool: click a seam and the page shows the computation for it with the real numbers: the distance to stored examples,
the kernel weights, the nearest examples and their outcomes, the per-expert evidence and weights, p-hat with its Beta bounds, and the
statement (`Session.explain`, endpoint `/explain`, equations rendered with KaTeX from cdnjs; plain text if it cannot load).

## The latent space (foundations)

`objectlearn/traits.py` describes a candidate part (the smaller side of a straight cut across an object, swept at four directions)
by physical, signal-agnostic traits: width profile either side of the cut relative to the cut, centre-line shift, how far each
side extends, size of the signal change at the cut against the usual change between rows, and a few shape ratios.
No colour value, position or absolute size is used, so candidates of different objects can be compared.

`objectlearn/latent.py` is the memory: each entry is a trait vector plus recorded behaviours (1 seen, 0 seen not to happen, NaN
not observed); here the one behaviour is "separates" (decided by the definition). A new candidate is read from its nearest entries;
how much each trait counts comes from how it co-varies with the behaviour across the stored entries. It abstains far from everything stored.

`experiments/latent_pilot.py` trains and tests it on the procedural bottles (outcomes of the cut-and-move protocol are written in
directly, not simulated). Not yet connected to the formal layer or the app.

### Latent space in the app

    PYTHONPATH=. .venv/bin/python web/server.py --latent --mode patch --patch 5 --memory test_bottles/latent_bottles_dragfirst.npz

Events train it (a part that comes away gives a "separates" entry for exactly that cut; an object or piece carried whole gives weak
"carried" entries for its cuts). In a still picture each object the definition has is scored on every candidate cut and the cuts the
latent space says separate are drawn as predicted objects (hypotheses; the formal layer is untouched). Inspect explains a cut.
`experiments/latent_events.py` trains through events and scores with the old strict check and a tolerant one (cap within two patch rows).

## Parts of any shape, and the paper's experiments

`objectlearn/regions.py` generalises the latent space from straight cuts to parts of any shape (a book among books, one cup among
several on a table, a drawer, fruit in a bowl). Candidates come from straight cuts and from regions of similar sensed signal (holes
filled); every candidate is described by the same 26 physical traits (size and shape, how it touches the rest, how the signal changes
across that contact, where it sits). The session learns from the definition: each piece that came away gives a "separates" entry with
the traits of exactly those patches; pieces carried whole give weak "carried" entries. Its settings (`LATENT_SETTINGS`,
`HELD_WEIGHT` in `objectlearn/session.py`) were chosen on validation scenes.

    PYTHONPATH=. .venv/bin/python web/server.py --latent --mode patch --patch 5 --memory test_bottles/latent_scenes.npz

`test_bottles/latent_scenes.npz` is the memory from all 840 training scenes of the paper (21 families).

`experiments/paper/`: `scenes.py` (21 procedural families: single objects with caps and lids, arrangements, things that never come
apart), `collect.py` (runs the real system on every scene with a scripted hand and caches its entries and candidates),
`validate.py` (chooses the settings on validation scenes), `run_experiments.py` and `run_all.sh` (every experiment),
`check_equivalence.py` (the cached scoring reproduces the live session), `figures.py` (figures, tables and the numbers quoted in
the text). The paper is in `paper/`.
