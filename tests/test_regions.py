import numpy as np

from objectlearn.regions import T, TRAIT_NAMES, ObjectGeom, candidate_masks, candidates, part_traits


def shelf(s=5.0):
    """A shelf (frame and back) with three books standing on a board: patches on a lattice, a signal per patch."""
    pts, sig, book = [], [], []
    for gy in range(20):
        for gx in range(16):
            frame = gx < 1 or gx > 14 or gy < 1 or gy > 17
            b = None if frame else (1 + (gx - 1) // 5 if gy >= 6 else None)
            pts.append((gx * s, gy * s))
            val = 60.0 if b is None else [0, 200.0, 120.0, 30.0][b]
            sig.append((val, 128.0, 128.0 + (0 if b is None else 10 * b)))
            book.append(0 if b is None else b)
    return np.array(pts), np.array(sig), np.array(book)


def test_region_traits_shape_and_finite():
    pos, sig, _ = shelf()
    X, masks, kind = candidates(pos, sig, 5.0)
    assert X.shape == (len(masks), T) and len(TRAIT_NAMES) == T and np.isfinite(X).all()
    assert set(kind) == {"cut", "region"}


def test_a_book_in_the_middle_is_a_candidate():
    pos, sig, book = shelf()
    _, masks, kind = candidates(pos, sig, 5.0)
    want = book == 2                                            # bounded by the rest on three sides: no straight cut gives it
    hits = [k for k in range(len(masks)) if (masks[k] == want).all()]
    assert hits and all(kind[k] == "region" for k in hits)


def test_traits_ignore_position_and_signal_offset():
    pos, sig, book = shelf()
    g0, g1 = ObjectGeom(pos, sig, 5.0), ObjectGeom(pos + np.array([33.0, -7.0]), sig + 25.0, 5.0)
    m = book == 2
    assert np.allclose(part_traits(g0, m), part_traits(g1, m), atol=1e-6)


def test_contact_and_edge_traits_describe_the_book():
    pos, sig, book = shelf()
    geo = ObjectGeom(pos, sig, 5.0)
    t = dict(zip(TRAIT_NAMES, part_traits(geo, book == 2)))
    assert t["contact_share"] > 0.5                             # most of its outline touches the rest
    assert t["edge_strong"] > 0.9                               # the signal changes all along that contact
    assert t["rest_pieces"] == 0                                # taking it out leaves the rest in one piece
    u = dict(zip(TRAIT_NAMES, part_traits(geo, (book == 2) & (pos[:, 1] < 50))))   # the top half of the book only
    assert u["edge_strong"] < t["edge_strong"]


def test_candidates_have_no_duplicates():
    pos, sig, _ = shelf()
    masks, _ = candidate_masks(ObjectGeom(pos, sig, 5.0))
    assert len({np.packbits(m).tobytes() for m in masks}) == len(masks)
