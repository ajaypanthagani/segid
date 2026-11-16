"""Paint world and the model watching it."""
import json
import threading
import urllib.request

import numpy as np

from objectlearn.paint import PaintApp, PaintWorld
from web.server import serve

RED, BLUE = [220, 40, 40], [40, 80, 220]


def rect(x0, y0, x1, y1):
    ys = np.linspace(y0, y1, 5)
    return [[x, y] for y in ys for x in (x0, x1)] + [[x, y] for x in np.linspace(x0, x1, 12) for y in (y0, y1)]


def fill(x0, y0, x1, y1, color, r=6):
    pts, y = [], y0
    while y <= y1:
        pts += [[x0, y], [x1, y]]
        y += r
    return dict(t="stroke", pts=pts, color=color, r=r)


def run(app, actions_per_tick, t0=0.0, dt=0.125):
    out, t = None, t0
    for acts in actions_per_tick:
        out = app.tick(acts, t)
        t += dt
    return out, t


def test_parts_exist_only_where_something_is_drawn():
    w = PaintWorld()
    assert w.n == 0
    f0, _ = w.frame()
    assert not f0.vis.any()                                   # a blank canvas gives the model nothing
    w.stroke([[100, 100], [160, 100]], (30, 30, 200), 8)
    f1, canvas = w.frame()
    assert w.n > 5 and f1.vis[:w.n].all()
    assert tuple(canvas[100, 130]) == (30, 30, 200) and tuple(canvas[200, 130]) != (30, 30, 200)
    assert (w.p_local[:w.n, 1] > 85).all() and (w.p_local[:w.n, 1] < 115).all()     # only on the painted strip


def test_move_carries_parts():
    w = PaintWorld()
    w.stroke([[100, 100], [160, 100]], (30, 30, 200), 8)
    pid = w.active
    before = w.frame()[0].pos[:w.n, :2].copy()
    w.grab(130, 100)
    w.drag(0, 80)
    f, canvas = w.frame()
    assert w.pieces[pid].off.tolist() == [0, 80]
    assert tuple(canvas[180, 130]) == (30, 30, 200) and tuple(canvas[100, 130]) != (30, 30, 200)
    assert np.allclose(f.pos[:w.n, :2] - before, [0, 80]) and f.vis[:w.n].all()


def test_cut_splits_a_piece_into_two_that_move_independently():
    w = PaintWorld()
    w.stroke([[80, 100], [200, 100], [80, 120], [200, 120]], (30, 160, 30), 10)
    assert len(w.order) == 1
    assert w.cut([[140, 80], [140, 140]])
    assert len(w.order) == 2
    a, b = w.order
    xa = np.flatnonzero(w.pieces[a].mask.any(0)); xb = np.flatnonzero(w.pieces[b].mask.any(0))
    assert xa.max() < xb.min() or xb.max() < xa.min()
    assert not (w.pieces[a].mask & w.pieces[b].mask).any()
    # patches followed their pixels
    for pid in (a, b):
        sel = w.p_piece[:w.n] == pid
        ys = np.rint(w.p_local[:w.n][sel, 1]).astype(int); xs = np.rint(w.p_local[:w.n][sel, 0]).astype(int)
        assert sel.any() and w.pieces[pid].mask[ys, xs].all()
    assert not w.cut([[10, 10], [20, 20]])               # nothing there


def test_dragging_one_of_two_touching_things_away_separates_it_and_the_model_learns_labels():
    app = PaintApp()
    run(app, [[], [fill(100, 100, 150, 140, RED)], [], [dict(t="newobject"), fill(150, 100, 200, 140, BLUE)], [], []])
    seq = [[dict(t="grab", x=180, y=120)]]
    seq += [[dict(t="drag", dx=0, dy=-10)] for _ in range(6)]
    seq += [[dict(t="release")]] + [[] for _ in range(16)]                         # hold still: persistence + end of the event
    out, _ = run(app, seq, t0=1.0)
    assert out["stats"]["formal_objects"] == 2
    assert out["stats"]["pairs_opened"] > 3
    assert out["stats"]["samples"] > 0 and out["stats"]["events"] >= 1


def test_moving_a_lone_object_changes_nothing():
    app = PaintApp()
    run(app, [[], [fill(100, 100, 150, 140, RED)], [], []])
    seq = [[dict(t="grab", x=125, y=120)]] + [[dict(t="drag", dx=0, dy=-10)] for _ in range(6)] + [[dict(t="release")]] + [[] for _ in range(6)]
    out, _ = run(app, seq, t0=1.0)
    assert out["stats"]["formal_objects"] == 1 and out["stats"]["pairs_opened"] == 0


def test_cutting_an_object_and_dragging_one_half_away_divides_it():
    app = PaintApp()
    out, t = run(app, [[fill(90, 100, 190, 140, BLUE)]] + [[] for _ in range(6)])
    assert len(out["formal"]) == 1
    old = out["formal"][0]["id"]
    out, t = run(app, [[dict(t="cut", pts=[[140, 90], [140, 150]])]] + [[] for _ in range(3)], t0=t)
    assert [o["id"] for o in out["formal"]] == [old]            # cutting alone changes nothing
    seq = [[dict(t="grab", x=165, y=120)]] + [[dict(t="drag", dx=0, dy=9)] for _ in range(14)] + [[dict(t="release")]] + [[] for _ in range(16)]
    out, _ = run(app, seq, t0=t)
    assert out["stats"]["formal_objects"] == 2                 # the two halves
    ids = {o["id"] for o in out["formal"]}
    assert old not in ids                                      # the original object no longer exists
    assert all(o["parent"] == old for o in out["formal"])      # both parts record where they came from


def test_server_round_trip():
    srv, _ = serve(port=0)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{port}"
        assert b"Paint" in urllib.request.urlopen(base + "/").read()
        body = json.dumps({"actions": [fill(100, 100, 150, 130, RED)]}).encode()
        out = json.loads(urllib.request.urlopen(urllib.request.Request(base + "/tick", data=body, method="POST")).read())
        assert out["image"].startswith("data:image/png;base64,") and 10 < len(out["pts"]) < 120
        assert set(out) >= {"formal", "predicted", "stats", "events", "pieces", "seams"}
        bad = urllib.request.Request(base + "/tick", data=b"{nope", method="POST")
        try:
            urllib.request.urlopen(bad)
            raise AssertionError("expected 400")
        except urllib.error.HTTPError as e:
            assert e.code == 400
    finally:
        srv.shutdown()


def test_background_is_not_an_object_and_painted_things_are():
    app = PaintApp()
    out, t = run(app, [[], [], []])
    assert out["stats"]["formal_objects"] == 0 and out["formal"] == []           # blank paper: nothing
    out, t = run(app, [[fill(100, 100, 160, 150, RED)], [], [], [dict(t="cut", pts=[[130, 90], [130, 160]])], [], []], t0=t)
    assert out["stats"]["formal_objects"] == 1                                   # painted and cut, nothing moved: one object
    assert 5 < out["formal"][0]["n"] < 120                                       # only the painted patches
    assert len(out["pts"]) == out["formal"][0]["n"]                              # the model was sent nothing else


def test_separate_things_are_separate_objects_and_touching_things_are_one():
    app = PaintApp()
    out, t = run(app, [[], [fill(40, 40, 90, 80, RED)], [], [dict(t="newobject"), fill(200, 150, 250, 190, BLUE)], [], []])
    assert out["stats"]["formal_objects"] == 2                                   # far apart: never connected
    app = PaintApp()
    out, t = run(app, [[], [fill(100, 100, 150, 140, RED)], [], [dict(t="newobject"), fill(150, 100, 200, 140, BLUE)], [], []])
    assert out["stats"]["formal_objects"] == 1                                   # touching: one object, as the definition says


def test_an_image_becomes_one_object_that_can_be_cut_and_split():
    import cv2, base64
    import numpy as np
    from objectlearn.paint import PaintApp
    img = np.zeros((60, 100, 3), np.uint8); img[:, :50] = (40, 60, 220); img[:, 50:] = (60, 200, 60)
    png = "data:image/png;base64," + base64.b64encode(cv2.imencode(".png", img)[1].tobytes()).decode()
    app = PaintApp(); t = 0.0
    def tick(*a):
        nonlocal t; t += 0.1; return app.tick(list(a), t)
    r = tick({"t": "image", "data": png})
    assert len(r["formal"]) == 1
    tick({"t": "cut", "pts": [[160, 50], [160, 190]]})
    for _ in range(3): tick()
    tick({"t": "grab", "x": 200, "y": 120})
    for _ in range(12): r = tick({"t": "drag", "dx": 4, "dy": 0})
    tick({"t": "release"})
    for _ in range(8): r = tick()
    assert len(r["formal"]) == 2
