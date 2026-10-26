import json, os, sys, time
B, hold = sys.argv[1], sys.argv[2]
cells = json.load(open(B + "/full/cells-125.json"))["cells"]
rt = [c for c in cells if c["environment"] == "robotwin"]
for d in ("full", "control"):
    out = B + "/" + d
    held = []
    for c in rt:
        k = "rt-%s-s%s" % (c["task"], c["seed"])
        r = os.path.join(out, k, "result.json")
        if os.path.exists(r) and json.load(open(r)).get("status") != "boot_error":
            continue
        claim = os.path.join(out, ".claim-" + k)
        live = False
        if os.path.exists(claim):
            try:
                pid = int(open(claim).read().split()[0]); os.kill(pid, 0); live = True
            except Exception:
                pass
        if live:
            continue
        with open(claim, "w") as f:
            f.write("%s %.0f drain: paused until a router booking exists\n" % (hold, time.time()))
        held.append(k)
    print(d, "held", len(held), held[:6])
