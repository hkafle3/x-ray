import numpy as np, struct

d = np.load("data/test_set.npz")
X, y = d["X"], d["y"]

with open("test_all.bin", "wb") as f:
    f.write(struct.pack("<I", len(X)))
    for x, label in zip(X, y):
        f.write(x.astype("<f4").tobytes())  
        f.write(struct.pack("<f", label))    # 0 = normal, 1 = pneumonia

print(f"Wrote {len(X)} test scans to test_all.bin")