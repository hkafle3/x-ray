import numpy as np, torch, torch.nn as nn, struct
from pathlib import Path
from PIL import Image

DATA = Path("data/chest_xray")
SIZE = 64          # images shrunk to 64x64
HIDDEN = 128       # neurons
torch.manual_seed(0); np.random.seed(0)

def load_split(split):
    X, y = [], []
    for label, cls in [(0, "NORMAL"), (1, "PNEUMONIA")]:
        for p in sorted((DATA / split / cls).iterdir()):
            if p.suffix.lower() not in {".jpeg", ".jpg", ".png"}:
                continue
            img = Image.open(p).convert("L").resize((SIZE, SIZE))
            X.append(np.asarray(img, dtype=np.float32).flatten() / 255.0)
            y.append(label)
    return np.stack(X), np.array(y, dtype=np.float32)

print("Loading images (takes a minute)...")
X_train, y_train = load_split("train")
X_test, y_test = load_split("test")

idx = np.random.permutation(len(X_train))
n_val = len(idx) // 10
X_val, y_val = X_train[idx[:n_val]], y_train[idx[:n_val]]
X_tr, y_tr = X_train[idx[n_val:]], y_train[idx[n_val:]]

np.savez("data/test_set.npz", X=X_test, y=y_test)  # reused in Phase 3

model = nn.Sequential(nn.Linear(SIZE * SIZE, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 1))

pos_weight = torch.tensor([(y_tr == 0).sum() / (y_tr == 1).sum()], dtype=torch.float32)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

def balanced_acc(X, y):
    with torch.no_grad():
        pred = (model(torch.tensor(X)).squeeze(1) > 0).numpy()
    caught = (pred[y == 1] == 1).mean()      # pneumonia correctly flagged
    cleared = (pred[y == 0] == 0).mean()     # normal correctly cleared
    return (caught + cleared) / 2, caught, cleared

Xt, yt = torch.tensor(X_tr), torch.tensor(y_tr)
for epoch in range(15):
    perm = torch.randperm(len(Xt))
    for i in range(0, len(Xt), 64):
        b = perm[i:i + 64]
        loss = loss_fn(model(Xt[b]).squeeze(1), yt[b])
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"epoch {epoch + 1:2d}  val balanced acc {balanced_acc(X_val, y_val)[0]:.3f}")

ba, caught, cleared = balanced_acc(X_test, y_test)
print(f"\nTEST balanced acc {ba:.3f}  (pneumonia caught {caught:.3f}, normal cleared {cleared:.3f})")

W1 = model[0].weight.detach().numpy()  
b1 = model[0].bias.detach().numpy()  
W2 = model[2].weight.detach().numpy()  
b2 = model[2].bias.detach().numpy()    

with open("weights.bin", "wb") as f:
    f.write(struct.pack("<3I", SIZE * SIZE, HIDDEN, 1))   # header: layer sizes
    for arr in (W1, b1, W2, b2):
        f.write(np.ascontiguousarray(arr, dtype="<f4").tobytes())

pick = np.r_[0:10, len(X_test) - 10:len(X_test)]   # 10 normal + 10 pneumonia
with torch.no_grad():
    logits = model(torch.tensor(X_test[pick])).squeeze(1).numpy()

with open("test_cases.bin", "wb") as f:
    f.write(struct.pack("<I", len(pick)))
    for j, i in enumerate(pick):
        f.write(X_test[i].astype("<f4").tobytes())               # 4096 pixels
        f.write(struct.pack("<ff", y_test[i], logits[j]))       # true label, Python's score

print("Saved weights.bin and test_cases.bin")