import numpy as np, matplotlib.pyplot as plt, csv

N = 64 * 64
v = np.fromfile("results/example.bin", dtype="<f4")
x, adv, grad = (v[i * N:(i + 1) * N].reshape(64, 64) for i in range(3))
before, after, level = v[3 * N:]

fig, ax = plt.subplots(1, 4, figsize=(16, 4.6))
ax[0].imshow(x, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
ax[0].set_title(f"Original\nscore {before:+.2f} -> PNEUMONIA")
ax[1].imshow(adv, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
ax[1].set_title(f"Attacked (eps = {level:.0f}/255)\nscore {after:+.2f} -> NORMAL")
diff = adv - x
m = max(np.abs(diff).max(), 1e-8)
ax[2].imshow(diff, cmap="seismic", vmin=-m, vmax=m, interpolation="nearest")
ax[2].set_title("Change made (amplified)\nred = brighter, blue = darker")
ax[3].imshow(np.abs(grad), cmap="hot", interpolation="nearest")
ax[3].set_title("Pixel sensitivity |gradient|\nbrighter = more influence")
for a in ax:
    a.axis("off")
plt.tight_layout()
plt.savefig("results/attack_example.png", dpi=150)

rows = list(csv.DictReader(open("results/flip_rates.csv")))
eps = [float(r["eps_levels"]) for r in rows]
plt.figure(figsize=(6.5, 4.2))
for key, name in [("pneumonia_to_normal", "Pneumonia -> Normal (missed disease)"),
                  ("normal_to_pneumonia", "Normal -> Pneumonia (false alarm)"),
                  ("overall", "Overall")]:
    plt.plot(eps, [float(r[key]) for r in rows], marker="o", label=name)
plt.xscale("log", base=2)
plt.xticks(eps, [f"{int(e)}/255" for e in eps])
plt.xlabel("Perturbation size per pixel (eps)")
plt.ylabel("% of correct diagnoses flipped")
plt.title("FGSM attack on the hand-written Rust engine")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("results/flip_rates.png", dpi=150)
print("Saved results/attack_example.png and results/flip_rates.png")