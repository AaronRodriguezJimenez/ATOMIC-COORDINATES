
import matplotlib.pyplot as plt

# Use LaTeX with Times-compatible text and math fonts
plt.rcParams["text.usetex"] = True
plt.rcParams["text.latex.preamble"] = (
    r"\usepackage{newtxtext,newtxmath}"
)


def create_label(filename, equation, figsize=(9, 1.5), fontsize=40):
    fig = plt.figure(figsize=figsize)

    fig.text(
        0.5, 0.5,
        equation,
        ha="center",
        va="center",
        fontsize=fontsize
    )

    # Remove axes and save a transparent, high-resolution PNG
    plt.axis("off")
    fig.savefig(
        filename,
        transparent=True,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.09
    )
    plt.close(fig)

    print(f"Saved: {filename}")


# 1. Initial state after the first Hadamard
create_label(
    "hadamard_1_plus_state.png",
    r"$|+\rangle = \frac{1}{\sqrt{2}}"
    r"\left(|0\rangle + |1\rangle\right)$",
    figsize=(6, 1.5)
)


# 2. State after controlled-U
create_label(
    "hadamard_2_controlled_U.png",
    r"$\frac{1}{\sqrt{2}}\left("
    r"|0\rangle|\psi\rangle"
    r"+|1\rangle U|\psi\rangle\right)$",
    figsize=(9, 1.5),
    fontsize=34
)


# 3. State after the second Hadamard
create_label(
    "hadamard_3_second_H.png",
    r"$\frac{1}{2}\left["
    r"|0\rangle\left(|\psi\rangle+U|\psi\rangle\right)"
    r"+|1\rangle\left(|\psi\rangle-U|\psi\rangle\right)"
    r"\right]$",
    figsize=(12, 1.8),
    fontsize=30
)


# 4. Measurement probabilities: real part
create_label(
    "hadamard_4_real_probabilities.png",
    r"$P(0)=\frac{1+\operatorname{Re}\langle\psi|U|\psi\rangle}{2}"
    r"\qquad{}"
    r"P(1)=\frac{1-\operatorname{Re}\langle\psi|U|\psi\rangle}{2}$",
    figsize=(13, 1.5),
    fontsize=27
)


# 5. Imaginary part: insert S-dagger before the final Hadamard
create_label(
    "hadamard_5_imaginary_probabilities.png",
    r"$P(0)=\frac{1+\operatorname{Im}\langle\psi|U|\psi\rangle}{2}"
    r"\qquad{}"
    r"P(1)=\frac{1-\operatorname{Im}\langle\psi|U|\psi\rangle}{2}$",
    figsize=(13, 1.5),
    fontsize=27
)


# 6. General phase-shifted test
create_label(
    "hadamard_6_phase_shift.png",
    r"$P(0)=\frac{1}{2}\left[1+\operatorname{Re}"
    r"\left(e^{i\phi}\langle\psi|U|\psi\rangle\right)\right]$",
    figsize=(12, 1.5),
    fontsize=30
)

print("All six figures created successfully.")
