import matplotlib.pyplot as plt

# Use LaTeX with Times-compatible text and math fonts
plt.rcParams["text.usetex"] = True
plt.rcParams["text.latex.preamble"] = (
    r"\usepackage{newtxtext,newtxmath}"
)

def create_label(filename):
    fig = plt.figure(figsize=(5, 1.5))

    # Equation
    fig.text(
        0.5, 0.65,
        # The |+> state:
      #  r"$|+\rangle = \frac{1}{\sqrt{2}} "
      #  r"\left(|0\rangle + |1\rangle\right)$",
#        r"$T = a_1a_2^\dagger - a_2a_1^\dagger$",
        #Trotterized operator
#        r"$e^{a\sum_j P_j}\approx \prod_j e^{aP_j}$",
        #Exact UCC operator
        #r"$|\psi(a)\rangle = \prod_j e^{aP_j} |\psi_0\rangle$",
        #r"$|\phi_1\rangle=a_0^\dagger a_1^\dagger|\mathrm{vac}\rangle$",
        #r"$|\phi_2\rangle=a_0^\dagger a_2^\dagger|\mathrm{vac}\rangle$",
        r"$|\psi(a)\rangle = e^{aT}|\psi_0\rangle$",
        ha="center",
        va="center",
        fontsize=40
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


create_label("eq2.png")
