
import matplotlib.pyplot as plt

# Use LaTeX with Times-compatible text and math fonts
plt.rcParams["text.usetex"] = True
plt.rcParams["text.latex.preamble"] = (
    r"\usepackage{newtxtext,newtxmath}"
)


def create_label(name, bond, distance, filename):
    fig = plt.figure(figsize=(5, 1.5))

    # Molecule name
    fig.text(
        0.5, 0.65,
        rf"\textrm{{{name}}}",
        ha="center",
        va="center",
        fontsize=30
    )

    # Bond label and distance
    fig.text(
        0.5, 0.35,
        rf"$\left(\mathrm{{{bond}}}: {distance}\,\AA\right)$",
        ha="center",
        va="center",
        fontsize=24
    )

    # Remove axes and save a transparent, high-resolution PNG
    plt.axis("off")
    fig.savefig(
        filename,
        transparent=True,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.05
    )
    plt.close(fig)

    print(f"Saved: {filename}")


# Generate all five labels
create_label("Adrenaline",    "Ga-N", "1.62", "adrenaline.png")
create_label("Dopamine",      "Ga-O", "1.63", "Dopa.png")
create_label("Noradrenaline", "Ga-O", "1.60", "Noradren.png")
create_label("Psilocin",      "Ga-O", "1.64", "Psilocin.png")
create_label("Serotonin",     "Ga-O", "1.63", "Serotonin.png")
