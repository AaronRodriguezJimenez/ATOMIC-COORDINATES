# expand_and_write_xyz.py
# Expands Wyckoff reps with pymatgen and writes .xyz (1x1x1 and 2x2x1 supercell)
from pymatgen.core import Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# --- lattice (conventional orthorhombic) ---
a, b, c = 5.36, 13.10, 5.54
lat = Lattice.from_parameters(a, b, c, 90, 90, 90)

# --- Wyckoff representatives (fractional coords) ---
species = ["Cu","O","La","O"]
frac_coords = [
    [0.0,      0.0,      0.0    ],   # Cu  4a
    [1/4.0,    0.988766, 3/4.0  ],   # O   8e
    [0.0,      0.638989, 0.985346],  # La  8f
    [1/2.0,    0.68678,  0.940211]   # O   8f
]

s = Structure(lat, species, frac_coords)

# expand to conventional standard cell (pymatgen will create full conv cell)
sga = SpacegroupAnalyzer(s, symprec=1e-2)
conv = sga.get_conventional_standard_structure()

print("Space group:", sga.get_space_group_symbol(), sga.get_space_group_number())
print("Number of sites in conventional cell:", len(conv))

def write_xyz_from_structure(structure, filename):
    # structure: pymatgen Structure with sites having .coords (Cartesian) and .species_string
    with open(filename, "w") as fh:
        fh.write(f"{len(structure)}\n")
        fh.write(f"{filename} generated from pymatgen (Cartesian Å)\n")
        for site in structure:
            x,y,z = site.coords
            fh.write(f"{site.species_string} {x:12.6f} {y:12.6f} {z:12.6f}\n")

# write 1x1x1
write_xyz_from_structure(conv, "La2CuO4_conventional_full.xyz")

# write supercell (2x2x1)
conv_super = conv * (2,2,2)
write_xyz_from_structure(conv_super, "La2CuO4_conventional_2x2x1_supercell.xyz")

print("Wrote: La2CuO4_conventional_full.xyz and La2CuO4_conventional_2x2x1_supercell.xyz")