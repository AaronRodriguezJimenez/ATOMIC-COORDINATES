"""
 This process the result from SIESTA to create an xyz file
 for visualization of a given lattice
"""
import numpy as np

# Optimized Lattice from PBE/DZP SIESTA calculation:
#!/usr/bin/env python3
#!/usr/bin/env python3
"""
parse_lattice.py

Parse a simple lattice+atom block of the form:

  ax ay az
  bx by bz
  cx cy cz
  N
  s Z x y z
  s Z x y z
  ...

Where:
 - the first three lines are lattice vectors (Angstrom)
 - next line is integer N (# atoms)
 - next N lines: species_index atomic_number x y z

Produces an XYZ file; supports fractional coords, supercells, centering.

Usage:
  python parse_lattice.py input.txt out.xyz
  python parse_lattice.py input.txt out.xyz --frac      # if coords are fractional
  python parse_lattice.py input.txt out.xyz --rep 2 2 1 --center
"""

from typing import List, Tuple, Dict, Optional
import numpy as np
import argparse
import sys
import re

# Minimal atomic-number -> symbol mapping for typical elements (extend as needed)
ATOMIC_NUMBER_TO_SYMBOL = {
    1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 10: "Ne",
    11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P", 16: "S", 17: "Cl", 18: "Ar",
    19: "K", 20: "Ca", 21: "Sc", 22: "Ti", 23: "V", 24: "Cr", 25: "Mn", 26: "Fe",
    27: "Co", 28: "Ni", 29: "Cu", 30: "Zn", 47: "Ag", 79: "Au", 82: "Pb", 57:"La"
}

def parse_block(lines: List[str]) -> Tuple[np.ndarray, List[Tuple[str, np.ndarray]]]:
    """
    Parse lines in the specified format.
    Returns:
      lattice: 3x3 numpy array with row vectors
      atoms: list of (element_symbol, [x,y,z]) -> coords as floats (not converted)
    """
    # Remove empty lines and strip
    ln_clean = [l.strip() for l in lines if l.strip() != ""]
    if len(ln_clean) < 4:
        raise ValueError("Not enough lines to parse lattice + atom count.")

    # Read first 3 lines as lattice vectors (allow multiple spaces, tabs)
    lattice = []
    for i in range(3):
        parts = re.findall(r"[-+]?\d*\.\d+|\d+", ln_clean[i])
        if len(parts) < 3:
            raise ValueError(f"Cannot parse lattice vector on line {i+1}: '{ln_clean[i]}'")
        lattice.append([float(parts[0]), float(parts[1]), float(parts[2])])
    lattice = np.array(lattice, dtype=float)

    # Next line is number of atoms
    nat_line = ln_clean[3]
    try:
        nat = int(re.findall(r"\d+", nat_line)[0])
    except Exception as e:
        raise ValueError("Could not parse number of atoms from line: " + nat_line) from e

    # Following nat lines are atoms
    if len(ln_clean) < 4 + nat:
        raise ValueError(f"Expected {nat} atom lines but found {len(ln_clean)-4}.")

    atoms = []
    for i in range(nat):
        ln = ln_clean[4 + i]
        # Expect: species_index  atomic_number   x   y   z
        parts = ln.split()
        if len(parts) < 5:
            # try numeric extraction
            parts = re.findall(r"[-+]?\d*\.\d+|\d+|[A-Za-z]+", ln)
        # We expect numeric species_index and atomic_number in first two tokens
        try:
            atomic_number = int(parts[1])
            x = float(parts[2]); y = float(parts[3]); z = float(parts[4])
        except Exception as e:
            raise ValueError(f"Can't parse atom line: '{ln}'") from e

        symbol = ATOMIC_NUMBER_TO_SYMBOL.get(atomic_number, f"Z{atomic_number}")
        atoms.append((symbol, np.array([x, y, z], dtype=float)))

    return lattice, atoms

# Coordinate transforms
def frac_to_cart(lattice: np.ndarray, frac_coords: np.ndarray) -> np.ndarray:
    """Assume lattice rows are vectors a,b,c. cart = frac @ lattice"""
    return np.dot(frac_coords, lattice)

def cart_to_frac(lattice: np.ndarray, cart_coords: np.ndarray) -> np.ndarray:
    inv = np.linalg.inv(lattice)
    return np.dot(cart_coords, inv)

def build_supercell(lattice: np.ndarray, atoms: List[Tuple[str, np.ndarray]], reps: Tuple[int,int,int]) -> Tuple[np.ndarray, List[Tuple[str, np.ndarray]]]:
    na, nb, nc = reps
    # new lattice: scale each lattice vector by replication factor
    new_lattice = np.array([lattice[0]*na, lattice[1]*nb, lattice[2]*nc])
    # convert atom positions to fractional in original cell
    coords = np.array([pos for (_, pos) in atoms])
    fracs = cart_to_frac(lattice, coords)
    new_atoms: List[Tuple[str, np.ndarray]] = []
    for i in range(na):
        for j in range(nb):
            for k in range(nc):
                shift = np.array([i, j, k], dtype=float)
                for s, f in zip([s for (s,_) in atoms], fracs):
                    new_frac = (f + shift) / np.array([na, nb, nc], dtype=float)
                    cart = frac_to_cart(new_lattice, new_frac)
                    new_atoms.append((s, cart))
    return new_lattice, new_atoms

def center_slab_in_c(lattice: np.ndarray, atoms: List[Tuple[str, np.ndarray]]) -> List[Tuple[str, np.ndarray]]:
    coords = np.array([pos for (_, pos) in atoms])
    zmin, zmax = coords[:,2].min(), coords[:,2].max()
    slab_center = 0.5 * (zmin + zmax)
    cvec = lattice[2]
    c_len = np.linalg.norm(cvec)
    target = 0.5 * c_len
    shift = target - slab_center
    new_atoms = [(s, np.array([p[0], p[1], p[2] + shift])) for (s,p) in atoms]
    return new_atoms

def write_xyz(path: str, atoms: List[Tuple[str, np.ndarray]], comment: Optional[str] = None):
    with open(path, "w") as f:
        f.write(f"{len(atoms)}\n")
        f.write((comment if comment is not None else "Generated by parse_lattice.py") + "\n")
        for s, p in atoms:
            f.write(f"{s} {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n")

# --- CLI / main ------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description="Parse lattice+atoms block and write XYZ.")
    p.add_argument("infile", help="input text file containing the block")
    p.add_argument("outxyz", help="output xyz filename")
    p.add_argument("--frac", action="store_true", help="treat coordinates in file as fractional (direct). Default: Cartesian (Ang).")
    p.add_argument("--rep", nargs=3, type=int, default=(1,1,1), metavar=("NA","NB","NC"),
                   help="replication factors along a,b,c (default 1 1 1)")
    p.add_argument("--center", action="store_true", help="center slab in c-direction inside cell before writing")
    args = p.parse_args(argv)

    text = open(args.infile).read()
    lattice, atoms = parse_block(text.splitlines())

    # If coords are fractional, convert to cart
    if args.frac:
        frac_coords = np.array([pos for (_,pos) in atoms])
        cart_coords = frac_to_cart(lattice, frac_coords)
        atoms = [(s, cart_coords[i]) for i,(s,_) in enumerate(atoms)]

    # Optionally replicate into supercell
    reps = tuple(args.rep)
    if reps != (1,1,1):
        lattice, atoms = build_supercell(lattice, atoms, reps)

    # Optionally center slab in c
    if args.center:
        atoms = center_slab_in_c(lattice, atoms)

    write_xyz(args.outxyz, atoms, comment=f"lattice rows: {lattice.tolist()}")

    print(f"Wrote {args.outxyz} (atoms: {len(atoms)})")


if __name__ == "__main__":
    main()

    """
    Instructions usage:
    python parse_lattice.py in.txt fe.xyz
    
    If the coordinates in your file are fractional (direct), run:
    python parse_lattice.py in.txt fe.xyz --frac
    
    To make a 2×2 in-plane supercell and center the slab:
    python parse_lattice.py in.txt fe2x2.xyz --rep 2 2 1 --center
    """