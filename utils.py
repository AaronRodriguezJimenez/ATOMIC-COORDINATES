def center_xyz(input_xyz, output_xyz):
    """
    Read an XYZ file, center the atomic coordinates at the origin,
    and write a new centered XYZ file.

    Parameters
    ----------
    input_xyz : str
        Path to input .xyz file
    output_xyz : str
        Path to output centered .xyz file
    """

    with open(input_xyz, "r") as f:
        lines = f.readlines()

    # Number of atoms
    n_atoms = int(lines[0].strip())

    # Comment line (kept as-is)
    comment = lines[1]

    atoms = []
    coords = []

    # Read atom labels and coordinates
    for line in lines[2:2 + n_atoms]:
        parts = line.split()
        atom = parts[0]
        x, y, z = map(float, parts[1:4])
        atoms.append(atom)
        coords.append([x, y, z])

    # Compute geometric center
    center_x = sum(c[0] for c in coords) / n_atoms
    center_y = sum(c[1] for c in coords) / n_atoms
    center_z = sum(c[2] for c in coords) / n_atoms

    # Center coordinates
    centered_coords = [
        [x - center_x, y - center_y, z - center_z]
        for x, y, z in coords
    ]

    # Write centered XYZ file
    with open(output_xyz, "w") as f:
        f.write(f"{n_atoms}\n")
        f.write(comment)
        for atom, (x, y, z) in zip(atoms, centered_coords):
            f.write(f"{atom:2s}  {x:15.8f}  {y:15.8f}  {z:15.8f}\n")

if __name__ == "__main__":
    center_xyz("./ga12as12.xyz", "./ga12as12_centerd.xyz")