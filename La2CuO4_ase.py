from ase import Atoms
from ase.visualize import view

# Structure from "materialsproject web"
a=5.36
b=13.10
c=5.54

atoms = Atoms(
    ['Cu','O','La','O'],
    scaled_positions=[
        (0,0,0),
        (0.25,0.988766,0.75),
        (0,0.638989,0.985346),
        (0.5,0.68678,0.940211)
    ],
    cell=[a,b,c],
    pbc=True
)

view(atoms)