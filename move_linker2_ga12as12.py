import numpy as np


# ================= USER PARAMETERS =================
DELTA = 2.2          # ligand translation distance (Å)
CLUSTER_SEP = 5.0     # cluster–ligand separation factor
LINK_INDICES1 = [20]    # catechol oxygen indices
LINK_INDICES2 = [21]
CLUSTER_INDICES = [1,5,11,19]  # Cluster1's atoms to be linked
CLUSTER_LINK_INDICES2 = [2,6,12,20]  # Cluster2's atoms to be linked
# ==================================================


# ---- Ligand coordinates ----
ligand = np.array([
        [-1.37844,  0.72042, -0.29894],
        [-0.11316,  1.34697,  0.28932],
        [ 1.13535,  0.66079, -0.25617],
        [ 2.39964,  1.30283,  0.31701],
        [ 1.10751, -0.41845,  0.00498],
        [-0.07703,  2.42704,  0.03338],
        [ 2.37793,  2.38441,  0.05163],
        [-1.37503, -0.36167, -0.03479],
        [-2.57557,  1.35934,  0.25177],
        [ 3.59711,  0.68026, -0.25048],
        [-2.80146,  2.62804, -0.41406],
        [-4.28335,  2.79990, -0.34889],
        [ 3.86547, -0.56983,  0.43409],
        [ 5.35015, -0.70765,  0.34508],
        [-3.75658,  0.57305, -0.11900],
        [ 4.76827,  1.50011,  0.07714],
        [ 3.33940, -1.43078, -0.02857],
        [ 5.76442, -1.43395,  1.07481],
        [-4.66642,  3.54973, -1.07190],
        [-2.26487,  3.46763,  0.07485],
        [-4.80388,  1.47402, -0.60252],
        [ 5.84315,  0.63452,  0.56524]
])

elements = [
    "C","C","C","C","H","H","H","H",
    "N","N","C","C","C","C","C","C",
    "H","H","H","H","N","N"
]


# ---- Mn@Ga12As12 cluster ----
cluster = np.array([
     [ 0.00000000,  0.00000000,  0.00000000],
     [ 3.03344398, -0.32868592,  0.92707762],
     [-1.12891140, -0.36561852, -2.95874281],
     [ 3.65823061, -0.84611087, -1.26477810],
     [ 1.02032644, -0.86901981, -3.72591870],
     [ 1.12869074,  0.36560216,  2.95838088],
     [-3.03298550,  0.32866575, -0.92673006],
     [-1.02038109,  0.86892582,  3.72591639],
     [-3.65854929,  0.84610006,  1.26486480],
     [ 2.12015880,  0.69993949, -2.27521965],
     [-1.65636783,  2.08359544,  1.75188530],
     [ 2.86774632,  1.77351138,  2.07989246],
     [-2.28692536,  1.72784172, -2.73330423],
     [ 1.76604600,  2.98225566, -1.91814296],
     [-0.62755806,  3.86038498,  0.63554650],
     [ 1.50749509,  2.77028092,  0.46194651],
     [-0.58894711,  2.75216200, -1.49664487],
     [ 1.65612182, -2.08352559, -1.75179211],
     [-2.12017112, -0.69991074,  2.27503181],
     [ 2.28692021, -1.72777427,  2.73359529],
     [-2.86789060, -1.77341270, -2.07978315],
     [ 0.62763758, -3.86051720, -0.63559081],
     [-1.76590607, -2.98220203,  1.91801222],
     [ 0.58915676, -2.75217284,  1.49659791],
     [-1.50738092, -2.77031489, -0.46210025]
])

cluster_elements = ['Mn',
     'Ga', 'Ga', 'As', 'As', 'Ga', 'Ga', 'As', 'As', 'Ga', 'Ga', 'As',
     'As', 'As', 'As', 'Ga', 'Ga', 'Ga', 'Ga', 'As', 'As', 'As','As',
     'Ga', 'Ga'
]

# ==================================================
#               GEOMETRY OPERATIONS
# ==================================================

def rotation_matrix_from_vectors(v1, v2):
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    cross = np.cross(v1, v2)
    dot = np.dot(v1, v2)

    if np.isclose(dot, 1.0):
        return np.eye(3)

    if np.isclose(dot, -1.0):
        # 180° rotation
        axis = np.array([1.0, 0.0, 0.0])
        if abs(v1 @ axis) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        axis -= axis.dot(v1) * v1
        axis /= np.linalg.norm(axis)
        return rotation_matrix_from_vectors(v1, axis) @ rotation_matrix_from_vectors(axis, v2)

    K = np.array([
        [0, -cross[2], cross[1]],
        [cross[2], 0, -cross[0]],
        [-cross[1], cross[0], 0]
    ])

    return np.eye(3) + K + K @ K / (1 + dot)


# ---------- First cluster + ligand ----------

cluster_center = cluster.mean(axis=0)
cluster_anchor1 = cluster[CLUSTER_INDICES].mean(axis=0)

lig1 = ligand[LINK_INDICES1].mean(axis=0)
lig2 = ligand[LINK_INDICES2].mean(axis=0)

# attach ligand to first cluster
attach_vec = cluster_anchor1 - lig1
attach_vec /= np.linalg.norm(attach_vec)

ligand_out = ligand + (cluster_anchor1 - lig1) + DELTA * attach_vec

# updated ligand positions
lig1_new = ligand_out[LINK_INDICES1].mean(axis=0)
lig2_new = ligand_out[LINK_INDICES2].mean(axis=0)


# ---------- Second cluster orientation ----------

# cluster linkage definition (second cluster)
cluster_link2 = cluster[CLUSTER_LINK_INDICES2].mean(axis=0)

# cluster intrinsic binding direction
cluster_bind_vec = cluster_center - cluster_link2
cluster_bind_vec /= np.linalg.norm(cluster_bind_vec)

# ligand approach direction (must point away from ligand core)
ligand_vec = lig2_new - lig1_new
ligand_vec /= np.linalg.norm(ligand_vec)

# rotate cluster so binding face looks at ligand atom 21
R = rotation_matrix_from_vectors(cluster_bind_vec, -ligand_vec)

cluster_rot = (cluster - cluster_center) @ R.T + cluster_center

# ---------- Translate second cluster to ligand ----------

new_link = cluster_rot[CLUSTER_LINK_INDICES2].mean(axis=0)
cluster2 = cluster_rot + (lig2_new + new_link) + DELTA * attach_vec #tune DELTA change

# ==================================================
#                 XYZ OUTPUT
# ==================================================

total_atoms = len(cluster) + len(ligand_out) + len(cluster2)
print(total_atoms)
print("Ga12As12–ligand–Ga12As12 assembled structure")

for el, xyz in zip(cluster_elements, cluster):
    print(f"{el:2s} {xyz[0]:12.6f} {xyz[1]:12.6f} {xyz[2]:12.6f}")

for el, xyz in zip(elements, ligand_out):
    print(f"{el:2s} {xyz[0]:12.6f} {xyz[1]:12.6f} {xyz[2]:12.6f}")

for el, xyz in zip(cluster_elements, cluster2):
    print(f"{el:2s} {xyz[0]:12.6f} {xyz[1]:12.6f} {xyz[2]:12.6f}")

with open("mngaas_link2.xyz", "w") as f:
    f.write(str(total_atoms) + "\n")
    f.write("Ga12As12–ligand–Ga12As12 assembled structure" + "\n")
    for el, xyz in zip(cluster_elements, cluster):
        f.write(f"{el:2s} {xyz[0]:12.6f} {xyz[1]:12.6f} {xyz[2]:12.6f}"+ "\n")

    for el, xyz in zip(elements, ligand_out):
        f.write(f"{el:2s} {xyz[0]:12.6f} {xyz[1]:12.6f} {xyz[2]:12.6f}"+ "\n")

    for el, xyz in zip(cluster_elements, cluster2):
        f.write(f"{el:2s} {xyz[0]:12.6f} {xyz[1]:12.6f} {xyz[2]:12.6f}"+ "\n")
