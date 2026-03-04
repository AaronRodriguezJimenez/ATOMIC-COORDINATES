import numpy as np


# ================= USER PARAMETERS =================
DELTA = 2.0          # ligand translation distance (Å)
CLUSTER_SEP = 5.0     # cluster–ligand separation factor
LINK_INDICES1 = [13]    # catechol oxygen indices
LINK_INDICES2 = [12]
CLUSTER_INDICES = [1,5,11,19]  # Cluster1's atoms to be linked
CLUSTER_LINK_INDICES2 = [2,6,12,20]  # Cluster2's atoms to be linked
LIGAND_CORE_INDICES = [2,3,4,5]
# ==================================================


# ---- Ligand coordinates ----
ligand = np.array([
    [-2.04741063002852,  0.54864045338507, -1.09993554401867],
    [-0.96253389173497,  1.31859862545151, -0.97185707134376],
    [ 0.25303973112792,  0.89527942267282, -0.30811447212848],
    [ 1.33792199713171,  1.66523269777867, -0.18004791537267],
    [-2.02611813265845, -0.46141192692147, -0.69507156623353],
    [-0.96483635927380,  2.32920188390800, -1.37678022980561],
    [ 0.25533835582121, -0.11532326941074,  0.09681052588254],
    [ 1.31663346090280,  2.67528246661371, -0.58491900212525],
    [ 3.86601862764014,  1.14045824014352, -0.36828389303439],
    [ 2.68292764329242, -0.27534654082594,  0.84773541793709],
    [-3.39240286339738,  2.48921467741994, -2.12773676839283],
    [-4.57551454429636,  1.07340417314835, -0.91174180409857],
    [-3.63654634723248,  0.98668053116625, -2.11118083299175],
    [ 2.92707295270577,  1.22718756547033,  0.83117215572590],
])

elements = [
    "C","C","C","C","H","H","H","H","H","H","H","H","As","As"
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
        # 180° rotation: choose orthogonal axis
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


# ---------- Cluster references ----------
cluster_center = cluster.mean(axis=0)
cluster_anchor = cluster[CLUSTER_INDICES].mean(axis=0)

# intrinsic cluster binding direction (outward)
cluster_vec = cluster_anchor - cluster_center
cluster_vec /= np.linalg.norm(cluster_vec)


# ---------- Ligand intrinsic axis ----------
lig1 = ligand[LINK_INDICES1].mean(axis=0)
lig2 = ligand[LINK_INDICES2].mean(axis=0)

lig_vec = lig2 - lig1
lig_vec /= np.linalg.norm(lig_vec)


# ---------- Rotate ligand so it aligns with cluster ----------
R_lig = rotation_matrix_from_vectors(lig_vec, cluster_vec)
ligand_rot = (ligand - lig1) @ R_lig.T + lig1

# update ligand anchors
lig1_rot = ligand_rot[LINK_INDICES1].mean(axis=0)
lig2_rot = ligand_rot[LINK_INDICES2].mean(axis=0)


# ---------- Translate ligand to first cluster ----------
ligand_out = ligand_rot + (cluster_anchor - lig1_rot) + DELTA * cluster_vec

lig1_new = ligand_out[LINK_INDICES1].mean(axis=0)
lig2_new = ligand_out[LINK_INDICES2].mean(axis=0)


# ---------- Second cluster: rotate + translate ----------
# second cluster must face BACK toward ligand end
target_vec = lig1_new - lig2_new
target_vec /= np.linalg.norm(target_vec)

R2 = rotation_matrix_from_vectors(cluster_vec, target_vec)
cluster_rot = (cluster - cluster_center) @ R2.T + cluster_center

new_anchor = cluster_rot[CLUSTER_INDICES].mean(axis=0)
cluster2 = cluster_rot + (lig2_new + new_anchor) + DELTA * cluster_vec

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

with open("mngaas_link3.xyz", "w") as f:
    f.write(str(total_atoms) + "\n")
    f.write("Ga12As12–ligand–Ga12As12 assembled structure" + "\n")
    for el, xyz in zip(cluster_elements, cluster):
        f.write(f"{el:2s} {xyz[0]:12.6f} {xyz[1]:12.6f} {xyz[2]:12.6f}"+ "\n")

    for el, xyz in zip(elements, ligand_out):
        f.write(f"{el:2s} {xyz[0]:12.6f} {xyz[1]:12.6f} {xyz[2]:12.6f}"+ "\n")

    for el, xyz in zip(cluster_elements, cluster2):
        f.write(f"{el:2s} {xyz[0]:12.6f} {xyz[1]:12.6f} {xyz[2]:12.6f}"+ "\n")