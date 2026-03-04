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
    [-2.12377368760962,  0.73737136806478, -1.00993979349655],
    [-0.95011382185740,  1.38053151580810, -0.93993813063581],
    [ 0.24093688229274,  0.83339448451051, -0.33941192752514],
    [ 1.41459519342249,  1.47655766268162, -0.26940916942353],
    [-2.22153510369547, -0.25721750746213, -0.58299403429641],
    [-0.87365010849610,  2.37797595583961, -1.37187528719673],
    [ 0.16447527369447, -0.16404959896800,  0.09252592347602],
    [ 1.51235479410487,  2.47114695944593, -0.69635475868358],
    [ 3.42333074651125,  1.31402569271952, -0.03219008849952],
    [ 2.55368943936999,  0.04830959517774,  0.61237610935633],
    [-3.26287005748831,  2.16562606230487, -1.89171427960012],
    [-4.13251032764742,  0.89990208793525, -1.24716118342135],
    [-3.26248516114929,  1.18157428889225, -1.66996866266199],
    [ 2.55330593854780,  1.03235943304996,  0.39062328260839]
])

elements = [
    "C","C","C","C","H","H","H","H","H","H","H","H","N","N"
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
cluster2 = cluster_rot + (lig2_new - new_anchor) + DELTA * cluster_vec

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

with open("mngaas_link4.xyz", "w") as f:
    f.write(str(total_atoms) + "\n")
    f.write("Ga12As12–ligand–Ga12As12 assembled structure" + "\n")
    for el, xyz in zip(cluster_elements, cluster):
        f.write(f"{el:2s} {xyz[0]:12.6f} {xyz[1]:12.6f} {xyz[2]:12.6f}"+ "\n")

    for el, xyz in zip(elements, ligand_out):
        f.write(f"{el:2s} {xyz[0]:12.6f} {xyz[1]:12.6f} {xyz[2]:12.6f}"+ "\n")

    for el, xyz in zip(cluster_elements, cluster2):
        f.write(f"{el:2s} {xyz[0]:12.6f} {xyz[1]:12.6f} {xyz[2]:12.6f}"+ "\n")