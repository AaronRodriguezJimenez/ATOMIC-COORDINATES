from mp_api.client import MPRester
with MPRester(api_key="uEdzHMTC6l2o6s3imuFKbCSGZ0d2pQIC") as mpr:
    data = mpr.materials.search(material_ids=["mp-1180676"])

    print(data)
    structure = data[0]["structure"]
    print(structure)
    lattice_constants = structure.lattice.a
    print(lattice_constants)
