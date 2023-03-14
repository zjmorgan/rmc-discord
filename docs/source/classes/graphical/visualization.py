from disorder.graphical.visualization import CrystalStructure
from disorder.material import structure

uc = structure.UnitCell('Tb2Ir3Ga9.mcif', tol=1e-4)

xyz = uc.get_unit_cell_cartesian_atomic_coordinates()
occ = uc.get_occupancies()
U = uc.get_cartesian_anisotropic_displacement_parameters()
mxmymz = uc.get_cartesian_magnetic_moments()

A = uc.get_fractional_cartesian_transform()

atms = uc.get_unit_cell_atoms()
colors = uc.get_atom_colors()
radii = uc.get_atom_radii()

cs = CrystalStructure(A, *xyz, atms, colors)

cs.draw_cell_edges()
cs.draw_basis_vectors()
cs.atomic_radii(radii, occ)
#cs.atomic_displacement_ellipsoids(*U, p=1.0)
cs.magnetic_vectors(*mxmymz)
cs.view_direction(3, 2, 1)

cs.show_figure()
