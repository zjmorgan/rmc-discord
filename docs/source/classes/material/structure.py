import numpy as np

from disorder.material import structure

from disorder.graphical.canvas import Canvas
from disorder.graphical.plots import Line

uc = structure.UnitCell('Cu3Au.cif', tol=1e-4)

uvw = uc.get_fractional_coordinates()
atm = uc.get_unit_cell_atoms()
occ = uc.get_occupancies()

U = uc.get_anisotropic_displacement_parameters()
constants = uc.get_all_lattice_constants()
symops = uc.get_space_group_symmetry_operators()

*hkl, d, F, mult = structure.factor(*uvw, atm, occ, *U, *constants,
                                    symops, dmin=0.7, source='neutron')

inv_d = 1/d
F2 = np.abs(F)**2

inv_d_line = np.repeat(inv_d,3)
F2_line = np.repeat(F2,3)

F2_line[0::3] = 0
F2_line[2::3] = np.nan

canvas = Canvas()

line = Line(canvas)
line.plot_data(inv_d_line, F2_line, marker='-')
line.plot_data(inv_d, F2, marker='o')
line.set_labels(r'', r'$Q/2\pi$ [$\AA^{-1}$]', r'$|F|^2$')

canvas.close()

data = np.stack((*hkl,d,F.real,F.imag,mult)).T

np.savetxt('Cu3Au.csv', data, fmt='%d,%d,%d,%.4f,%.4f,%.4f,%d')
