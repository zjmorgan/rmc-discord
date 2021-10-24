import mayavi.mlab as mlab
import numpy as np
from scipy.spatial.transform.rotation import Rotation

from mayavi.api import Engine
from mayavi.sources.api import ParametricSurface
from mayavi.modules.api import Surface
from tvtk.tools import visual

from disorder.material import crystal
from disorder.material import tables

def probability_ellipsoid(Uxx, Uyy, Uzz, Uyz, Uxz, Uxy, p=0.99):
    
    U = np.array([[Uxx,Uxy,Uxz],
                  [Uxy,Uyy,Uyz],
                  [Uxz,Uyz,Uzz]])
    
    w, v = np.linalg.eig(np.linalg.inv(U))
    
    r_eff = -2*np.log(1-p)
    
    radii = np.sqrt(r_eff/w)
    
    rot = Rotation.from_matrix(v)
    
    euler_angles = rot.as_euler('ZXY', degrees=True)
    
    return radii, euler_angles

def draw_cell_edges(ucx, ucy, ucz):
    
    connections = ((0,1),(0,2),(0,4),(1,3),(1,5),(2,3),
                   (2,6),(3,7),(4,5),(4,6),(5,7),(6,7)) 
    
    pts = mlab.points3d(ucx, ucy, ucz, np.zeros(8))
    
    pts.mlab_source.dataset.lines = np.array(connections)
    
    tube = mlab.pipeline.tube(pts, tube_radius=0.05)
    tube.filter.radius_factor = 0.
    mlab.pipeline.surface(tube, color=(0.0,0.0,0.0))
    
def draw_basis_vectors(ar, br, cr, av, bv, cv):
    
    ca = mlab.quiver3d(ar[0], ar[1], ar[2], av[0], av[1], av[2], color=(1,0,0), 
                       resolution=60, scale_factor=1, mode='arrow')
    
    cb = mlab.quiver3d(br[0], br[1], br[2], bv[0], bv[1], bv[2], color=(0,1,0), 
                       resolution=60, scale_factor=1, mode='arrow')
    
    cc = mlab.quiver3d(cr[0], cr[1], cr[2], cv[0], cv[1], cv[2], color=(0,0,1), 
                       resolution=60, scale_factor=1, mode='arrow')
    
    ca.glyph.glyph_source.glyph_position = 'tail'    
    cb.glyph.glyph_source.glyph_position = 'tail'
    cc.glyph.glyph_source.glyph_position = 'tail'
    
    ta = mlab.text3d(ar[0], ar[1], ar[2], 'a')
    tb = mlab.text3d(br[0], br[1], br[2], 'b')
    tc = mlab.text3d(cr[0], cr[1], cr[2], 'c')
    
def atomic_radii(ux, uy, uz, radii, colors):
    
    n_atm = ux.shape[0]
    
    for i in range(n_atm):
        p = mlab.points3d(ux[i], uy[i], uz[i], radii[i], 
                          color=colors[i], resolution=60, scale_factor=1)

def magnetic_vectors(ux, uy, uz, sx, sy, sz):
    
    n_atm = ux.shape[0]
    
    for i in range(n_atm):
        
        v = mlab.quiver3d(ux[i], uy[i], uz[i], sx[i], sy[i], sz[i], 
                          color=(1,0,0), line_width=60, resolution=60, 
                          scale_factor=1, mode='arrow')
        
        v.glyph.glyph_source.glyph_position = 'tail'
        
        t = mlab.quiver3d(ux[i], uy[i], uz[i], -sx[i], -sy[i], -sz[i], 
                         color=(1,0,0), line_width=60, resolution=60, 
                         scale_factor=1, mode='arrow')
        
        t.glyph.glyph_source.glyph_position = 'tail'
        t.glyph.glyph_source.glyph_source.tip_radius = 0

def atomic_displacement_ellipsoids(ux, uy, uz, Uxx, Uyy, Uzz, Uyz, Uxz, Uxy,
                                   colors, p=0.99):

    n_atm = ux.shape[0]
    
    for i in range(n_atm):
        
        s, a = probability_ellipsoid(Uxx[i], Uyy[i], Uzz[i], 
                                     Uyz[i], Uxz[i], Uxy[i], p)
        
        source = ParametricSurface()
        source.function = 'ellipsoid'
        engine.add_source(source)
        
        surface = Surface()
        source.add_module(surface)
        
        actor = surface.actor 
        #actor.property.ambient = 1 
        actor.property.opacity = 1.0
        actor.property.color = colors[i]
        
        actor.mapper.scalar_visibility = False 
        actor.property.backface_culling = True 
        
        actor.property.diffuse = 1.0
        actor.property.specular = 0.0
        
        actor.actor.origin = np.array([0,0,0])
        actor.actor.position = np.array([ux[i],uy[i],uz[i]])
        
        actor.actor.scale = np.array([s[0],s[1],s[2]])
        actor.actor.orientation = np.array([a[0],a[1],a[2]]) #ZXY

        actor.enable_texture = True
        actor.property.representation = 'surface'

a, b, c = 5, 5, 12
alpha, beta, gamma = np.deg2rad(90), np.deg2rad(90), np.deg2rad(120)

A = crystal.cartesian(a, b, c, alpha, beta, gamma)

atms = np.array(['Co', 'Fe', 'Mn'])

n_atm = atms.shape[0]

u = np.array([0.5,0.0,0.0])
v = np.array([0.0,0.5,0.0])
w = np.array([0.5,0.5,0.0])

ux, uy, uz = np.dot(A, [u,v,w])

radii = np.array([tables.r.get(atm)[0] for atm in atms])

Uxx, Uyy, Uzz, Uyz, Uxz, Uxy = [], [], [], [], [], []
rots = Rotation.random(n_atm).as_matrix()
ae = np.random.random(n_atm)
be = np.random.random(n_atm)
ce = np.random.random(n_atm)

for i in range(n_atm):
    U = np.dot(np.dot(rots[i],np.diag([ae[i],be[i],ce[i]])),rots[i].T)
    Uxx.append(U[0,0])
    Uyy.append(U[1,1])
    Uzz.append(U[2,2])
    Uyz.append(U[1,2])
    Uxz.append(U[0,2])
    Uxy.append(U[0,1])

Uxx = np.array(Uxx)
Uyy = np.array(Uyy)
Uzz = np.array(Uzz)
Uyz = np.array(Uyz)
Uxz = np.array(Uxz)
Uxy = np.array(Uxy)

colors = [tables.rgb.get(atm) for atm in atms]

uc = np.array([0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0])
vc = np.array([0.0,0.0,1.0,1.0,0.0,0.0,1.0,1.0])
wc = np.array([0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0])

ucx, ucy, ucz = np.dot(A, [uc,vc,wc])

scene = mlab.figure(fgcolor=(0,0,0), bgcolor=(1,1,1))
engine = mlab.get_engine()

scale = np.min([a,b,c])
offset = 0.5

ar = np.dot(A, [1.0+offset*scale/a,0,0])
br = np.dot(A, [0,1.0+offset*scale/b,0])
cr = np.dot(A, [0,0,1.0+offset*scale/c])

av = np.dot(A, [offset/scale*a,0,0])
bv = np.dot(A, [0,offset/scale*b,0])
cv = np.dot(A, [0,0,offset*scale/c])

engine.start()
#scene = engine.new_scene()

draw_basis_vectors(ar, br, cr, av, bv, cv)
draw_cell_edges(ucx, ucy, ucz)

#atomic_radii(ux, uy, uz, radii, colors)

scene.scene.disable_render = True 
atomic_displacement_ellipsoids(ux, uy, uz, Uxx, Uyy, Uzz, 
                               Uyz, Uxz, Uxy, colors, 0.99)
scene.scene.disable_render = False 

mlab.show()
