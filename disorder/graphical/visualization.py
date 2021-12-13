import mayavi.mlab as mlab
import numpy as np

from scipy.stats import chi2
from scipy.spatial.transform.rotation import Rotation

from mayavi.sources.api import ParametricSurface
from mayavi.modules.api import Surface
    
class CrystalStructure:
    
    def __init__(self):
        
        self.fig = mlab.figure(fgcolor=(0,0,0), bgcolor=(1,1,1))
        self.engine = mlab.get_engine()
        self.engine.start()
        
        self.fig.scene.parallel_projection = True
        
    def __probability_ellipsoid(self, Uxx, Uyy, Uzz, Uyz, Uxz, Uxy, p=0.99):
        
        U = np.array([[Uxx,Uxy,Uxz],
                      [Uxy,Uyy,Uyz],
                      [Uxz,Uyz,Uzz]])
        
        w, v = np.linalg.eig(np.linalg.inv(U))
        
        r_eff = chi2.ppf(1-p, 3)
        
        radii = np.sqrt(r_eff/w)
        
        rot = Rotation.from_matrix(v)
        
        euler_angles = rot.as_euler('ZXY', degrees=True)
        
        return radii, euler_angles
    
    def save_figure(self, filename):
        
        mlab.savefig(filename, figure=self.fig)
            
    def view_direction(self, u, v, w):
        
        A = self.A
        
        x, y, z = np.dot(A, [u,v,w])
        
        r = np.sqrt(x**2+y**2+z**2)
        theta = np.rad2deg(np.arccos(z/r))
        phi = np.rad2deg(np.arctan2(y,x))
        
        mlab.view(azimuth=phi, elevation=theta, distance=None, focalpoint=None, 
                  roll=None, reset_roll=True, figure=self.fig)
        
    def draw_basis_vectors(self):
        
        A = self.A
        a, b, c = self.a, self.b, self.c
    
        scale = np.min([a,b,c])
        offset = 0.5
        
        ar = np.dot(A, [1.0+offset*scale/a,0,0])
        br = np.dot(A, [0,1.0+offset*scale/b,0])
        cr = np.dot(A, [0,0,1.0+offset*scale/c])
        
        av = np.dot(A, [offset/scale*a,0,0])
        bv = np.dot(A, [0,offset/scale*b,0])
        cv = np.dot(A, [0,0,offset*scale/c])
        
        ca = mlab.quiver3d(ar[0], ar[1], ar[2], av[0], av[1], av[2], 
                           color=(1,0,0), resolution=60, scale_factor=1, 
                           mode='arrow', figure=self.fig)
        
        cb = mlab.quiver3d(br[0], br[1], br[2], bv[0], bv[1], bv[2], 
                           color=(0,1,0), resolution=60, scale_factor=1, 
                           mode='arrow', figure=self.fig)
        
        cc = mlab.quiver3d(cr[0], cr[1], cr[2], cv[0], cv[1], cv[2], 
                           color=(0,0,1), resolution=60, scale_factor=1, 
                           mode='arrow', figure=self.fig)
        
        ca.glyph.glyph_source.glyph_position = 'tail'    
        cb.glyph.glyph_source.glyph_position = 'tail'
        cc.glyph.glyph_source.glyph_position = 'tail'
        
        mlab.text3d(ar[0], ar[1], ar[2], 'a', figure=self.fig)
        mlab.text3d(br[0], br[1], br[2], 'b', figure=self.fig)
        mlab.text3d(cr[0], cr[1], cr[2], 'c', figure=self.fig)
            
    def draw_cell_edges(self, ucx, ucy, ucz):
        
        connections = ((0,1),(0,2),(0,4),(1,3),(1,5),(2,3),
                       (2,6),(3,7),(4,5),(4,6),(5,7),(6,7)) 
        
        pts = mlab.points3d(ucx, ucy, ucz, np.zeros(8), figure=self.fig)
        
        pts.mlab_source.dataset.lines = np.array(connections)
        
        tube = mlab.pipeline.tube(pts, tube_radius=0.05)
        tube.filter.radius_factor = 0.
        mlab.pipeline.surface(tube, color=(0.0,0.0,0.0), figure=self.fig)
        
    def atomic_displacement_ellipsoids(self, ux, uy, uz, 
                                       Uxx, Uyy, Uzz, Uyz, Uxz, Uxy,
                                       colors, p=0.99):
    
        self.fig.scene.disable_render = True 
        
        n_atm = ux.shape[0]
                
        for i in range(n_atm):
            
            s, a = self.__probability_ellipsoid(Uxx[i], Uyy[i], Uzz[i], 
                                                Uyz[i], Uxz[i], Uxy[i], p)
            
            source = ParametricSurface()
            source.function = 'ellipsoid'
            
            self.engine.add_source(source)
            
            surface = Surface()
            source.add_module(surface)
            
            actor = surface.actor 
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
            
        self.fig.scene.disable_render = False 
            
    def atomic_radii(self, ux, uy, uz, radii, colors):
        
        n_atm = ux.shape[0]
        
        points = []
        for i in range(n_atm):
            p = mlab.points3d(ux[i], uy[i], uz[i], radii[i], color=colors[i], 
                              resolution=60, scale_factor=1, figure=self.fig)
            points.append(p)
        
        return points
    
    def magnetic_vectors(self, ux, uy, uz, sx, sy, sz):
        
        n_atm = ux.shape[0]
        
        for i in range(n_atm):
            
            v = mlab.quiver3d(ux[i], uy[i], uz[i], sx[i], sy[i], sz[i], 
                              color=(1,0,0), line_width=60, resolution=60, 
                              scale_factor=1, mode='arrow', figure=self.fig)
            
            v.glyph.glyph_source.glyph_position = 'tail'
            
            t = mlab.quiver3d(ux[i], uy[i], uz[i], -sx[i], -sy[i], -sz[i], 
                             color=(1,0,0), line_width=60, resolution=60, 
                             scale_factor=1, mode='arrow', figure=self.fig)
            
            t.glyph.glyph_source.glyph_position = 'tail'
            t.glyph.glyph_source.glyph_source.tip_radius = 0