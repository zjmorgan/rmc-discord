#!/usr/bin/env python3

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.transforms as mtransforms

from mpl_toolkits.mplot3d import axes3d

def save(folder='', filename=None, ext='pdf'):
    
    if (filename != None):

        plt.savefig(folder+filename+'.'+ext)

def refinement(chi_sq, acc_moves, rej_moves):
    
    plt.figure(dpi=100)
    plt.plot(chi_sq)
    plt.ylabel(r'Reduced chi-squared statistic change, $\Delta\chi^2$')
    plt.xlabel(r'Total moves, $t$')
    
    plt.figure(dpi=100)
    plt.semilogy(acc_moves)
    plt.ylabel(r'Reduced chi-squared statistic, $\chi^2$')
    plt.xlabel(r'Accepted moves, $n$')
    
    plt.figure(dpi=100)
    plt.semilogy(rej_moves)
    plt.ylabel(r'Reduced chi-squared statistic, $\chi^2$')
    plt.xlabel(r'Rejected moves, $n$')
    
def extents(umin, umax, nu, vmin, vmax, nv):
    du = np.float(umax-umin)/nu
    dv = np.float(vmax-vmin)/nv
    return [umin-du/2, umax+dv/2, vmin-du/2, vmax+dv/2]

def evalstr(x):
    return str(np.round(x,4))

def intensity(I, 
              h_range, 
              k_range, 
              l_range, 
              plane='l', 
              index=0, 
              norm='log', 
              aspect=None,
              colormap=plt.cm.viridis,
              B=np.eye(3),
              T=np.eye(3)):
    
    nh, nk, nl = I.shape[0], I.shape[1], I.shape[2]
    
    h_min = h_range[0]
    h_max = h_range[1]
    
    k_min = k_range[0]
    k_max = k_range[1]
    
    l_min = l_range[0]
    l_max = l_range[1]
    
    if (norm == 'log'):
        normalize = colors.LogNorm(vmin=np.nanmin(np.abs(I)), 
                                   vmax=np.nanmax(I))
    else:
        normalize = colors.Normalize(vmin=np.nanmin(I), vmax=np.nanmax(I))
        
    P = np.full((3,3), '', dtype='<U4').tolist()
    
    for i in range(3):
        for j in range(3):
            P[i][j] = str(int(T[j,i])).replace('1','')
   
    P[0][0] += '*h'
    P[1][0] += '*k'
    P[2][0] += '*l'
    
    P[0][1] += '*h'
    P[1][1] += '*k'
    P[2][1] += '*k'
    
    P[0][2] += '*h'
    P[1][2] += '*k'
    P[2][2] += '*l'
        
    for i in range(3):
        for j in range(3):
            if (P[i][j][0] == '0'):
                P[i][j] = '0'
            elif (P[i][j][0] == '*'):
                P[i][j] = P[i][j][1:]
            elif (P[i][j][:2] == '-*'):
                P[i][j] = '-'+P[i][j][2:]
                
    plt.figure(dpi=100)
                
    if (plane == 'h'):
    
        if (nh == 1):
            ih = 0
            h = h_min
        else:
            ih = int(round((index-h_min)/(h_max-h_min)*(nh-1)))
            h = h_min+((h_max-h_min)/(nh-1))*ih            
        
        im = plt.imshow(I[ih,:,:].T, 
                        norm=normalize,
                        interpolation='nearest', 
                        origin='lower', 
                        cmap=colormap,
                        extent=extents(k_min, 
                                       k_max, 
                                       nk,
                                       l_min, 
                                       l_max,
                                       nl))
        
        S = np.copy(P).tolist()
        for i in range(3):
            for j in range(3):
                if ('h' in P[i][j]):
                    S[i][j] = str(np.round(eval(P[i][j]),4))
                    if (S[i][j][0] != '-'):
                        S[i][j] = '+'+S[i][j]
                if (S[i][j] == '0'):
                    S[i][j] = ''
                if (S[i][j] == '+0'):
                    S[i][j] = ''
                if (S[i][j] == 'k'):
                    S[i][j] = '+k'
                elif (S[i][j] == 'l'):
                    S[i][j] = '+l'                    
                                
        H = (S[0][0]+S[1][0]+S[2][0]).replace('*','')
        K = (S[0][1]+S[1][1]+S[2][1]).replace('*','')
        L = (S[0][2]+S[1][2]+S[2][2]).replace('*','')
                
        if (H == ''):
            H = '0'
        elif (H[0] == '+'):
            H = H[1:]
            
        if (K == ''):
            K = '0'
        elif (K[0] == '+'):
            K = K[1:]
            
        if (L == ''):
            L = '0'
        elif (L[0] == '+'):
            L = L[1:]
                    
        plt.title(r'$('+H+','+K+','+L+')$')       
        plt.xlabel(r'$k$')
        plt.ylabel(r'$l$')
        
        dk = (k_max-k_min)/(nk-1)
        dl = (l_max-l_min)/(nl-1)   
        extents_h = [k_min-dk/2, k_max+dk/2, 
                     l_min-dl/2, l_max+dl/2]
            
        trans = mtransforms.Affine2D()
        
        M = np.array([[B[1,1]/B[2,2],B[1,2]/B[2,2],0],
                      [B[2,1]/B[2,2],B[2,2]/B[2,2],0],
                      [0,0,1]])
        
        Q = np.eye(3)
        Q[0:2,0:2] = T.T.copy()[1:3,1:3]
        
        N = np.dot(M.T,M)
        
        scale_trans = N[1,1].copy()
        
        N[0:2] /= scale_trans
        N[0:2] /= scale_trans
        
        M = np.linalg.cholesky(np.dot(Q,np.dot(N,Q.T))).T
        
        scale_rot = M[1,1].copy()
        
        M[0,1] /= scale_rot
        M[0,0] /= scale_rot
        M[1,1] /= scale_rot
                
        scale = M[0,0].copy()
        
        M[0,1] /= scale
        M[0,0] /= scale
        
        trans.set_matrix(M)
        
        offset = -np.dot(M,[0,l_min,0])[0]
        
        shift = mtransforms.Affine2D().translate(offset,0)
                
        trans_data = trans+shift+plt.gca().transData
        
        im.set_transform(trans_data)
        
        ext_min = np.dot(M[0:2,0:2],extents_h[0::2])
        ext_max = np.dot(M[0:2,0:2],extents_h[1::2])
        
        plt.gca().set_xlim(ext_min[0]+offset,ext_max[0]+offset)
        plt.gca().set_ylim(ext_min[1],ext_max[1])
        
        plt.gca().set_aspect(1/scale/scale_rot)
        plt.gca().set_transform(trans_data)    
        
    elif (plane == 'k'):
        
        if (nk == 1):
            ik = 0
            k = k_min
        else:
            ik = int(round((index-k_min)/(k_max-k_min)*(nk-1)))
            k = k_min+((k_max-k_min)/(nk-1))*ik 
        
        im = plt.imshow(I[:,ik,:].T, 
                        norm=normalize,
                        interpolation='nearest', 
                        origin='lower', 
                        cmap=colormap,
                        extent=extents(h_min, 
                                       h_max, 
                                       nh,
                                       l_min, 
                                       l_max,
                                       nl))
        
        S = np.copy(P).tolist()
        for i in range(3):
            for j in range(3):
                if ('k' in P[i][j]):
                    S[i][j] = str(np.round(eval(P[i][j]),4))
                    if (S[i][j][0] != '-'):
                        S[i][j] = '+'+S[i][j]
                if (S[i][j] == '0'):
                    S[i][j] = ''
                if (S[i][j] == '+0'):
                    S[i][j] = ''
                if (S[i][j] == 'k'):
                    S[i][j] = '+k'
                elif (S[i][j] == 'l'):
                    S[i][j] = '+l'                    
                                
        H = (S[0][0]+S[1][0]+S[2][0]).replace('*','')
        K = (S[0][1]+S[1][1]+S[2][1]).replace('*','')
        L = (S[0][2]+S[1][2]+S[2][2]).replace('*','')
                
        if (H == ''):
            H = '0'
        elif (H[0] == '+'):
            H = H[1:]
            
        if (K == ''):
            K = '0'
        elif (K[0] == '+'):
            K = K[1:]
            
        if (L == ''):
            L = '0'
        elif (L[0] == '+'):
            L = L[1:]
                    
        plt.title(r'$('+H+','+K+','+L+')$')   
        plt.xlabel(r'$h$')
        plt.ylabel(r'$l$')
        
        dh = (h_max-h_min)/(nh-1)
        dl = (l_max-l_min)/(nl-1)   
        extents_k = [h_min-dh/2, h_max+dh/2, 
                     l_min-dl/2, l_max+dl/2]
        
        trans = mtransforms.Affine2D()
        
        M = np.array([[B[0,0]/B[2,2],B[0,2]/B[2,2],0],
                      [B[2,0]/B[2,2],B[2,2]/B[2,2],0],
                      [0,0,1]])
        
        Q = np.eye(3)
        Q[0:2,0:2] = T.T.copy()[0:3:2,0:3:2]
        
        N = np.dot(M.T,M)
        
        scale_trans = N[1,1].copy()
        
        N[0:2] /= scale_trans
        N[0:2] /= scale_trans
        
        M = np.linalg.cholesky(np.dot(Q,np.dot(N,Q.T))).T
        
        scale_rot = M[1,1].copy()
        
        M[0,1] /= scale_rot
        M[0,0] /= scale_rot
        M[1,1] /= scale_rot
                
        scale = M[0,0].copy()
        
        M[0,1] /= scale
        M[0,0] /= scale
                
        trans.set_matrix(M)
        
        offset = -np.dot(M,[0,l_min,0])[0]
        
        shift = mtransforms.Affine2D().translate(offset,0)
        
        trans_data = trans+shift+plt.gca().transData
        
        plt.gca().set_aspect(1/scale)
        
        im.set_transform(trans_data)
        
        ext_min = np.dot(M[0:2,0:2],extents_k[0::2])
        ext_max = np.dot(M[0:2,0:2],extents_k[1::2])
        
        plt.gca().set_xlim(ext_min[0]+offset,ext_max[0]+offset)
        plt.gca().set_ylim(ext_min[1],ext_max[1])
        
        plt.gca().set_aspect(1/scale/scale_rot)
        plt.gca().set_transform(trans_data)    
    
    else:
 
        if (nl == 1):
            il = 0
            l = l_min
        else:
            il = int(round((index-l_min)/(l_max-l_min)*(nl-1)))
            l = l_min+((l_max-l_min)/(nl-1))*il 
        
        im = plt.imshow(I[:,:,il].T, 
                        norm=normalize,
                        interpolation='nearest', 
                        origin='lower', 
                        cmap=colormap,
                        extent=extents(h_min, 
                                       h_max, 
                                       nh,
                                       k_min, 
                                       k_max,
                                       nk))
        
        S = np.copy(P).tolist()
        for i in range(3):
            for j in range(3):
                if ('l' in P[i][j]):
                    S[i][j] = str(np.round(eval(P[i][j]),4))
                    if (S[i][j][0] != '-'):
                        S[i][j] = '+'+S[i][j]
                if (S[i][j] == '0'):
                    S[i][j] = ''
                if (S[i][j] == '+0'):
                    S[i][j] = ''
                if (S[i][j] == 'k'):
                    S[i][j] = '+k'
                elif (S[i][j] == 'l'):
                    S[i][j] = '+l'                    
                                
        H = (S[0][0]+S[1][0]+S[2][0]).replace('*','')
        K = (S[0][1]+S[1][1]+S[2][1]).replace('*','')
        L = (S[0][2]+S[1][2]+S[2][2]).replace('*','')
                
        if (H == ''):
            H = '0'
        elif (H[0] == '+'):
            H = H[1:]
            
        if (K == ''):
            K = '0'
        elif (K[0] == '+'):
            K = K[1:]
            
        if (L == ''):
            L = '0'
        elif (L[0] == '+'):
            L = L[1:]
                    
        plt.title(r'$('+H+','+K+','+L+')$')   
        plt.xlabel(r'$h$')
        plt.ylabel(r'$k$')
        
        dh = (h_max-h_min)/(nh-1)
        dk = (k_max-k_min)/(nk-1)
        extents_l = [h_min-dh/2, h_max+dh/2, 
                     k_min-dk/2, k_max+dk/2]
        
        trans = mtransforms.Affine2D()
        
        M = np.array([[B[0,0]/B[1,1],B[0,1]/B[1,1],0],
                      [B[1,0]/B[1,1],B[1,1]/B[1,1],0],
                      [0,0,1]])
        
        Q = np.eye(3)
        Q[0:2,0:2] = T.T.copy()[0:2,0:2]
        
        N = np.dot(M.T,M)
        
        scale_trans = N[1,1].copy()
        
        N[0:2] /= scale_trans
        N[0:2] /= scale_trans
        
        M = np.linalg.cholesky(np.dot(Q,np.dot(N,Q.T))).T
        
        scale_rot = M[1,1].copy()
        
        M[0,1] /= scale_rot
        M[0,0] /= scale_rot
        M[1,1] /= scale_rot
                
        scale = M[0,0].copy()
        
        M[0,1] /= scale
        M[0,0] /= scale
               
        trans.set_matrix(M)
        
        offset = -np.dot(M,[0,k_min,0])[0]
        
        shift = mtransforms.Affine2D().translate(offset,0)
                
        trans_data = trans+shift+plt.gca().transData
        
        im.set_transform(trans_data)
        
        ext_min = np.dot(M[0:2,0:2],extents_l[0::2])
        ext_max = np.dot(M[0:2,0:2],extents_l[1::2])
        
        plt.gca().set_xlim(ext_min[0]+offset,ext_max[0]+offset)
        plt.gca().set_ylim(ext_min[1],ext_max[1])
                
        plt.gca().set_aspect(1/scale/scale_rot)
        plt.gca().set_transform(trans_data)    

    plt.gca().xaxis.tick_bottom()

    cb = plt.colorbar()
    if (norm == 'linear'):
        cb.formatter.set_powerlimits((0, 0))
        cb.update_ticks()
        
    if (aspect is not None):
        
        plt.gca().set_aspect(aspect)
    
def linecut(I, 
            h_range, 
            k_range, 
            l_range, 
            error=None, 
            line='l', 
            indices=[0,0], 
            norm='log',
            T=np.eye(3)):
    
    nh, nk, nl = I.shape[0], I.shape[1], I.shape[2]
    
    h_min = h_range[0]
    h_max = h_range[1]
    
    k_min = k_range[0]
    k_max = k_range[1]
    
    l_min = l_range[0]
    l_max = l_range[1]
    
    plt.figure(dpi=100)
        
    if (line == 'h'):

        if (nk == 1):
            ik = 0
            k = k_min
        else:
            ik = int(round((indices[0]-k_min)/(k_max-k_min)*(nk-1)))
            k = k_min+((k_max-k_min)/(nk-1))*ik  
            
        if (nk == 1):
            il = 0
            l = l_min
        else:
            il = int(round((indices[1]-l_min)/(l_max-l_min)*(nl-1)))
            l = l_min+((l_max-l_min)/(nl-1))*il  
        
        h = np.linspace(h_range[0],h_range[1],nh)
        
        if (error != None):
            plt.errorbar(h, I[:,ik,il], yerr=np.sqrt(error[:,ik,il]))   
        else:
            plt.errorbar(h, I[:,ik,il])       
            
        plt.title(r'($h$,'+str(np.round(k*T[1,1]+l*T[1,2],4))+','\
                          +str(np.round(k*T[2,1]+l*T[2,2],4))+')')  
        plt.xlabel(r'$h$')
            
    elif (line == 'k'):

        if (nl == 1):
            il = 0
            l = l_min
        else:
            il = int(round((indices[0]-l_min)/(l_max-l_min)*(nl-1)))
            l = l_min+((l_max-l_min)/(nl-1))*il  

        if (nh == 1):
            ih = 0
            h = h_min
        else:
            ih = int(round((indices[1]-h_min)/(h_max-h_min)*(nh-1)))
            h = h_min+((h_max-h_min)/(nh-1))*ih  
        
        k = np.linspace(k_range[0],k_range[1],nk)
        
        if (error != None):
            plt.errorbar(k, I[ih,:,il], yerr=np.sqrt(error[ih,:,il]))
        else:
            plt.errorbar(k, I[ih,:,il])

        plt.title(r'('+str(np.round(h*T[0,0]+l*T[0,2],4))+',$k$,'\
                      +str(np.round(h*T[2,0]+l*T[2,2],4))+')')       
        plt.xlabel(r'$k$')
    
    else:
        
        if (nh == 1):
            ih = 0
            h = h_min
        else:
            ih = int(round((indices[0]-h_min)/(h_max-h_min)*(nh-1)))
            h = h_min+((h_max-h_min)/(nh-1))*ih  
            
        if (nk == 1):
            ik = 0
            k = k_min
        else:
            ik = int(round((indices[1]-k_min)/(k_max-k_min)*(nk-1)))
            k = k_min+((k_max-k_min)/(nk-1))*ik  
        
        l = np.linspace(l_range[0],l_range[1],nl)
        
        if (error != None):
            plt.errorbar(l, I[ih,ik,:], yerr=np.sqrt(error[ih,ik,:]))
        else:
            plt.errorbar(l, I[ih,ik,:])
            
        plt.title(r'('+str(np.round(h*T[0,0]+k*T[0,1],4))+','\
                      +str(np.round(h*T[1,0]+k*T[1,1],4))+',$l$)')    
        plt.xlabel(r'$l$')
       
    if (norm == 'log'):       
         plt.gca().set_yscale(norm)
    else:     
        plt.gca().ticklabel_format(style='sci', scilimits=(-2,2))

def spin(Sx, 
         Sy, 
         Sz, 
         rx, 
         ry, 
         rz, 
         atm,
         nu,
         nv,
         nw,
         xlim=[0,1], 
         ylim=[0,1], 
         zlim=[0,1], 
         elev=None, 
         azim=None,
         length=1):

    atoms = np.tile(atm, nu*nv*nw)
        
    fig = plt.figure(dpi=100)
    ax = fig.gca(projection='3d')
    
    ax.view_init(elev, azim)
    
    n_atm = atm.shape[0]
    
    i0, i1 = xlim[0], xlim[1]
    j0, j1 = ylim[0], ylim[1]
    k0, k1 = zlim[0], zlim[1]
    
    mask = np.full((nu,nv,nw,n_atm), fill_value=False)
    mask[i0:i1,j0:j1,k0:k1,:] = True
    mask = mask.flatten()
         
    for atom in np.unique(atm):
    
        veil = atoms[mask] == atom
        
        color = ax._get_lines.get_next_color()
        
        ax.quiver(rx[mask][veil], 
                  ry[mask][veil], 
                  rz[mask][veil], 
                  Sx[mask][veil],
                  Sy[mask][veil], 
                  Sz[mask][veil],
                  color=color,
                  pivot='middle',
                  length=length)
        
        ax.scatter(rx[mask][veil], 
                   ry[mask][veil], 
                   rz[mask][veil], 
                   c=color, 
                   depthshade=False)
        
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    
    ax.grid(False)
    
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_zlabel(r'$z$')

    ax.set_aspect('equal')
    
def correlation(r, S_corr, vmin=-1, vmax=1, disorder='magnetic'):
    
    if (disorder == 'magnetic'):
        label = r'$\overline{\mathbf{S}(0)\cdot \mathbf{S}(r)}$'
    elif (disorder == 'displacement'):
        label = r'$\overline{\mathbf{u}(0)\cdot \mathbf{u}(r)}$'    
    else:
        label = r'$\overline{a(0)a(r)}$'
        
    plt.figure(dpi=100)
    plt.plot(r, S_corr, 'o')
    plt.axhline(y=0,color='k',lw=1,zorder=-1)
    plt.ylim([vmin, vmax])
    plt.xlabel(r'$r$')
    plt.ylabel(label)

def collinearity(r, S_coll, vmin=0, vmax=1, disorder='magnetic'):
    
    if (disorder == 'magnetic'):
        label = r'$\overline{|\mathbf{S}(0)\cdot \mathbf{S}(r)|^2}$'
    elif (disorder == 'displacement'):
        label = r'$\overline{|\mathbf{u}(0)\cdot \mathbf{u}(r)|^2}$'
    else:
        label = r'$\overline{|a(0)a(r)|^2}$'
    
    plt.figure(dpi=100)
    plt.plot(r, S_coll, 'o')
    plt.ylim([vmin, vmax])
    plt.xlabel(r'$r$')
    plt.ylabel(label)

def correlation3d(rx, 
                  ry, 
                  rz, 
                  S_corr, 
                  plane='z', 
                  index=0, 
                  vmin=-1, 
                  vmax=1, 
                  tol=1e-4,
                  disorder='magnetic'):
    
    if (disorder == 'magnetic'):
        label = r'$\overline{\mathbf{S}(\mathbf{0})\cdot '+\
                           r'\mathbf{S}(\mathbf{r})}$'
    elif (disorder == 'displacement'):
        label = r'$\overline{\mathbf{u}(\mathbf{0})\cdot '+\
                           r'\mathbf{u}(\mathbf{r})}$'  
    else:
        label = r'$\overline{a(\mathbf{0})'+\
                           r'a(\mathbf{r})}$'
    
    plt.figure(dpi=100)
    
    if (plane == 'x'):
        
        mask = np.isclose(rx, index, rtol=tol)

        plt.figure(dpi=100)
        plt.scatter(ry[mask], 
                    rz[mask], 
                    c=S_corr[mask], 
                    vmin=vmin, 
                    vmax=vmax, 
                    cmap=plt.cm.bwr)
        plt.title(r'$x='+str(index)+'$')
        plt.xlabel(r'$y$')
        plt.ylabel(r'$z$')
        
    elif (plane == 'y'):
        
        mask = np.isclose(ry, index, rtol=tol)

        plt.figure(dpi=100)
        plt.scatter(rx[mask], 
                    rz[mask], 
                    c=S_corr[mask], 
                    vmin=vmin, 
                    vmax=vmax, 
                    cmap=plt.cm.bwr)
        plt.title(r'$y='+str(index)+'$')
        plt.xlabel(r'$x$')
        plt.ylabel(r'$z$')
        
    else:
        
        mask = np.isclose(rz, index, rtol=tol)

        plt.figure(dpi=100)
        plt.scatter(rx[mask], 
                    ry[mask], 
                    c=S_corr[mask], 
                    vmin=vmin, 
                    vmax=vmax, 
                    cmap=plt.cm.bwr)
        plt.title(r'$z='+str(index)+'$')
        plt.xlabel(r'$x$')
        plt.ylabel(r'$y$')
        
    cb = plt.colorbar()
    cb.set_label(label)
    plt.gca().set_aspect('equal')
    plt.gca().ticklabel_format(style='sci', scilimits=(-2,2))
    
def collinearity3d(rx, 
                   ry, 
                   rz, 
                   S_corr, 
                   plane='z',
                   norm='log',
                   index=0, 
                   vmin=0, 
                   vmax=1, 
                   tol=1e-4,
                   disorder='magnetic'):
    
    if (disorder == 'magnetic'):
        label = r'$\overline{|\mathbf{S}(\mathbf{0})\cdot '+\
                            r'\mathbf{S}(\mathbf{r})|^2}$'
    elif (disorder == 'displacement'):
        label = r'$\overline{|\mathbf{u}(\mathbf{0})\cdot '+\
                            r'\mathbf{u}(\mathbf{r})|^2}$'
    else:
        label = r'$\overline{|a(\mathbf{0})'+\
                            r'a(\mathbf{r})|^2}$'

    if (norm == 'log'):
        normalize = colors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        normalize = colors.Normalize(vmin=vmin, vmax=vmax)
    
    plt.figure(dpi=100)
    
    if (plane == 'x'):
        
        mask = np.isclose(rx, index, rtol=tol)

        plt.figure(dpi=100)
        plt.scatter(ry[mask], 
                    rz[mask], 
                    c=S_corr[mask], 
                    vmin=vmin, 
                    vmax=vmax,
                    norm=normalize,
                    cmap=plt.cm.viridis)
        plt.title(r'$x='+str(index)+'$')
        plt.xlabel(r'$y$')
        plt.ylabel(r'$z$')
        
    elif (plane == 'y'):
        
        mask = np.isclose(ry, index, rtol=tol)

        plt.figure(dpi=100)
        plt.scatter(rx[mask], 
                    rz[mask], 
                    c=S_corr[mask], 
                    vmin=vmin, 
                    vmax=vmax,
                    norm=normalize,
                    cmap=plt.cm.viridis)
        plt.title(r'$y='+str(index)+'$')
        plt.xlabel(r'$x$')
        plt.ylabel(r'$z$')
        
    else:
        
        mask = np.isclose(rz, index, rtol=tol)

        plt.figure(dpi=100)
        plt.scatter(rx[mask], 
                    ry[mask], 
                    c=S_corr[mask], 
                    vmin=vmin, 
                    vmax=vmax, 
                    norm=normalize,
                    cmap=plt.cm.viridis)
        plt.title(r'$z='+str(index)+'$')
        plt.xlabel(r'$x$')
        plt.ylabel(r'$y$')
        
    cb = plt.colorbar()
    cb.set_label(label)
    plt.gca().set_aspect('equal')
    plt.gca().ticklabel_format(style='sci', scilimits=(-2,2))

def coolwarm():
    return plt.cm.coolwarm

def viridis():
    return plt.cm.viridis

def powder(Q, I):
    
    plt.figure(dpi=100)
    plt.plot(Q/2/np.pi, I)
    plt.xlabel(r'$Q/2\pi$')    
    plt.ylabel(r'$I(Q)$')