# import commands
import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as widgets
import matplotlib.gridspec as gridspec
import time
import ipywidgets as widgets
import warnings
warnings.filterwarnings('ignore')
from numpy import trapz, ma
from scipy import integrate
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def doubleintegral():
    func = widgets.Dropdown(\
        options=['Ex_1','Ex_2','Ex_3'],\
        value='Ex_2',\
        description='Select Example')

    inner = widgets.RadioButtons(\
        options=['with respect to x', 'with respect to y'],\
        value='with respect to x',\
        description='Inner Integral:',\
        continuous_update=False,\
        disabled=False)    

    nslices = widgets.BoundedIntText(\
        value=3,\
        min=1,\
        max=10,\
        step=1,\
        description='number of slices',\
        continuous_update=False,\
        disabled=False)

    view_hor = widgets.BoundedIntText(\
        value=120,\
        min=0,\
        max=360,\
        step=10,\
        description='horizontal viewing angle',\
        continuous_update=False,\
        disabled=False)

    view_vert = widgets.BoundedIntText(\
        value=30,\
        min=0,\
        max=45,\
        step=5,\
        description='vertical viewing angle',\
        continuous_update=False,\
        disabled=False)
    
    return widgets.VBox([\
        widgets.HBox([func]),\
        widgets.HBox([inner,nslices]),\
        widgets.HBox([view_hor,view_vert]),\
        widgets.interactive_output(doubleintegral_run,\
        {'func':func,'inner':inner,'nslices':nslices,\
        'view_hor':view_hor,'view_vert':view_vert})])


def intcalc_func(func,opt,x,y):
    if opt == 'mesh_x' or opt == 'mesh_y':
        nxy = 40
        xmesh = np.empty([nxy,nxy])
        ymesh = np.empty([nxy,nxy])
    if func == 'Ex_1':
        if opt == 'surf':
            surf = x*x*y
        if opt == 'mesh_x':
            for i in range(nxy):
                ymesh[i][:] = 3.+i*1./(nxy*1.-1.)
                for j in range(nxy):
                    xmesh[i][j] = j*ymesh[i][0]/(nxy*1.-1.)
        if opt == 'mesh_y':
            for i in range(nxy):
                xmesh[i][:] = 0.+i*2./(nxy*1.-1.)
                for j in range(nxy):
                    ymesh[i][j] = j*2.*xmesh[i][0]/(nxy*1.-1.)         
    elif func == 'Ex_2':
        if opt == 'surf':
            surf = y
        if opt == 'mesh_x':
            for i in range(nxy):
                ymesh[i][:] = 0.+i*4./(nxy*1.-1.)
                for j in range(nxy):
                    xmesh[i][j] = 0.5*ymesh[i][0]+j*(2.-0.5*ymesh[i][0])/(nxy*1.-1.)
        if opt == 'mesh_y':
            for i in range(nxy):
                xmesh[i][:] = 0.+i*2./(nxy*1.-1.)
                for j in range(nxy):
                    ymesh[i][j] = j*2.*xmesh[i][0]/(nxy*1.-1.)
    elif func == 'Ex_3':
        if opt == 'surf':
            surf = np.exp(x*x*x)
        elif opt == 'mesh':
            xmesh = np.linspace(0.,1.,nxy)
            ymesh = np.linspace(0.,4.,nxy)
    
    # return appropriate output
    if opt == 'surf':
        return surf
    if opt == 'mesh_x' or opt == 'mesh_y':
        return xmesh, ymesh


def doubleintegral_run(func,inner,nslices,view_hor,view_vert):
    
    # set some basic meshgrids, arrays and values
    nxy = 40
    if inner == 'with respect to x':
        X,Y = intcalc_func(func, 'mesh_x', 0., 0.)
    elif inner == 'with respect to y':
        X,Y = intcalc_func(func, 'mesh_y', 0., 0.)
    Z = np.nan_to_num(intcalc_func(func, 'surf', X, Y))
    
    # use number of slices to find integer points for meshgrid to choose
    slice_spacing = nxy/nslices
    slices = np.empty(nslices, dtype='int')
    for i in range(nslices):
        slices[i] = np.ceil(slice_spacing/2 + i*slice_spacing)
    
    # create the basic plotting setup
    fig = plt.figure(figsize=(16, 8))

    # plot the new 3d surface wireframe and integration region on xy plane
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('f(x,y)')
    ax1.set_zlim([0,Z.max()])
    ax1.plot_surface(X, Y, 0*Z, rstride=1, cstride=1, color='0.75', linewidth=0, antialiased=True, alpha=0.4)
    ax1.plot_wireframe(X, Y, Z, rstride=1, cstride=1, color='black', linewidth=0.4)    

    for i in range(nslices):
        n = slices[i]
        if inner == 'with respect to x':
            verts = np.array([[X[n][-1],Y[n][0],0.],[X[n][0],Y[n][0],0.]])            
            for j in range(nxy):
                verts = np.append(verts, [[X[n][j],Y[n][0],Z[n][j]]], axis=0)
            verts = np.append(verts, [[X[n][-1],Y[n][0],0.]], axis=0)

        elif inner == 'with respect to y':
            verts = np.array([[X[n][0],Y[-1][n],0.],[X[n][0],Y[0][n],0.]])            
            for j in range(nxy):
                verts = np.append(verts, [[X[n][0],Y[j][n],Z[j][n]]], axis=0)
            
        # add polygon to the current plot
        face = Poly3DCollection([verts], linewidth=1, alpha=0.5)
        face.set_facecolor((0, 0, 1, 0.2))
        ax1.add_collection3d(face)
            
        # empty out the current polygon array
        verts = np.delete(verts, np.s_[::1], None)
            
        # change the viewing angle
        ax1.view_init(view_vert, view_hor)

# some old code to ignore
#             verts = np.array([[curr_x,y[-1],0.],[curr_x,y[0],0.]])            
#             for j in range(len(y)):
#                 curr_y = y[j]
#                 curr_z = intcalc_func(func,'surf',curr_x,curr_y)
#                 verts = np.append(verts, [[curr_x,curr_y,curr_z]], axis=0)
#             verts = np.append(verts, [[curr_x,y[-1],0.]], axis=0)


def gradient():
    func = widgets.Dropdown(\
        options=['Ex_4','Ex_5'],\
        value='Ex_4',\
        description='Select Example')

    model = widgets.RadioButtons(\
        options=['gradient descent (min)', 'gradient ascent(max)'],\
        description='Model Option:',\
        continuous_update=False,\
        disabled=False)
    
    iteration = widgets.IntSlider(\
        value = 0,\
        min=0,\
        max=20,\
        step=1,\
        description='Iteration number:',\
        disabled=False,\
        continuous_update=False)
    
    step = widgets.BoundedFloatText(\
        value=0.1,\
        min=0.01,\
        max=0.2,\
        step=0.01,\
        description='step size for gradient ascent/descent',\
        continuous_update=False,\
        disabled=False)
    
    # spatial sliders
    xzoom = widgets.FloatRangeSlider(\
        value=[-1.2,1.2],\
        min=-2.,\
        max=2.,\
        step=0.1,\
        continuous_update=False,\
        description='x range for contour plot')
    
    yzoom = widgets.FloatRangeSlider(\
        value=[-1.2,1.2],\
        min=-2,\
        max=2.,\
        step=0.1,\
        continuous_update=False,\
        description='y range for contour plot')
    
    view_grad = widgets.Checkbox(\
        value=False,\
        description='View gradient vector field')
    
    # point at which to evaluate partial derivatives
    pt_x = widgets.BoundedFloatText(\
        value=0.,\
        min=-2.,\
        max=2.,\
        step=0.1,\
        description='x coordinate of starting point',\
        continuous_update=False,\
        disabled=False)
    
    pt_y = widgets.BoundedFloatText(\
        value=0.,\
        min=-2.,\
        max=2.,\
        step=0.1,\
        description='y coordinate of starting point',\
        continuous_update=False,\
        disabled=False)
    
    # interactive output
    return widgets.VBox([widgets.HBox([func]),\
        widgets.HBox([model,iteration,step]),\
        widgets.HBox([xzoom,yzoom,view_grad]),\
        widgets.HBox([pt_x,pt_y]),\
        widgets.interactive_output(gradient_run,\
        {'func':func,'model':model,'iteration':iteration,'step':step,\
        'xzoom':xzoom,'yzoom':yzoom,'view_grad':view_grad,'pt_x':pt_x,'pt_y':pt_y})])


def directionalderivative():
    func = widgets.Dropdown(options=['Ex_4','Ex_5'], value='Ex_5', description='Select Example')
    
    # spatial slider
    xzoom = widgets.FloatRangeSlider(value=[-2.,2.], min=-5., max=5., step=0.1, continuous_update=False, description='x range for contour plot')
    yzoom = widgets.FloatRangeSlider(value=[-2.,2.], min=-5, max=5., step=0.1, continuous_update=False, description='y range for contour plot')
    view_grad = widgets.Checkbox(value=False, description='View gradient vector field')
    
    # point at which to evaluate partial derivatives
    pt_x = widgets.BoundedFloatText(value=0.4, min=-5., max=5., step=0.1, description='x coordinate of starting point', continuous_update=False, disabled=False)
    pt_y = widgets.BoundedFloatText(value=0.2, min=-5., max=5., step=0.1, description='y coordinate of starting point', continuous_update=False, disabled=False)

    # point at which to evaluate partial derivatives
    dir_x = widgets.BoundedFloatText(value=1., min=-2., max=2., step=0.1, description='x of direction vector', continuous_update=False, disabled=False)
    dir_y = widgets.BoundedFloatText(value=1., min=-2., max=2., step=0.1, description='y of direction vector', continuous_update=False, disabled=False)
    dir_norm = widgets.Checkbox(value=True, description='normalise the vector direction', disabled=True)
    
    # interactive output
    return widgets.VBox([widgets.HBox([func]),\
        widgets.HBox([xzoom,yzoom,view_grad]),\
        widgets.HBox([pt_x,pt_y]),\
        widgets.HBox([dir_x,dir_y,dir_norm]),\
        widgets.interactive_output(directionalderivative_run,\
        {'func':func,'xzoom':xzoom,'yzoom':yzoom,'view_grad':view_grad,\
         'pt_x':pt_x,'pt_y':pt_y,\
        'dir_x':dir_x,'dir_y':dir_y,'dir_norm':dir_norm})])





# select examples for differential calc notebooks
def diffcalc_func(func,opt,x,y):   
    if func == 'Ex_1':
        if opt == 'surf':
            surf = 1.-x*x
        elif opt == 'dfdx':
            dfdx = -2.*x
        elif opt == 'dfdy':
            dfdy = 0.*x
    elif func == 'Ex_2':
        if opt == 'surf':
            surf = 2.*x*y + y*y
        elif opt == 'dfdx':
            dfdx = 2.*y
        elif opt == 'dfdy':
            dfdy = 2.*x+2.*y
    elif func == 'Ex_3':
        if opt == 'surf':
            surf = x*x*x*y + x*np.sin(y*y)
        elif opt == 'dfdx':
            dfdx = 3.*x*x*y+np.sin(y*y)
        elif opt == 'dfdy':
            dfdy = x*x*x + 2.*x*y*np.cos(y*y)
    elif func == 'Ex_4':
        if opt == 'surf':
            surf = x*x*x/3.-y*y*y/3.-x+y+3.
        elif opt == 'dfdx':
            dfdx = x*x - 1.
        elif opt == 'dfdy':
            dfdy = -y*y + 1.
        elif opt == 'stpt':
            stx = np.array([-1., -1., 1., 1.])
            sty = np.array([1., -1., 1., -1.])
        elif opt == 'mesh':
            xrange = np.array([-2.,2.])
            yrange = np.array([-2.,2.])
    elif func == 'Ex_5':
        if opt == 'surf':
            surf = x*x-y*y
        elif opt == 'dfdx':
            dfdx = 2.*x
        elif opt == 'dfdy':
            dfdy = -2.*y
        elif opt == 'stpt':
            stx = np.array([0.])
            sty = np.array([0.])
        elif opt == 'mesh':
            xrange = np.array([-5.,5.])
            yrange = np.array([-5.,5.])
    # return requested output
    if opt == 'surf':
        return surf
    elif opt == 'dfdx':
        return dfdx
    elif opt == 'dfdy':
        return dfdy
    elif opt == 'stpt':
        return stx,sty
    elif opt == 'mesh':
        return xrange,yrange


def directionalderivative_run(func,xzoom,yzoom,view_grad,pt_x,pt_y,dir_x,dir_y,dir_norm):
    
    # create domain in a few different forms
    dxy = 0.1
    xrange,yrange = diffcalc_func(func, 'mesh', 0., 0.)
    x = np.arange(xrange[0],xrange[1],dxy)
    y = np.arange(yrange[0],yrange[1],dxy)
    X,Y = np.meshgrid(x, y)
    Z = np.nan_to_num(diffcalc_func(func, 'surf', X, Y))

    # set numpy arrays for fixed point to include in the plots
    ptx = np.array([pt_x])
    pty = np.array([pt_y])
    ptz = diffcalc_func(func, 'surf', ptx, pty)

    # rate of change in x, y direction at all meshgrid points
    GY, GX = np.gradient(Z)

    # coordinates of stationary points
    stx, sty = diffcalc_func(func,'stpt',0.,0.)
    stz = diffcalc_func(func,'surf',stx,sty)

    # find gradient vector at point of interest
    dfdx = diffcalc_func(func,'dfdx', pt_x, pt_y)
    dfdy = diffcalc_func(func,'dfdy', pt_x, pt_y)
    grad_vec = np.array([dfdx,dfdy])

    # find vector of direction of interest, and normalise if user-selected
    dir_vec = np.array([dir_x,dir_y])
    if dir_norm:
        dir_vec_length = np.sqrt(np.dot(dir_vec,dir_vec))
        dir_vec = dir_vec / dir_vec_length

    # calculate directional derivative and apply to scale the direction vector
    dirder = np.dot(grad_vec,dir_vec)
    dot_vec = dirder*dir_vec
    
    # setup basic plotting structure
    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122)

    # plot 1: surface with stationary points and initial point for gradient ascent/descent
    norm = plt.Normalize(Z.min(), Z.max())
    colors = cm.viridis(norm(Z))
    rcount, ccount, _ = colors.shape
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z(x,y)')
    surf = ax1.plot_surface(X, Y, Z, rcount=rcount, ccount=ccount, facecolors=colors, shade=False)
    surf.set_facecolor((0,0,0,0))
    ax1.plot(stx, sty, stz, 'ro')
    ax1.plot([ptx,ptx], [Y.min(),pty], [Z.min(),Z.min()], 'k--')
    ax1.plot([X.min(),ptx], [pty,pty], [Z.min(),Z.min()], 'k--')
    ax1.plot([ptx,ptx], [pty,pty], [Z.min(),ptz], 'k--')
    ax1.plot(ptx,pty,ptz, 'ko')    
#     ax1.quiver((a,a), (b,b),(stz[0],stz[0]), (grad_vec[0],dir_vec[0]), (grad_vec[1],dir_vec[1]),(stz[0],stz[0]), colors=['k','g'], pivot='tail', arrow_length_ratio = 0.1)
    
    # contour plot with vector arrows
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_xlim([xzoom[0],xzoom[1]])
    ax2.set_ylim([yzoom[0],yzoom[1]])
    print(Z.min(),Z.max(),(Z.max()-Z.min())/50.)
    levels = np.arange(Z.min(),Z.max(),(Z.max()-Z.min())/20.)
    cont = ax2.contour(X, Y, Z, facecolors=colors, levels=levels)
    ax2.plot(stx, sty, 'ro')
    ax2.plot(ptx, pty, 'ko')
    ax2.quiver(pt_x, pt_y, grad_vec[0], grad_vec[1], angles='xy', scale_units='xy', scale=1., width=0.005, color='blue', pivot='tail', label=r'$\nabla f$')
    ax2.quiver(pt_x, pt_y, dir_vec[0], dir_vec[1], angles='xy', scale_units='xy', scale=1., width=0.005, color='red', pivot='tail', label=r'$\hat{a}$')
    ax2.quiver(pt_x, pt_y, dot_vec[0], dot_vec[1], angles='xy', scale_units='xy', scale=1., width=0.005, color='green', pivot='tail', label=r'$\left(\nabla f \cdot \hat{a}\right) \hat{a}$')
    textstr = r'$\nabla f \cdot \hat{a} = %.2f$' % (dirder,)    
    ax2.text(0.05, 0.95, textstr, backgroundcolor='white', transform=ax2.transAxes, fontsize=14, verticalalignment='top')
    if view_grad:
        ax2.quiver(X, Y, GX, GY, angles='xy', scale_units='xy', scale=1., width=0.005, color='black', pivot='tail')
    ax2.legend()


def gradient_run(func,model,iteration,step,xzoom,yzoom,view_grad,pt_x,pt_y):
 
    # create domain in a few different forms
    dxy = 0.1
    xrange,yrange = diffcalc_func(func, 'mesh', 0., 0.)
    x = np.arange(xrange[0],xrange[1],dxy)
    y = np.arange(yrange[0],yrange[1],dxy)
    X,Y = np.meshgrid(x, y)
    Z = np.nan_to_num(diffcalc_func(func, 'surf', X, Y))

    # set numpy arrays for fixed point to include in the plots
    ptx = np.array([pt_x])
    pty = np.array([pt_y])
    ptz = diffcalc_func(func, 'surf', ptx, pty)

    # rate of change in x, y direction at all meshgrid points
    GY, GX = np.gradient(Z)

    # coordinates of stationary points
    stx, sty = diffcalc_func(func,'stpt',0.,0.)
    stz = diffcalc_func(func,'surf',stx,sty)

    # setup basic plotting structure
    fig = plt.figure(figsize=(18, 10))
    ax1 = fig.add_subplot(121, projection='3d')

    # plot 1: surface with stationary points and initial point for gradient ascent/descent
    norm = plt.Normalize(Z.min(), Z.max())
    colors = cm.viridis(norm(Z))
    rcount, ccount, _ = colors.shape
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z(x,y)')
    surf = ax1.plot_surface(X, Y, Z, rcount=rcount, ccount=ccount, facecolors=colors, shade=False)
    surf.set_facecolor((0,0,0,0))
    ax1.plot(stx, sty, stz, 'ro')
    ax1.plot([ptx,ptx], [Y.min(),pty], [Z.min(),Z.min()], 'k--')
    ax1.plot([X.min(),ptx], [pty,pty], [Z.min(),Z.min()], 'k--')
    ax1.plot([ptx,ptx], [pty,pty], [Z.min(),ptz], 'k--')
    ax1.plot(ptx,pty,ptz, 'ko')
    
    # contour plot with gradient ascent/descent
    ax2 = fig.add_subplot(122)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_xlim([xzoom[0],xzoom[1]])
    ax2.set_ylim([yzoom[0],yzoom[1]])
    levels = np.arange(Z.min(),Z.max(),(Z.max()-Z.min())/20.)
    cont = ax2.contour(X, Y, Z, facecolors=colors, levels=levels)
    ax2.plot(stx, sty, 'ro')
    ax2.plot(ptx, pty, 'ko')
    
    if view_grad:
        ax2.quiver(X, Y, GX, GY, angles='xy', scale_units='xy', scale=1., width=0.005, color='black', pivot='tail')
    
    # set initial location of gradient ascent or descent, save in vector 
    old_x = ptx
    old_y = pty
    old_z = diffcalc_func(func,'surf', old_x, old_y)    
    coord = [[old_x, old_y, old_z]]
    
    # learning rate/ step size
    alpha = step
    
    # set iteration counter
    i = 0
    
    # start iteration loop
    while i < iteration:
        
    # compute steepest descent direction
        dfdx = diffcalc_func(func,'dfdx', old_x, old_y)
        dfdy = diffcalc_func(func,'dfdy', old_x, old_y)

        # find updated point location after an iteration
        if model == 'gradient ascent(max)':
            update_x = dfdx*alpha
            update_y = dfdy*alpha
        elif model == 'gradient descent (min)':
            update_x = -1.*dfdx*alpha
            update_y = -1.*dfdy*alpha
        new_x = old_x + update_x
        new_y = old_y + update_y
        new_z = diffcalc_func(func,'surf', new_x, new_y)
        
        # add iteration to the two plots
        ax1.plot(new_x, new_y, new_z, 'bo', linestyle='None', label='Label', zorder=10)
        ax2.plot(new_x, new_y, 'bo', linestyle='None', label='Label', zorder=10)
        ax2.quiver(old_x, old_y, update_x, update_y, angles='xy', scale_units='xy', scale=1., width=0.005, color='black', pivot='tail')

        # set old point based on latest iteration
        old_x = new_x
        old_y = new_y
        
        # update iteration counter
        i += 1


def partialderivative():
    # choose example from list  
    func = widgets.Dropdown(options=['Ex_1','Ex_2','Ex_3'], value='Ex_1', description='Select Example')

    # spatial slider
    xyrange = widgets.IntRangeSlider(value=[-2,2], min=-4, max=4, step=1, continuous_update=False, description='x and y range')

    # point at which to evaluate partial derivatives
    pt_x = widgets.BoundedFloatText(value=0., min=-4., max=4., step=0.1, description='x coordinate of point', continuous_update=False, disabled=False)
    pt_y = widgets.BoundedFloatText(value=0., min=-4., max=4., step=0.1, description='y coordinate of point', continuous_update=False, disabled=False)
    
    # interactive output
    return widgets.VBox([widgets.HBox([func]),\
        widgets.HBox([xyrange,pt_x,pt_y]),\
        widgets.interactive_output(partialderivative_run,\
            {'func':func,'xyrange':xyrange,\
            'pt_x':pt_x,'pt_y':pt_y})])


# define plot function
def partialderivative_run(func,xyrange,pt_x,pt_y):
    
    # create domain in a few different forms
    dxy = 0.1
    x = np.arange(xyrange[0],xyrange[1],dxy)
    y = np.arange(xyrange[0],xyrange[1],dxy)
    X,Y = np.meshgrid(x, y)
    pt_x_rep = np.repeat(pt_x, len(x))
    pt_y_rep = np.repeat(pt_y, len(y))
    Z = np.nan_to_num(diffcalc_func(func, 'surf', X, Y))

    # set numpy arrays for fixed point to include in the plots
    stx = np.array([pt_x])
    sty = np.array([pt_y])
    stz = diffcalc_func(func, 'surf', stx, sty)
    
    # determine f(x,pt_y) and f(pt_x,y)
    z1 = np.nan_to_num(diffcalc_func(func, 'surf', pt_x_rep, pt_y))
    z2 = np.nan_to_num(diffcalc_func(func, 'surf', pt_x, pt_y_rep))

    # determine f(x,pt_y) and f(pt_x,y)
    pt1 = np.nan_to_num(diffcalc_func(func, 'surf', x, pt_y_rep))
    pt2 = np.nan_to_num(diffcalc_func(func, 'surf', pt_x_rep, y))    
    
    # find slope at current point in each plot
    pt_dfdx = np.nan_to_num(diffcalc_func(func, 'dfdx', pt_x, pt_y))
    pt_dfdy = np.nan_to_num(diffcalc_func(func, 'dfdy', pt_x, pt_y))

    # create repeated set of pt_x and pt_y values
    tan_dfdx = pt_dfdx*x + (stz-pt_x*pt_dfdx)
    tan_dfdy = pt_dfdy*y + (stz-pt_y*pt_dfdy)
    
    # create the basic plotting environment
    gs = gridspec.GridSpec(2, 2)
    fig = plt.figure(figsize=(14, 8))
    ax1 = fig.add_subplot(gs[:,0], projection='3d')
    ax2 = fig.add_subplot(gs[0,1])
    ax3 = fig.add_subplot(gs[1, 1])

    # create some buffers for plotting windows
    xbuffer = (x.max()-x.min())/5.
    ybuffer = (y.max()-y.min())/5.
    pt1buffer = (pt1.max()-pt1.min())/5.
    pt2buffer = (pt2.max()-pt2.min())/5.

    # plot 1: surface
    norm = plt.Normalize(Z.min(), Z.max())
    colors = cm.viridis(norm(Z))
    rcount, ccount, _ = colors.shape
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z(x,y)')
    surf = ax1.plot_surface(X, Y, Z, rcount=rcount, ccount=ccount, facecolors=colors, shade=False)
    surf.set_facecolor((0,0,0,0))
    ax1.plot(stx, sty, stz, 'ko')
    ax1.plot([stx,stx], [Y.min(),sty], [Z.min(),Z.min()], 'k--')
    ax1.plot([X.min(),stx], [sty,sty], [Z.min(),Z.min()], 'k--')
    ax1.plot([stx,stx], [sty,sty], [Z.min(),stz], 'k--')
    
    
    # plot 2a: partial derivative in x direction
    ax2.set_xlabel('x')
    ax2.set_ylabel('f(x,pt_y)')
    ax2.set_xlim([x.min()-xbuffer, x.max()+xbuffer])
    ax2.set_ylim([pt1.min()-pt1buffer, pt1.max()+pt1buffer])
    ax2.plot(x, pt1)
    ax2.plot(pt_x, stz, 'ko')
    ax2.plot(x, tan_dfdx, 'b--')
    textstr = r'$\frac{df}{dx} = %.2f$' % (pt_dfdx,)    
    ax2.text(0.05, 0.95, textstr , transform=ax2.transAxes, fontsize=14, verticalalignment='top')
    
    ax3.set_xlabel('y')
    ax3.set_ylabel('f(pt_x,y)')
    ax3.set_xlim([y.min()-ybuffer, y.max()+xbuffer])
    ax3.set_ylim([pt2.min()-pt2buffer, pt2.max()+pt2buffer])
    ax3.plot(y, pt2)
    ax3.plot(pt_y, stz, 'ko')
    ax3.plot(y, tan_dfdy, 'b--')
    textstr3 = r'$\frac{df}{dy} = %.2f$' % (pt_dfdy,)    
    ax3.text(0.05, 0.95, textstr3 , transform=ax3.transAxes, fontsize=14, verticalalignment='top')


def spacecurve():
    # choose example from list  
    func = widgets.Dropdown(options=['Ex_1','Ex_2','Ex_3'], value='Ex_1', description='Select Example')

    # time sliders
    ts = widgets.FloatSlider(value=0., min=0, max=1.*np.pi, step=0.1*np.pi, continuous_update=False, description='Time Start')
    te = widgets.FloatSlider(value=2.*np.pi, min=0.1*np.pi, max=4.*np.pi, step=0.1*np.pi, continuous_update=False, description='Time End')
    dt = widgets.FloatSlider(value=0.1*np.pi, min=0.01*np.pi, max=0.5*np.pi, step=0.01*np.pi, continuous_update=False, description='Time Step')

    # plot 1 space curve: position, velocity, total acceleration
    view_pos = widgets.Checkbox(value=True, description='View position', disabled=False)
    view_vel = widgets.Checkbox(value=False, description='View velocity', disabled=False)
    view_acc = widgets.Checkbox(value=False, description='View total acceleration', disabled=False)
    
    # plot 1 vectors: unit tangent vector, and acceleration components
    view_utan = widgets.Checkbox(value=False, description='View unit tangent vector')
    view_atan = widgets.Checkbox(value=False, description='View tangential acceleration vector')
    view_anrm = widgets.Checkbox(value=False, description='View normal acceleration vector')
    
    # plot 2 speed
    view_sped = widgets.Checkbox(value=True, description='View speed as function of time', disabled=True)
    view_area = widgets.Checkbox(value=True, description='View distance travelled along space curve', disabled=False)

    # reparameterise by arc length
    para_arcl = widgets.Checkbox(value=False, description='Re-parameterise by arc length', disabled=False)
    
    # interactive output
    return widgets.VBox([widgets.HBox([func]),\
        widgets.HBox([ts,te,dt]),\
        widgets.HBox([view_pos,view_vel,view_acc]),\
        widgets.HBox([view_utan,view_atan,view_anrm]),\
        widgets.HBox([view_sped,view_area]),\
        widgets.HBox([para_arcl]),\
        widgets.interactive_output(spacecurve_run,\
            {'func':func,'ts':ts,'te':te,'dt':dt,\
            'view_pos':view_pos,'view_vel':view_vel,'view_acc':view_acc,\
            'view_utan':view_utan,'view_atan':view_atan,'view_anrm':view_anrm,\
            'view_sped':view_sped,'view_area':view_area,\
            'para_arcl':para_arcl})])


def spacecurve_func(func,para,t):
    # provide examples of space curves
    rz = None
    vz = None
    az = None
    if func == 'Ex_1':
        if para == 'time':
            ndim = 2
            rx = -2+np.cos(t)
            ry = 2+np.sin(t)
            vx = -1.*np.sin(t)
            vy = 1.*np.cos(t)
            ax = -1.*np.cos(t)
            ay = -1.*np.sin(t)
        elif para == 'arcl':
            rx = -2+np.cos(t)
            ry = 2+np.sin(t)
            vx = -1.*np.sin(t)
            vy = 1.*np.cos(t)
    if func == 'Ex_2':
        if para == 'time':
            ndim = 3
            rx = np.cos(t)
            ry = np.sin(t)
            rz = t
            vx = -1.*np.sin(t)
            vy = np.cos(t)
            vz = 0.*t+1.
            ax = -1.*np.cos(t)
            ay = -1.*np.sin(t)
            az = 0.*t
        elif para == 'arcl':
            rx = np.cos(t/np.sqrt(2.))
            ry = np.sin(t/np.sqrt(2.))
            rz = t/np.sqrt(2.)
            vx = -1.*np.sin(t/np.sqrt(2.))/np.sqrt(2.)
            vy = np.cos(t/np.sqrt(2.))/np.sqrt(2.)
            vz = 0.*t+1./np.sqrt(2.)
    if func == 'Ex_3':
        if para == 'time':
            ndim = 2
            rx = t*t/np.sqrt(2)
            ry = t*t/np.sqrt(2)
            vx = np.sqrt(2)*t
            vy = np.sqrt(2)*t
            ax = 0.*t+np.sqrt(2)
            ay = 0.*t+np.sqrt(2)
        elif para == 'arcl':
            rx = t/np.sqrt(2)
            ry = t/np.sqrt(2)
            vx = 0*t+1./np.sqrt(2)
            vy = 0*t+1./np.sqrt(2)
    if para == 'time':   
        return rx,ry,rz,vx,vy,vz,ax,ay,az,ndim        
    if para == 'arcl':
        return rx,ry,rz,vx,vy,vz


# calculate the magnitude of the velocity vector
def spacecurve_sped(ndim,vx,vy,vz):
    if ndim == 2:
        sped = np.sqrt(vx*vx+vy*vy)
    elif ndim == 3:
        sped = np.sqrt(vx*vx+vy*vy+vz*vz)
    return sped


# calculate the arc length function for each example
def spacecurve_arcl_func(func,t):
    arcl = t*0.
    if func == 'Ex_1':
        arcl_func = lambda t: 1.
    if func == 'Ex_2':
        arcl_func = lambda t: np.sqrt(2)
    if func == 'Ex_3':
        arcl_func = lambda t: 2.*t
    for i in range(len(t)):
        arcl[i] = integrate.quad(arcl_func, 0, t[i])[0]
    return arcl


# main script for runnint the space curves demo
def spacecurve_run(func,ts,te,dt,\
            view_pos,view_vel,view_acc,\
            view_utan,view_atan,view_anrm,\
            view_sped,view_area,\
            para_arcl):

    # set time range using input values from sliders
    t = np.arange(ts+0.000001,te+dt,dt)

    # determine position and acceleration as function of time
    rx_t,ry_t,rz_t,vx_t,vy_t,vz_t,ax_t,ay_t,az_t,ndim = spacecurve_func(func,'time',t)

    # determine the speed as magnitude of the velocity vector as a function of time
    sp_t = spacecurve_sped(ndim,vx_t,vy_t,vz_t)

    # if re-parameterising by arc length
    if para_arcl:
        # find the arc length function for given timespan and function
        arcl = spacecurve_arcl_func(func,t)
        
        # calculate position and velocity as a function of arc length
        rx_s,ry_s,rz_s,vx_s,vy_s,vz_s = spacecurve_func(func,'arcl',arcl)

        # determine the speed as function of arc length, should be approx. 1
        sp_s = spacecurve_sped(ndim,vx_s,vy_s,vz_s)
    
    # find unit tangent vector as a function of time
    if view_utan:
        utx = vx_t/sp_t
        uty = vy_t/sp_t
        if ndim == 3:
            utz = vz_t/sp_t          

    # calculate the tangential acceleration components
    if view_atan or view_anrm:
        if ndim == 2:
            const = (ax_t*vx_t+ay_t*vy_t)/(sp_t*sp_t)
        elif ndim == 3:
            const = (ax_t*vx_t+ay_t*vy_t+az_t*vz_t)/(sp_t*sp_t)
        atx_t = vx_t*const
        aty_t = vy_t*const
        if ndim == 3:
            atz_t = vz_t*const

    # calculate the normal acceleration components
    if view_anrm:
        anx_t = ax_t - atx_t
        any_t = ay_t - aty_t
        if ndim == 3:
            anz_t = az_t - atz_t
            
    # calculate and print line integral of speed for distance travelled
    if view_area:
        area_t = np.trapz(sp_t, t)
        print('distanced travelled from integral of |v(t)| over time span: ',area_t)
        if para_arcl:
            area_s = np.trapz(sp_s, arcl)
            print('distance travelled from integral of |v(s)| over arc length span: ',area_s)
        
        
    # setup the basic plotting window and axis
    fig = plt.figure(figsize=(14, 8))
    if view_sped:
        if ndim == 2:
            ax1 = fig.add_subplot(121)
        elif ndim == 3:
            ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122)
    else:
        if ndim == 2:
            ax1 = fig.add_subplot(111)
        elif ndim == 3:
            ax1 = fig.add_subplot(111, projection='3d')        

    # set axis 1 labels
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    if ndim == 3:
        ax1.set_zlabel('z')
        
    # plot 1: space curve
    if view_pos:
        if ndim == 2:
            if not para_arcl:
                ax1.plot(rx_t,ry_t,'k-o', label=r'$\vec{r}(t)$')
            if para_arcl:
                ax1.plot(rx_s,ry_s,'k-o', label=r'$\vec{r}(s)$')
        elif ndim == 3:
            if not para_arcl:
                ax1.plot(rx_t,ry_t,rz_t,'k-o', label=r'$\vec{r}(t)$')
            if para_arcl:
                ax1.plot(rx_s,ry_s,rz_s,'k-o', label=r'$\vec{r}(s)$')
    if view_vel:
        if ndim == 2:
            if not para_arcl:
                ax1.plot(vx_t,vy_t,'b-o', label=r'$\vec{v}(t)$')
            if para_arcl:
                ax1.plot(vx_s,vy_s,'m-o', label=r'$\vec{v}(s)$')
        elif ndim == 3:
            if not para_arcl:
                ax1.plot(vx_t,vy_t,vz_t,'b-o', label=r'$\vec{v}(t)$')
            if para_arcl:
                ax1.plot(vx_s,vy_s,vz_s,'m-o', label=r'$\vec{v}(s)$')
    if view_acc:
        if ndim == 2:
            ax1.plot(ax_t,ay_t,'r-o', label=r'$\vec{a}(t)$')
        elif ndim == 3:
            ax1.plot(ax_t,ay_t,az_t,'r-o', label=r'$\vec{a}(t)$')
    if view_utan:
        if ndim == 2:
            ax1.quiver(rx_t,ry_t,utx,uty, angles='xy', scale_units='xy', scale=2., color='green', pivot='tail', label=r'$\vec{\tau}(t)$')
        elif ndim == 3:
            ax1.quiver(rx_t,ry_t,rz_t,utx,uty,utz, color='green', pivot='tail', label=r'$\vec{\tau}(t)$')
    if view_atan:
        if ndim == 2:
            ax1.quiver(rx_t,ry_t,atx_t,aty_t, angles='xy', scale_units='xy', scale=1., color='orange', pivot='tail', label=r'$\vec{a}_{tan}(t)$')
        elif ndim == 3:
            ax1.quiver(rx_t,ry_t,rz_t,atx_t,aty_t,atz_t, color='orange', pivot='tail', label=r'$\vec{a}_{tan}(t)$')      
    if view_anrm:
        if ndim == 2:
            ax1.quiver(rx_t,ry_t,anx_t,any_t, angles='xy', scale_units='xy', scale=2., color='purple', pivot='tail', label=r'$\vec{a}_{norm}(t)$')
        elif ndim == 3:
            ax1.quiver(rx_t,ry_t,rz_t,anx_t,any_t,anz_t, color='purple', pivot='tail', label=r'$\vec{a}_{norm}(t)$')  
    ax1.legend()

    # plot 2: speed vs independent variable
    if view_sped:
        ax2.plot(t,sp_t,'b-o', label='|v(t)|')
        ax2.set_xlabel('t')
        ax2.set_ylabel('speed, |v|')
        if view_area:
            ax2.fill_between(t, 0, sp_t, alpha=0.8, facecolor='white', edgecolor='blue', hatch='/', label=r'$\int{|v(t)|\,dt}$') 
        if para_arcl:
            ax2.plot(arcl,sp_s,'m-o', label=r'$|v(s)|$')
            ax2.set_xlabel('independent variable, t or s')
            if view_area:
                ax2.fill_between(arcl, 0, sp_s, alpha=0.8, facecolor='white', edgecolor='magenta', hatch='\\', label=r'$\int{|v(s)|\,ds}$')
        ax2.legend()



def mc3_example_3_1(option):
	nxy = 101
	x = np.linspace(0.,2.,nxy)
	y = 2.*x
	X,Y = np.meshgrid(x, y)
	Z = mc3_example_3_1_func(X, Y)

	verts = []
	verts.append([x[-1],y[0],0.])
	for i in range(nxy):
		verts.append([x[i],y[i],0.])
	verts.append([x[-1],y[0],0.])

	if option != 'default':
		verts1 = []
		verts2 = []
		verts3 = []
		
		if option == 'x':
			tmp = np.int(nxy/2)
			verts1.append([x[tmp],y[0],0.])
			for i in range(tmp):
				verts1.append([x[tmp],y[i],mc3_example_3_1_func(x[tmp],y[i])])
			verts1.append([x[tmp],y[tmp],0.])

			tmp = np.int(nxy/4)
			verts2.append([x[tmp],y[0],0.])
			for i in range(tmp):
				verts2.append([x[tmp],y[i],mc3_example_3_1_func(x[tmp],y[i])])
			verts2.append([x[tmp],y[tmp],0.])

			tmp = np.int(nxy/1.2)
			verts3.append([x[tmp],y[0],0.])
			for i in range(tmp):
				verts3.append([x[tmp],y[i],mc3_example_3_1_func(x[tmp],y[i])])
			verts3.append([x[tmp],y[tmp],0.])	

		if option == 'y':
			tmp = np.int(nxy/2)
			verts1.append([x[tmp],y[tmp],0.])
			for i in range(nxy-tmp):
				verts1.append([x[tmp+i],y[tmp],mc3_example_3_1_func(x[tmp+i],y[tmp])])
			verts1.append([x[-1],y[tmp],0.])

			tmp = np.int(nxy/4)
			verts2.append([x[tmp],y[tmp],0.])
			for i in range(nxy-tmp):
				verts2.append([x[tmp+i],y[tmp],mc3_example_3_1_func(x[tmp+i],y[tmp])])
			verts2.append([x[-1],y[tmp],0.])

			tmp = np.int(nxy/1.2)
			verts3.append([x[tmp],y[tmp],0.])
			for i in range(nxy-tmp):
				verts3.append([x[tmp+i],y[tmp],mc3_example_3_1_func(x[tmp+i],y[tmp])])
			verts3.append([x[-1],y[tmp],0.])
		
	fig = plt.figure(figsize=(16, 8))
	if option == 'default':
		ax1 = fig.add_subplot(121, projection='3d')
		ax1.set_xlabel('x')
		ax1.set_ylabel('y')
		ax1.set_zlabel('f(x,y)')
		ax1.set_zlim([0,4])
		ax1.set_title('Surface')
		ax1.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.plasma, linewidth=0, antialiased=False)
		ax1.view_init(30, 325)
		ax2 = fig.add_subplot(122)
		ax2.set_xlabel('x')
		ax2.set_ylabel('y')
		ax2.set_title('Integration region')
		ax2.plot([x[0],x[-1]],[y[0],y[0]], 'k--o')
		ax2.plot([x[-1],x[-1]],[y[0],y[-1]], 'k--o')
		ax2.plot([x[-1],x[0]],[y[-1],y[0]], 'k--o')
		r = Poly3DCollection([verts], linewidth=1, alpha=0.5)
		r.set_facecolor((0.2, 0.2, 0.2, 0.5))
		ax1.add_collection3d(r)
	else:
		ax1 = fig.add_subplot(111, projection='3d')
		ax1.set_xlabel('x')
		ax1.set_ylabel('y')
		ax1.set_zlabel('f(x,y)')
		ax1.set_zlim([0,4])
		#ax1.plot_wireframe(X, Y, 0*Z, rstride=nxy, cstride=nxy, color='k', linewidth=1.0, antialiased=True)
		ax1.plot_wireframe(X, Y, Z, rstride=1, cstride=1, color='black', linewidth=0.4)
		face1 = Poly3DCollection([verts1], linewidth=1, alpha=0.5)
		face2 = Poly3DCollection([verts2], linewidth=1, alpha=0.5)
		face3 = Poly3DCollection([verts3], linewidth=1, alpha=0.5)
		face1.set_facecolor((0, 0, 1, 0.5))
		face2.set_facecolor((0, 0, 1, 0.5))
		face3.set_facecolor((0, 0, 1, 0.5))
		ax1.add_collection3d(face1)
		ax1.add_collection3d(face2)
		ax1.add_collection3d(face3)
		ax1.view_init(30, 315)
		r = Poly3DCollection([verts], linewidth=1, alpha=0.5)
		r.set_facecolor((0.2, 0.2, 0.2, 0.5))
		ax1.add_collection3d(r)



def mc3_example_3_2(option):
	nxy = 101
	y = np.linspace(0.,4.,nxy)
	x = np.sqrt(y)
	X,Y = np.meshgrid(x, y)
	Z = mc3_example_3_2_func(X, Y)

	intreg_x = []
	intreg_y = []
	for i in range(nxy):
		intreg_x.append(x[i])
		intreg_y.append(y[i])
	intreg_x.append(x[-1])
	intreg_y.append(y[0])
	intreg_x.append(x[0])
	intreg_y.append(y[0])

	if option != 'default':
		verts1 = []
		verts2 = []
		verts3 = []
		
		if option == 'x':
			tmp = np.int(nxy/2)
			verts1.append([x[tmp],y[0],0.])
			for i in range(tmp):
				verts1.append([x[tmp],y[i],mc3_example_3_2_func(x[tmp],y[i])])
			verts1.append([x[tmp],y[tmp],0.])

			tmp = np.int(nxy/4)
			verts2.append([x[tmp],y[0],0.])
			for i in range(tmp):
				verts2.append([x[tmp],y[i],mc3_example_3_2_func(x[tmp],y[i])])
			verts2.append([x[tmp],y[tmp],0.])

			tmp = np.int(nxy/1.2)
			verts3.append([x[tmp],y[0],0.])
			for i in range(tmp):
				verts3.append([x[tmp],y[i],mc3_example_3_2_func(x[tmp],y[i])])
			verts3.append([x[tmp],y[tmp],0.])	

		if option == 'y':
			tmp = np.int(nxy/2)
			verts1.append([x[tmp],y[tmp],0.])
			for i in range(nxy-tmp):
				verts1.append([x[tmp+i],y[tmp],mc3_example_3_2_func(x[tmp+i],y[tmp])])
			verts1.append([x[-1],y[tmp],0.])

			tmp = np.int(nxy/4)
			verts2.append([x[tmp],y[tmp],0.])
			for i in range(nxy-tmp):
				verts2.append([x[tmp+i],y[tmp],mc3_example_3_2_func(x[tmp+i],y[tmp])])
			verts2.append([x[-1],y[tmp],0.])

			tmp = np.int(nxy/1.2)
			verts3.append([x[tmp],y[tmp],0.])
			for i in range(nxy-tmp):
				verts3.append([x[tmp+i],y[tmp],mc3_example_3_2_func(x[tmp+i],y[tmp])])
			verts3.append([x[-1],y[tmp],0.])
		
	fig = plt.figure(figsize=(16, 8))
	if option == 'default':
		ax1 = fig.add_subplot(121, projection='3d')
		ax1.set_xlabel('x')
		ax1.set_ylabel('y')
		ax1.set_zlabel('f(x,y)')
		#ax1.set_zlim([0,4])
		ax1.set_title('Surface')
		ax1.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.plasma, linewidth=0, antialiased=False)
		ax1.view_init(30, 325)
		ax2 = fig.add_subplot(122)
		ax2.set_xlabel('x')
		ax2.set_ylabel('y')
		ax2.set_title('Integration region')
		ax2.plot(intreg_x,intreg_y, 'k--')
		#ax2.plot([x[0],[-1]],[y[0],y[0]], 'k--o')
		#ax2.plot([x[-1],x[-1]],[y[0],y[-1]], 'k--o')
		#ax2.plot([x[-1],x[0]],[y[-1],y[0]], 'k--o')
		#r = Poly3DCollection([verts], linewidth=1, alpha=0.5)
		#r.set_facecolor((0.2, 0.2, 0.2, 0.5))
		#ax1.add_collection3d(r)
	else:
		ax1 = fig.add_subplot(111, projection='3d')
		ax1.set_xlabel('x')
		ax1.set_ylabel('y')
		ax1.set_zlabel('f(x,y)')
		#ax1.set_zlim([0,4])
		#ax1.plot_wireframe(X, Y, 0*Z, rstride=nxy, cstride=nxy, color='k', linewidth=1.0, antialiased=True)
		ax1.plot_wireframe(X, Y, Z, rstride=1, cstride=1, color='black', linewidth=0.4)
		face1 = Poly3DCollection([verts1], linewidth=1, alpha=0.5)
		face2 = Poly3DCollection([verts2], linewidth=1, alpha=0.5)
		face3 = Poly3DCollection([verts3], linewidth=1, alpha=0.5)
		face1.set_facecolor((0, 0, 1, 0.5))
		face2.set_facecolor((0, 0, 1, 0.5))
		face3.set_facecolor((0, 0, 1, 0.5))
		ax1.add_collection3d(face1)
		ax1.add_collection3d(face2)
		ax1.add_collection3d(face3)
		ax1.view_init(30, 315)
		r = Poly3DCollection([verts], linewidth=1, alpha=0.5)
		r.set_facecolor((0.2, 0.2, 0.2, 0.5))
		ax1.add_collection3d(r)