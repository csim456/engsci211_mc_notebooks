import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
from sympy.utilities.lambdify import lambdify
# from collections import namedtuple
# from easing import easing
# from matplotlib.animation import FuncAnimation


class Vector2D(object):
    def __init__(self, x, y, u, v, label, color):
        self.x = x
        self.y = y
        self.u = u
        self.v = v
        self.length = np.sqrt((u - x) * (u - x) + (v - y) * (v - y))
        self.label = label
        self.color = color

    def axes_limits(self):
        min_x_tmp = np.array([np.min(self.x), np.min(self.u)])
        max_x_tmp = np.array([np.max(self.x), np.max(self.u)])
        min_y_tmp = np.array([np.min(self.y), np.min(self.v)])
        max_y_tmp = np.array([np.max(self.y), np.max(self.v)])
        min_x = np.min(min_x_tmp)
        max_x = np.max(max_x_tmp)
        min_y = np.min(min_y_tmp)
        max_y = np.max(max_y_tmp)
        buffer_x = (np.abs(max_x) - np.abs(min_x)) / 5.
        buffer_y = (np.abs(max_y) - np.abs(min_y)) / 5.
        ax_x = [min_x - buffer_x, max_x + buffer_x]
        ax_y = [min_y - buffer_y, max_y + buffer_y]
        return ax_x, ax_y

    def plot_vectors(self, filename=None):
        fig = plt.figure(figsize=[4, 4.], dpi=300)
        ax1 = fig.add_subplot(1, 1, 1, adjustable='box')
        for i in range(len(self.x)):
            ax1.quiver(self.x[i], self.y[i], self.u[i], self.v[i], self.length[i],
                       angles='xy', scale=1, scale_units='xy', color=self.color[i], label=self.label[i])
        ax_x, ax_y = self.axes_limits()
        ax1.set_xlim(ax_x[0], ax_x[1])
        ax1.set_ylim(ax_y[0], ax_y[1])
        ax1.set_xlabel(r'$x$')
        ax1.set_ylabel(r'$y$')
        ax1.legend()
        if filename is not None:
            plt.savefig(filename, dpi=300)
        else:
            plt.show()

    # def plot_vectors_3d(self):
    #     fig = plt.figure(figsize=[4, 4.], dpi=300)
    #     ax1 = fig.add_subplot(1, 1, 1, adjustable='box', aspect=1., projection='3d')
    #     ax1.quiver(self.x, self.y, self.z, self.u, self.v, self.w)
    #     ax1.set_xlabel('x')
    #     ax1.set_ylabel('y')
    #     ax1.set_zlabel('z')
    #     plt.show()


class ScalarFunction1Var(object):
    def __init__(self, sp_x, sp_f, np_x):
        self.sp_func = {"x": sp_x, "f": sp_f}
        self.np_func = {"f": np_lambdify(sp_x, sp_f)}
        self.np_array = {"x": np_x}
        self.np_array["f"] = self.np_func["f"](self.np_array["x"])
        self.gradient = None

    def calculate_derivatives(self):
        self.sp_func["dfdx"] = sp.diff(self.sp_func["f"])
        self.sp_func["d2fdx2"] = sp.diff(self.sp_func["dfdx"])
        self.np_func["dfdx"] = np_lambdify(self.sp_func["x"], self.sp_func["dfdx"])
        self.np_func["d2fdx2"] = np_lambdify(self.sp_func["x"], self.sp_func["d2fdx2"])
        self.np_array["dfdx"] = self.np_func["dfdx"](self.np_array["x"])
        self.np_array["d2fdx2"] = self.np_func["d2fdx2"](self.np_array["x"])

    def plot_curve(self, filename=None):
        fig = plt.figure(figsize=[5., 5.], dpi=300)
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.plot(self.np_array["x"], self.np_array["f"])
        ax1.set_xlabel(r'$x$')
        ax1.set_ylabel(r'$f(x)$')
        ax1.axhline(color='grey', alpha=0.8)
        ax1.axvline(color='grey', alpha=0.8)
        if filename is not None:
            plt.savefig(filename, dpi=300)
        else:
            plt.show()

    def plot_derivatives(self, filename=None):
        fig = plt.figure(figsize=[4., 3.], dpi=300)
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.plot(self.np_array["x"], self.np_array["f"], 'k-', label=r'$f\,(x)$')
        ax1.plot(self.np_array["x"], self.np_array["dfdx"], 'b--', label=r'$f^{\;\prime}\,(x)$')
        ax1.plot(self.np_array["x"], self.np_array["d2fdx2"], 'g-.', label=r'$f^{\;\prime\prime}\,(x)$')
        ax1.set_xlabel(r'$x$')
        ax1.set_ylabel(r'$f(x)$')
        ax1.axhline(color='grey', alpha=0.8)
        ax1.axvline(color='grey', alpha=0.8)
        ax1.legend()
        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename, dpi=300)
        else:
            plt.show()


class ScalarFunction2Var(object):
    def __init__(self, sp_x, sp_y, sp_f, np_x_1d, np_y_1d, cmap=None):
        self.sp_func = {"x": sp_x, "y": sp_y, "f": sp_f}
        self.np_func = {"f": lambdify([sp_x, sp_y], sp_f, 'numpy')}
        np_x2d, np_y2d = np.meshgrid(np_x_1d, np_y_1d)
        self.np_array = {"x_1d": np_x_1d, "y_1d": np_y_1d, "x_2d": np_x2d, "y_2d": np_y2d}
        self.np_array["f_2d"] = self.np_func["f"](self.np_array["x_2d"], self.np_array["y_2d"])
        if cmap is None:
            self.cmap = cm.inferno
        else:
            self.cmap = cmap
        self.gradient = None

    def calculate_partial_derivatives(self):
        self.sp_func["dfdx"] = sp.diff(self.sp_func["f"], self.sp_func["x"])
        self.sp_func["dfdy"] = sp.diff(self.sp_func["f"], self.sp_func["y"])
        self.sp_func["d2fdx2"] = sp.diff(self.sp_func["f"], self.sp_func["x"], self.sp_func["x"])
        self.sp_func["d2fdy2"] = sp.diff(self.sp_func["f"], self.sp_func["y"], self.sp_func["y"])
        self.sp_func["d2fdxy"] = sp.diff(self.sp_func["f"], self.sp_func["x"], self.sp_func["y"])
        self.sp_func["d2fdyx"] = sp.diff(self.sp_func["f"], self.sp_func["y"], self.sp_func["x"])

        # self.np_func["dfdx_alt"] = np_lambdify([self.sp_func["x"], self.sp_func["y"]], self.sp_func["dfdx"])
        self.np_func["dfdx"] = lambdify([self.sp_func["x"], self.sp_func["y"]], self.sp_func["dfdx"], 'numpy')
        self.np_func["dfdy"] = lambdify([self.sp_func["x"], self.sp_func["y"]], self.sp_func["dfdy"], 'numpy')
        self.np_func["d2fdx2"] = np_lambdify_v2([self.sp_func["x"], self.sp_func["y"]], self.sp_func["d2fdx2"])
        # self.np_func["d2fdy2"] = lambdify([self.sp_func["x"], self.sp_func["y"]], self.sp_func["d2fdy2"], 'numpy')
        self.np_func["d2fdy2"] = np_lambdify_v2([self.sp_func["x"], self.sp_func["y"]], self.sp_func["d2fdy2"])
        self.np_func["d2fdxy"] = lambdify([self.sp_func["x"], self.sp_func["y"]], self.sp_func["d2fdxy"], 'numpy')
        self.np_func["d2fdyx"] = lambdify([self.sp_func["x"], self.sp_func["y"]], self.sp_func["d2fdyx"], 'numpy')

    def calculate_gradient(self, spacing=5):
        self.gradient = VectorFunction2Var(self.np_array["x_1d"][::spacing], self.np_array["y_1d"][::spacing],
                                           self.np_func["dfdx"], self.np_func["dfdy"])

    def plot_surface(self, colorbar=True, angle=None, filename=None):
        fig = plt.figure(figsize=[5, 5.], dpi=300)
        ax1 = fig.add_subplot(1, 1, 1, adjustable='box', aspect=1., projection='3d')
        surface = ax1.plot_surface(self.np_array["x_2d"], self.np_array["y_2d"], self.np_array["f_2d"], cmap=self.cmap)
        ax1.set_xlabel(r'$x$')
        ax1.set_ylabel(r'$y$')
        ax1.set_zlabel(r'$f(x,y)$')
        if angle is not None:
            ax1.view_init(angle['elev'], angle['azim'])
        if colorbar:
            fig.colorbar(surface, ax=ax1, shrink=0.65)
        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename, dpi=300)
        else:
            plt.show()

    def plot_contour(self, filename=None):
        fig = plt.figure(figsize=[5., 5.], dpi=300)
        ax1 = fig.add_subplot(1, 1, 1, adjustable='box', aspect=1.)
        contour = ax1.contour(self.np_array["x_2d"], self.np_array["y_2d"], self.np_array["f_2d"],
                              cmap=self.cmap)
        ax1.clabel(contour, fmt=r'%.0f')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename, dpi=300)
        else:
            plt.show()

    def plot_contour_and_partial_derivatives(self, point, filename=None):
        fig = plt.figure(figsize=[5., 5.], dpi=300)
        gs = gridspec.GridSpec(2, 1)
        # ax1 = fig.add_subplot(gs[:, 0])
        ax2 = fig.add_subplot(gs[0, 0])
        ax3 = fig.add_subplot(gs[1, 0])

        fixed_x_x = np.repeat(point[0], len(self.np_array["y_1d"]))
        fixed_x_y = self.np_array["y_1d"]
        fixed_y_x = self.np_array["x_1d"]
        fixed_y_y = np.repeat(point[1], len(self.np_array["x_1d"]))

        # ax1.contour(self.np_array["x_2d"], self.np_array["y_2d"], self.np_array["f_2d"], cmap=self.cmap)
        # ax1.plot(fixed_x_x, fixed_x_y, 'k--')
        # ax1.plot(fixed_y_x, fixed_y_y, 'k--')
        # ax1.plot(point[0], point[1], 'ko')
        # ax1.set_xlabel('x')
        # ax1.set_ylabel('y')

        ax2.set_xlabel('x')
        ax2.set_ylabel('f(x,%d)' % point[1])
        ax2.plot(point[0], self.np_func["f"](point[0], point[1]), 'ko')
        ax2.plot(point[0], self.np_func["dfdx"](point[0], point[1]), 'bo')
        ax2.plot(point[0], self.np_func["d2fdx2"](point[0], point[1]), 'go')
        ax2.plot(fixed_y_x, self.np_func["f"](fixed_y_x, fixed_y_y), 'k-', label=r'$f$')
        ax2.plot(fixed_y_x, self.np_func["dfdx"](fixed_y_x, fixed_y_y), 'b--', label=r'$\frac{df}{dx}$')
        ax2.plot(fixed_y_x, self.np_func["d2fdx2"](fixed_y_x, fixed_y_y), 'g-.', label=r'$\frac{d^2f}{dx^2}$')
        ax2.axhline(color='grey', alpha=0.8)
        ax2.axvline(color='grey', alpha=0.8)
        ax2.legend(fontsize='small', bbox_to_anchor=(1.2, 1.0), markerscale=0.5, handletextpad=0.5)

        ax3.set_xlabel('y')
        ax3.set_ylabel('f(%d,y)' % point[0])
        ax3.plot(point[1], self.np_func["f"](point[0], point[1]), 'ko')
        ax3.plot(point[1], self.np_func["dfdy"](point[0], point[1]), 'bo')
        ax3.plot(point[1], self.np_func["d2fdy2"](point[0], point[1]), 'go')
        ax3.plot(fixed_x_y, self.np_func["f"](fixed_x_x, fixed_x_y), 'k-', label=r'$f$')
        ax3.plot(fixed_x_y, self.np_func["dfdy"](fixed_x_x, fixed_x_y), 'b--', label=r'$\frac{df}{dy}$')
        ax3.plot(fixed_x_y, self.np_func["d2fdy2"](fixed_x_x, fixed_x_y), 'g-.', label=r'$\frac{d^2f}{dy^2}$')
        ax3.axhline(color='grey', alpha=0.8)
        ax3.axvline(color='grey', alpha=0.8)
        ax3.legend(fontsize='small', bbox_to_anchor=(1.2, 1.0), markerscale=0.5, handletextpad=0.5)

        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename, dpi=300)
        else:
            plt.show()

    def plot_contour_gradient(self, scale=1., width=0.005, filename=None):
        fig = plt.figure(figsize=[5., 5.], dpi=300)
        ax1 = fig.add_subplot(1, 1, 1, adjustable='box', aspect=1.)
        contour = ax1.contour(self.np_array["x_2d"], self.np_array["y_2d"], self.np_array["f_2d"], cmap=self.cmap)
        ax1.quiver(self.gradient.x_2d, self.gradient.y_2d,
                   self.gradient.vector_x, self.gradient.vector_y,
                   angles='xy', scale_units='xy', scale=scale, width=width, color='black', pivot='mid')
        ax1.clabel(contour, fmt=r'%.0f')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename, dpi=300)
        else:
            plt.show()

    def plot_directional_derivative(self, point=[0., 0.], vector=[1., 1.], scale=5., width=0.005, filename=None):
        fig = plt.figure(figsize=[5., 5.], dpi=300)
        ax1 = fig.add_subplot(1, 1, 1, adjustable='box', aspect=1.)
        contour = ax1.contour(self.np_array["x_2d"], self.np_array["y_2d"], self.np_array["f_2d"], cmap=self.cmap)
        vector = np.array(vector)/np.linalg.norm(vector)
        ax1.quiver(point[0], point[1],
                   self.gradient.function_x(point[0], point[1]), self.gradient.function_y(point[0], point[1]),
                   angles='xy', scale_units='xy', scale=scale, width=width, color='black', pivot='tail')
        ax1.quiver(point[0], point[1],
                   vector[0], vector[1],
                   angles='xy', scale_units='xy', scale=scale, width=width, color='green', pivot='tail')
        ax1.clabel(contour, fmt=r'%.0f')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename, dpi=300)
        else:
            plt.show()


class VectorFunction2Var(object):
    def __init__(self, x, y, func_x, func_y):
        self.x_1d = x
        self.y_1d = y
        self.function_x = func_x
        self.function_y = func_y
        self.x_2d, self.y_2d = np.meshgrid(x, y)
        self.vector_x = func_x(self.x_2d, self.y_2d)
        self.vector_y = func_y(self.x_2d, self.y_2d)

    def plot_vector_field(self, scale=1., width=0.005, label=None, filename=None):
        fig = plt.figure(figsize=[5., 5.], dpi=300)
        ax1 = fig.add_subplot(1, 1, 1, adjustable='box', aspect=1.)
        ax1.quiver(self.x_2d, self.y_2d, self.vector_x, self.vector_y, angles='xy', scale_units='xy', scale=scale,
                   width=width, color='black', pivot='mid')
        ax1.set_xlabel(r'$x$')
        ax1.set_ylabel(r'$y$')
        if label is not None:
            ax1.set_title(label)
        if filename is not None:
            plt.savefig(filename, dpi=300)
        else:
            plt.show()


class VectorFunction2VarSP(object):
    def __init__(self, sp_x, sp_y, sp_fx, sp_fy, np_x_1d, np_y_1d):
        self.sp_func = {'x': sp_x, 'y': sp_y, 'fx': sp_fx, 'fy': sp_fy}
        self.np_func = {'fx': np_lambdify_v2([self.sp_func['x'], self.sp_func['y']], self.sp_func['fx']),
                        'fy': np_lambdify_v2([self.sp_func['x'], self.sp_func['y']], self.sp_func['fy'])}
        np_x_2d, np_y_2d = np.meshgrid(np_x_1d, np_y_1d)
        self.np_array = {"x_1d": np_x_1d, "y_1d": np_y_1d, "x_2d": np_x_2d, "y_2d": np_y_2d}
        self.np_array['fx_2d'] = self.np_func['fx'](self.np_array["x_2d"], self.np_array["y_2d"])
        self.np_array['fy_2d'] = self.np_func['fy'](self.np_array["x_2d"], self.np_array["y_2d"])

    def calculate_dots(self, dots_x_1d, dots_y_1d):
        dots_x_2d, dots_y_2d = np.meshgrid(dots_x_1d, dots_y_1d)
        self.np_array['dots_x_1d'] = dots_x_1d
        self.np_array['dots_y_1d'] = dots_y_1d
        self.np_array['dots_x_2d'] = dots_x_2d
        self.np_array['dots_y_2d'] = dots_y_2d

    def plot_vector_field(self, dots=None, spacing=10, scale=1., width=0.005, label=None, filename=None):
        fig = plt.figure(figsize=[5., 5.], dpi=300)
        ax1 = fig.add_subplot(1, 1, 1, adjustable='box', aspect=1.)
        if dots is not None:
            ax1.plot(dots['x_2d'], dots['y_2d'], 'bo', ms=2.)
        ax1.quiver(self.np_array['x_2d'][::spacing], self.np_array['y_2d'][::spacing],
                   self.np_array['fx_2d'][::spacing], self.np_array['fy_2d'][::spacing],
                   angles='xy', scale_units='xy', scale=scale, width=width, color='black', pivot='mid')
        ax1.set_xlabel(r'$x$')
        ax1.set_ylabel(r'$y$')
        ax1.axhline(color='grey', alpha=0.8)
        ax1.axvline(color='grey', alpha=0.8)
        ax1.set_xlim(np.min(self.np_array['x_1d']), np.max(self.np_array['x_1d']))
        ax1.set_ylim(np.min(self.np_array['y_1d']), np.max(self.np_array['y_1d']))
        plt.tight_layout()
        if label is not None:
            ax1.set_title(label)
        if filename is not None:
            plt.savefig(filename, dpi=300)
        else:
            plt.show()

    def animation_vector_field(self, frames=20, spacing=20, step=0.1, scale=1., width=0.005,
                                    label=None, filename=None):
        mywriter = animation.FFMpegWriter(fps=30)
        plt.rcParams['animation.ffmpeg_path'] = r'C:\\Users\\colinsimpson\\Documents\\ffmpeg\\bin\\ffmpeg.exe'
        fig = plt.figure(figsize=[5., 5.], dpi=300)
        ax1 = fig.add_subplot(1, 1, 1, adjustable='box', aspect=1.)
        plotdots, = ax1.plot([], [], 'bo', ms=3.)
        def init():
            ax1.quiver(self.np_array['x_2d'][::spacing], self.np_array['y_2d'][::spacing],
                       self.np_array['fx_2d'][::spacing], self.np_array['fy_2d'][::spacing],
                       angles='xy', scale_units='xy', scale=scale, width=width, color='black', pivot='mid')
            ax1.set_xlabel(r'$x$')
            ax1.set_ylabel(r'$y$')
            ax1.axhline(color='grey', alpha=0.8)
            ax1.axvline(color='grey', alpha=0.8)
            ax1.set_xlim(np.min(self.np_array['x_1d']), np.max(self.np_array['x_1d']))
            ax1.set_ylim(np.min(self.np_array['y_1d']), np.max(self.np_array['y_1d']))
            if label is not None:
                ax1.set_title(label)
            return plotdots,
        def update(frame):
            self.np_array['dots_fx_2d'] = self.np_func['fx'](self.np_array["dots_x_2d"], self.np_array["dots_y_2d"])
            self.np_array['dots_fy_2d'] = self.np_func['fy'](self.np_array["dots_x_2d"], self.np_array["dots_y_2d"])
            self.np_array['dots_x_2d'] += step * self.np_array['dots_fx_2d']
            self.np_array['dots_y_2d'] += step * self.np_array['dots_fy_2d']
            plotdots.set_data(self.np_array['dots_x_2d'], self.np_array['dots_y_2d'])
            return plotdots,
        animate = animation.FuncAnimation(fig, func=update, frames=frames, init_func=init, blit=True)
        if filename is not None:
            # animate.save(filename)
            animate.save(filename, writer=mywriter)
        else:
            plt.show()


class Spacecurve2D(object):
    def __init__(self, time, step, sp_t, sp_x, sp_y):
        self.sp_func = {"t": sp_t, "x": sp_x, "y": sp_y}
        self.np_func = {"x": np_lambdify_v2(sp_t, sp_x), "y": np_lambdify_v2(sp_t, sp_y)}
        self.np_array = {"t": np.arange(time[0], time[1] + step, step)}
        self.np_array["x"] = self.np_func["x"](self.np_array["t"])
        self.np_array["y"] = self.np_func["y"](self.np_array["t"])

    def calculate_velocity(self):
        self.sp_func["vx"] = sp.diff(self.sp_func["x"])
        self.sp_func["vy"] = sp.diff(self.sp_func["y"])
        self.np_func["vx"] = lambdify(self.sp_func["t"], self.sp_func["vx"], 'numpy')
        self.np_func["vy"] = lambdify(self.sp_func["t"], self.sp_func["vy"], 'numpy')
        self.np_array["vx"] = self.np_func["vx"](self.np_array["t"])
        self.np_array["vy"] = self.np_func["vy"](self.np_array["t"])

    def calculate_speed(self):
        self.sp_func["speed"] = sp.sqrt(self.sp_func["x"] * self.sp_func["x"] +
                                        self.sp_func["y"] * self.sp_func["y"])
        self.np_func["speed"] = lambdify(self.sp_func["t"], self.sp_func["speed"], 'numpy')
        self.np_array["speed"] = self.np_func["speed"](self.np_array["t"])

    def calculate_arc_length(self):
        self.sp_func["arc"] = sp.integrate(self.sp_func["speed"], self.sp_func["t"])
        self.np_func["arc"] = lambdify(self.sp_func["t"], self.sp_func["arc"], 'numpy')
        self.np_array["arc"] = self.np_func["arc"](self.np_array["t"])

    def calculate_tau(self):
        self.sp_func["tau_x"] = self.sp_func["vx"] / self.sp_func["speed"]
        self.sp_func["tau_y"] = self.sp_func["vy"] / self.sp_func["speed"]
        self.np_func["tau_x"] = lambdify(self.sp_func["t"], self.sp_func["tau_x"], 'numpy')
        self.np_func["tau_y"] = lambdify(self.sp_func["t"], self.sp_func["tau_y"], 'numpy')
        self.np_array["tau_x"] = self.np_func["tau_x"](self.np_array["t"])
        self.np_array["tau_y"] = self.np_func["tau_y"](self.np_array["t"])

    def calculate_acceleration(self):
        self.sp_func["ax"] = sp.diff(self.sp_func["vx"])
        self.sp_func["ay"] = sp.diff(self.sp_func["vy"])
        self.np_func["ax"] = lambdify(self.sp_func["t"], self.sp_func["ax"], 'numpy')
        self.np_func["ay"] = lambdify(self.sp_func["t"], self.sp_func["ay"], 'numpy')
        self.np_array["ax"] = self.np_func["ax"](self.np_array["t"])
        self.np_array["ay"] = self.np_func["ay"](self.np_array["t"])

    def calculate_acceleration_tangent(self):
        atan_magnitude = self.sp_func["ax"] * self.sp_func["tau_x"] + \
                         self.sp_func["ay"] * self.sp_func["tau_y"]
        self.sp_func["atan_x"] = atan_magnitude * self.sp_func["tau_x"]
        self.sp_func["atan_y"] = atan_magnitude * self.sp_func["tau_y"]
        # self.sympy_function["atan_mag"] = atan_magnitude
        self.np_func["atan_x"] = lambdify(self.sp_func["t"], self.sp_func["atan_x"], 'numpy')
        self.np_func["atan_y"] = lambdify(self.sp_func["t"], self.sp_func["atan_y"], 'numpy')
        self.np_array["atan_x"] = self.np_func["atan_x"](self.np_array["t"])
        self.np_array["atan_y"] = self.np_func["atan_y"](self.np_array["t"])

    def calculate_acceleration_normal(self):
        self.sp_func["anorm_x"] = self.sp_func["ax"] - self.sp_func["atan_x"]
        self.sp_func["anorm_y"] = self.sp_func["ay"] - self.sp_func["atan_y"]
        self.np_func["anorm_x"] = lambdify(self.sp_func["t"], self.sp_func["anorm_x"], 'numpy')
        self.np_func["anorm_y"] = lambdify(self.sp_func["t"], self.sp_func["anorm_y"], 'numpy')
        self.np_array["anorm_x"] = self.np_func["anorm_x"](self.np_array["t"])
        self.np_array["anorm_y"] = self.np_func["anorm_y"](self.np_array["t"])

    def plot_parametric_position(self, figsize=[5., 5.], filename=None):
        fig = plt.figure(figsize=figsize, dpi=300)
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        ax1.plot(self.np_array['t'], self.np_array['x'], 'b-o')
        ax2.plot(self.np_array['t'], self.np_array['y'], 'b-o')
        ax1.set_ylabel(r'$x(t)$')
        ax2.set_xlabel(r'$t$')
        ax2.set_ylabel(r'$y(t)$')
        ax1.axvline(color='grey')
        ax1.axhline(color='grey')
        ax2.axvline(color='grey')
        ax2.axhline(color='grey')
        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename, dpi=300)
        else:
            plt.show()

    def plot_position(self, figsize=[5., 5.], filename=None):
        fig = plt.figure(figsize=figsize, dpi=300)
        ax1 = fig.add_subplot(111, adjustable='box')
        ax1.plot(self.np_array['x'], self.np_array['y'], 'b-o', label=r'$\vec{r} \left( t \right)$')
        ax1.set_xlabel(r'$x(t)$')
        ax1.set_ylabel(r'$y(t)$')
        ax1.axvline(color='grey')
        ax1.axhline(color='grey')
        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename, dpi=300)
        else:
            plt.show()

    def plot_velocity_vector(self, scale=1., width=0.005, figsize=[5., 5.], filename=None):
        fig = plt.figure(figsize=figsize, dpi=300)
        ax1 = fig.add_subplot(111, adjustable='box')
        ax1.plot(self.np_array['x'], self.np_array['y'], 'bo', label=r'$\vec{r} \left( t \right)$')
        ax1.quiver(
            self.np_array['x'], self.np_array['y'],
            self.np_array['vx'], self.np_array['vy'],
            angles='xy', scale_units='xy', scale=scale, width=width, color='green', pivot='mid',
            label=r'$\vec{v} \left( t \right)$'
        )
        ax1.set_xlabel(r'$x(t)$')
        ax1.set_ylabel(r'$y(t)$')
        ax1.axvline(color='grey')
        ax1.axhline(color='grey')
        ax1.legend()
        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename, dpi=300)
        else:
            plt.show()

    def plot_tau_vector(self, scale=1., width=0.005, figsize=[5., 5.], filename=None):
        fig = plt.figure(figsize=figsize, dpi=300)
        ax1 = fig.add_subplot(111, adjustable='box')
        ax1.plot(self.np_array['x'], self.np_array['y'], 'bo', label=r'$\vec{r} \left( t \right)$')
        ax1.quiver(
            self.np_array['x'], self.np_array['y'],
            self.np_array['tau_x'], self.np_array['tau_y'],
            angles='xy', scale_units='xy', scale=scale, width=width, color='green', pivot='mid',
            label=r'$\vec{\tau} \left( t \right)$'
        )
        ax1.set_xlabel(r'$x(t)$')
        ax1.set_ylabel(r'$y(t)$')
        ax1.axvline(color='grey')
        ax1.axhline(color='grey')
        ax1.legend()
        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename, dpi=300)
        else:
            plt.show()

    def plot_acceleration_total(self, scale=1., width=0.005, figsize=[5., 5.], filename=None):
        fig = plt.figure(figsize=figsize, dpi=300)
        ax1 = fig.add_subplot(111, adjustable='box')
        ax1.plot(self.np_array['x'], self.np_array['y'], 'b-o', label=r'$\vec{r} \left( t \right)$')
        ax1.quiver(
            self.np_array['x'], self.np_array['y'],
            self.np_array['ax'], self.np_array['ay'],
            angles='xy', scale_units='xy', scale=scale, width=width, color='red', pivot='tail',
            label=r'$\vec{a} \left( t \right)$'
        )
        ax1.set_xlabel(r'$x(t)$')
        ax1.set_ylabel(r'$y(t)$')
        ax1.axvline(color='grey', alpha=0.5)
        ax1.axhline(color='grey', alpha=0.5)
        ax1.legend()
        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename, dpi=300)
        else:
            plt.show()

    def plot_acceleration_components(self, scale=1., width=0.005, figsize=[5., 5.], filename=None):
        fig = plt.figure(figsize=figsize, dpi=300)
        ax1 = fig.add_subplot(111, adjustable='box')
        ax1.plot(self.np_array['x'], self.np_array['y'], 'b-o', label=r'$\vec{r} \left( t \right)$')
        ax1.quiver(
            self.np_array['x'], self.np_array['y'],
            self.np_array['ax'], self.np_array['ay'],
            angles='xy', scale_units='xy', scale=scale, width=width, color='red', pivot='tail',
            label=r'$\vec{a} \left( t \right)$'
        )
        ax1.quiver(
            self.np_array['x'], self.np_array['y'],
            self.np_array['atan_x'], self.np_array['atan_y'],
            angles='xy', scale_units='xy', scale=scale, width=width, color='green', pivot='tail',
            label=r'$\vec{a}_{tan} \left( t \right)$'
        )
        ax1.quiver(
            self.np_array['x'], self.np_array['y'],
            self.np_array['anorm_x'], self.np_array['anorm_y'],
            angles='xy', scale_units='xy', scale=scale, width=width, color='orange', pivot='tail',
            label=r'$\vec{a}_{norm} \left( t \right)$'
        )
        ax1.set_xlabel(r'$x(t)$')
        ax1.set_ylabel(r'$y(t)$')
        ax1.axvline(color='grey', alpha=0.5)
        ax1.axhline(color='grey', alpha=0.5)
        ax1.legend()
        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename, dpi=300)
        else:
            plt.show()

    def plot_vfield_background(self, vfield, scale_sc=1., width_sc=0.005, scale_vf=1., width_vf=0.005,
                               spacing_vf=1, figsize=[5., 5.], filename=None):
        fig = plt.figure(figsize=figsize, dpi=300)
        ax1 = fig.add_subplot(111, adjustable='box')
        ax1.quiver(vfield.np_array['x_2d'][::spacing_vf], vfield.np_array['y_2d'][::spacing_vf],
                   vfield.np_array['fx_2d'][::spacing_vf], vfield.np_array['fy_2d'][::spacing_vf],
                   angles='xy', scale_units='xy', scale=scale_vf, width=width_vf, color='black', pivot='tail',
                   label=r'$\vec{F} \left(x,y\right)$')
        ax1.quiver(self.np_array['x'], self.np_array['y'],
                   self.np_array['vx'], self.np_array['vy'],
                   angles='xy', scale_units='xy', scale=scale_sc, width=width_sc, color='green', pivot='tail',
                   label=r'$\vec{v} \left(t\right)$')
        ax1.plot(self.np_array['x'], self.np_array['y'], 'bo', label=r'$\vec{r} \left( t \right)$')
        ax1.set_xlabel(r'$x(t)$')
        ax1.set_ylabel(r'$y(t)$')
        ax1.axvline(color='grey', alpha=0.5)
        ax1.axhline(color='grey', alpha=0.5)
        ax1.legend()
        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename, dpi=300)
        else:
            plt.show()

    def plot_vfield_local(self, vfield, scale_sc=1., width_sc=0.005, scale_vf=1., width_vf=0.005, figsize=[5., 5.],
                          filename=None):
        fig = plt.figure(figsize=figsize, dpi=300)
        ax1 = fig.add_subplot(111, adjustable='box')
        fx = vfield.np_func['fx'](self.np_array["x"], self.np_array["y"])
        fy = vfield.np_func['fy'](self.np_array["x"], self.np_array["y"])
        ax1.quiver(self.np_array['x'], self.np_array['y'], fx, fy,
                   angles='xy', scale_units='xy', scale=scale_vf, width=width_vf, color='black', pivot='tail',
                   label=r'$\vec{F} \left(t\right)$')
        ax1.quiver(self.np_array['x'], self.np_array['y'],
                   self.np_array['vx'], self.np_array['vy'],
                   angles='xy', scale_units='xy', scale=scale_sc, width=width_sc, color='green', pivot='tail',
                   label=r'$\vec{v} \left(t\right)$')
        ax1.plot(self.np_array['x'], self.np_array['y'], 'bo', label=r'$\vec{r} \left( t \right)$')
        ax1.set_xlabel(r'$x(t)$')
        ax1.set_ylabel(r'$y(t)$')
        ax1.axvline(color='grey', alpha=0.5)
        ax1.axhline(color='grey', alpha=0.5)
        ax1.legend()
        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename, dpi=300)
        else:
            plt.show()

    def plot_speed(self, figsize=[5., 5.], filename=None):
        fig = plt.figure(figsize=figsize, dpi=300)
        ax1 = fig.add_subplot(111, adjustable='box')
        ax1.plot(self.np_array['t'], self.np_array['speed'], 'b-o')
        ax1.set_xlabel(r'$t$')
        ax1.set_ylabel(r'$\Vert \vec{v} \left( t \right) \Vert$')
        ax1.axvline(color='grey')
        ax1.axhline(color='grey')
        # ax1.legend()
        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename, dpi=300)
        else:
            plt.show()

    def plot_arc_length(self, figsize=[5., 5.], filename=None):
        fig = plt.figure(figsize=figsize, dpi=300)
        ax1 = fig.add_subplot(111, adjustable='box')
        ax1.plot(self.np_array['t'], self.np_array['arc'], 'b-o')
        ax1.set_xlabel(r'$t$')
        ax1.set_ylabel(r'$s(t)$')
        ax1.axvline(color='grey')
        ax1.axhline(color='grey')
        # ax1.legend()
        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename, dpi=300)
        else:
            plt.show()

    def plot_speed_and_arc_length(self, figsize=[5., 5.], filename=None):
        fig = plt.figure(figsize=figsize, dpi=300)
        ax1 = fig.add_subplot(111, adjustable='box')
        ax1.plot(self.np_array['t'], self.np_array['speed'], 'b-o', label=r'$\Vert \vec{v} \left( t \right) \Vert$')
        ax1.plot(self.np_array['t'], self.np_array['arc'], 'g-o', label=r'$s \left( t \right)$')
        # ax1.plot([self.np_array['t'][-1], self.np_array['t'][-1]], [self.np_array['arc'][0], self.np_array['arc'][-1]],
        #          'r-', label=r'$L = s \left( t_e \right) - s \left( t_s \right)$')
        ax1.set_xlabel(r'$t$')
        ax1.set_ylabel(r'$f(t)$')
        ax1.axvline(color='grey')
        ax1.axhline(color='grey')
        ax1.legend()
        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename, dpi=300)
        else:
            plt.show()



class Spacecurve3D(object):
    def __init__(self, time, step, sp_t, sp_x, sp_y, sp_z):
        self.sp_func = {"t": sp_t, "x": sp_x, "y": sp_y, "z": sp_z}
        self.np_func = {
            "x": np_lambdify_v2(sp_t, sp_x),
            "y": np_lambdify_v2(sp_t, sp_y),
            "z": np_lambdify_v2(sp_t, sp_z)
        }
        self.np_array = {"t": np.arange(time[0], time[1] + step, step)}
        self.np_array["x"] = self.np_func["x"](self.np_array["t"])
        self.np_array["y"] = self.np_func["y"](self.np_array["t"])
        self.np_array["z"] = self.np_func["z"](self.np_array["t"])

    def plot_parametric_position(self, figsize=[5., 5.], filename=None):
        fig = plt.figure(figsize=figsize, dpi=300)
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312)
        ax3 = fig.add_subplot(313)
        ax1.plot(self.np_array['t'], self.np_array['x'], 'b-o')
        ax2.plot(self.np_array['t'], self.np_array['y'], 'b-o')
        ax3.plot(self.np_array['t'], self.np_array['z'], 'b-o')
        ax1.set_ylabel(r'$x(t)$')
        ax2.set_ylabel(r'$y(t)$')
        ax3.set_ylabel(r'$z(t)$')
        ax3.set_xlabel(r'$t$')
        ax1.axvline(color='grey')
        ax1.axhline(color='grey')
        ax2.axvline(color='grey')
        ax2.axhline(color='grey')
        ax3.axvline(color='grey')
        ax3.axhline(color='grey')
        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename, dpi=300)
        else:
            plt.show()

    def plot_position(self, figsize=[5., 5.], view=[30., 0.], filename=None):
        fig = plt.figure(figsize=figsize, dpi=300)
        ax1 = fig.add_subplot(111, projection='3d')
        ax1.plot(self.np_array['x'], self.np_array['y'], self.np_array['z'], 'b-o')
        ax1.set_xlabel(r'$x(t)$')
        ax1.set_ylabel(r'$y(t)$')
        ax1.set_ylabel(r'$z(t)$')
        ax1.axvline(color='grey')
        ax1.axhline(color='grey')
        ax1.view_init(view[0], view[1])
        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename, dpi=300)
        else:
            plt.show()


def np_lambdify(var, func):
    lamb = sp.lambdify(var, func, modules=['numpy'])
    if func.is_constant():
        return lambda t: np.full_like(t, lamb(t))
    else:
        return lambda t: lamb(np.array(t))

def np_lambdify_v2(var, func):
    lamb = sp.lambdify(var, func, modules=['numpy'])
    if not isinstance(var, list):
    # if len(var) == 1:
        if func.is_constant():
            return lambda t: np.full_like(t, lamb(t))
        else:
            return lambda t: lamb(np.array(t))
    else:
        if func.is_constant():
            return lambda x, y: np.full_like(x, lamb(x, y))
        else:
            return lambda x, y: lamb(np.array(x), np.array(y))
