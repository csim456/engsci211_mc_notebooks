from plotting_functions_mc import *
import ipywidgets as widgets
import warnings
warnings.filterwarnings('ignore')


def parameterisation(example):
    func = widgets.fixed(example)
    # func = widgets.Dropdown(options=['Example_2_1', 'Example_2_2'], value='Example_2_1', description='Example')
    ts = widgets.FloatText(value=0., description='Time Start:', disabled=False)
    te = widgets.FloatText(value=5., description='Time End:', disabled=False)
    dt = widgets.IntText(value=20, description='Evaluations:', disabled=False)
    fig_x = widgets.FloatText(value=10., description='Figure Size X:', disabled=False)
    fig_y = widgets.FloatText(value=6., description='Figure Size Y:', disabled=False)
    plot_parametric = widgets.Checkbox(value=False, description='View Parametric Equations', disabled=False)
    plot_position = widgets.Checkbox(value=True, description='View Position Vector', disabled=False)
    return widgets.VBox([
        # widgets.HBox([func]),
        widgets.HBox([ts, te, dt]),
        widgets.HBox([plot_position, plot_parametric]),
        widgets.HBox([fig_x, fig_y]),
        widgets.interactive_output(run_parameterisation, {
            'func': func, 'ts': ts, 'te': te, 'dt': dt,
            'fig_x': fig_x, 'fig_y': fig_y,
            'plot_position': plot_position, 'plot_parametric': plot_parametric})
    ])


def run_parameterisation(func, ts, te, dt, fig_x, fig_y, plot_position, plot_parametric):

    time = [ts, te]
    step = (time[1] - time[0]) / dt
    sp_t = sp.Symbol('t')

    if func == 'Example_2_1':
        sp_x = sp_t
        sp_y = sp_t
        spacecurve = Spacecurve2D(time, step, sp_t, sp_x, sp_y)
        if plot_position:
            spacecurve.plot_position(figsize=[fig_x, fig_y])
        if plot_parametric:
            spacecurve.plot_parametric_position(figsize=[fig_x, fig_y])

    if func == 'Example_2_2':
        sp_x = sp.cos(sp_t)
        sp_y = sp.sin(sp_t)
        sp_z = sp_t
        spacecurve = Spacecurve3D(time, step, sp_t, sp_x, sp_y, sp_z)
        if plot_position:
            spacecurve.plot_position(figsize=[fig_x, fig_y])
        if plot_parametric:
            spacecurve.plot_parametric_position(figsize=[fig_x, fig_y])


def velocity_and_speed(example):
    func = widgets.fixed(example)
    ts = widgets.FloatText(value=0., description='Time Start:', disabled=False)
    te = widgets.FloatText(value=5., description='Time End:', disabled=False)
    dt = widgets.IntText(value=10, description='Evaluations:', disabled=False)
    arrow_length = widgets.FloatText(value=2., step=0.1, description='Arrow Length:', disabled=False)
    arrow_width = widgets.FloatText(value=0.005, step=0.001, description='Arrow Width:', disabled=False)
    fig_x = widgets.FloatText(value=10., description='Figure Size X:', disabled=False)
    fig_y = widgets.FloatText(value=6., description='Figure Size Y:', disabled=False)
    plot_velocity = widgets.Checkbox(value=True, description='View Velocity Vector', disabled=False)
    plot_speed = widgets.Checkbox(value=True, description='View Speed', disabled=False)
    return widgets.VBox([
        # widgets.HBox([func]),
        widgets.HBox([ts, te, dt]),
        widgets.HBox([arrow_length, arrow_width]),
        widgets.HBox([plot_velocity, plot_speed]),
        widgets.HBox([fig_x, fig_y]),
        widgets.interactive_output(run_velocity_and_speed, {
            'func': func, 'ts': ts, 'te': te, 'dt': dt,
            'arrow_length': arrow_length, 'arrow_width': arrow_width,
            'fig_x': fig_x, 'fig_y': fig_y,
            'plot_velocity': plot_velocity, 'plot_speed': plot_speed})
    ])


def run_velocity_and_speed(func, ts, te, dt, arrow_length, arrow_width, fig_x, fig_y, plot_velocity, plot_speed):
    time = [ts, te]
    step = (time[1] - time[0]) / dt
    sp_t = sp.Symbol('t')

    if func == 'Example_2_3':
        sp_x = sp.cos(sp_t)
        sp_y = sp.sin(sp_t)
        spacecurve = Spacecurve2D(time, step, sp_t, sp_x, sp_y)
        spacecurve.calculate_velocity()
        spacecurve.calculate_speed()
        if plot_velocity:
            spacecurve.plot_velocity_vector(scale=arrow_length, width=arrow_width, figsize=[fig_x, fig_y])
        if plot_speed:
            spacecurve.plot_speed(figsize=[fig_x, fig_y])


def distance_and_arc(example):
    func = widgets.fixed(example)
    ts = widgets.FloatText(value=0., description='Time Start:', disabled=False)
    te = widgets.FloatText(value=5., description='Time End:', disabled=False)
    dt = widgets.IntText(value=10, description='Evaluations:', disabled=False)
    fig_x = widgets.FloatText(value=10., description='Figure Size X:', disabled=False)
    fig_y = widgets.FloatText(value=6., description='Figure Size Y:', disabled=False)
    plot_position = widgets.Checkbox(value=True, description='View Position Vector', disabled=False)
    plot_speed_and_arc = widgets.Checkbox(value=True, description='View Speed and Arc Length', disabled=False)
    return widgets.VBox([
        # widgets.HBox([func]),
        widgets.HBox([ts, te, dt]),
        widgets.HBox([plot_position, plot_speed_and_arc]),
        widgets.HBox([fig_x, fig_y]),
        widgets.interactive_output(run_distance_and_arc, {
            'func': func, 'ts': ts, 'te': te, 'dt': dt,
            'fig_x': fig_x, 'fig_y': fig_y,
            'plot_position': plot_position, 'plot_speed_and_arc': plot_speed_and_arc})
    ])


def run_distance_and_arc(func, ts, te, dt, fig_x, fig_y, plot_position, plot_speed_and_arc):
    time = [ts, te]
    step = (time[1] - time[0]) / dt
    sp_t = sp.Symbol('t')

    if func == 'Example_2_5':
        sp_x = sp.cos(sp_t)
        sp_y = sp.sin(sp_t)
        spacecurve = Spacecurve2D(time, step, sp_t, sp_x, sp_y)
        spacecurve.calculate_velocity()
        spacecurve.calculate_speed()
        spacecurve.calculate_arc_length()
        if plot_position:
            spacecurve.plot_position(figsize=[fig_x, fig_y])
        if plot_speed_and_arc:
            spacecurve.plot_speed_and_arc_length(figsize=[fig_x, fig_y])


def acceleration(example):
    func = widgets.fixed(example)
    ts = widgets.FloatText(value=0., description='Time Start:', disabled=False)
    te = widgets.FloatText(value=5., description='Time End:', disabled=False)
    dt = widgets.IntText(value=10, description='Evaluations:', disabled=False)
    arrow_length = widgets.FloatText(value=2., step=0.1, description='Arrow Length:', disabled=False)
    arrow_width = widgets.FloatText(value=0.005, step=0.001, description='Arrow Width:', disabled=False)
    fig_x = widgets.FloatText(value=10., description='Figure Size X:', disabled=False)
    fig_y = widgets.FloatText(value=6., description='Figure Size Y:', disabled=False)
    plot_acceleration = widgets.Checkbox(value=True, description='View Total Acceleration', disabled=False)
    plot_components = widgets.Checkbox(value=True, description='View Acceleration Components', disabled=False)
    return widgets.VBox([
        widgets.HBox([ts, te, dt]),
        widgets.HBox([plot_acceleration, plot_components]),
        widgets.HBox([arrow_length, arrow_width]),
        widgets.HBox([fig_x, fig_y]),
        widgets.interactive_output(run_acceleration, {
            'func': func, 'ts': ts, 'te': te, 'dt': dt,
            'arrow_length': arrow_length, 'arrow_width': arrow_width,
            'fig_x': fig_x, 'fig_y': fig_y,
            'plot_acceleration': plot_acceleration, 'plot_components': plot_components})
    ])


def run_acceleration(func, ts, te, dt, arrow_length, arrow_width, fig_x, fig_y, plot_acceleration, plot_components):
    time = [ts, te]
    step = (time[1] - time[0]) / dt
    sp_t = sp.Symbol('t')

    if func == 'Additional_Example_1':
        sp_x = sp.cos(sp_t)
        sp_y = sp.sin(sp_t)
        spacecurve = Spacecurve2D(time, step, sp_t, sp_x, sp_y)
        spacecurve.calculate_velocity()
        spacecurve.calculate_speed()
        spacecurve.calculate_tau()
        spacecurve.calculate_arc_length()
        spacecurve.calculate_acceleration()
        spacecurve.calculate_acceleration_tangent()
        spacecurve.calculate_acceleration_normal()
        if plot_acceleration:
            spacecurve.plot_acceleration_total(scale=arrow_length, width=arrow_width, figsize=[fig_x, fig_y])
        if plot_components:
            spacecurve.plot_acceleration_components(scale=arrow_length, width=arrow_width, figsize=[fig_x, fig_y])

    if func == 'Additional_Example_2':
        sp_x = sp_t*sp.cos(sp_t)
        sp_y = sp_t*sp.sin(sp_t)
        spacecurve = Spacecurve2D(time, step, sp_t, sp_x, sp_y)
        spacecurve.calculate_velocity()
        spacecurve.calculate_speed()
        spacecurve.calculate_tau()
        # spacecurve.calculate_arc_length()
        spacecurve.calculate_acceleration()
        spacecurve.calculate_acceleration_tangent()
        spacecurve.calculate_acceleration_normal()
        if plot_acceleration:
            spacecurve.plot_acceleration_total(scale=arrow_length, width=arrow_width, figsize=[fig_x, fig_y])
        if plot_components:
            spacecurve.plot_acceleration_components(scale=arrow_length, width=arrow_width, figsize=[fig_x, fig_y])


def work_done(example):
    func = widgets.fixed(example)
    ts = widgets.FloatText(value=0., description='Time Start:', disabled=False)
    te = widgets.FloatText(value=1., description='Time End:', disabled=False)
    dt = widgets.IntText(value=10, description='Evaluations:', disabled=False)
    arrow_length_sc = widgets.FloatText(value=5., step=0.5, description='Velocity Arrow Length:', disabled=False)
    arrow_width_sc = widgets.FloatText(value=0.005, step=0.001, description='Velocity Arrow Width:', disabled=False)
    arrow_spacing_vf = widgets.IntText(value=3, step=1, description='Force Arrow Spacing:', disabled=False)
    arrow_length_vf = widgets.FloatText(value=20., step=0.5, description='Force Arrow Length:', disabled=False)
    arrow_width_vf = widgets.FloatText(value=0.005, step=0.001, description='Force Arrow Width:', disabled=False)
    fig_x = widgets.FloatText(value=10., description='Figure Size X:', disabled=False)
    fig_y = widgets.FloatText(value=6., description='Figure Size Y:', disabled=False)
    plot_background = widgets.Checkbox(value=True, description='View Background Vector Field', disabled=False)
    plot_local = widgets.Checkbox(value=True, description='View Local Vector Field', disabled=False)
    return widgets.VBox([
        widgets.HBox([ts, te, dt]),
        widgets.HBox([plot_background, plot_local]),
        widgets.HBox([arrow_length_sc, arrow_width_sc]),
        widgets.HBox([arrow_length_vf, arrow_width_vf, arrow_spacing_vf]),
        widgets.HBox([fig_x, fig_y]),
        widgets.interactive_output(run_work_done, {
            'func': func, 'ts': ts, 'te': te, 'dt': dt,
            'arrow_length_sc': arrow_length_sc, 'arrow_width_sc': arrow_width_sc,
            'arrow_length_vf': arrow_length_vf, 'arrow_width_vf': arrow_width_vf, 'arrow_spacing_vf': arrow_spacing_vf,
            'fig_x': fig_x, 'fig_y': fig_y,
            'plot_background': plot_background, 'plot_local': plot_local})
    ])


def run_work_done(func, ts, te, dt, arrow_length_sc, arrow_width_sc, arrow_length_vf, arrow_width_vf, arrow_spacing_vf,
                  fig_x, fig_y, plot_background, plot_local):
    if func == 'Example_2_7':
        # space curve
        time = [ts, te]
        step = (time[1] - time[0]) / dt
        sp_t = sp.Symbol('t')
        sp_sc_x = 1. * sp_t
        sp_sc_y = 2. * sp_t
        spacecurve = Spacecurve2D(time, step, sp_t, sp_sc_x, sp_sc_y)
        spacecurve.calculate_velocity()

        # force vector field
        xmin = np.min(spacecurve.np_array['x'])
        xmax = np.max(spacecurve.np_array['x'])
        ymin = np.min(spacecurve.np_array['y'])
        ymax = np.max(spacecurve.np_array['y'])
        np_x_1d = np.arange(xmin, xmax+(xmax-xmin)/dt, (xmax-xmin)/dt)
        np_y_1d = np.arange(ymin, ymax+(ymax-ymin)/dt, (xmax-xmin)/dt)
        sp_vf_x = sp.Symbol('x')
        sp_vf_y = sp.Symbol('y')
        sp_fx = -1. * sp_vf_y
        sp_fy = sp_vf_x * sp_vf_y
        force = VectorFunction2VarSP(sp_vf_x, sp_vf_y, sp_fx, sp_fy, np_x_1d, np_y_1d)

        if plot_background:
            spacecurve.plot_vfield_background(force, scale_sc=arrow_length_sc, width_sc=arrow_width_sc,
                                              scale_vf=arrow_length_vf, width_vf=arrow_width_vf,
                                              spacing_vf=arrow_spacing_vf, figsize=[fig_x, fig_y])
        if plot_local:
            spacecurve.plot_vfield_local(force, scale_sc=arrow_length_sc, width_sc=arrow_width_sc,
                                         scale_vf=arrow_length_vf, width_vf=arrow_width_vf, figsize=[fig_x, fig_y])
