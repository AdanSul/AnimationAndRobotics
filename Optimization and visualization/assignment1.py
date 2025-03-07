#%% imports
import vedo as vd
import numpy as np
from vedo.pyplot import plot
from vedo import Latex
from vedo import Button
import time

vd.settings.default_backend = 'vtk'

#%% Callbacks
msg = vd.Text2D(pos='bottom-left', font="VictorMono")  # an empty text

# Global variables
Xi = np.empty((0, 3))
candidate_position = np.empty((0, 2))
gd_path = np.empty((0, 2))
nm_path = np.empty((0, 2))

add_G = False
candidate_marker_added = False   
previous_plot = None
current_plot = None
last_point = None

path_colors = ['orange', 'red', 'green', 'blue', 'yellow', 'purple']
current_color_index = 0
arrows = []  # List to store all arrows

# Create the main plotter for 3D visualization
plt = vd.Plotter()

def update_graph():
    global previous_plot, current_plot,Xi

    if previous_plot is not None:
        plt.remove(previous_plot)
    
    if len(Xi) < 2:
        return  # Exit if there is less than two points

    x_values = np.arange(len(Xi))
    y_values = Xi[:, 2]

    if len(y_values) > 0:
        try:
            current_plot = plot(
                x_values, y_values,
                title='Function Values',
                c='green', marker='d', markersize=6, lw=3,
                xtitle="Points", ytitle="Value"
            ).clone2d(pos=(-1, 0.15), size=0.8)
            
            # Add the new plot to the scene
            plt.add(current_plot)
            previous_plot = current_plot

        except Exception as e:
            print("Error while creating or adding plot:", e)
    else:
        print("No valid y_values to plot")

def update_arrow_colors(color):
    for arrow in arrows:
        arrow.color(color)

def OnMouseMove(evt):  ### called every time mouse moves!
    global Xi, arrows, add_G

    if evt.object is None:  # mouse hits nothing, return.
        return

    pt = evt.picked3d  # 3d coords of point under mouse
    if pt is not None and len(pt) == 3:  # Ensure pt is a valid 3D point
        X = np.array([pt[0], pt[1], objective([pt[0], pt[1]])])  # X = (x,y,e(x,y))

        # to prevent forming a path with mouse movement.
        #Xi = np.append(Xi, [X], axis=0)  # append to the list of points

        if len(Xi) > 1:  # need at least two points to compute a distance
            txt = (
                f"X:  {vd.precision(X, 2)}\n"
                f"dX: {vd.precision(Xi[-1, 0:2] - Xi[-2, 0:2], 2)}\n"
                f"dE: {vd.precision(Xi[-1, 2] - Xi[-2, 2], 2)}\n"
            )
            # to prevent adding arrows for mouse movement.
            # ar = vd.Arrow(Xi[-2, :], Xi[-1, :], s=0.001, c=path_colors[current_color_index])
            # plt.add(ar)  # add the arrow
            # arrows.append(ar)  # store the arrow
        else:
            txt = f"X: {vd.precision(X, 2)}"
            
        msg.text(txt)  # update text message

        # c = vd.Cylinder([np.append(Xi[-1, 0:2], 0.0), Xi[-1, :]], r=0.01, c='orange5')
        plt.remove("Cylinder")
        fp = fplt3d[0].flagpole(txt, point=X, s=0.08, c='k', font="Quikhand")
        fp.follow_camera()  # make it always face the camera
        plt.remove("FlagPole")  # remove the old flagpole

        # plt.add(fp, c)  # add the new flagpole and new cylinder
        plt.add(fp)  # add the new flagpole and new cylinder

        if len(Xi) > 7:
            add_G=True
            update_graph()  # Update the graph with the latest data

        plt.render()  # re-render the scene

def OnKeyPress(evt):  ### called every time a key is pressed
    global Xi, previous_plot, current_plot

    if evt.keypress in ['c', 'C']:  # reset Xi and the arrows
        Xi = np.empty((0, 3))
        plt.remove("Arrow").render()
        plt.remove("Sphere").render()
        if previous_plot is not None:
            plt.remove(previous_plot)
        if current_plot is not None:
            plt.remove(current_plot)
        previous_plot = None
        current_plot = None
        msg.text("Path cleared.")
        plt.render()

def OnSliderAlpha(widget, event):  ### called every time the slider is moved
    val = widget.value  # get the slider value
    fplt3d[0].alpha(val)  # set the alpha (transparency) value of the surface
    fplt3d[1].alpha(val)  # set the alpha (transparency) value of the isolines

#--------- Task 1.1 --------------------
def OnRightButtonDown(evt):  ### called every time the right mouse button is pressed 
    global current_color_index
    pt = evt.picked3d
    if pt is not None:
        print(f"Right mouse button pressed at coordinate: {pt}")
        # Create and add a sphere at the point of the click
        sphere = vd.Sphere(pos=pt, r=0.05, c='white', alpha=0.5)
        plt.add(sphere)
        # Change path color
        current_color_index = (current_color_index + 1) % len(path_colors)
        new_color = path_colors[current_color_index]
        update_arrow_colors(new_color)
        plt.render()

#--------- Task 2.2 --------------------
def OnLeftButton(evt): ### called every time the right mouse button is pressed
    global Xi,current_candidate_marker, candidate_marker_added, gd_path, nm_path,candidate_position, add_G,previous_plot,current_plot

    # Ensure a valid 3D point is clicked
    if evt.picked3d is None:
        return
    
    # Clear arrays and reset global variables
    Xi = np.empty((0, 3))
    gd_path = np.empty((0, 2)) 
    nm_path = np.empty((0, 2))  

    # Remove visual elements
    try:
        plt.remove("Arrow").render()
        plt.remove("Point").render()
        plt.remove("Cylinder")
        plt.remove(current_candidate_marker)
        if previous_plot is not None:
            plt.remove(previous_plot)
        if current_plot is not None:
            plt.remove(current_plot)
        previous_plot = None
        current_plot = None
        print("Previous markers and plots removed")
    except Exception as e:
        print("No elements to remove:", e)

    # Reset candidate marker
    current_candidate_marker = None
    candidate_marker_added = False
    plt.render()

    # Set new candidate position 
    pt = evt.picked3d
    candidate_position = np.array([pt[0], pt[1]])

    X_candidate = np.array([pt[0], pt[1], objective([pt[0], pt[1]])])

    # Print candidate point coordinates
    print(f"New optimization candidate set to: {X_candidate}")

    gd_path = np.append(gd_path, [candidate_position], axis=0)
    nm_path = np.append(nm_path, [candidate_position], axis=0)

    # Remove the current candidate marker if it exists
    if candidate_marker_added:
        plt.remove(current_candidate_marker)
    else:
        candidate_marker_added = True
    
    # Create a candidate marker at the clicked position
    candidate_z = objective([pt[0], pt[1]])
    current_candidate_marker = vd.Sphere(
        pos=np.array([pt[0], pt[1], candidate_z]), 
        r=0.03, c='orange'
    )

    plt.add(current_candidate_marker)
    msg.text(f"Starting optimization at {vd.precision(X_candidate, 2)}")
    plt.render()

#--------- Task 2.3 + Task 2.4 ----------------
# def perform_step(method_type): #perform a single optimization step based on the selected method on seperate plots
#     global gd_path, nm_path, previous_plot, arrows,add_G
#     add_G = True

#     # Select the correct path and direction based on the method
#     if method_type == "gradient":
#         path_data = gd_path
#         direction = gradient_direction
#         color = "orange"
#         title = "Path of Gradient Descent"
#         label = "Gradient Descent Progression"
#         position = 'bottom-left'
#     elif method_type == "newton":
#         path_data = nm_path
#         direction = Newton_direction
#         color = "purple"
#         title = "Newton's Method Pathway"
#         label = "Newton's Path Progress"
#         position = 'bottom-right'
#     else:
#         raise ValueError("Invalid method_type. Choose 'gradient' or 'newton'.")

#     # If path_data is empty, do nothing and return immediately
#     if (path_data.size) == 0:
#         print(f"No starting point for {method_type}.")
#         return

#     # Perform the optimization step
#     try:
#         new_candidate = step(objective, path_data[-1], direction)
#         path_data = np.append(path_data, [new_candidate], axis=0)

#     except Exception as e:
#         print(f"Error in optimization step for {method_type}:", e)
#         return

#     # Update and plot the path if there are enough points
#     if len(path_data) > 1:
#         if previous_plot is not None:
#             plt.remove(previous_plot).render()

#         try:
#             # Plot the updated path graph
#             previous_plot = plot(
#                 np.arange(len(path_data)),
#                 [objective([pt[0], pt[1]]) for pt in path_data],
#                 f"{color} -",
#                 title=title,
#                 xtitle="Step",
#                 ytitle="Objective Value",
#                 ylim=(0, 1),
#                 xlim=(0, len(path_data)),
#                 label=label
#             )
#             # Clone and position the graph
#             previous_plot = previous_plot.clone2d(pos=position, size=0.9)
#             plt.add(previous_plot).render()

#         except Exception as e:
#             print(f"Error updating {method_type} graph:", e)

#         # Draw arrows for visualization
#         try:
#             arrow = vd.Arrow(
#                 np.array([path_data[-2, 0], path_data[-2, 1], objective([path_data[-2, 0], path_data[-2, 1]])]),
#                 np.array([path_data[-1, 0], path_data[-1, 1], objective([path_data[-1, 0], path_data[-1, 1]])]),
#                 s=0.001, c=color
#             )
#             arrows.append(arrow)
#             plt.add(arrow).render()
#         except Exception as e:
#             print(f"Error adding arrow for {method_type} step:", e)

#     # Save the updated path back to the global variables
#     if method_type == "gradient":
#         gd_path = path_data
#     elif method_type == "newton":
#         nm_path = path_data

def update_paths_plot(): #Update the plot to show both paths.
    global previous_plot

    if previous_plot is not None:
        plt.remove(previous_plot)

    # Check if there is valid data to plot
    gd_steps, gd_values = [], []
    nm_steps, nm_values = [], []

    if gd_path.size > 1:
        gd_steps = np.arange(len(gd_path))
        gd_values = [objective(pt) for pt in gd_path]
    if nm_path.size > 1:
        nm_steps = np.arange(len(nm_path))
        nm_values = [objective(pt) for pt in nm_path]

    # Ensure at least one path has valid data
    if not gd_values and not nm_values:
        print("No valid data to plot for either path.")
        return

    try:
        # Determine axis limits
        x_max = max(len(gd_steps), len(nm_steps)) or 1
        y_min = min(gd_values + nm_values) if gd_values + nm_values else 0
        y_max = max(gd_values + nm_values) if gd_values + nm_values else 1

        # Plot paths with consistent axes
        plot_data = None
        if gd_values:
            plot_data = plot(
                gd_steps, gd_values,
                title="Optimization Progress",
                c='orange', lw=2,
                xtitle="Steps", ytitle="Objective Value",
                xlim=(0, x_max), ylim=(y_min, y_max)
            )
        if nm_values:
            nm_plot = plot(
                nm_steps, nm_values,
                c='purple', lw=2,
                xlim=(0, x_max), ylim=(y_min, y_max)
            )
            plot_data = plot_data + nm_plot if plot_data else nm_plot

        if plot_data:
            current_plot = plot_data.clone2d(pos=(-1, 0.15), size=0.8)
            plt.add(current_plot)
            previous_plot = current_plot

    except Exception as e:
        print("Error updating plot:", e)

#----------- Task 3.1---------------
def perform_step(method_type): #perform a single optimization step based on the selected method
    global gd_path, nm_path, arrows

    if method_type == "gradient":
        path_data = gd_path
        direction = gradient_direction
        color = "orange"
    elif method_type == "newton":
        path_data = nm_path
        direction = Newton_direction
        color = "purple"
    else:
        raise ValueError("Invalid method_type. Choose 'gradient' or 'newton'.")

    if path_data.size == 0:
        print(f"No starting point for {method_type}.")
        return

    try:
        # Perform optimization step
        new_candidate = step(objective, path_data[-1], direction)
        path_data = np.append(path_data, [new_candidate], axis=0)

        # Update the global path
        if method_type == "gradient":
            gd_path = path_data
        elif method_type == "newton":
            nm_path = path_data

        # Draw the arrow for visualization
        try:
            arrow = vd.Arrow(
                np.array([path_data[-2, 0], path_data[-2, 1], objective([path_data[-2, 0], path_data[-2, 1]])]),
                np.array([path_data[-1, 0], path_data[-1, 1], objective([path_data[-1, 0], path_data[-1, 1]])]),
                s=0.001, c=color
            )
            arrows.append(arrow)
            plt.add(arrow)
        except Exception as e:
            print(f"Error adding arrow for {method_type} step:", e)

        # Update the plot with both paths
        update_paths_plot()

    except Exception as e:
        print(f"Error in optimization step for {method_type}:", e)

def add_step_button(method, pos, state_texts, colors):
    button = Button(
        pos=pos,  # Position of the button in the window
        states=state_texts,  # Text for each state of the button
        c=['white', 'white'],  # Text color for each state
        bc=colors,  # Background color for each state
        font="VictorMono",
        size=20
    )
    button.status(0)  # Set initial state
    plt.add(button)
    return button

def check_button_action(evt):
    if evt.object.name == 'gradient_button' and gradient_button.status() == 'Gradient Step':
        perform_step('gradient')
        gradient_button.status(0)  
    elif evt.object.name == 'newton_button' and newton_button.status() == 'Newton Step':
        perform_step('newton')
        newton_button.status(0) 

#%% Optimization functions
def objective(X):
    x, y = X[0], X[1]
    return np.sin(2*x*y) * np.cos(3*y)/2+1/2

def gradient_fd(func, X, h=0.001): # finite difference gradient
    x, y = X[0], X[1]
    gx = (func([x+h, y]) - func([x-h, y])) / (2*h)
    gy = (func([x, y+h]) - func([x, y-h])) / (2*h)
    return gx, gy

def Hessian_fd(func, X, h=0.001): # finite difference Hessian
    x, y = X[0], X[1]
    gxx = (func([x+h, y]) - 2*func([x, y]) + func([x-h, y])) / h**2
    gyy = (func([x, y+h]) - 2*func([x, y]) + func([x, y-h])) / h**2
    gxy = (func([x+h, y+h]) - func([x+h, y-h]) - func([x-h, y+h]) + func([x-h, y-h])) / (4*h**2)
    H = np.array([[gxx, gxy], [gxy, gyy]])
    return H

def gradient_direction(func, X): # compute gradient step direction
    g = gradient_fd(func, X)
    return -np.array(g)

def Newton_direction(func, X):   # compute Newton step direction
    g = gradient_fd(func, X)
    H = Hessian_fd(func, X)
    d = -np.linalg.solve(H, np.array(g))
    return np.array(d[0],d[1])

def line_search(func, X, d): 
    alpha = 1.0
    while func(X + d*alpha) > func(X):  # If the function value increases, reduce alpha
        alpha *= 0.5                               
    return alpha

def step(func, X, search_direction_function):
    d = search_direction_function(func, X)
    alpha = line_search(func, X, d)
    return X + d*alpha

def optimize(func, X, search_direction_function, tol=1e-6, iter_max=10):
    for i in range(iter_max):
        X = step(func, X, search_direction_function)
        if np.linalg.norm(gradient_fd(func, X)) < tol:
            break
    return X

Xi = np.empty((0, 3))
# test the optimization functions
X = optimize(objective, [0.6, 0.6], Newton_direction, tol=1e-6, iter_max=100)

#%% Plotting
plt = vd.Plotter(bg2='lightblue')  # Create the plotter
fplt3d = plot(lambda x, y: objective([x, y]), c='terrain')  # create a plot from the function e. fplt3d is a list containing surface mesh, isolines, and axis
fplt2d = fplt3d.clone()  # clone the plot to create a 2D plot

fplt2d[0].lighting('off')  # turn off lighting for the 2D plot
fplt2d[0].vertices[:, 2] = 0  # set the z-coordinate of the mesh to 0
fplt2d[1].vertices[:, 2] = 0  # set the z-coordinate of the isolines to 0

plt.add_callback('LeftButtonDown', OnLeftButton) # add Left mouse button callback
plt.add_callback('mouse move', OnMouseMove)  # add Mouse move callback
plt.add_callback('key press', OnKeyPress)  # add Keyboard callback
plt.add_callback('RightButtonDown', OnRightButtonDown)  # add Right mouse button callback

# Add gradient step button and its action check
gradient_button = add_step_button('gradient', (0.8, 0.85), ['Gradient Step', 'Running...'], ['blue', 'red'])
gradient_button.name = 'gradient_button'
plt.add(gradient_button)

# Add Newton step button and its action check
newton_button = add_step_button('newton', (0.8, 0.8), ['Newton Step', 'Running...'], ['green', 'orange'])
newton_button.name = 'newton_button'
plt.add(newton_button)

plt.add_callback('button press', check_button_action)

plt.add_slider(OnSliderAlpha, 0., 1., 1., title="Alpha")  # add a slider for the alpha value of the surface

plt.show([fplt3d, fplt2d], msg, __doc__, viewup='z')
plt.close()

# %%