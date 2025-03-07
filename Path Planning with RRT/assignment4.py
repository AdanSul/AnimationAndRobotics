#%%
from scipy.spatial import KDTree
import numpy as np
import math
import random
import vedo as vd
from vedo.pyplot import DirectedGraph
from vedo import Button
from skimage.draw import line
import time

vd.settings.default_backend = 'vtk'
#%%
class Sample:
    def __init__(self, x, parent = None):
        self.x = x
        self.parent = parent

# find the nearest sample using KDTree
def NearestSample(samples, x):
    tree = KDTree([sample.x for sample in samples])
    dist, sid = tree.query(x)
    return sid


# check if the line between x1 and x2 intersects with the obstacles
def Collision(img, x1, x2):
    # Validate input points
    if x1 is None or x2 is None:
        print("Collision Error: Points are None.")
        return True
    # Convert to integer coordinates
    try:
        x1, y1 = map(int, x1)
        x2, y2 = map(int, x2)
    except ValueError:
        print(f"Collision Error: Invalid points {x1}, {x2}")
        return True
    # Generate pixel coordinates along the line
    rr, cc = line(y1, x1, y2, x2)
    # Check each point for collision
    for r, c in zip(rr, cc):
        if 0 <= r < img.shape[0] and 0 <= c < img.shape[1]:
            if not img[r, c]:  # Obstacle detected
                return True
    return False


# get the next sample point
def getNextSample(dim, samples, stepsize):
    x = np.array([random.randint(0, d) for d in dim])
    sid = NearestSample(samples,x)
    nearest_sample = samples[sid]

    direction = x - nearest_sample.x
    distance = np.linalg.norm(direction)

    if distance == 0:
        return None

    direction_normalized = direction / distance
    nx = nearest_sample.x + direction_normalized * stepsize

    nx = np.round(nx).astype(int) 

    ns = Sample(nx, sid)
    return ns

#%%
def bfunc(obj=None, ename=None):
    global timer_id

    plt.timer_callback("destroy", timer_id)
    if "Run" in run_button.status():
        timer_id = plt.timer_callback("create", dt=10)
    run_button.switch()

current_mode = "Set Source"  # Initial mode

# Button callback to toggle between source and destination modes
def toggleSetMode(obj=None, ename=None):
    global current_mode
    if current_mode == "Set Source":
        current_mode = "Set Destination"
        set_mode_button.switch()  # Change button text
    else:
        current_mode = "Set Source"
        set_mode_button.switch()
    print(f"Current mode: {current_mode}")


is_drawing_active = False
is_erasing_active = False

# Toggle Draw Mode
def toggleDraw(obj=None, ename=None):
    global is_drawing_active, is_erasing_active
    is_drawing_active = not is_drawing_active
    if is_drawing_active:
        is_erasing_active = False  # Disable erase mode if draw mode is active
        erase_button.status(0)  # Reset erase button
    draw_button.switch()  # Update draw button state
    print(f"Draw Mode {'activated' if is_drawing_active else 'deactivated'}")

# Toggle Erase Mode
def toggleErase(obj=None, ename=None):
    global is_drawing_active, is_erasing_active
    is_erasing_active = not is_erasing_active
    if is_erasing_active:
        is_drawing_active = False  # Disable draw mode if erase mode is active
        draw_button.status(0)  # Reset draw button
    erase_button.switch()  # Update erase button state
    print(f"Erase Mode {'activated' if is_erasing_active else 'deactivated'}")


destination_reached = False
TARGET_REACHED_THRESHOLD = 10
MAX_ITERATIONS = 1500
iteration_count = 0 

# Function to stop the iteration
def stopIteration():
    global destination_reached
    destination_reached = True
    print("RRT algorithm stopped.")

# Recursive path highlighting from the destination to the source
def highlightPath(sample):
    if sample.parent is None:  # Base case: reached the source
        print("Path tracing complete.")
        return

    # Visualize the edge from the current sample to its parent
    parent_sample = samples[sample.parent]
    edge = vd.Line(sample.x, parent_sample.x, lw=3, c="blue")
    plt.add(edge)

    # Recursively highlight the path
    highlightPath(parent_sample)

# Event handler to check if destination is reached
def checkForCompletion(ns, dest):
    distance = np.linalg.norm(ns.x - dest)
    if distance < TARGET_REACHED_THRESHOLD:
        stopIteration()
        highlightPath(ns)
        return True
    return False


sample_count = 0
collisions = 0
start_time = time.time()  # Track the start time of the algorithm

# Function to update statistics
def update_statistics():
    global sample_count, collisions, start_time

    elapsed_time = time.time() - start_time  # Calculate elapsed time
    distance_to_dest = np.linalg.norm(samples[-1].x - dest)  # Calculate distance from source to destination

    # Create the text label for the statistics
    stats_text = f"Samples: {sample_count}\nCollisions: {collisions}\nDistance to Goal: {distance_to_dest:.2f}\nTime Elapsed: {elapsed_time:.2f}s"

    # Update or create a new label for the statistics
    if hasattr(update_statistics, "stats_label"):
        update_statistics.stats_label.text(stats_text)  # Update the existing label
    else:
        update_statistics.stats_label = vd.Text2D(stats_text,  pos="top-center",c="black")
        plt.add(update_statistics.stats_label)  # Create a new label and add to plot

def doRRTIteration(event):
    global destination_reached, iteration_count, destination_reached, sample_count, collisions

    sample_count += 1

    # If the destination is already reached, stop the iteration
    if destination_reached:
        return

    y, x = img.shape  # Get the image size; switch x and y due to conventions
    ns = getNextSample([x, y], samples, stepSize)  # Generate a new sample

    iteration_count += 1

    # Check if maximum iterations are exceeded
    if iteration_count >= MAX_ITERATIONS:
        print("Maximum iterations reached. Stopping RRT.")
        stopIteration()  # Stop the iteration
        return

    # Check if the new sample is valid
    if ns is None:  # Avoid processing if no sample is generated
        print("No valid sample generated.")
        return

    # Ensure the new sample does not collide with the destination
    if Collision(img, ns.x, samples[ns.parent].x):  # Check collision with parent
        collisions += 1
        return

    # Add the new node to the tree (samples list)
    samples.append(ns)
  
    # Update statistics
    update_statistics()

    if checkForCompletion(ns, dest):
        return  # Stop further iterations if destination is reached

    # Add the new point to the points' vertices
    points.vertices = np.vstack([points.vertices, np.hstack([ns.x, 0])])
    # Add the edge connecting the new point to its parent
    edge = vd.Line(ns.x, samples[ns.parent].x, lw=2, c="pink")
    edges.append(edge)

    # Update the plot
    plt.add(points)  # Add updated points
    plt.remove("Line")  # Clear old lines from visualization
    plt.add(edges)  # Add updated edges
    plt.render()  # Render the updated plot


source_point = None
dest_point = None

# Mouse right-click event for setting points based on the mode
def onRightClick(event):
    global source, dest, samples, points, source_point, dest_point
    # Check if the right mouse button was clicked
    if "RightButtonPress" in event.name and event.picked3d is not None:
        clicked_point = np.round(event.picked3d[:2]).astype(int)
        if current_mode == "Set Source":
            source = clicked_point
            samples = [Sample(source)]  # Reset samples with new source

            # Remove the old source point if it exists
            if source_point:
                plt.remove(source_point)
            
            # Add the new source point
            source_point = vd.Point(source, r=10, c="green")
            plt.add(source_point)

            points.vertices = np.array([source])  # Reset points to the new source
            edges.clear()  # Clear existing edges
            plt.remove('Line')  # Remove old lines
            plt.add(points)

            print(f"Source updated to: {source}")
        
        elif current_mode == "Set Destination":
            dest = clicked_point

            # Remove the old destination point if it exists
            if dest_point:
                plt.remove(dest_point)
            
            # Add the new destination point
            dest_point = vd.Point(dest, r=10, c="red")
            plt.add(dest_point)
            print(f"Destination updated to: {dest}")
    elif event.picked3d is None:
        print("No valid point picked. Please click on the scene.")

# Callback function to reset the GUI
def resetGUI(obj=None, ename=None):
    global source, dest, samples, points, edges, source_point, dest_point, destination_reached
    global sample_count, collisions, start_time

    source = np.array([0,0])
    dest = np.array([50,50])
    destination_reached = False

    # Reset statistics to initial state
    sample_count = 0
    collisions = 0
    start_time = time.time()  # Reset the start time


    # Clear source and destination points
    if source_point:
        plt.remove(source_point)
        source_point = None
    if dest_point:
        plt.remove(dest_point)
        dest_point = None

    # Re-initialize samples/points and clear tree visualization
    samples = [Sample(np.array([0,0]))]
    points = vd.Points([np.array([0,0])])

    edges.clear()  
    plt.remove("Line") 
    plt.add(points)  

    # Remove the statistics label (if it exists)
    if hasattr(update_statistics, "stats_label"):
        plt.remove(update_statistics.stats_label)
        del update_statistics.stats_label  # Clean up the reference

    print("GUI has been reset.")


brush_size = 6

def onMouseDrag(event):
    global obstacle_map, vd_img, brush_size
    if event.picked3d is not None:
        clicked_point = np.round(event.picked3d[:2]).astype(int)
        x, y = clicked_point

        # Ensure the point is within bounds
        if 0 <= y < obstacle_map.shape[0] and 0 <= x < obstacle_map.shape[1]:
            # Define the range of the brush
            y_min = max(0, y - brush_size // 2)
            y_max = min(obstacle_map.shape[0], y + brush_size // 2 + 1)
            x_min = max(0, x - brush_size // 2)
            x_max = min(obstacle_map.shape[1], x + brush_size // 2 + 1)

            # Apply the drawing or erasing action
            if is_drawing_active:  # Draw mode
                obstacle_map[y_min:y_max, x_min:x_max] = False
            elif is_erasing_active:  # Erase mode
                obstacle_map[y_min:y_max, x_min:x_max] = True

            # Update visualization
            plt.remove(vd_img)
            vd_img = vd.Image(obstacle_map.astype(np.uint8) * 255).bw()
            plt.add(vd_img)
            plt.render()


# function to update step size
def updateStepSize(widget, event):
    global stepSize
    stepSize = int(widget.GetRepresentation().GetValue()) 
    print(f"Step size updated to: {stepSize}")

def add_step_button(method, pos, state_texts, colors):
    button = Button(
        pos=pos,  # Position of the button in the window
        states=state_texts,  # Text for each state of the button
        c=['white', 'white'],  # Text color for each state
        bc=colors,  # Background color for each state
        font="VictorMono",
        size=24
    )
    # button.status(0)  # Set initial state
    plt.add(button)
    return button

def check_button_action(evt):
    if evt.object.name == 'run_button':
        bfunc(None,None)  
    elif evt.object.name == 'set_mode_button':
        toggleSetMode(None,None)  
    elif evt.object.name == 'reset_button':
        resetGUI(None,None) 
    elif evt.object.name == 'draw_button':
        toggleDraw(None,None) 
    elif evt.object.name == 'erase_button':
        toggleErase(None,None) 
    else:
        return

imagePath = 'obstacle_map.png'
stepSize = 10
source = np.array([0,0])
dest = np.array([50,50])
tree = [Sample(source)]

vd_img = vd.Image(imagePath).bw().binarize()
img = vd_img.tonumpy().astype(bool)
# obstacle_map = img.copy()
obstacle_map = np.flipud(img.copy()) 

samples = [Sample(source)] # list to store all the node points
points = vd.Points([source]) # list to store all the node points
edges = []
plt = vd.Plotter()
plt += vd_img
plt.user_mode('2d')

# slider to change the step size
slider = plt.add_slider(
    updateStepSize,  
    1, 50, 
    value=10,          
    title="Step Size", 
    show_value=True,                         
    pos=[(0.25, 0.05), (0.75, 0.05)]              
)

# Add the Run/Pause button
run_button = add_step_button(
    bfunc,  # Callback function for button interaction
    pos=(0.1, 0.8),  # Position of the button
    state_texts=[" Run ", "Pause"],  # Button states
    colors=["#9E9E9E", "#FF5722"]
) 
run_button.name = 'run_button'

# Add the Set Source/Destination button
set_mode_button = add_step_button(
    toggleSetMode,
    pos=(0.1, 0.9),
    state_texts=["Set Source", "Set Destination"],
    colors=["#9E9E9E", "#FF5722"]
)
set_mode_button.name = 'set_mode_button'

# Add the Reset button
reset_button = add_step_button(
    resetGUI,
    pos=(0.1, 0.7),
    state_texts=["Reset"],
    colors=["#9E9E9E", "#FF5722"]
)
reset_button.name = 'reset_button'

# Add the Draw Mode button
draw_button = add_step_button(
    toggleDraw,
    pos=(0.1, 0.6),
    state_texts=["Activate Draw", "Deactivate Draw"],
    colors=["#9E9E9E", "#FF5722"]
)
draw_button.name = 'draw_button'

# Add the Erase Mode button
erase_button = add_step_button(
    toggleErase,
    pos=(0.1, 0.5),
    state_texts=["Activate Erase", "Deactivate Erase"],
    colors=["#9E9E9E", "#FF5722"]
)
erase_button.name = 'erase_button'

# Add buttons to the Plotter
plt.add(run_button)
plt.add(set_mode_button)
plt.add(reset_button)
plt.add(draw_button)
plt.add(erase_button)

plt.add_callback('button press', check_button_action)

plt.add_callback("MouseMove", onMouseDrag)

evntid = plt.add_callback("timer", doRRTIteration, enable_picking=False)
timer_id = -1

plt.add_callback("RightButtonPress", onRightClick)


plt.show(zoom="tightest").close()


# %%