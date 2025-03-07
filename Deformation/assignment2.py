#%%
import vedo as vd
vd.settings.default_backend= 'vtk'
import keyboard

from vedo import show
import numpy as np

from abc import ABC, abstractmethod
import numdifftools as nd
from scipy.sparse import coo_matrix
import triangle as tr # pip install triangle    

#%% Stencil class
# Abstract class for a stencil, which is a local description of the elements in a mesh. The stencil
# is used to extract the elements from the mesh and to extract the different variables from the vectors
# that represent the elements. 
# The ExtractElementsFromMesh method take a Vedo Mesh object as input and returns a list of elements
# The ExtractVariblesFromVectors method takes a vector as input and returns the variables that represent the element
class Stencil(ABC):
    @abstractmethod
    def ExtractElementsFromMesh(F):
        return 0

    @abstractmethod
    def ExtractVariblesFromVectors(x):
        return 0

class EdgeStencil(Stencil):
    # Extract the edges from a mesh
    @staticmethod
    def ExtractElementsFromMesh(F):
        edges = {tuple(sorted((F[i, j], F[i, (j+1) % 3]))) for i in range(F.shape[0]) for j in range(3)}
        return list(edges)
    
    # Extract x1, x2, the two vertices that define the edge, from the vector x, assuming that the variables are stored in the order x1, x2, x3, y1, y2, y3, z1, z2, z3
    # or as a 3x2 matrix, where the columns are the vertices
    @staticmethod
    def ExtractVariblesFromVectors(x):
        return x.flat[0:2], x.flat[2:4]

#%% Energy functions
# Abstract element energy class that implements finite differences for the gradient and hessian
# of the energy function. The energy function is defined in the derived classes, which must implement
# the energy method. The gradient and hessian methods should override the finite differences implementation.
# X is the undeformed configuration and x is the deformed configuration. The order of the variables in X and x
# is implementation dependant, but should be consistent between the energy, gradient and hessian methods.
# The current implementation assumes that the variables are stored in a 1D array, in and x1, x2, x3,..., y1, y2, y3, ..., z1, z2, z3 order.

class ElementEnergy(ABC):    
    @abstractmethod
    def energy(X, x):
        return 0

    # should be overridden by the derived class, otherwise the finite difference implementation will be used
    def gradient(self, X, x):
        return self.gradient_fd(X, x)

    def hessian(self, X, x):
        return self.hessian_fd(X, x)
    
    # finite difference gradient and hessian
    def gradient_fd(self, X, x):
        return nd.Gradient(lambda X, x: self.energy(X, x.flatten()))

    def hessian_fd(self, X, x):
        return nd.Hessian(lambda X, x: self.energy(X, x.flatten()))

    # check that the gradient is correct by comparing it to the finite difference gradient
    def check_gradient(self, X, x):
        grad = self.gradient(X, x)
        grad_fd = self.gradient_fd(X, x)
        return np.linalg.norm(grad - grad_fd)

# Spring energy function for a zero-length spring, defined as E = 0.5*||x1-x2||^2, regardless of the undeformed configuration
class ZeroLengthSpringEnergy(ElementEnergy):
    def __init__(self):
        self.stencil = EdgeStencil()
    
    def energy(self, X, x):
        x1,x2 = self.stencil.ExtractVariblesFromVectors(x)
        return 0.5*np.linalg.norm(x1 - x2)**2

    def gradient(self, X, x):
        dx = x[0, 0] - x[1, 0] 
        dy = x[0, 1] - x[1, 1]  

        return  np.array([dx, dy, -dx, -dy])

    def hessian(self, X, x):
        # The hessian is constant and is shapes like [I -I; -I I], where I is the identity matrix
        I = np.eye(2)
        return np.block([[I, -I], [-I, I]])
    
# Spring energy function for a spring with a rest length, defined as E = 0.5*(||x1-x2|| - l)^2, where l is the rest length
class SpringEnergy(ElementEnergy):
    def __init__(self):
        self.stencil = EdgeStencil()
    
    def energy(self, X, x):
        x1,x2 = self.stencil.ExtractVariblesFromVectors(x)
        X1,X2 = self.stencil.ExtractVariblesFromVectors(X)
        energy_value = 0.5 * (np.linalg.norm(x1 - x2) - np.linalg.norm(X1 - X2)) ** 2
        return energy_value if energy_value > 0 else 0.5 * np.linalg.norm(x[0] - x[1]) ** 2
        

    def gradient(self, undeformed_positions, deformed_positions):
        undeformed_length = np.linalg.norm(undeformed_positions[0] - undeformed_positions[1])
        deformed_length = np.linalg.norm(deformed_positions[0] - deformed_positions[1])

        if undeformed_length == deformed_length:
            scale = 1
        else:
          scale = (deformed_length - undeformed_length) / deformed_length

        # Differences in x and y components
        dx = deformed_positions[0, 0] - deformed_positions[1, 0]
        dy = deformed_positions[0, 1] - deformed_positions[1, 1]

        # Gradient array (x1, y1, x2, y2)
        gradient_array = np.array([dx, dy, -dx, -dy])

        scaled_gd = scale * gradient_array

        return scaled_gd


    # def hessian(self, undeformed_positions, deformed_positions):
    #     # Compute rest and current lengths
    #     undeformed_length = np.linalg.norm(undeformed_positions[0] - undeformed_positions[1])
    #     deformed_length = np.linalg.norm(deformed_positions[0] - deformed_positions[1])

    #     # Handle zero-length springs
    #     if deformed_length == 0:
    #         return np.zeros((4, 4))

    #     # Differences in x and y components
    #     dx = deformed_positions[1, 0] - deformed_positions[0, 0]
    #     dy = deformed_positions[1, 1] - deformed_positions[0, 1]

    #     # Precompute scale and other factors
    #     scale = (deformed_length - undeformed_length) / deformed_length
    #     factor = 1 / (deformed_length ** 3)  # Used for second derivatives

    #     # Hessian matrix (4x4) initialization
    #     hessian = np.zeros((4, 4))

    #     # Diagonal terms
    #     hessian[0, 0] = scale / deformed_length - scale * (1 - dx ** 2 * factor)
    #     hessian[1, 1] = scale / deformed_length - scale * (1 - dy ** 2 * factor)
    #     hessian[2, 2] = scale / deformed_length - scale * (1 - dx ** 2 * factor)
    #     hessian[3, 3] = scale / deformed_length - scale * (1 - dy ** 2 * factor)

    #     # Off-diagonal terms
    #     hessian[0, 2] = hessian[2, 0] = -scale / deformed_length + scale * dx ** 2 * factor
    #     hessian[1, 3] = hessian[3, 1] = -scale / deformed_length + scale * dy ** 2 * factor
    #     hessian[0, 3] = hessian[3, 0] = scale * dx * dy * factor
    #     hessian[1, 2] = hessian[2, 1] = scale * dx * dy * factor

    #     return hessian

    def hessian(self, undeformed_positions, deformed_positions):
        rest_length = np.linalg.norm(undeformed_positions[0] - undeformed_positions[1])
        deformed_length = np.linalg.norm(deformed_positions[0] - deformed_positions[1])

        if deformed_length == 0:
            return np.zeros((4, 4))

        dx = deformed_positions[1, 0] - deformed_positions[0, 0]
        dy = deformed_positions[1, 1] - deformed_positions[0, 1]

        scale = (deformed_length - rest_length) / deformed_length
        factor = 1 / (deformed_length ** 3)

        hessian = np.zeros((4, 4))
        hessian[0, 0] = scale / deformed_length - dx * dx * factor
        hessian[1, 1] = scale / deformed_length - dy * dy * factor
        hessian[0, 1] = hessian[1, 0] = -dx * dy * factor
        hessian += 1e-6 * np.eye(hessian.shape[0])  # Regularization

        return hessian

#%% Mesh class
class FEMMesh:
    def __init__(self, V, F, energy, stencil):
        self.V = V
        self.F = F
        self.energy = energy
        self.stencil = stencil
        self.elements = self.stencil.ExtractElementsFromMesh(F)
        self.X = self.V.copy()
        self.nV = self.V.shape[0]

    def compute_energy(self,x):
        energy = 0
        for element in self.elements:
            Xi = self.X[element,:]
            xi = x[element,:]
            energy += self.energy.energy(Xi, xi)
        return energy
    

    # def compute_gradient(self,x):
    #     grad = np.zeros(2 * x.shape[0])
    #     for element in self.elements:
    #         Xi = self.X[element, :]
    #         xi = x[element, :]

    #         gi = self.energy.gradient(Xi, xi)
            
    #         grad[2 * element[0]] += gi[0]
    #         grad[2 * element[0] + 1] += gi[1]
    #         grad[2 * element[1]] += gi[2]
    #         grad[2 * element[1] + 1] += gi[3]
    #     return grad

    # def compute_hessian(self, x):
    #     spring_hessian = np.zeros((2 * self.nV, 2 * self.nV))
    #     for element in self.elements:
    #         Xi = self.X[element, :]
    #         xi = x[element, :]
            
    #         # Calculate the Hessian for the current element
    #         norm = np.linalg.norm(x[element[0]] - x[element[1]])
    #         if norm < 1e-8:  # Avoid division by zero
    #             norm = 1e-8
    #         hess = self.energy.hessian(Xi, xi) / norm
            
    #         # Determine the indices for the vertices
    #         indices = [2 * element[0], 2 * element[0] + 1, 2 * element[1], 2 * element[1] + 1]
            
    #         # Update the spring Hessian matrix at the specified indices
    #         spring_hessian[np.ix_(indices, indices)] += hess
        
    #     # Add regularization to improve conditioning
    #     spring_hessian += 1e-3 * np.eye(spring_hessian.shape[0])  # Regularization term

    #     # Debugging: Check symmetry
    #     assert np.allclose(spring_hessian, spring_hessian.T), "Hessian matrix is not symmetric!"

    #     return spring_hessian

    def compute_gradient(self, x):
        grad = np.zeros(2 * x.shape[0])
        for element in self.elements:
            Xi = self.X[element, :]
            xi = x[element, :]
            gi = self.energy.gradient(Xi, xi)
            grad[2 * element[0]] += gi[0]
            grad[2 * element[0] + 1] += gi[1]
            grad[2 * element[1]] += gi[2]
            grad[2 * element[1] + 1] += gi[3]

        # Zero out gradient for pinned vertices
        for v_id in pinned_vertices:
            grad[2 * v_id:2 * v_id + 2] = 0
        return grad
    

    # def compute_hessian(self, x):
    #     hessian = np.zeros((2 * self.nV, 2 * self.nV))
    #     for element in self.elements:
    #         Xi = self.X[element, :]
    #         xi = x[element, :]
    #         hess = self.energy.hessian(Xi, xi)
    #         indices = [2 * element[0], 2 * element[0] + 1, 2 * element[1], 2 * element[1] + 1]
    #         hessian[np.ix_(indices, indices)] += hess

    #     # Zero out rows and columns for pinned vertices
    #     for v_id in pinned_vertices:
    #         idx = [2 * v_id, 2 * v_id + 1]
    #         hessian[idx, :] = 0
    #         hessian[:, idx] = 0
    #         hessian[idx, idx] = np.eye(2)  # Preserve diagonal to prevent singularity
    #     return hessian

    def compute_hessian(self, x):
        hessian = np.zeros((2 * self.nV, 2 * self.nV))
        for element in self.elements:
            Xi = self.X[element, :]
            xi = x[element, :]
            hess = self.energy.hessian(Xi, xi)
            indices = [2 * element[0], 2 * element[0] + 1, 2 * element[1], 2 * element[1] + 1]
            hessian[np.ix_(indices, indices)] += hess

        # Zero out rows and columns for pinned vertices
        for v_id in pinned_vertices:
            idx = [2 * v_id, 2 * v_id + 1]  # Indices for x and y components of the vertex

            # Clear rows and columns
            hessian[idx, :] = 0  # Clear rows
            hessian[:, idx] = 0  # Clear columns

            # Preserve diagonal entries for stability
            for i in idx:
                hessian[i, i] = 1  # Set diagonal to 1 to prevent singularity

        return hessian


            
#%% Optimization
class MeshOptimizer:
    def __init__(self, femMesh):
        self.femMesh = femMesh
        self.SearchDirection = self.Newton
        self.LineSearch = self.BacktrackingLineSearch
        self.pinned_vertices = []  # List of pinned vertex indices

    def BacktrackingLineSearch(self, x, d, alpha=1):
        if d is None:
            print("No direction provided. Skipping line search.")
            return x, alpha  # Return the current x and alpha unchanged

        x0 = x.copy()
        f0 = self.femMesh.compute_energy(x0)
        while self.femMesh.compute_energy(x0 + alpha * d.reshape((self.femMesh.nV, 2))) > f0:
            alpha *= 0.5
        return x0 + alpha * d.reshape((self.femMesh.nV, 2)), alpha

    def apply_pinned_constraints(self, x):
        x_flat = x.flatten()  # Flatten the 2D array for easier indexing
        for i in self.pinned_vertices:
            x_flat[2 * i] = original_positions[i, 0]  # x-coordinate
            x_flat[2 * i + 1] = original_positions[i, 1]  # y-coordinate
        return x_flat.reshape(x.shape)  # Reshape to original 2D dimensions


    def GradientDescent(self, x):
        d = self.femMesh.compute_gradient(x)
        return -d

    def Newton(self, x):
        grad = self.femMesh.compute_gradient(x)
        hess = self.femMesh.compute_hessian(x)

        # Regularization for singular Hessian
        epsilon = 1e-6
        hess += epsilon * np.eye(hess.shape[0])

        try:
            d = -np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            print("Hessian is singular, skipping Newton step.")
            return None

        return d

    def step(self, x):
        d = self.SearchDirection(x)
        if d is None:
            print("No valid direction found. Skipping this step.")
            return x  # Return the same x without changes
        new_x, alpha = self.LineSearch(x, d)
        return self.apply_pinned_constraints(new_x)  # Enforce pinned constraints

    def optimize(self, x, max_iter=100, tol=1e-6):
        """
        Optimize the mesh while enforcing pinned vertex constraints.
        """
        global plt
        method_type = "Gradient Descent" if self.SearchDirection == self.GradientDescent else "Newton"
        print(f"Starting optimization using {method_type}...")

        for i in range(max_iter):
            x_prev = x.copy()
            x = self.step(x)

            # Compute energy for this iteration
            current_energy = self.femMesh.compute_energy(x)
            print(f"Energy of {method_type} Iteration {i + 1} = {current_energy}")
            plt.clear()  # Clear the plotter
            updated_mesh = vd.Mesh([x, self.femMesh.F]).linecolor("black")
            plt += updated_mesh
            plt += vd.Points(x[pinned_vertices, :], r=10, c="blue")  # Highlight pinned vertices
            plt.render()

            # Check for convergence based on tolerance
            if current_energy < tol:
                print(f"Converged after {i + 1} iterations.")
                break

            # Stop if x does not change significantly
            if np.allclose(x, x_prev, atol=1e-9):
                print(f"No significant update after {i + 1} iterations. Stopping.")
                break

        print(f"Finished optimization using {method_type}. Final Energy = {current_energy}.")
        return x


#%% Main program
vertices = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]) # square
tris = tr.triangulate({"vertices":vertices[:,0:2]}, f'qa0.01') # triangulate the square
V = tris['vertices'] # get the vertices of the triangles
F = tris['triangles'] # get the triangles

message = vd.Text2D("", pos="top-left", c="black")

#-------------Task 1.2--------------------
# segments = [[0, 1], [1, 2], [2, 3], [3, 0]]  # Connect vertices to form a square

# # Input for triangulation
# triangulation_input = {
#     "vertices": vertices,
#     "segments": segments
# }

# tris_no_interior  = tr.triangulate(triangulation_input, 'p')
# # V = tris_no_interior['vertices']
# # F = tris_no_interior['triangles']

# tris_approx_100 = tr.triangulate({"vertices": vertices[:, 0:2]}, 'qa0.009')
# V = tris_approx_100['vertices']
# F = tris_approx_100['triangles']

# boundary_vertex_indices = set(np.unique(tris_no_interior['segments']))  # Boundary indices
# interior_vertex_indices = set(range(len(V))) - boundary_vertex_indices  # Interior vertices

# # # Print the number of interior vertices
# print(f"Number of vertices: {len(V)}")
# print(f"Number of interior vertices: {len(interior_vertex_indices)}")


# #-------------Task 1.1----------------------
# # Define a function for the Star
# def create_star():
#     # First triangle 
#     triangle1 = np.array([[0, np.sqrt(3) / 3], [0.5, -np.sqrt(3) / 6], [-0.5, -np.sqrt(3) / 6]])

#     # Second triangle 
#     triangle2 = np.array([[0, -np.sqrt(3) / 3], [0.5, np.sqrt(3) / 6], [-0.5, np.sqrt(3) / 6]])

#     return triangle1, triangle2

# # Create the Star 
# triangle1, triangle2 = create_star()

# # Perform triangulation for each triangle
# tris1 = tr.triangulate({"vertices": triangle1}, 'qa0.01')
# tris2 = tr.triangulate({"vertices": triangle2}, 'qa0.01')

# # Extract vertices and faces for each triangle
# V1, F1 = tris1['vertices'], tris1['triangles']
# V2, F2 = tris2['vertices'], tris2['triangles']


# #-----------------Task 2.1----------------------
# def redraw():
#     plt.remove("Mesh")
#     mesh = vd.Mesh([V,F]).linecolor('black')
#     plt.add(mesh)
#     plt.remove("Points")
#     plt.add(vd.Points(V[pinned_vertices,:],r=10))
#     plt.render()

# def OnLeftButtonPress(event):
#     if event.object is None:  # Mouse hits nothing
#         message.text("Mouse hits nothing")
#     elif isinstance(event.object, vd.mesh.Mesh):  # Mouse hits the mesh
#         Vi = vdmesh.closest_point(event.picked3d, return_point_id=True)
#         message.text(f"Mouse hits the mesh\nCoordinates: {event.picked3d}\nPoint ID: {Vi}")
#         if Vi not in pinned_vertices:
#             pinned_vertices.append(Vi)
#         else:
#             pinned_vertices.remove(Vi)
#     redraw()


#---------------Task 2.2-------------------------
pinned_vertices = []  # Use a list to track pinned vertex IDs
action_stack = []  # Stack to track actions for undo
original_positions = V.copy()  # Store original positions
dragging_vertex = None


# Function to redraw the plotter
def redraw():
    plt.remove("Mesh")
    mesh = vd.Mesh([V, F]).linecolor('black')
    plt.add(mesh)
    plt.remove("Points")
    if pinned_vertices:
        plt.add(vd.Points(V[pinned_vertices, :], r=12, c="red"))
    plt.render()

# Define the Left Button Press callback
def OnLeftButtonPress(event):
    global pinned_verticesdd
    if event.object is None:  # Mouse hits nothing
        message.text("Mouse hits nothing")
    elif isinstance(event.object, vd.mesh.Mesh):  # Mouse hits the mesh
        Vi = vdmesh.closest_point(event.picked3d, return_point_id=True)
        if keyboard.is_pressed("ctrl"):  # Check if Ctrl is pressed
            if Vi not in pinned_vertices:  # Pin the vertex
                pinned_vertices.append(Vi)  # Add to pinned vertices list
                action_stack.append(("pin", Vi))  # Record action
                message.text(f"Vertex {Vi} pinned")
            else:  # Unpin the vertex
                pinned_vertices.remove(Vi)  # Remove from pinned vertices list
                action_stack.append(("unpin", Vi))  # Record action
                message.text(f"Vertex {Vi} unpinned")
            redraw()

# Define the Mouse Move callback for dragging
def OnMouseMove(event):
    global dragging_vertex

    # Start dragging when "D" is pressed and a pinned vertex is clicked
    if keyboard.is_pressed("d"):
        if dragging_vertex is None and isinstance(event.object, vd.mesh.Mesh):  # No vertex locked yet
            if event.picked3d is not None:  # Ensure event.picked3d is not None
                Vi = vdmesh.closest_point(event.picked3d, return_point_id=True)
                if Vi in pinned_vertices:  # Start dragging this vertex
                    dragging_vertex = Vi  # Lock this vertex for dragging

        if dragging_vertex is not None and event.picked3d is not None:  # A vertex is being dragged, and a valid position is detected
            previous_position = V[dragging_vertex].copy()  # Save previous position for undo
            new_position = event.picked3d[:2]  # Get new position from the event

            # Enforce movement within the bounds of the original square polygon
            min_x, min_y = np.min(vertices, axis=0)  # Bottom-left corner of the square
            max_x, max_y = np.max(vertices, axis=0)  # Top-right corner of the square

            # Clamp the new position to stay within bounds
            clamped_x = np.clip(new_position[0], min_x, max_x)
            clamped_y = np.clip(new_position[1], min_y, max_y)

            # Update the vertex position
            V[dragging_vertex] = [clamped_x, clamped_y]

            # Record the movement action for undo functionality
            action_stack.append(("move", dragging_vertex, previous_position))

            # Provide feedback to the user
            message.text(f"Pinned vertex {dragging_vertex} moved to {V[dragging_vertex]}")
        redraw()
    else:
        # If "D" is released, stop dragging
        if dragging_vertex is not None:
            message.text(f"Stopped dragging vertex {dragging_vertex}")
            dragging_vertex = None  # Unlock the vertex

# Define the Key Press callback for resetting vertex positions
def OnKeyPress(event):
    global optimizer, V

    if event.keypress in ['g', 'G']: # Start Gradient Descent
        print("Running Gradient Descent with Pinned Vertices...")
        optimizer.SearchDirection = optimizer.GradientDescent  # Set search direction
        final_x = optimizer.optimize(V.copy())  # Run optimization
        V = final_x.copy()  # Update the global vertex positions
        final_energy = optimizer.femMesh.compute_energy(final_x)
        print(f"Final Energy (Gradient Descent with Pinned Vertices): {final_energy:.6f}")

        # Visualize the results
        optimized_mesh = vd.Mesh([final_x, F]).linecolor("blue")
        plt.add(optimized_mesh)
        plt.add(vd.Points(final_x[optimizer.pinned_vertices, :], r=12, c="red"))  # Highlight pinned vertices
        plt.render()


    if event.keypress in ['n', 'N']:  # Start Newton's Method
        print("Running Newton's Method with Pinned Vertices...")
        optimizer.SearchDirection = optimizer.Newton  # Set search direction
        final_x = optimizer.optimize(V.copy())  # Run optimization
        V = final_x.copy()  # Update the global vertex positions
        final_energy = optimizer.femMesh.compute_energy(final_x)
        print(f"Final Energy (Newton's Method with Pinned Vertices): {final_energy:.6f}")

        # Visualize the results
        optimized_mesh = vd.Mesh([final_x, F]).linecolor("green")
        plt.add(optimized_mesh)
        plt.add(vd.Points(final_x[optimizer.pinned_vertices, :], r=12, c="red"))  # Highlight pinned vertices
        plt.render()


    if event.keypress in ['z', 'Z']:
        if action_stack:
            last_action = action_stack.pop()  # Get the last action
            if last_action[0] == "pin":  # Undo pinning
                pinned_vertices.remove(last_action[1])  # Remove the pin
                message.text(f"Undo: Vertex {last_action[1]} unpinned")
            elif last_action[0] == "unpin":  # Undo unpinning
                pinned_vertices.append(last_action[1])  # Re-pin the vertex
                message.text(f"Undo: Vertex {last_action[1]} pinned back")
            elif last_action[0] == "move":  # Undo moving
                Vi, previous_position = last_action[1], last_action[2]
                V[Vi] = previous_position  # Restore the previous position
                message.text(f"Undo: Vertex {Vi} reset to {previous_position}")
        else:
            message.text("Nothing to undo!")
        redraw()
    elif event.keypress in ['c', 'C']:  # Clear all pinned vertices and reset positions
        if pinned_vertices or not np.array_equal(V, original_positions):
            pinned_vertices.clear()
            action_stack.clear()  # Clear action history
            V[:] = original_positions  # Reset positions
            message.text("All pinned vertices cleared, positions reset!")
        else:
            message.text("No changes to reset.")
        redraw()

guide_text = """
- Left Click + Ctrl: Pin/Unpin a vertex.
- Move Mouse + 'D': Drag a pinned vertex.
- Press 'Z': Undo the last action.
- Press 'C': Clear all pins and reset positions.
- Press 'g': Run Gradient Descent with Pinned Vertices.
- Press 'n': Run Newton's Method with Pinned Vertices.
"""

guide = vd.Text2D(guide_text, pos="bottom-left", c="black" ,font="VictorMono",s=0.8)


plt = vd.Plotter(size="full")
vdmesh = vd.Mesh([V,F]).linecolor('black')
plt += vdmesh
plt += vd.Points(V[pinned_vertices,:], r=10, c="red")

#----------Task 3.1-------------------
# # Clear pinned vertices to ensure none are present
# pinned_vertices.clear()

# # Initialize FEMMesh and Optimizer
# zero_length_spring_energy = ZeroLengthSpringEnergy()  # Use ZeroLengthSpringEnergy
# femMesh = FEMMesh(V, F, zero_length_spring_energy, EdgeStencil())  # Initialize FEMMesh
# optimizer = MeshOptimizer(femMesh)  # Create optimizer

# # Run optimization with Gradient Descent
# print("Running with Gradient Descent:")
# optimizer.SearchDirection = optimizer.GradientDescent
# final_x_gd = optimizer.optimize(V.copy())
# V = final_x_gd.copy()
# final_energy_gd = femMesh.compute_energy(final_x_gd)
# print(f"Final Energy (Gradient Descent): {final_energy_gd:.6f}")

# #Visualize Results for Gradient Descent
# gradient_descent_mesh = vd.Mesh([final_x_gd, F]).linecolor("blue")
# plt.add(gradient_descent_mesh)

# # #Run optimization with Newton's Method
# # print("\nRunning with Newton's Method:")
# # optimizer.SearchDirection = optimizer.Newton
# # x = V.copy()  # Reset the mesh to original positions
# # final_x_newton = optimizer.optimize(x)
# # V = final_x_newton.copy()
# # final_energy_newton = femMesh.compute_energy(final_x_newton)
# # print(f"Final Energy (Newton's Method): {final_energy_newton:.6f}")

# # #Visualize Results for Newton's Method
# # newton_method_mesh = vd.Mesh([final_x_newton, F]).linecolor("blue")
# # plt.add(newton_method_mesh)

#-----------------------------------------------------


#----------------------Task 3.2 ------------------------
# # Clear pinned vertices to ensure none are present
# pinned_vertices.clear()

# spring_energy = SpringEnergy()  # Use SpringEnergy
# femMesh = FEMMesh(V, F, spring_energy, EdgeStencil())  # Initialize FEMMesh with SpringEnergy
# optimizer = MeshOptimizer(femMesh)  # Create optimizer


# print("Running with Gradient Descent for SpringEnergy:")
# optimizer.SearchDirection = optimizer.GradientDescent
# final_x_gd = optimizer.optimize(V.copy())
# V = final_x_gd.copy()
# final_energy_gd = femMesh.compute_energy(final_x_gd)
# print(f"Final Energy (Gradient Descent with SpringEnergy): {final_energy_gd:.6f}")

# gradient_descent_mesh = vd.Mesh([final_x_gd, F]).linecolor("blue")
# plt.add(gradient_descent_mesh)

# print("\nRunning with Newton's Method for SpringEnergy:")
# optimizer.SearchDirection = optimizer.Newton
# final_x_newton = optimizer.optimize(V.copy())
# V = final_x_newton.copy()
# final_energy_newton = femMesh.compute_energy(final_x_newton)
# print(f"Final Energy (Newton's Method with SpringEnergy): {final_energy_newton:.6f}")

# newton_method_mesh = vd.Mesh([final_x_newton, F]).linecolor("blue")
# plt.add(newton_method_mesh)

#-------------------------------------------------------

#--- Task 3.3 -----------

# Initialize FEMMesh and Optimizer
spring_energy = SpringEnergy()  # Use SpringEnergy
femMesh = FEMMesh(V, F, spring_energy, EdgeStencil())  # Initialize FEMMesh with SpringEnergy
optimizer = MeshOptimizer(femMesh)  # Create optimizer

# Ensure original_positions is set
original_positions = V.copy()  # Store the original positions of vertices



plt.add_callback('LeftButtonPress', OnLeftButtonPress) # add Keyboard callback
plt.add_callback('MouseMove', OnMouseMove)  # Add mouse move callback
plt.add_callback('KeyPress', OnKeyPress)  # Add key press callback


# mesh1 = vd.Mesh([V1, F1]).linecolor('white').color('pink')  # Upper triangle
# mesh2 = vd.Mesh([V2, F2]).linecolor('white').color('pink')  # Lower triangle 
# plt.add(mesh1)
# plt.add(mesh2)
# plt += vd.Points(np.vstack([V1, V2])[pinned_vertices, :])

plt += message
plt += guide
plt.user_mode('2d').show().close()

# %%