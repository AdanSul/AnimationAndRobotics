#%%
import vedo as vd
vd.settings.default_backend= 'vtk'
import numpy as np
import time


#%% class for a robot arm
def Rot(angle, axis):
    # calculate the rotation matrix for a given angle and axis using Rodrigues' formula
    # return a 3x3 numpy array
    # also see scipy.spatial.transform.Rotation.from_rotvec
    axis = np.array(axis)
    axis = axis/np.linalg.norm(axis)
    I = np.eye(3)
    K = np.array([[0, -axis[2], axis[1]],
                    [axis[2], 0, -axis[0]],
                    [-axis[1], axis[0], 0]])
    R = I + np.sin(angle)*K + (1-np.cos(angle))*np.dot(K,K)
    return R
    
class SimpleArm:
    #---------- Task 1.1 -------------
    def __init__(self, n=3, link_lengths=None):
        self.n = n  # Number of links
        self.angles = [0] * self.n  # Joint angles initialized to 0

        # Set link lengths
        if link_lengths is None:
            self.link_lengths = [1] * self.n  # Default to unit lengths
        elif not isinstance(link_lengths, (list, tuple)) or len(link_lengths) != n:
            raise ValueError("The length of 'link_lengths' must match the number of links 'n'.")
        else:
            self.link_lengths = link_lengths

        # Initialize joint positions in local coordinates
        self.Jl = np.zeros((self.n + 1, 3))
        for i in range(1, n + 1):
            self.Jl[i, :] = np.array([sum(self.link_lengths[:i]), 0, 0])

        # Initialize joint positions in world coordinates
        self.Jw = np.zeros((self.n + 1, 3))
        self.FK()

    def FK(self, angles=None):
        # calculate the forward kinematics of the arm

        # angles is a list of joint angles. If angles is None, the current joint angles are used
        if angles is not None:
            self.angles = angles

        # Initial rotation matrix
        Ri = np.eye(3)

        #--------- Task 1.2-----------------
        # Update joint positions based on rotations
        for i in range(1, self.n + 1):
            Ri = Rot(self.angles[i - 1], [0, 0, 1]) @ Ri
            self.Jw[i, :] = Ri @ (self.Jl[i, :] - self.Jl[i - 1, :]) + self.Jw[i - 1, :]

        return self.Jw[-1, :]  # Return the position of the end effector
   
    #----------------Task 3.1 ----------------------------
    # Update to the IK method to include Gauss-Newton optimization
    def IK(self, target, method='gradient_descent', max_iterations=1000, learning_rate=0.01, threshold=1e-3):

        target = np.array(target)
        max_reach = sum(self.link_lengths)
        distance_to_target = np.linalg.norm(target)
        
        if distance_to_target > max_reach:
            print("Target is unreachable. Moving to the closest reachable position.")
            target = target / distance_to_target * max_reach

        for iteration in range(max_iterations):
            current_position = self.FK()
            loss = np.linalg.norm(current_position - target)
            if loss < threshold:
                print(f"Converged after {iteration} iterations.")
                return self.angles
            
            J = self.VelocityJacobian()
            residual = target - current_position
            
            if method == 'gradient_descent':
                gradient = 2 * J.T @ residual
                self.angles += learning_rate * gradient
            
            elif method == 'gauss_newton':
                # Use Gauss-Newton update
                lambda_damping = 0.01  # Damping factor
                J_damped = np.linalg.inv(J.T @ J + lambda_damping * np.eye(J.shape[1])) @ J.T
                delta_angles = J_damped @ residual
                self.angles += delta_angles
            
            else:
                raise ValueError("Invalid method. Choose 'gradient_descent' or 'gauss_newton'.")
        
        # # Visualization update during each iteration
        # plt.remove("Assembly")
        # plt.add(self.draw())  # Update the arm's visualization
        # plt.render()
        # print(f"Iteration {iteration}: Loss = {loss:.4f}")
        # time.sleep(0.1)

        return self.angles


    def VelocityJacobian(self, angles=None):
        # calculate the velocity jacobian of the arm
        # return a 3x3 numpy array

        # Use the provided angles if given
        if angles is not None:
            self.FK(angles)
        # Ensure the arm has exactly 3 joints
        if self.n != 3:
            raise ValueError("VelocityJacobian is implemented for a 3-joint arm only.")
        # Initialize Jacobian and Position of the end effector
        J = np.zeros((3, 3))
        p_end = self.Jw[-1, :]

        Ri = np.eye(3)  # Rotation matrix starts as identity
        for i in range(self.n):
            # Axis of rotation (z-axis for planar rotation)
            z_i = Ri @ np.array([0, 0, 1])
            # Position of the current joint
            p_i = self.Jw[i, :]
            # Compute the contribution to the Jacobian
            J[:, i] = np.cross(z_i, p_end - p_i)
            # Update rotation matrix for the next joint
            Ri = Rot(self.angles[i], [0, 0, 1]) @ Ri

        return J

    def draw(self):
        vd_arm = vd.Assembly()
        vd_arm += vd.Sphere(pos = self.Jw[0,:], r=0.05)
        for i in range(1,self.n+1):
            vd_arm += vd.Cylinder(pos = [self.Jw[i-1,:], self.Jw[i,:]], r=0.02)
            vd_arm += vd.Sphere(pos = self.Jw[i,:], r=0.05)
        return vd_arm
    
    def draw_with_jacobian(self):
        vd_arm = self.draw()
        J = self.VelocityJacobian()
        
        # Add Jacobian vectors for visualization
        for i in range(3):  # Only 3 columns as requested
            start = self.Jw[i, :]
            end = start + J[:, i]
            vd_arm += vd.Arrow(start, end, c="green")
        
        return vd_arm 

#%%
activeJoint = 0
IK_target = [1,1,0]

def OnSliderAngle(widget, event):
    global activeJoint
    arm.angles[activeJoint] = widget.value
    arm.FK()
    plt.remove("Assembly")
    plt.add(arm.draw())
    #plt.add(arm.draw_with_jacobian())

    plt.render()

def OnCurrentJoint(widget, event):
    global activeJoint
    activeJoint = round(widget.value)
    sliderAngle.value = arm.angles[activeJoint]

# Add a global variable to track the current feedback text object
current_text = None
current_distance_text = None

#-----------------------Task 3.1 ---------------------------------
def LeftButtonPress(evt):
    global IK_target, current_text, current_distance_text
    IK_target = evt.picked3d

    if IK_target is not None:
       
        # Remove any existing target sphere
        plt.remove("Sphere")
        max_reach = sum(arm.link_lengths)  # Calculate the arm's maximum reach
        distance_to_target = np.linalg.norm(IK_target)

        # Add a new sphere to mark the target position
        if distance_to_target > max_reach:
            print("Target is unreachable. Clamping to the workspace boundary.")
            plt.add(vd.Sphere(pos=IK_target, r=0.05, c='white'))
            feedback_message = "Target is unreachable and clamped to the workspace boundary."
            IK_target = IK_target / distance_to_target * max_reach  # Clamp target
        else:
            print("Target is reachable.")
            plt.add(vd.Sphere(pos=IK_target, r=0.05, c='blue'))
            feedback_message = "Target is reachable and within the workspace."

        # Perform IK step by step and visualize each iteration
        for iteration in range(500): 
            current_position = arm.FK()
            loss = np.linalg.norm(current_position - IK_target)
            if loss < 1e-3:  # Convergence threshold
                print(f"Converged after {iteration} iterations.")
                break

            # Update the arm's position using the selected method
            arm.IK(IK_target, method=optimization_method, max_iterations=1)  # Run one iteration at a time

            # Update the visualization after each iteration
            plt.remove("Assembly")
            plt.add(arm.draw())
            plt.render()

        # Add feedback text
        if current_text is not None:
            plt.remove(current_text)
        current_text = vd.Text2D(feedback_message, c="red", pos="bottom-center")
        plt.add(current_text)

# Add buttons for optimization method selection
def OnSelectGradientDescent(widget, event):
    global optimization_method
    if optimization_method != 'gradient_descent':  # Avoid redundant state changes
        optimization_method = 'gradient_descent'
        print("Optimization method set to: Gradient Descent")
        updateMethodDisplay()

def OnSelectGaussNewton(widget, event):
    global optimization_method
    if optimization_method != 'gauss_newton':  # Avoid redundant state changes
        optimization_method = 'gauss_newton'
        print("Optimization method set to: Gauss-Newton")
        updateMethodDisplay()

# Function to update the display of the selected method
def updateMethodDisplay():
    global current_method_text
    if current_method_text:
        plt.remove(current_method_text)
    current_method_text = vd.Text2D(
        f"Selected Method: {optimization_method}",
        c="white",
        pos="top-center",
        s=1.2
    )
    plt.add(current_method_text)

# Global variable to track selected method
optimization_method = 'gradient_descent'
current_method_text = None

arm = SimpleArm(3)

plt = vd.Plotter()
plt +=arm.draw()

plt += vd.Sphere(pos = IK_target, r=0.05, c='b').draggable(True)
plt += vd.Plane(s=[2.1*arm.n,2.1*arm.n]) # a plane to catch mouse events
sliderCurrentJoint = plt.add_slider(OnCurrentJoint, 0, arm.n-1, 0, title="Current joint", pos=3, delayed=True)
sliderAngle =  plt.add_slider(OnSliderAngle,-np.pi,np.pi,0., title="Joint Angle", pos=4)
plt.add_callback('LeftButtonPress', LeftButtonPress) # add Mouse callback

# Add buttons for optimization method selection
gradient_button = plt.add_button(OnSelectGradientDescent, pos=(0.35, 0.9), states=["Gradient Descent"], c=["white"], bc=["darkblue"])
gauss_button = plt.add_button(OnSelectGaussNewton, pos=(0.35, 0.83), states=["Gauss-Newton"], c=["white"], bc=["darkblue"])

# Add a text display to show the current selected method
updateMethodDisplay()

plt.user_mode('2d').show(zoom="tightest")

plt.close()

# %%