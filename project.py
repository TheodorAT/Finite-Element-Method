import numpy as np
import calfem.core as cfc
import calfem.vis_mpl as cfv
import calfem.geometry as cfg
import calfem.mesh as cfm
import calfem.utils as cfu

## Defining functions, later used in the program
def plantml(ex: np.array, ey: np.array, s: float):
    """
    Computes the integral of the form-functions over a 3-node triangle element
    Me = int(s*N^T*N)dA

    Inputs:
        ex: element x-coordinates
        ey: element y-coordinates
        s: constant scalar, e.g. density*thickness

    Outputs:
        Me: integrated element matrix
    """
    if not ex.shape == (3,) or not ey.shape == (3,):
        raise Exception("Incorrect shape of ex or ey: {0}, {1} but should be (3,)".format(ex.shape, ey.shape))

    # Compute element area
    Cmat = np.vstack((np.ones((3, )), ex, ey))
    A = np.linalg.det(Cmat)/2

    # Set up quadrature
    g1 = [0.5, 0.0, 0.5]
    g2 = [0.5, 0.5, 0.0]
    g3 = [0.0, 0.5, 0.5]
    w = (1/3)

    # Perform numerical integration
    Me = np.zeros((3, 3))
    for i in range(0, 3):
        Me += w*np.array([
            [g1[i]**2, g1[i]*g2[i], g1[i]*g3[i]],
            [g2[i]*g1[i], g2[i]**2, g2[i]*g3[i]],
            [g3[i]*g1[i], g3[i]*g2[i], g3[i]**2]])
    Me *= A*s
    return Me

def defineGeometry():
    ## Drawing the geometry
    g = cfg.Geometry()

    ## The nylon part:
    g.point([0, 0])             # point 0
    g.point([c+d, 0])           # point 1, also used in copper lines
    g.point([c+d, 0.3*L])       # point 2
    g.point([a+t, 0.3*L])       # point 3
    g.point([a+t, 0.3*L - h])   # point 4
    g.point([a, 0.3*L - h])     # point 5
    g.point([a, 0.3*L])         # point 6
    g.point([0, 0.3*L])         # point 7, also used in copper lines

    g.spline([0,1])                             # curve 0
    g.spline([1,2])                             # curve 1
    g.spline([2,3])                             # curve 2
    g.spline([3,4])                             # curve 3
    g.spline([4,5])                             # curve 4
    g.spline([5,6])                             # curve 5
    g.spline([6,7])                             # curve 6
    g.spline([7,0], marker=mark_immovable)      # curve 7

    ## Copper part:
    g.point([c + d + a, 0])         # point 8
    g.point([L-2*d, 0.3*L-d])       # point 9
    g.point([L, 0.3*L-d])           # point 10
    g.point([L, 0.3*L])             # point 11
    g.point([L-2*d, 0.3*L])         # point 12
    g.point([c + d + a, d])         # point 13
    g.point([c + d + a, c + a - d]) # point 14
    g.point([c + a, c + a])         # point 15
    g.point([a, c + a])             # point 16
    g.point([a, c + a + b])         # point 17
    g.point([0, c + a + b])         # point 18
    g.point([0, c + a])             # point 19

    g.spline([1, 8])                            # curve 8
    g.spline([8, 9], marker = mark_Newton)      # curve 9
    g.spline([9, 10], marker = mark_Newton)     # curve 10
    g.spline([10, 11], marker = mark_immovableX)# curve 11
    g.spline([11, 12], marker = mark_Newton)    # curve 12
    g.spline([12, 13], marker = mark_Newton)    # curve 13
    g.spline([13, 14], marker = mark_Newton)    # curve 14
    g.spline([14, 15], marker = mark_Newton)    # curve 15
    g.spline([15, 16], marker = mark_Newton)    # curve 16
    g.spline([16, 17], marker = mark_Newton)    # curve 17
    g.spline([17, 18], marker = mark_immovableY)# curve 18
    g.spline([18, 19], marker = mark_hFlow)     # curve 19
    g.spline([19, 7], marker = mark_immovable)  # curve 20

    g.surface([0, 1, 2, 3, 4, 5, 6, 7], marker = mark_nylon)
    g.surface([8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 6, 5, 4, 3, 2, 1], marker = mark_copper)
    return g

def drawGeometry(g: cfg.Geometry):
    cfv.draw_geometry(g)
    cfv.addText("x [m]", [0.0022, -0.0003])
    cfv.addText("y [m]", [-0.0006, 0.0027])
    cfv.showAndWait()

def drawMesh():
    cfv.figure(fig_size=(10, 10))
    cfv.draw_mesh(
        coords = coords,
        edof = edof,
        dofs_per_node=mesh.dofs_per_node,
        el_type=mesh.el_type,
        filled=True,
        title="Mesh of geometry"
    )
    cfv.addText("x [m]", [0.00225, -0.0004])
    cfv.addText("y [m]", [-0.0006, 0.0028])
    cfv.showAndWait()


## Defining material properties:
# Nylon
E_nylon = 3.00           # Young modulus [GPa]
v_nylon = 0.39           # Poisson ratio [-]
alpha_nylon = 80e-6      # Expansion coefficient [1/K]
rho_nylon = 1100         # Density [kg/m^3]
cp_nylon = 1500          # Specific heat capacity [J/(kg*K)]
k_nylon = 0.26           # Thermal conductivity [W/(m*K)]
D_nylon = np.matrix([
  [k_nylon,0.],
  [0.,k_nylon]
])
D_nylon_strain = cfc.hooke(2,E_nylon, v_nylon)[np.ix_([0, 1, 3], [0, 1, 3])]


# Copper
E_copper = 128            # Young modulus [GPa]
v_copper = 0.36           # Poisson ratio [-]
alpha_copper = 17.6e-6    # Expansion coefficient [1/K]
rho_copper = 8930         # Density [kg/m^3]
cp_copper = 386           # Specific heat capacity [J/(kg*K)]
k_copper = 385            # Thermal conductivity [W/(m*K)]
D_copper = np.matrix([
  [k_copper,0.],
  [0.,k_copper]
])
D_copper_strain = cfc.hooke(2, E_copper, v_copper)[np.ix_([0, 1, 3], [0, 1, 3])]


## Definie geometry
L = 0.005           # L in meters
a = 0.1*L
b = 0.1*L
c = 0.3*L
d = 0.05*L
h = 0.15*L
t = 0.05*L
thickness = L


## Define heat flow and surrounding temperature
heatFlow = -1e5             # Heat flow into the module in one part of the boundary [W/m^2]
T_surrounding = 18 + 273    # Temperature in the surrounding environment [Degrees C]
alpha_convectionCoeff = 40  # Convection coefficient for the boundary subject to Newton convection [W/(m^2K)]
T_initial = 18+273          # Initial temperature [Degrees C]


## Define marker constants instead of using numbers in the code
# Surface markers
mark_nylon = 55
mark_copper = 66
# Flow boundary conditions
mark_hFlow = 70
mark_Newton = 90
# Strain boundary conditions, these are used for the tension problem.
mark_immovable = 100
mark_immovableX = 110
mark_immovableY = 120


## Define and draw geometry
g = defineGeometry()
drawGeometry(g)


#### The stationary heat problem ####
## Generate and draw mesh
mesh = cfm.GmshMesh(g)
mesh.elType = 2                 # Type of mesh
mesh.dofs_per_node = 1          # Degrees of freedom per node
mesh.el_size_factor = 0.03      # Element size factor
mesh.return_boundary_elements = True
coords, edof, dofs, boundary_dofs, element_markers, boundaryElements = mesh.create()
drawMesh()


## Mirroring the coordinates to plot the whole solution, in y = 0.5L and x = L
yAxis = 0.5*L
xAxis = L
coordRows, coordCols  = coords.shape
coords_mirrorY = np.zeros([coordRows, coordCols])
coords_mirrorX = np.zeros([coordRows, coordCols])
coords_mirrorXY = np.zeros([coordRows, coordCols])

coords_mirrorY[:, 0] = coords[:, 0]
coords_mirrorY[:, 1] = (2*yAxis - coords[:, 1])

coords_mirrorX[:, 0] = (2*xAxis - coords[:, 0])
coords_mirrorX[:, 1] = coords[:, 1]

coords_mirrorXY[: , 0] = (2*xAxis - coords[:, 0])
coords_mirrorXY[:, 1] = (2*yAxis - coords[:, 1])


## Solve problem
n_dofs = np.size(dofs)                      # Degrees of freedom
ex, ey = cfc.coordxtr(edof, coords, dofs)   # Element coordinates
K = np.zeros([n_dofs, n_dofs])
f = np.zeros([n_dofs, 1])
C = np.zeros([n_dofs, n_dofs])


## Extract nodes on specific boundary, to incorporate natural boundary conditions
# Boundary with known flow
all_elements_hFlow = boundaryElements[mark_hFlow]
boundaryNodePairs_hFlow = []
for element in all_elements_hFlow:
    newElement = element['node-number-list']
    boundaryNodePairs_hFlow.append(newElement)

# Boundary with Newton convection
all_elements_Newton = boundaryElements[mark_Newton]
boundaryNodePairs_Newton = []
for element in all_elements_Newton:
    newElement = element['node-number-list']
    boundaryNodePairs_Newton.append(newElement)


## Create the Ke, Ce and fe matrixes element-wise and use assem to get them into the global matrixes
for el_topo, elx, ely, marker in zip(edof, ex, ey, element_markers):

    # Check for different materials inside the mesh
    if marker == mark_nylon:
        Ke = cfc.flw2te(elx, ely, [thickness], D_nylon)  
        Ce = plantml(elx,ely, thickness*cp_nylon*rho_nylon)  # Calculating the C matrix for the transient solution
    else:
        Ke = cfc.flw2te(elx,ely,[thickness],D_copper)
        Ce = plantml(elx,ely,thickness*cp_copper*rho_copper)

    # Check for element on boundary with h_flow marker
    assemed = False
    for boundaryNodePair in boundaryNodePairs_hFlow:
        if boundaryNodePair[0] in el_topo and boundaryNodePair[1] in el_topo:
            # Get indexes from the nodes in the element
            index1 = np.where(el_topo == boundaryNodePair[0])
            index2 = np.where(el_topo == boundaryNodePair[1])
            
            # Calculate the distance between nodes
            length = np.sqrt((elx[index1]-elx[index2])**2 + (ely[index1]-ely[index2])**2) 
            fe_input = -1*heatFlow*length*thickness*0.5
           
            fe = np.zeros([3,1])
            fe[index1] = fe_input
            fe[index2] = fe_input
           
            cfc.assem(el_topo, K, Ke, f, fe)
            assemed = True
   
    # Check for element on boundary with Newton marker
    for boundaryNodePair in boundaryNodePairs_Newton:
        if boundaryNodePair[0] in el_topo and boundaryNodePair[1] in el_topo:
            # Get indexes from the nodes in the element
            index1 = np.where(el_topo == boundaryNodePair[0])
            index2 = np.where(el_topo == boundaryNodePair[1])

            # Calculate the distance between nodes
            length = np.sqrt((elx[index1]-elx[index2])**2 + (ely[index1]-ely[index2])**2) 
            fe_input = T_surrounding*alpha_convectionCoeff*length*thickness*0.5
           
            fe = np.zeros([3,1])
            fe[index1] = fe_input
            fe[index2] = fe_input

            # Create element matrix responsible for Newton convection
            Ke_Convection = np.zeros([3,3])
            Ke_Convection[index1,index1] = 2
            Ke_Convection[index1,index2] = 1
            Ke_Convection[index2,index1] = 1
            Ke_Convection[index2,index2] = 2
            Ke_Convection = Ke_Convection*alpha_convectionCoeff*thickness*length/6
            Ke = Ke + Ke_Convection
         
            cfc.assem(el_topo,K,Ke,f,fe)
            assemed = True

    # Assemble K-matrix if it has not already been done on the boundary
    if (assemed == False):
        cfc.assem(el_topo, K, Ke)
 
    # Assemble the C-matrix
    cfc.assem(el_topo,C,Ce)


## Handling remaining boundary conditions, and solving the system
# We have no essential boundary conditions for the temperature problem
bc = np.array([], 'i')
bc_val = np.array([], 'f')


## Solve with solveq
a_stat, r = cfc.solveq(K,f,bc,bc_val)


## Plotting the solution
cfv.figure(fig_size=(10,10))
cfv.draw_nodal_values_contourf(a_stat-273, coords, edof, title="Stationary temperature distribution", dofs_per_node=mesh.dofs_per_node, el_type=mesh.el_type, draw_elements=True)
cfv.addText("x [m]", [0.0022, -0.0003])
cfv.addText("y [m]", [-0.0006, 0.0027])
cfv.addText("Temperature [" + chr(176) + "C]", [0.0057, 0.0012])
cfv.colorbar()
cfv.show()


## Plotting the solution, mirrored to the whole gripper
cfv.figure(fig_size=(10,10))
cfv.draw_nodal_values_contourf(a_stat-273, coords, edof, title="Stationary temperature distribution", dofs_per_node=mesh.dofs_per_node, el_type=mesh.el_type, draw_elements=True)
cfv.draw_nodal_values_contourf(a_stat-273, coords_mirrorX, edof, title="Stationary temperature distribution", dofs_per_node=mesh.dofs_per_node, el_type=mesh.el_type, draw_elements=True)
cfv.draw_nodal_values_contourf(a_stat-273, coords_mirrorY, edof, title="Stationary temperature distribution", dofs_per_node=mesh.dofs_per_node, el_type=mesh.el_type, draw_elements=True)
cfv.draw_nodal_values_contourf(a_stat-273, coords_mirrorXY, edof, title="Stationary temperature distribution", dofs_per_node=mesh.dofs_per_node, el_type=mesh.el_type, draw_elements=True)
cfv.addText("x [m]", [0.0045, -0.0007])
cfv.addText("y [m]", [-0.0006, 0.0053])
cfv.addText("Temperature [" + chr(176) + "C]", [0.0115, 0.0025])
cfv.colorbar()
cfv.show()


#### The transient solution ####
## Time stepping for the transient equation
a_0 = T_initial*np.ones((n_dofs,1), dtype=np.int8)
a_transient = a_0
a_transient_new = a_0
currentTime = 0
deltaTime = 0.1


## Timestepping until the maximal temperature is 90% of stationary solutions max temp.
factor1 = np.linalg.inv(C+deltaTime*K)
while np.max(a_transient) <= 0.9*np.max(a_stat):
    a_transient = np.matmul(factor1 ,f*deltaTime + np.matmul(C,a_transient))
    currentTime = currentTime+deltaTime

# The time to reach 90% of the stationary max value
t_90 = currentTime
print("Time to reach 90% of the stationary max value: " + str(round(t_90, 3)) + "s")


## Resetting the currentTime, and preparing to plot the early stages of the heat distributions
currentTime = 0
a_transient = a_0
t_final = t_90*0.03
t_increment = t_final/4
t_snapshot = t_increment
deltaTime = 0.01

## Timestepping agian, but for 3% of the previous time, and creating snapshots. 
factor1 = np.linalg.inv(C + deltaTime*K)
firstTime = True
while(currentTime <= t_final):
    a_transient = np.matmul(factor1,f*deltaTime + np.matmul(C, a_transient))
    currentTime = currentTime + deltaTime

    if (round(currentTime, 2) == round(t_snapshot, 2) or firstTime):
        if (firstTime):
            firstTime = False
        else:
            t_snapshot = t_snapshot + t_increment
        cfv.figure(fig_size=(10,10))
        cfv.draw_nodal_values_contourf(a_transient-273, coords, edof, title="Temperature at time t: " + str(round(currentTime, 5)) + " s", dofs_per_node=mesh.dofs_per_node, el_type=mesh.el_type, draw_elements=True)
        cfv.addText("x [m]", [0.0022, -0.0003])
        cfv.addText("y [m]", [-0.0006, 0.0027])
        cfv.addText("Temperature [" + chr(176) + "C]", [0.0057, 0.0012])
        cfv.colorbar()
     
cfv.show()


#### The displacements due to thermal expansion ####
## Create new matrices from same mesh, to update new deg of freedom for displacement problem.
n_dofs = np.size(dofs) # Degrees of freedom
n_edof_rows, n_edof_cols = edof.shape

dof_tension = np.zeros([n_dofs, 2], dtype = int)
edof_tension = np.zeros([n_edof_rows, 2*n_edof_cols], dtype = int)
bdofs_tension = {}

# The dof matrix
for i in range(1, n_dofs+1):
    dof_tension[i-1, :] = [2*i - 1, 2*i]

# The edof matrix
for i in range(1, n_edof_rows+1):
    for j in range(1, n_edof_cols+1):
        edof_tension[i-1, 2*j-2] = 2*edof[i-1, j-1] - 1
        edof_tension[i-1, 2*j-1] = 2*edof[i-1, j-1]

# The boundary_dofs 
for marker in boundary_dofs:
    bdofs_tension.setdefault(marker, [])
    for dof in boundary_dofs[marker]:
        bdofs_tension[marker].append(dof*2-1)
        bdofs_tension[marker].append(dof*2)


## Create K and f
n_dofs_tension = np.size(dof_tension)
ex, ey = cfc.coordxtr(edof_tension, coords, dof_tension)
K = np.zeros([n_dofs_tension, n_dofs_tension])
strain_deltaT = np.zeros([n_dofs_tension,1])


## Iterating over every element to calculate displacements
for el_topo, el_topo_old, elx, ely, marker in zip(edof_tension, edof, ex, ey, element_markers):
    # Calculating the average temperature in the element
    deltaTime = (1/3) * (a_stat[el_topo_old[0]-1] + a_stat[el_topo_old[1]-1] + a_stat[el_topo_old[2]-1]).item() - T_surrounding
   
    # Checking for material parameters, and calculating the element strain matrices
    if marker == mark_nylon:
        Ke = cfc.plante(elx, ely, [2,thickness], D_nylon_strain)
        element_stress = (np.matmul(D_nylon_strain,np.array([1,1,0,]))*alpha_nylon*deltaTime)
        strain_deltaT_e = cfc.plantf(elx, ely, [2,thickness], element_stress).reshape(6,1)
    else:
        Ke = cfc.plante(elx,ely,[2,thickness],D_copper_strain)
        element_stress = (np.matmul(D_copper_strain,np.array([1,1,0,]))*alpha_copper*deltaTime)
        strain_deltaT_e = cfc.plantf(elx, ely, [2,thickness], element_stress).reshape(6,1)

    cfc.assem(el_topo,K,Ke,strain_deltaT,strain_deltaT_e)


## Apply essential boundary conditions
bc_tension = np.array([],'i')
bcVal_tension = np.array([],'f')

bc_tension, bcVal_tension = cfu.applybc(bdofs_tension, bc_tension, bcVal_tension, mark_immovable, 0, 0)
bc_tension, bcVal_tension = cfu.applybc(bdofs_tension, bc_tension, bcVal_tension, mark_hFlow, 0, 0)
bc_tension, bcVal_tension = cfu.applybc(bdofs_tension, bc_tension, bcVal_tension, mark_immovableX, 0, 1)
bc_tension, bcVal_tension = cfu.applybc(bdofs_tension, bc_tension, bcVal_tension, mark_immovableY, 0, 2)


## Solving using solveq
a_displacement, r = cfc.solveq(K, strain_deltaT, bc_tension, bcVal_tension)


## Mirroring appropriately to plot the whole gripper
a_displacements_mirrorY = np.copy(a_displacement)
a_displacements_mirrorY[::2] = a_displacements_mirrorY[::2]*-1
a_displacements_mirrorY = a_displacements_mirrorY*-1

a_displacements_mirrorX = np.copy(a_displacement)
a_displacements_mirrorX[::2] = a_displacements_mirrorY[::2]*-1

a_displacements_mirrorXY = a_displacement*-1


## Plotting the results
cfv.figure(fig_size=(10,10))
cfv.draw_displacements(a_displacement, coords, edof_tension, dofs_per_node=2,
                             el_type=mesh.el_type, draw_undisplaced_mesh=True, title = "Displacements due to thermal expansion",
                             magnfac=1)
cfv.draw_displacements(a_displacements_mirrorY, coords_mirrorY, edof_tension, dofs_per_node=2,
                             el_type=mesh.el_type, draw_undisplaced_mesh=True, title = "Displacements due to thermal expansion",
                             magnfac=1)
cfv.draw_displacements(a_displacements_mirrorX, coords_mirrorX, edof_tension, dofs_per_node=2,
                             el_type=mesh.el_type, draw_undisplaced_mesh=True, title = "Displacements due to thermal expansion",
                             magnfac=1)
cfv.draw_displacements(a_displacements_mirrorXY, coords_mirrorXY, edof_tension, dofs_per_node=2,
                             el_type=mesh.el_type, draw_undisplaced_mesh=True, title = "Displacements due to thermal expansion",
                             magnfac=1)
cfv.addText("x [m]", [0.00445, -0.00075])
cfv.addText("y [m]", [-0.00075, 0.0055])
cfv.show_and_wait()


cfv.figure(fig_size=(10,10))
cfv.draw_displacements(a_displacement, coords, edof_tension, dofs_per_node=2,
                             el_type=mesh.el_type, draw_undisplaced_mesh=True, title = "",
                             magnfac=5)
cfv.draw_displacements(a_displacements_mirrorY, coords_mirrorY, edof_tension, dofs_per_node=2,
                             el_type=mesh.el_type, draw_undisplaced_mesh=True, title = "",
                             magnfac=5)
cfv.draw_displacements(a_displacements_mirrorX, coords_mirrorX, edof_tension, dofs_per_node=2,
                             el_type=mesh.el_type, draw_undisplaced_mesh=True, title = "",
                             magnfac=5)
cfv.draw_displacements(a_displacements_mirrorXY, coords_mirrorXY, edof_tension, dofs_per_node=2,
                             el_type=mesh.el_type, draw_undisplaced_mesh=True, title = "Displacements due to thermal expansion, magnified with a factor of 5",
                             magnfac=5)
cfv.addText("x [m]", [0.00445, -0.00075])
cfv.addText("y [m]", [-0.00075, 0.0055])
cfv.show_and_wait()


#### The stresses from the displacements ####
## Compute element forces
element_displacements = cfc.extract_eldisp(edof_tension,a_displacement)
vonMises = []


## Iterating over every element, to calculate the von Mises stress distribution
for i, marker, el_topo_thermal in zip(np.arange(edof.shape[0]), element_markers, edof):
    deltaTime = (1/3) * (a_stat[el_topo_thermal[0]-1] + a_stat[el_topo_thermal[1]-1] + a_stat[el_topo_thermal[2]-1]).item() - T_surrounding
    
    # If copper element
    if marker == mark_copper:
        es_thermal = np.matmul(D_copper_strain,np.array([1,1,0,]))*alpha_copper*deltaTime
        element_stress, e_strain = cfc.plants(ex[i,:], ey[i,:], [2, thickness], D_copper_strain, element_displacements[i,:])
        e_stress_zz = ((v_copper*E_copper/((1+v_copper)*(1-2*v_copper)))*(e_strain[0,0]+e_strain[0,1]) - E_copper*alpha_copper*deltaTime/(1-2*v_copper)).reshape(1,1) # Stress out of plane    
        element_stress = element_stress - es_thermal
        element_stress = np.concatenate([element_stress,e_stress_zz], axis = 1)
        
    # If nylon element 
    else:
        es_thermal = np.matmul(D_nylon_strain,np.array([1,1,0,]))*alpha_nylon*deltaTime
        element_stress, e_strain = cfc.plants(ex[i,:], ey[i,:], [2, thickness], D_nylon_strain, element_displacements[i,:])
        e_stress_zz = ((v_nylon*E_nylon/((1+v_nylon)*(1-2*v_nylon)))*(e_strain[0,0]+e_strain[0,1]) - E_nylon*alpha_nylon*deltaTime/(1-2*v_nylon)).reshape(1,1) # Stress out of plane
        element_stress = element_stress - es_thermal  
        element_stress = np.concatenate([element_stress,e_stress_zz], axis = 1)

    # Calc and append effective stress to list
    vonMises.append(np.sqrt(np.power(element_stress[0,0],2) + np.power(element_stress[0,1],2) + np.power(element_stress[0,3],2)  - element_stress[0,0]*element_stress[0,1]
                           -element_stress[0,0]*element_stress[0,3]  - element_stress[0,1]*element_stress[0,3]  + 3*np.power(element_stress[0,2],2)))


## Printing the elements with largest von Mises stresses 
sorted_Index_Array = np.argsort(vonMises)
listSize = sorted_Index_Array.size
print("The elements with largest von Mises stress:")
for i in range(0,5):
    index = sorted_Index_Array[listSize-1-i]
    print("Element: " + str(index))
    print("Von Mises stress = " + str(vonMises[index]) + " GPa")


## Plotting the von Mises stress distribution
cfv.figure(fig_size=(10,10))
cfv.draw_element_values(vonMises, coords, edof_tension, dofs_per_node = 2, el_type = mesh.el_type, displacements = a_displacement,
                        draw_elements = False, draw_undisplaced_mesh = True,
                        title = 'Effective stress due to thermal expansion', magnfac = 1)
cfv.addText("x [m]", [0.0022, -0.0005])
cfv.addText("y [m]", [-0.0006, 0.00283])
cfv.addText("Von Mises Stress [GPa]", [0.006, 0.0012])
cfv.colorbar()
cfv.show_and_wait()

# The mirrorred plots for the whole gripper
cfv.figure(fig_size=(15,15))
cfv.draw_element_values(vonMises, coords, edof_tension, dofs_per_node = 2, el_type = mesh.el_type, displacements = a_displacement,
                        draw_elements = False, draw_undisplaced_mesh = True,
                        title = 'Effective stress due to thermal expansion', magnfac = 1)
cfv.draw_element_values(vonMises, coords_mirrorX, edof_tension, dofs_per_node = 2, el_type = mesh.el_type, displacements = a_displacements_mirrorX,
                        draw_elements = False, draw_undisplaced_mesh = True,
                        title = 'Effective stress due to thermal expansion', magnfac = 1)
cfv.draw_element_values(vonMises, coords_mirrorY, edof_tension, dofs_per_node = 2, el_type = mesh.el_type, displacements = a_displacements_mirrorY,
                        draw_elements = False, draw_undisplaced_mesh = True,
                        title = 'Effective stress due to thermal expansion', magnfac = 1)
cfv.draw_element_values(vonMises, coords_mirrorXY, edof_tension, dofs_per_node = 2, el_type = mesh.el_type, displacements = a_displacements_mirrorXY,
                        draw_elements = False, draw_undisplaced_mesh = True,
                        title = 'Effective stress due to thermal expansion', magnfac = 1)
cfv.colorbar()
cfv.addText("x [m]", [0.0046, -0.0007])
cfv.addText("y [m]", [-0.0006, 0.0055])
cfv.addText("Von Mises Stress [GPa]", [0.012, 0.0025])
cfv.show_and_wait()
