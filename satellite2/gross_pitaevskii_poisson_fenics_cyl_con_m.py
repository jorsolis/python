#!/usr/bin/env python
# coding: utf-8
from dolfin import (CompiledSubDomain, Constant, DirichletBC, dot, ds, dx,
                    Expression, File, FiniteElement, Function, FunctionSpace,
                    grad, HDF5File, Measure, Mesh, MeshFunction, MixedElement,
                    nabla_grad, near, plot, Point, project, solve, split,
                    SubDomain, TestFunctions, triangle, VectorFunctionSpace,
                    assemble,div)
from mshr import Rectangle,generate_mesh, Circle
from schrodinger_poisson_fenics_plotting import (plots_pyplot, graf,
                                                 density3d_files)
import numpy as np

def mainprog(ncor, Rf = 10, E= 0.034465, Vf = -0.1, fif = .001,Nf=1.,
             direct = '/home/jordi/gross', bet = 0., eme = 0):
    a = 1.
    b = 0.
    eme2 = eme*eme
    ###########################     MESH         ##################################
    circulo = Circle(Point(0.0,0.0), Rf) - Rectangle(Point(-Rf,-Rf),Point(0.,Rf)) - Rectangle(Point(0.,-Rf), Point(Rf,0.))
    mesh = generate_mesh(circulo, 164)#
#############               Save mesh to a file           #####################
    File('%s/mesh_schrodinger_poisson_m%d_%d.xml.gz'%(direct,eme,ncor)) << mesh
#############           Define boundaries for  Newman conditions       ########
    boundary_markers = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)    
    tol = 1E-14
    class leftwall(SubDomain):
        def inside(self, x, on_boundary):
            if on_boundary:
                if near(x[0], 0, tol):
                    return True
                else:
                    return False
            else:
                return False       

    class bottomwall(SubDomain):
        def inside(self, x, on_boundary):
#            return on_boundary and near(x[1], 0, tol)
            if on_boundary:
                if near(x[1], 0, tol):
                    return True
                else:
                    return False
            else:
                return False
    
    left = leftwall()
    bottom = bottomwall()
   
    left.mark(boundary_markers, 2)
    bottom.mark(boundary_markers, 1)   
    ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)
####################     Define function space       ##########################
    P1 = FiniteElement('P', triangle, 1)
    V = FunctionSpace(mesh, MixedElement([P1, P1]))
###################      Define test functions      ############################
    v_1, v_2= TestFunctions(V)
############          Define functions for SF and potential      ##############
    u = Function(V)
#    u_0 = Function(V)
#    t = 0
#############         Split system functions to access components      ########
    psi, Phi = split(u)
###############          Define expressions used in variational forms       ###
    fx = Expression('a0*x[0] + b0', degree=1, a0 = a, b0= b)#
    fx2 = Expression('a0*x[0]*x[0]', degree=2, a0 = a)#
#    function = project(fx, FunctionSpace(mesh, 'P', 3))
#    plot(function)
#    plt.show()
####################         Constants           #############################
    k = Constant(E)
    g1 = Constant(0.)
    beta = Constant(bet)
    m = Constant(eme2)
##############          Define variational problem     #########################
    F = fx*dot(grad(psi), grad(v_1))*dx  + fx*Phi*Phi*v_1*dx \
    + fx2*dot(grad(Phi), grad(v_2))*dx + fx*Phi.dx(0)*v_2*dx \
    + 2.*fx2*psi*Phi*v_2*dx + 2.*fx2*k*Phi*v_2*dx \
    + 2.*fx2*beta*Phi*Phi*Phi*v_2*dx \
    + m*Phi*v_2*dx \
    - g1*v_1*ds(2) - g1*v_1*ds(1) - g1*v_2*ds(2)
#############           Define boundaries             #########################
    def circlebord(x, on_boundary):
        if on_boundary:
            if near(x[0], 0, tol) or near(x[1],0, tol):
                return False
            else:
                return True
        else:
            return False 
##############          boundary conditions         ###########################
    bc = DirichletBC(V.sub(0), Constant(Vf) , circlebord) # on psi
    bc2 = DirichletBC(V.sub(1), Constant(fif) , circlebord)  # on Phi
    bc3 = DirichletBC(V.sub(1), Constant(0.) , bottom)  # on Phi
###############           Solve variational problem         ###################
    solve(F == 0, u, [bc, bc2, bc3]) 
################          Save solution               #########################
    output_file = HDF5File(mesh.mpi_comm(), "%s/u_m%d_%d.h5"%(direct,eme,ncor), "w")
    output_file.write(u, "solution")
    output_file.close()
    
#def save_sol_to_vtk(u, ncor, direct = '/home/jordi/satellite/schrodinger_poisson'): 
#    psi_1, phi_1 = u.split()
#    vtkfile_psi = File('%s/psi_%d.pvd'%(direct,ncor))
#    vtkfile_phi = File('%s/phi_%d.pvd'%(direct,ncor))
#    vtkfile_psi << (psi_1)
#    vtkfile_phi << (phi_1)
    
def read_saved_file(ncor,m, direct = '/home/jordi/satellite/schrodinger_poisson'):
    ################         Read mesh         #################################
    mesh = Mesh('%s/mesh_schrodinger_poisson_m%d_%d.xml.gz'%(direct,m, ncor))
    P1 = FiniteElement('P', triangle, 1)
    V = FunctionSpace(mesh, MixedElement([P1, P1]))
    ###############    Load solution        ###################################
    u = Function(V)
    input_file = HDF5File(mesh.mpi_comm(), "%s/u_m%d_%d.h5"%(direct,m,ncor), "r")
    input_file.read(u, "solution")
    input_file.close()
    return u  
############################################################################
def numero_particulas(f,r,R,dr):
    "integral definida de r*f(r,z)**2 drdz  r de 0 a R y z de 0 a R"
    A=0.
    elem = int(np.rint(R/dr))
    for i in range(0,elem,1):
        for j in range(0,elem,1):
            A+= dr*f[i,j]*r[j]*dr       
    return A

def Km(f,r,R,dr,m):
    A=0.
    elem = int(np.rint(R/dr))
    for i in range(0,elem,1):
        for j in range(0,elem,1):
            A+= dr*f[i,j]*dr/r[j]       
    return 2.*np.pi*A*m**2

def fi(u, rho,z,Rf, n=1):
    Phi = []
    for i in range(0,np.shape(rho)[0]):
        Phii=[]
        for j in range(0,np.shape(z)[0]):
            point = (rho[i], z[j]) 
            if z[j]**2 + rho[i]**2 > Rf**2:
                Phii.append(0.)
            else:
                if z[j]<0:
                    if rho[i]<0:
                        point = (-rho[i], -z[j])
                        Phii.append((-u[1](point))**n)
                    else:
                        point = (rho[i], -z[j])
                        Phii.append((-u[1](point))**n)
                else:
                    if rho[i]<0:
                        point = (-rho[i], z[j])
                        Phii.append((u[1](point))**n)
                    else:
                        Phii.append((u[1](point))**n)    
        Phi.append(Phii)
    Phi = np.array(Phi)
    return np.transpose(Phi)

####################        number of particles       ########################
########                FALTA VER SI FUNCIONAN IGUAL QUE EN m=0
def print_num_part(u,Rf):
    tol = 0.001
    rho = np.linspace(tol, Rf - tol, 201) 
    dr = rho[1]-rho[0]
    N2 = numero_particulas(fi(u, rho,rho,Rf, n=2),rho,Rf,dr)
    psi, phi = u.split()
    fx = Expression('a0*x[0] + b0', degree=1, a0 = 1, b0= 0)
    ene = fx*phi*phi*dx  # corregida
    Nn = assemble(ene)
    print('N=',Nn, 'N/rf= ', -Nn/Rf)
    return N2, Nn 
#
#################       Viralization        ####################################
########                FALTA VER SI FUNCIONA kee2
def virializacion(u, m, R):
    psi, phi = u.split()
    V = phi.function_space()
    mesh = V.mesh()
    fx = Expression('a0*x[0] + b0', degree=1, a0 = 1, b0= 0, domain=mesh)
    we = 2.*np.pi*phi*phi*psi*fx*dx(domain=mesh)  ### antes faltaba el fx    
    kee = -2.*np.pi*phi*div(fx*grad(phi))*dx(domain=mesh)
#    mc = m**2
#    kee2 = 2.*np.pi*mc*phi*phi*dx(domain=mesh)/fx
#    Ke2 = assemble(kee2)
    
    tol = 0.001
    rho = np.linspace(tol, Rf - tol, 201) 
    dr = rho[1]-rho[0]
    Ke2 = Km(fi(u, rho,rho,Rf, n=2),rho,Rf,dr,m)
    We = assemble(we)
    Ke = assemble(kee)

    print('K=', Ke + Ke2, 'W=', We)
    print('K/W=', (Ke + Ke2)/abs(We))
    return Ke, We
    
#def write_catidades_file(u,Rf, Ener, beta, m)
def write_catidades_file(u,Rf, Ener, beta, m, nc, Nf):
    N2, Nn = print_num_part(u,Rf)
    Ke, We = virializacion(u, m, Rf)
    psi, phi = u.split()
    psi.set_allow_extrapolation(True)
    phi.set_allow_extrapolation(True)
    print('Psi(Rf,0)=',psi(Rf,0))
    rf = Rf
    thf = np.pi/4
    rhof = rf*np.sin(thf)
    zf = rf*np.cos(thf)
    print('Psi(rf,pi/4)=',psi(rhof,zf))
#    f= open("%s/%d/m%d/cantidades_%d.txt"%(dec,int(beta),int(m),ncor),"w+")
    f= open("%s/%d/%d/cantidades_m%d_%d.txt"%(dec,int(beta),nc,int(m),ncor),"w+")
    f.write(" Ni=%f \r\n " %Nf)
    f.write(" N= %f, N/rf= %f \r\n " % (Nn, Nn/Rf))
    f.write(" N= %f, N/rf= %f, Rf= %f \r\n " % (N2, N2/Rf, Rf))
    f.write(" E_t= %f, Psi(Rf,0)= %f, Psi(rf,pi/4) = %f \r\n " % (Ener, psi(Rf,0), psi(rhof,zf)))
    f.write(" We= %f, Ke= %f, Ke/We= %f \r\n " % (We, Ke, Ke/abs(We)))
    f.write(" beta= %f, m=%d, phif= %f \r\n " %(beta, m, phi(rhof,zf)))
    f.close()

################################################################################  
################################################################################
di2R100 = {1 :{"Rf" : 100., "En" : 1./2., "expon": -4, "Vf" : -1e-2/2.,
               "fif" : 1e-4/np.sqrt(2.), 'rlam':10}, 
               2 :{"Rf" : 100., "En": 2./2., "expon" : -4, "Vf" : -1e-2/2.,
                   "fif" : 1e-4/np.sqrt(2.), 'rlam':10},##con fif = 1e-7 tambien funciona
               3 :{"Rf" : 100., "En": 8./2., "expon" : -4, "Vf" : -1e-2/2.,
                   "fif" : 1e-4/np.sqrt(2.), 'rlam':10},##con fif = 1e-7 tambien funciona
               4 :{"Rf" : 100., "En": 9./2., "expon" : -4, "Vf" : -1e-2/2.,
                   "fif" : 1e-4/np.sqrt(2.), 'rlam':10},##con fif = 1e-7 tambien funciona
               5 :{"Rf" : 100., "En" : 12./2., "expon" : -4, "Vf" : -1e-2/2.,
                   "fif" : 1e-4/np.sqrt(2.), 'rlam':10},
               6 :{"Rf" : 100., "En": 16./2., "expon" : -4, "Vf" : -1e-2/2.,
                   "fif" : 1e-4/np.sqrt(2.), 'rlam':10},
               7 :{"Rf" : 100., "En": 17./2., "expon" : -4, "Vf" : -1e-2/2.,
                   "fif" : 1e-4/np.sqrt(2.), 'rlam':10},
               8 :{"Rf" : 100., "En" : 18./2., "expon" : -4, "Vf" : -1e-2/2.,
                   "fif" : 1e-4/np.sqrt(2.), 'rlam':10},
               9 :{"Rf" : 100., "En" : 40./2., "expon" : -4, "Vf" : -1e-2/2.,
                   "fif" : 1e-4/np.sqrt(2.), 'rlam':10},
               10 :{"Rf" : 100., "En": 49./2., "expon" : -4, "Vf" : -1e-2/2.,
                    "fif" : 1e-4/np.sqrt(2.), 'rlam':10},
               11 :{"Rf" : 100., "En": 50./2., "expon" : -4, "Vf" : -1e-2/2.,
                    "fif" : 1e-4/np.sqrt(2.), 'rlam':10},
               12 :{"Rf" : 100., "En" : 55./2., "expon" : -4, "Vf" : -1e-2/2.,
                    "fif" : 1e-4/np.sqrt(2.), 'rlam':10},
               13 :{"Rf" : 100., "En" : 83./2., "expon" : -4, "Vf" : -1e-2/2.,
                    "fif" : 1e-4/np.sqrt(2.), 'rlam':10},
               14 :{"Rf" : 100., "En" : 60./2., "expon" : -4, "Vf" : -1e-2/2.,
                    "fif" : 1e-4/np.sqrt(2.), 'rlam':10}}

di2R10 = {12:{"Rf" : 10., "En" : 1.74, "expon" : -2, "fif" : 1e-1,
              "N" : 1.2, "beta" : 0.0,'rlam': 1.}}

dec = '/home/jordi/gross_m'
################################################################################  
################################################################################

de = float(input("run"))
print(de, type(de))
#
nc = float(de)
#nc = 2750

#ncor = float(de)
ncor = 12


#for ncor in range(9,15,1):
Rf = di2R10[ncor]["Rf"]
expon = di2R10[ncor]["expon"]
En = di2R10[ncor]["En"] + 0.1
fif = di2R10[ncor]["fif"]


Nf = nc/1000.

#Nf = di2R10[ncor]["N"]
#nc = Nf*1000

Vf = -Nf/Rf

###############         Escalamiento     #####################################
#rlam = di2R10[ncor]['rlam']
#
#Rf = Rf/rlam
#Nf = Nf*rlam
#En = rlam**2*En
#Vf = rlam**2*Vf
#fif = rlam**2*fif
##############################################################################

#nc = Nf*1000

m = 1.
Ener = En*10**expon
beta = 0.

print('N=',nc/1000)
print('bet=',beta)
print('ncor=',ncor)
print('E=',Ener)

print('Rf=',Rf)
print('fif=',fif)
print('Nf=',Nf)

mainprog(ncor, Rf= Rf, E= Ener, Vf = Vf, fif =fif, 
         direct = '%s/%d/%d'%(dec,int(beta),nc), bet = beta, eme = m) 
#
u = read_saved_file(ncor,m, direct = '%s/%d/%d'%(dec,int(beta),nc))

u.set_allow_extrapolation(True)
plots_pyplot(u,Rf, En, expon, ncor, ruta='%s/%d/%d'%(dec,int(beta),nc),m=m,
             show = False, dens=True, pot=True, otras = False)

write_catidades_file(u,Rf, Ener, beta, m, nc, Nf)

  
#density3d_files(u,Rf,"/home/jordi/gross_m/%d/%d/m%d/matrix_%d.CSV"%(beta,nc,m,ncor))