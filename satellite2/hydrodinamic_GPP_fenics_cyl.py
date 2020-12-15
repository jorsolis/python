#!/usr/bin/env python
# coding: utf-8
##############################################################################
############### versión hidrodinamica de gross_pitaevskii_poisson_fenics_cyl
##############################################################################
from dolfin import (CompiledSubDomain, Constant, DirichletBC, dot, ds, dx,
                    Expression, File, FiniteElement, Function, FunctionSpace,
                    grad, HDF5File, Measure, Mesh, MeshFunction, MixedElement,
                    nabla_grad, near, plot, Point, project, solve, split,
                    SubDomain, TestFunctions, triangle, VectorFunctionSpace,
                    assemble,div,SpatialCoordinate, cos, UnitIntervalMesh)
from mshr import Rectangle,generate_mesh, Circle
from schrodinger_poisson_fenics_plotting import (plots_pyplot, graf,
                                                 density3d_files,plotses)
#from gross_pitaevskii_poisson_fenics_cyl_con_m.py import (read_saved_file, fi)
import numpy as np
import plots_jordi as pts

def mainprog(ncor, Rf = 10, E= 0.034465, Vf = -0.1, densf = .001, Nf=1.,
             direct = '/home/jordi/gross_hidro', lam = 0., R0 = 0.01):
###########################     MESH         ##################################
#    circulo = Circle(Point(0.0,0.0), Rf) - Circle(Point(0.0,0.0), R0) - Rectangle(Point(-Rf,-Rf),Point(0.,Rf)) - Rectangle(Point(0.,-Rf), Point(Rf,0.))
    
    circulo = Circle(Point(0.0,0.0), Rf) - Rectangle(Point(-Rf,-Rf),Point(0.,Rf)) - Rectangle(Point(0.,-Rf), Point(Rf,0.))
    mesh = generate_mesh(circulo, 164)
#############               Save mesh to a file           #####################
    File('%s/mesh_schrodinger_poisson_%d.xml.gz'%(direct,ncor)) << mesh
#############           Define boundaries for  Newman conditions       ########
    boundary_markers = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)    
    tol = 1E-14
    class leftwall(SubDomain):
        def inside(self, x, on_boundary):
            if on_boundary:
                if near(x[0], 0, tol):
                    return True
                
#                elif near(x[1], 0, R0) and near(x[0], 0, R0):
#                    return False 
                
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
                
#                elif near(x[1], 0, R0) and near(x[0], 0, R0):
#                    return False 
                
                else:
                    return False
            else:
                return False

    class no_orig(SubDomain):
        def inside(self, x, on_boundary):
            if on_boundary:
                if near(x[1], 0, R0) and near(x[0], 0, R0):
                    return True           
                else:
                    return False
            else:
                return False  
        
    left = leftwall()
    bottom = bottomwall()
    no_origen = no_orig()

    bottom.mark(boundary_markers, 1)        
    left.mark(boundary_markers, 2)
    no_origen.mark(boundary_markers, 3)
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
    psi, dens = split(u)
###############          Define expressions used in variational forms       ###
    fx = Expression('a0*x[0] + b0', degree=1, a0 = 1, b0= 0)#
    u_D = Expression('a*exp(-pow(x[1] + b , 2)/(2*pow(c,2)))', degree=2,
                     a= 0.171509, b=-2.6996, c=1.) 
####################         Constants           #############################
    k = Constant(E)
    g1 = Constant(0.)
    g2 = Constant(0.)
    ct = Constant(3./2.)
    lamb = Constant(lam)
##############          Define variational problem     #########################
    F = fx*dot(grad(psi), grad(v_1))*dx  + fx*dens*v_1*dx \
    + fx*dens*dot(grad(dens), grad(v_2))*dx \
    + ct*fx*v_2*dot(grad(dens), grad(dens))*dx \
    + 4.*fx*dens*dens*psi*v_2*dx - 4.*fx*dens*dens*k*v_2*dx \
    + 4.*fx*dens*dens*lamb*dens*v_2*dx \
    - g1*v_1*ds(1) - g2*v_1*ds(2) #- g1*v_2*ds(2)
#############           Define boundaries             #########################
    def circlebord(x, on_boundary):
        if on_boundary:
            if near(x[0], 0, tol) or near(x[1],0, tol):
                return False

#            elif near(x[1], 0, R0) and near(x[0], 0, R0):
#                return False

            else:
                return True
        else:
            return False  
##############          boundary conditions         ###########################
    bc = DirichletBC(V.sub(0), Constant(Vf) , circlebord) # on psi
    bc2 = DirichletBC(V.sub(1), Constant(densf) , circlebord)  # on Phi**2
    bc3 = DirichletBC(V.sub(1), Constant(0.) , bottom)  # on Phi**2
    bc4 = DirichletBC(V.sub(1), u_D , left)
###############           Solve variational problem         ###################
#    solve(F == 0, u, [bc, bc2, bc3]) 
    solve(F == 0, u, [bc, bc2, bc3,bc4]) 
################          Save solution               #########################
    output_file = HDF5File(mesh.mpi_comm(), "%s/u_%d.h5"%(direct,ncor), "w")
    output_file.write(u, "solution")
    output_file.close() 
     
def read_saved_file(ncor, direct = '/home/jordi/gross_hidro'):
    ################         Read mesh         #################################
    mesh = Mesh('%s/mesh_schrodinger_poisson_%d.xml.gz'%(direct,ncor))
    P1 = FiniteElement('P', triangle, 1)
    V = FunctionSpace(mesh, MixedElement([P1, P1]))
    ###############    Load solution        ###################################
    u = Function(V)
    input_file = HDF5File(mesh.mpi_comm(), "%s/u_%d.h5"%(direct,ncor), "r")
    input_file.read(u, "solution")
    input_file.close()
    return u  
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

###############################################################################
def numero_particulas(f,r,R,dr):
    "integral definida de rho*f(rho,z)**2 drho dz  r de 0 a R y z de 0 a R"
    A=0.
    elem = int(np.rint(R/dr))
    for i in range(0,elem,1):
        for j in range(0,elem,1):
            A+= dr*f[i,j]*r[j]*dr       
    return A
####################        number of particles       ########################
def print_num_part(u,Rf):
    tol = 0.001
    rho = np.linspace(tol, Rf - tol, 201) 
    dr = rho[1]-rho[0]
    N2 = numero_particulas(fi(u, rho,rho,Rf, n=1),rho,Rf,dr)
    _, dens = u.split()
    V = dens.function_space()
    mesh = V.mesh()    
    fx = Expression('a0*x[0] + b0', degree=1, a0 = 1, b0= 0, domain=mesh)
    ene = fx*dens*dx(domain=mesh)  # corregida
    Nn = assemble(ene)
    print('N=',Nn, 'N/rf= ', -Nn/Rf)
    return N2, Nn 

    ################       Viralization        ####################################
def virializacion(u):
    psi, phi = u.split()
    V = phi.function_space()
    mesh = V.mesh()
    fx = Expression('a0*x[0] + b0', degree=1, a0 = 1, b0= 0, domain=mesh)
    kee = -2.*np.pi*phi*div(fx*grad(phi))*dx(domain=mesh) 
#    we = 2.*np.pi*phi*phi*psi*dx 
    we = 2.*np.pi*phi*phi*psi*fx*dx(domain=mesh)  ### antes faltaba el fx    
    We = assemble(we)
    Ke = assemble(kee)
    print('K=', Ke, 'W=', We)
    print('K/W', Ke/abs(We))
    return Ke, We
    
def write_catidades_file(u,Rf, Ener, beta):
    N2, Nn = print_num_part(u,Rf)
    Ke, We = virializacion(u)
    psi, phi = u.split()
    psi.set_allow_extrapolation(True)
    phi.set_allow_extrapolation(True)
    print('Psi(Rf,0)=',psi(Rf,0))
    rf = Rf
    thf = np.pi/4
    rhof = rf*np.sin(thf)
    zf = rf*np.cos(thf)
    print('Psi(rf,pi/4)=',psi(rhof,zf))
    f= open("%s/%d/cantidades_%d.txt"%(dec,nc,ncor),"w+")
    f.write(" Ni=%f \r\n " %Nf)
    f.write(" N= %f, -N/rf= %f \r\n " % (Nn, -Nn/Rf))
    f.write(" N= %f, -N/rf= %f, Rf= %f \r\n " % (N2, -N2/Rf, Rf))
    f.write(" E_t= %f, Psi(Rf,0)= %f, Psi(rf,pi/4) = %f \r\n " % (Ener, psi(Rf,0), psi(rhof,zf)))
    f.write(" We= %f, Ke= %f, Ke/We= %f \r\n " % (We, Ke, Ke/abs(We)))
    f.write(" beta= %f, phif= %f \r\n " %(beta, phi(rhof,zf)))
    f.close()

def numero_particulas2(f,r,R,Z,dr,dz):
    "integral definida de r*f(r,z)**2 drdz  r de 0 a R y z de -R a R"
    A=0.
    elem = int(np.rint(R/dr))
    elemz = int(np.rint(Z/dz))
    for i in range(0,elemz,1):
        for j in range(0,elem,1):
            A+= dr*f[i,j]*dz*r[j]
    return A
###############################################################################
###############################################################################
###############################################################################
def plot_Nrhoz(u,Rf,ncor,direct=''):
    tol = 0.
    nr = 50
    nz = 51
    rho = np.linspace(tol, Rf - tol, nr + 1)
    z = np.linspace(tol, Rf - tol, nz + 1)
    Nxz=[]        
    for i in range(1,nr):
        print("%d percent"%i)
        aux = []
        for j in range(1,nz):
            if rho[i]**2 + z[j]**2 > Rf**2:
                aux.append(np.nan)
#                aux.append(0.)
            else:
                z2 = np.linspace(tol, z[j] - tol, j+1)
                rho2 = np.linspace(tol, rho[i] - tol, i+1)        
                dr = rho2[1]-rho2[0]
                dz = z2[1]-z2[0]
                Ne = numero_particulas2(fi(u, rho2,z2,Rf, n=2), rho2, rho[i], z[j], dr,dz)  
                aux.append(Ne)
        Nxz.append(aux)    
    Nxz = np.array(Nxz)
    Nxz = np.transpose(Nxz)    
    X, Z = np.meshgrid(rho[1:-1],z[1:-1])
    b = np.array([X,Z,Nxz])
    np.save("%s/Ntab_%d"%(direct,ncor),b)
    pts.densityplot(X,Z,Nxz,r"$\rho$ (kpc)", r"$z$ (kpc)",r'$N(\rho,z)$',
                    "", aspect = '1/1', name="%s/N_%d.png"%(direct,ncor))
################################################################################

def r95(ncor, dec=""):
    X, Z, Nxz = np.load("%s/Ntab_%d.npy"%(dec,ncor))
    print("Nmax=",np.amax(Nxz))
    N95 = 0.95*np.amax(Nxz)
    tol = 0.0008    
    u = np.where((Nxz < N95 + tol) & (Nxz > N95 -tol))
    print(u)
#    print(u[0][0],u[1][0])
    print("N95=",Nxz[u[0][0],u[1][0]])
    rho = X[u[0][0],u[1][0]]
    z = Z[u[0][0],u[1][0]]
#    print("rho",rho)
#    print("z",z)
    print("r95=",np.sqrt(rho**2 + z**2))
    return np.sqrt(rho**2 + z**2)
################################################################################
di2R10 = {1:{"Rf" : 9., "En" : 7.291, "expon" : -3, "fif" : 6.94e-2, #E_{r10}= 0.0059
              "N" : 2.289, "beta" : 0.,'rlam': 1.},### rescalamiento de la 10

          8:{"Rf" : 10., "En" : 2.45, "expon" : -1, "fif" : 7e-2,#E_{r10}= 0.245
              "N" : 2.750, "beta" : 0.,'rlam': 1.}, # falta virializar
          9:{"Rf" : 10., "En" : 2.45, "expon" : -1, "fif" : 7e-2,#E_{r10}= 0.245
              "N" : 3.100, "beta" : 0.,'rlam': 1.}, # falta virializar
             
          10:{"Rf" : 7.5, "En" : 1.05, "expon" : -2, "fif" : 1e-1,#E_{r10}=0.0059
              "N" : 2.747, "beta" : 0.,'rlam': 7.5/9.}, ## dipolo con toros 
              ##hay que bajar la N
##############              con autointeracción              ####################
##############    falta ver como calcular energia autointeraccion   ####################
          11:{"Rf" : 7.5, "En" : 1.04, "expon" : -2, "fif" : 1e-1,#E_{r10}=0.0058
              "N" : 2.750, "beta" : 0.005},             
##############              LA BUENA  ¿Dipole?             ####################
          12:{"Rf" : 7.5, "En" : 1.73, "expon" : -2, "fif" : 1e-1,#E_{r10}=0.0097
              "N" : 1.369, "beta" : 0.0,'rlam': 1.}, 
###############################################################################             

#          13:{"Rf" : 3.5, "En" : 10.4, "expon" : -1, "fif" : 0.5,#E_{r10}=1.04
#              "N" : 3.8, "beta" : 0.0},  ## art paco y luis
#          13:{"Rf" : 5.0, "En" : 10.4, "expon" : -1, "fif" : 0.4,#E_{r10}=1.04
#              "N" : 4.04, "beta" : 0.0},  ## art paco y luis
          17 :{"Rf" : 10., "En": 2.45, "expon" : -1, "fif" : 1e-2/np.sqrt(2.),#E_{r10}=0.245
               "N" : 5., "beta" : 0.0, 'rlam':1.}###la de las 4 bolitas
          }
#################################################################################
#################################################################################
#################################################################################           

#de = float(input("run"))
#print(de, type(de))

#nc = float(de)
#nc = 2750

#ncor = float(de)
ncor = 12

dec= '/home/jordi/gross_hidro'

#for ncor in (1,2,3,8,9,10,11,12,13,14,15,16,17):

expon = di2R10[ncor]["expon"]
En = di2R10[ncor]["En"]  
fif = di2R10[ncor]['fif']
Rf = di2R10[ncor]['Rf']

#Nf = nc/1000.

Nf = di2R10[ncor]["N"] 

Vf = -Nf/Rf  
 

###############         Escalamiento     #####################################
##if Rf ==7.5: 
##
#rlam = di2R10[ncor]['rlam']
##rlam = Rf/5.
#
#Rf = Rf/rlam
#Nf = Nf*rlam
#En = rlam**2*En
#Vf = rlam**2*Vf
#fif = rlam**2*fif

##############################################################################
nc = Nf*1000

Ener = En*10**expon
beta = di2R10[ncor]['beta']

print('N=',nc/1000)
print('bet=',beta)
print('Nr=', Nf)
print('ncor=',ncor)
print('Rf=',Rf)
print('E=',Ener)
print('nc=',nc)

print('fif=',fif)

mainprog(ncor, Rf= Rf, E= Ener, Vf = Vf, densf =fif, Nf = Nf, R0 = 0.1,
         lam = beta, direct = "%s/%d"%(dec,nc)) 

u = read_saved_file(ncor, direct = "%s/%d"%(dec,nc))
psi_1, dens_1 = u.split() 
u.set_allow_extrapolation(True)  

graf(dens_1, '', dec, name='dens_%d'%ncor, zlabel= r'$\Phi^2$', rango = False,
         vmin=0, vmax=10)

#write_catidades_file(u,Rf, Ener, beta)

#plot_Nrhoz(u,Rf, ncor, direct = "%s/%d"%(dec,nc)) 
#print_num_part(u,Rf)
#r95 = r95(ncor, dec = "%s/%d"%(dec,nc))
#density3d_files(u,Rf,"%s/%d/matrix_%d.CSV"%(dec,nc,ncor))
