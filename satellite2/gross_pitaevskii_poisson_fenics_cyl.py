#!/usr/bin/env python
# coding: utf-8
##############################################################################
###############         ES MAS GENERAL gross_pitaevskii_poisson_fenics_cyl_con_m
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

def mainprog(ncor, Rf = 10, E= 0.034465, Vf = -0.1, fif = .001,Nf=1.,
             direct = '/home/jordi/grossN', bet = 0., R0 = 0.01):
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
        
    left.mark(boundary_markers, 2)
    bottom.mark(boundary_markers, 1)
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
    psi, Phi = split(u)
###############          Define expressions used in variational forms       ###
    fx = Expression('a0*x[0] + b0', degree=1, a0 = 1, b0= 0)#
####################         Constants           #############################
    k = Constant(E)
    g1 = Constant(0.)
    g2 = Constant(0.) #ésta está bien
    beta = Constant(bet)
##############          Define variational problem     #########################
    F = fx*dot(grad(psi), grad(v_1))*dx  + fx*Phi*Phi*v_1*dx \
    + fx*dot(grad(Phi), grad(v_2))*dx + 2.*fx*psi*Phi*v_2*dx + 2.*fx*k*Phi*v_2*dx \
    + 2.*fx*beta*Phi*Phi*Phi*v_2*dx \
    - g1*v_1*ds(1) - g2*v_1*ds(2) - g1*v_2*ds(2)# \
#    - g2*v_2*ds(3) - g2*v_1*ds(3)
#############           Define boundaries             #########################
    def circlebord(x, on_boundary):
        if on_boundary:
            if near(x[0], 0, tol):
                return False
            elif near(x[1],0, tol):
                return False
#            elif near(x[1], 0, R0) and near(x[0], 0, R0):
#                return False
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
    output_file = HDF5File(mesh.mpi_comm(), "%s/u_%d.h5"%(direct,ncor), "w")
    output_file.write(u, "solution")
    output_file.close() 
     
def read_saved_file(ncor, direct = '/home/jordi/grossN'):
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
    N2 = numero_particulas(fi(u, rho,rho,Rf, n=2),rho,Rf,dr)
    _, phi = u.split()
    V = phi.function_space()
    mesh = V.mesh()    
    fx = Expression('a0*x[0] + b0', degree=1, a0 = 1, b0= 0, domain=mesh)
    ene = fx*phi*phi*dx(domain=mesh)
    Nn = assemble(ene)
    print('N=',Nn, 'N/rf= ', -Nn/Rf)
    return N2, Nn 

    ################       Viralization        ####################################
def virializacion(u, deg = 1):
    psi, phi = u.split()
    V = phi.function_space()
    mesh = V.mesh()
    fx = Expression('a0*x[0] + b0', degree=deg, a0 = 1, b0= 0, domain=mesh)
    kee = -2.*np.pi*phi*div(fx*grad(phi))*dx(domain=mesh) 
    we = 2.*np.pi*phi*phi*psi*fx*dx(domain=mesh)  ### antes faltaba el fx    
    We = assemble(we)
    Ke = assemble(kee)
    print('K = ', Ke, 'W = ', We)
    print('K/W = ', Ke/abs(We))
    return Ke, We
    
def write_catidades_file(u,Rf, Ener, beta, dec = ''):
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
#                aux.append(np.nan)
                aux.append(0.)
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
#    pts.densityplot(X,Z,Nxz,r"$\rho$ (kpc)", r"$z$ (kpc)",r'$N(\rho,z)$',
#                    "", aspect = '1/1', name="%s/N_%d.png"%(direct,ncor))
################################################################################

def r95(ncor, dec=""):
    X, Z, Nxz = np.load("%s/Ntab_%d.npy"%(dec,ncor))
    print("Nmax=",np.amax(Nxz))
    N95 = 0.95*np.amax(Nxz)
    tol = 0.0008    
    try:
        u = np.where((Nxz < N95 + tol) & (Nxz > N95 -tol))
    except IndexError:
        print('subir tolerancia')

    rho = X[u[0][0],u[1][0]]
    z = Z[u[0][0],u[1][0]]
    print("r95=",np.sqrt(rho**2 + z**2))
    f= open("%s/r95_%d.txt"%(dec,ncor),"w+")
    f.write("Nmax = %f \r\n " %np.amax(Nxz))
    f.write("r95 = %f \r\n " %np.sqrt(rho**2 + z**2))
    f.write("N95 = %f \r\n " %N95)
    f.close()
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

          13:{"Rf" : 5., "En" : 11.4, "expon" : -1, "fif" : 1e-5,
              "N" : 4.0399, "beta" : 0.0},  ## art paco y luis
          17 :{"Rf" : 10., "En": 2.45, "expon" : -1, "fif" : 1e-2/np.sqrt(2.),#E_{r10}=0.245
               "N" : 5., "beta" : 0.0, 'rlam':1.}###la de las 4 bolitas
          }
#################################################################################
#################################################################################
#################################################################################           
if __name__ == '__main__':
#    de = float(input("run"))
#    print(de, type(de))
    
#    nc = float(de)
#    nc = 2750
    
#    ncor = float(de)
    ncor = 13
    
    dec= '/home/jordi/grossN3'
    
    #for ncor in (1,2,3,8,9,10,11,12,13,14,15,16,17):
    
    expon = di2R10[ncor]["expon"]
    En = di2R10[ncor]["En"]  
    Rf = di2R10[ncor]['Rf']
    fif = di2R10[ncor]['fif']
    
#    Nf = nc/1000.
    
    Nf = di2R10[ncor]["N"] 
    
    Vf = -Nf/Rf  
     
    
    ##############         Escalamiento     #####################################
    #if Rf ==7.5: 
#    rlam = di2R10[ncor]['rlam']

#    rnew = 5.
#    rlam = Rf/rnew
#    Rf = Rf/rlam
#    Nf = Nf*rlam
#    En = rlam**2*En
#    Vf = rlam**2*Vf
#    fif = rlam**2*fif
#    print('E=', En*10**expon, 'Rf=', Rf, 'Nf =', Nf)
    
    #############################################################################
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
#    
    mainprog(ncor, Rf= Rf, E= Ener, Vf = Vf, fif =fif, Nf = Nf, R0 = 0.1,
             bet = beta, direct = "%s/%d"%(dec,nc)) 
#    
    u = read_saved_file(ncor, direct = "%s/%d"%(dec,nc))
    u.set_allow_extrapolation(True)  
    plots_pyplot(u,Rf, En, expon, ncor,ruta="%s/%d"%(dec,nc), show = False,
                 dens=True, pot=False, otras = False)
#    write_catidades_file(u,Rf, Ener, beta, dec = dec)
#    
#    plot_Nrhoz(u,Rf, ncor, direct = "%s/%d"%(dec,nc)) 
#    print_num_part(u,Rf)
#    r95 = r95(ncor, dec = "%s/%d"%(dec,nc))
#    density3d_files(u,Rf,"%s/%d/matrix_%d.CSV"%(dec,nc,ncor))
