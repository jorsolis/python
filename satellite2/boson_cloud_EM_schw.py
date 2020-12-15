#!/usr/bin/env python
# coding: utf-8
##############################################################################
######       Boson cloud in Schwarzschild With electromagnetic field      ####
#############################################################################
from dolfin import (CompiledSubDomain, Constant, DirichletBC, dot, ds, dx,
                    Expression, File, FiniteElement, Function, FunctionSpace,
                    grad, HDF5File, Measure, Mesh, MeshFunction, MixedElement,
                    nabla_grad, near, plot, Point, project, solve, split,
                    SubDomain, TestFunctions, triangle, VectorFunctionSpace,
                    assemble,div,SpatialCoordinate, cos, UnitIntervalMesh)
from mshr import Rectangle,generate_mesh, Circle
import numpy as np
import plots_jordi as pts

def mainprog(ncor, Rf = 10, fif = .001, M_BH = 0.1, mu_EM = 1, omega = 0.,
             jey = 1., mu_SF = 1., lambd = 0.,
             direct = '/home/jordi/boson_cloud', R0 = 0.25):
###########################     MESH         ##################################  
    circulo = Circle(Point(0.0,0.0), Rf) - Circle(Point(0.0,0.0), R0) - Rectangle(Point(-Rf,-Rf),Point(0.,Rf)) - Rectangle(Point(0.,-Rf), Point(Rf,0.))
    mesh = generate_mesh(circulo, 164)
#    plot(mesh)
#############               Save mesh to a file           #####################
    File('%s/mesh_boson_cloud_%d.xml.gz'%(direct,ncor)) << mesh
##############           Define boundaries for  Newman conditions       ########
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
            if on_boundary:
                if near(x[1], 0, tol):
                    return True                            
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
    def circlebord(x, on_boundary):
        if on_boundary:
            if near(x[0], 0, tol):
                return False
            elif near(x[1],0, tol):
                return False
            elif near(x[1], 0, R0) and near(x[0], 0, R0):
                return False
            else:
                return True
        else:
            return False        
        
    left = leftwall()
    bottom = bottomwall()
    no_origen = no_orig()
        
    left.mark(boundary_markers, 2)
    bottom.mark(boundary_markers, 1)
    no_origen.mark(boundary_markers, 3)
    ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)
#####################     Define function space       ##########################
    P1 = FiniteElement('P', triangle, 1)
    V = FunctionSpace(mesh, MixedElement([P1, P1, P1]))
####################      Define test functions      ############################
    v_1, v_2, v_3= TestFunctions(V)
#############          Define functions for SF and potential      ##############
    u = Function(V)
##############         Split system functions to access components      ########
    At, Aph, Phi = split(u)
################          Define expressions used in variational forms       ###
#    el = V.ufl_element()
    rfs = Expression('sqrt(x[0]*x[0] + x[1]*x[1]) - 2*M', degree=1, M = M_BH)
    rho = Expression('x[0]', degree=1)
    rho2 = Expression('x[0]*x[0]',degree=2)
    r2 = Expression('x[0]*x[0] + x[1]*x[1]', degree=2)
    r = Expression('sqrt(x[0]*x[0] + x[1]*x[1])', degree=1)
    rm1 = Expression('1/sqrt(x[0]*x[0] + x[1]*x[1])', degree=1)
    rfs2 = Expression('3*sqrt(x[0]*x[0] + x[1]*x[1]) - 4*M',degree=2, M = M_BH)
    x = Expression(('x[0]','x[1]'), degree=1)
#####################         Constants           #############################
    M = Constant(M_BH)
    mu0 = Constant(mu_EM)
    g1 = Constant(0.)
    g2 = Constant(0.)
    g3 = Constant(0.)
    g4 = Constant(0.)
    g5 = Constant(0.)
    g6 = Constant(0.)
    w = Constant(omega)
    j = Constant(jey)
    lam = Constant(lambd)
    mu = Constant(mu_SF)    
###############          Define variational problem     #########################
    F = rho*r*r2*rfs*dot(grad(At), grad(v_1))*dx \
    - rho*r*rfs2*v_1*dot(x, grad(At))*dx \
    + g1*v_1*ds(1) + g1*v_1*ds(2) + g1*v_1*ds(3) - g2*v_1*ds(1) - g2*v_1*ds(2) - g2*v_1*ds(3) \
    + 3.*rho*M*rfs*v_1*dot(x, grad(At))*dx + 2.*rho*M*rfs*dot(x, grad(v_1))*dot(x, grad(At))*dx \
    + 2.*v_1*M*rho*(rfs + r)*dot(x, grad(At))*dx + 4.*M*M*M*rm1*rho*At*v_1*dx \
    - 4.*mu0*rho*r2*r*Phi*Phi*(rfs*At - w*r)*v_1*dx
    - rho2*r2*r*dot(grad(Aph), grad(v_2))*dx + rho*v_2*r*r2*Aph.dx(0)*dx \
    - 2.*rho2*r*v_2*dot(x, grad(Aph))*dx \
    + g3*v_2*ds(1) + g3*v_2*ds(2) + g3*v_2*ds(3) - g4*v_2*ds(1) - g4*v_2*ds(2) - g4*v_2*ds(3) \
    + 2.*M*rho2*dot(x, grad(v_2))*dot(x, grad(Aph))*dx + 2.*v_2*M*(2.*rho2)*dot(x, grad(Aph))*dx \
    - 2.*M*rho2*v_2*Aph*dx - 4.*mu0*Phi*Phi*r*r2*(rho2*Aph + j)*v_2*dx \
    - rho2*r2*rfs*dot(grad(Phi), grad(v_3))*dx - rho*v_3*r2*rfs*Phi.dx(0)*dx \
    - rho2*v_3*rfs2*dot(x, grad(Phi))*dx \
    + g5*v_3*ds(1) + g5*v_3*ds(2) + g5*v_3*ds(3) - g6*v_3*ds(1) - g6*v_3*ds(2) - g6*v_3*ds(3) \
    + 4.*M*rfs*rm1*rho2*v_3*dot(x, grad(Phi))*dx + 2.*M*rfs*rm1*rho2*dot(x, grad(v_3))*dot(x, grad(Phi))*dx \
    + 2.*M*v_3*rho2*(r - M)*rm1*dot(x, grad(Phi))*dx \
    - rho2*(mu*mu + lam*Phi*Phi)*(r2 - 2.*M*r)*r*Phi*v_3*dx \
    - 2.*rho*(r2 - 2.*M*r)*r*(At*w + Aph*j)*Phi*v_3*dx \
    + rho*(r2 - 2.*M*r)*(rfs*At*At - r*rho2*Aph*Aph)*Phi*v_3*dx \
    - Phi*(rfs*j*j - rho2*r*w*w)*v_3*r2*dx
###############          boundary conditions         ###########################
    bcAt = DirichletBC(V.sub(0), Constant(0.) , bottom) # on At
    bcAt1 = DirichletBC(V.sub(0), Constant(0.) , left) # on At
    bcAt2 = DirichletBC(V.sub(0), Constant(0.) , circlebord) # on At

    bcAp = DirichletBC(V.sub(1), Constant(0.) , bottom) # on Aph
    bcAp1 = DirichletBC(V.sub(1), Constant(0.) , left) # on Aph
    bcAp2 = DirichletBC(V.sub(1), Constant(0.) , circlebord) # on Aph

    bcP = DirichletBC(V.sub(2), Constant(fif) , circlebord)  # on Phi
    bcP1 = DirichletBC(V.sub(2), Constant(0.) , bottom)  # on Phi
################           Solve variational problem         ###################
    solve(F == 0, u, [bcAt, bcAt1, bcAt2, bcAp, bcAp1, bcAp2, bcP, bcP1]) 
#################          Save solution               #########################
    output_file = HDF5File(mesh.mpi_comm(), "%s/u_%d.h5"%(direct,ncor), "w")
    output_file.write(u, "solution")
    output_file.close() 
#     
def read_saved_file(ncor, direct = '/home/jordi/boson_cloud'):
    ################         Read mesh         #################################
    mesh = Mesh('%s/mesh_boson_cloud_%d.xml.gz'%(direct,ncor))
    P1 = FiniteElement('P', triangle, 1)
    V = FunctionSpace(mesh, MixedElement([P1, P1, P1]))
    ###############    Load solution        ###################################
    u = Function(V)
    input_file = HDF5File(mesh.mpi_comm(), "%s/u_%d.h5"%(direct,ncor), "r")
    input_file.read(u, "solution")
    input_file.close()
    return u  
#################################################################################
#################################################################################
#################################################################################           
if __name__ == '__main__':
    ncor = 1    
    mainprog(ncor, Rf = 10, fif = .001, M_BH = 0.1, mu_EM = 0., omega = 0.,
             jey = 0., mu_SF = 1., lambd = 0., R0 = 0.25) 

