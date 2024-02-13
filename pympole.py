import numpy as np
import matplotlib.pyplot as plt
import os, math, time
from scipy import integrate, misc
import scipy.special as func
import scipy.constants as const
from scipy.interpolate import interpn, RegularGridInterpolator
from mpl_toolkits.mplot3d import Axes3D
import scipy.integrate as integ
import tmm
from numba import jit

class MultipoleDecomp:
     
    def __init__(self):
        self.__r0 = np.array([0,0,0])
        self.__is_subs = False
        self.__is_pec_subs = False
        self.__is_multilayer = False
        self.__is_set_eff_point = False


    def __SetParticleParameters(self, eps_p, mu_p = 1):
        """
        Initialization of input parameters:
        eps_p - dielectric ralative permittivity of the particle
        mu_p - magnetic ralative permittivity of the particle (by default 1)
        """
        self.__eps_p = eps_p
        self.__mu_p = mu_p


    def __SetMediumParameters(self, wl, eps_d = 1,  mu_d = 1):
        """
        Initialization of input parameters:
        wl - wavelength, nm
        eps_d - dielectric ralative permittivity of the surrounding medium (by default 1)
        mu_d - magnetic ralative permittivity of the surrounding medium (by default 1)
        """
        self.__wl = wl*1e-9
        self.__k_0 = 2*np.pi/self.__wl
        self.__omega = 2*np.pi*const.c/self.__wl
        self.__eps_d = eps_d
        self.__mu_d = mu_d
        self.__v_d = const.c/(self.__eps_d)**(1/2)
        self.__k_d = self.__k_0*(self.__eps_d)**(1/2)


    def __LoadMultilayers(self, eps_list, d_list):
        self.__is_multilayer = True
        self.__eps_list = eps_list
        self.__d_list = d_list


    def __DetermeineEffectiveDipolePoint(self):
        if not self.__is_set_eff_point:
            self.__r0 = np.array([
                (self.__particle_points_list_xyz[:,0].min()+self.__particle_points_list_xyz[:,0].max())/2,
                (self.__particle_points_list_xyz[:,1].min()+self.__particle_points_list_xyz[:,1].max())/2,
                (self.__particle_points_list_xyz[:,2].min()+self.__particle_points_list_xyz[:,2].max())/2])
        print("Dipole Point is " + str(self.__r0))


    def SetEffectiveDipolePoint(self, r0 = np.array([0,0,0])):   
        self.__r0 = r0
        self.__is_set_eff_point = True
   

    def __SetSubstrateParameters(self, eps_s, mu_s=1):
        self.__eps_s = eps_s
        self.__mu_s = mu_s
        self.__is_subs = True
        self.__is_pec_subs = False
    

    def SetPECSubstrate(self):
        self.__is_subs = True
        self.__is_pec_subs = True
    
    
    def __LoadEField(self, e_field):

        """
        Electric field interpolation

        Input file should have the following format:
        x[nm] y[nm] z[nm] ExRe[V/m] ExIm[V/m] EyRe[V/m] EyIm[V/m] EzRe[V/m] EzIm[V/m]
        -----------------------------------------------------------------------------

        return:
        Field[component, Real/Imaginary part]

        1 - x
        2 - y
        3 - z

        0 - Real
        1 - Imaginary


        Example:

        EField[0,1]([0,0,0]) 
        x-component; imaginary part in point: x=0; y=0; z=0 ( Im(Fx(0,0,0)) )

        """       
        # print(e_field)
#         step_x = abs(e_field[1,0]-e_field[0,0])
#         n_x = int(round((e_field[:,0].max()-e_field[:,0].min())/step_x)+1)
#         x = e_field[:n_x,0]

#         step_y = abs(e_field[x.shape[0]+1,1]-e_field[0,1])
#         n_y = int(round((e_field[:,1].max()-e_field[:,1].min())/step_y)+1)
#         y = e_field[:n_y*n_x:n_x,1]

#         z = e_field[::n_y*n_x,2]

        x = np.unique(e_field[:,0])
        y = np.unique(e_field[:,1])
        z = np.unique(e_field[:,2])

        e_field_x_Re =e_field[:,3].reshape((z.shape[0],y.shape[0],x.shape[0]))
        e_field_inter_x_Re = RegularGridInterpolator((z, y, x),e_field_x_Re)
        e_field_x_Im =e_field[:,4].reshape((z.shape[0],y.shape[0],x.shape[0]))
        e_field_inter_x_Im = RegularGridInterpolator((z, y, x),e_field_x_Im)


        e_field_y_Re =e_field[:,5].reshape((z.shape[0],y.shape[0],x.shape[0]))
        e_field_inter_y_Re = RegularGridInterpolator((z, y, x),e_field_y_Re)
        e_field_y_Im =e_field[:,6].reshape((z.shape[0],y.shape[0],x.shape[0]))
        e_field_inter_y_Im = RegularGridInterpolator((z, y, x),e_field_y_Im)


        e_field_z_Re =e_field[:,7].reshape((z.shape[0],y.shape[0],x.shape[0]))
        e_field_inter_z_Re = RegularGridInterpolator((z, y, x),e_field_z_Re)
        e_field_z_Im =e_field[:,8].reshape((z.shape[0],y.shape[0],x.shape[0]))
        e_field_inter_z_Im = RegularGridInterpolator((z, y, x),e_field_z_Im)


        e_field_inter_x = np.array([e_field_inter_x_Re,e_field_inter_x_Im])
        e_field_inter_y = np.array([e_field_inter_y_Re,e_field_inter_y_Im])
        e_field_inter_z = np.array([e_field_inter_z_Re,e_field_inter_z_Im])

        self.__full_interp_e_field = np.array([e_field_inter_x,e_field_inter_y,e_field_inter_z])
        

    def __StepDet(self, arr):
        temp = arr[0]
        for i in arr[1:]:
            step = abs(temp-i)
            if step>0:
                return step

    
    def __LoadParticlePointList(self, point_list, plotting=False, step_print = False):
        """
        Particle coordinate in cortesian coordinates in formart:
         x[nm] y[nm] z[nm]
        ------------------
        Create private variable __particle_points_list
        """
        self.__particle_points_list_xyz = np.array([point_list[:,0], point_list[:,1], point_list[:,2]]).transpose()
        self.__dx = self.__StepDet(self.__particle_points_list_xyz[:,0])
        self.__dy = self.__StepDet(self.__particle_points_list_xyz[:,1])
        self.__dz = self.__StepDet(self.__particle_points_list_xyz[:,2])
        
        if plotting:
            self.PlotParticle()
        if step_print:
            print ("dx = "+ str(self.__dx)+  " nm; dy = "+ str(self.__dy)+ " nm; dz = "+ str(self.__dz) + " nm")

    
    def PlotParticle(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title("Particle Points")
        ax.scatter(self.__particle_points_list_xyz[:,0], self.__particle_points_list_xyz[:,1], self.__particle_points_list_xyz[:,2])
        ax.set_xlabel("x, nm")
        ax.set_ylabel("y, nm")
        ax.set_zlabel("z, nm")
        plt.tight_layout()
        plt.show()
         
    
    def __ECart(self, x, y, z):
        """
        cartesian
        return complexe electric field in format array[Ex,Ey,Ez] in the point(x, y, z)
        """
        E = np.array([self.__full_interp_e_field[0,0]([z,y,x]) - 1j * self.__full_interp_e_field[0,1]([z,y,x]),
                      self.__full_interp_e_field[1,0]([z,y,x]) - 1j * self.__full_interp_e_field[1,1]([z,y,x]),
                      self.__full_interp_e_field[2,0]([z,y,x]) - 1j * self.__full_interp_e_field[2,1]([z,y,x])])
        return E
   
    
    def __CalculateMoments(self):
        nm = 1e-9
        U = np.identity(3)
        self.__p = np.zeros(3, dtype=complex) # electric dipole moment
        self.__m = np.zeros(3, dtype=complex) # magnetic dipole moment
        self.__T = np.zeros(3, dtype=complex) # toroidal moment
        self.__Q = np.zeros((3,3), dtype=complex) # electric quadrapole moment
        self.__M = np.zeros((3,3), dtype=complex) # magnetic quadrapole moment
        self.__O = np.zeros((3,3,3), dtype=complex) # electric octapole moment
        for r_ in self.__particle_points_list_xyz:
            r = r_ - self.__r0
            P = (const.epsilon_0 * (self.__eps_p - self.__eps_d) * self.__ECart(r_[0], r_[1], r_[2]))[:,0]
            self.__p += P 
            self.__m += np.cross(r * nm, P)
            self.__T += (2 * np.dot(r * nm, r * nm) * P - np.dot(r * nm, P) * (r * nm))
                        
            # self.__Q += np.outer(r * nm, P) + np.outer(P, r * nm)
            # self.__M += np.outer(np.cross(r * nm, P), r * nm) 

            self.__Q += np.outer(r * nm, P) + np.outer(P, r * nm) -2/3*np.dot(P, r * nm)*U
            self.__M += np.outer(np.cross(r * nm, P), r * nm) + np.outer(r * nm, np.cross(r * nm, P)) 

            self.__O += np.tensordot(np.tensordot(P, r * nm, axes=0), r * nm, axes=0) + np.tensordot(np.tensordot(r * nm, P, axes=0), r * nm, axes=0) + np.tensordot(np.tensordot(r * nm, r * nm, axes=0), P, axes=0)
            
        
        dr = (self.__dx * self.__dy * self.__dz) * nm**3
        self.__p *= dr
        self.__m *= -0.5j * self.__omega * dr 
        self.__T *= 0.1j * self.__omega * dr
        # self.__M *= -2j*self.__omega * dr/3
        self.__Q *= 3 * dr
        self.__M *= -1j*self.__omega * dr/3
        self.__O *= -dr
        
        return [self.__p, self.__T, self.__m, self.__Q, self.__M, self.__O]


    
    def __CalculateSCS(self, limits = [[0, np.pi/2], [0, np.pi*2]]):

        x_mat = np.zeros((3,3))
        y_mat = np.zeros((3,3))
        z_mat = np.zeros((3,3))
        x_mat[0,0]=1
        y_mat[1,1]=1
        z_mat[2,2]=1

        if not self.__is_subs and not self.__is_multilayer:
            self.__scs_ed = self.__k_0**4 / (6 * np.pi * const.epsilon_0**2) * ((abs(self.__p)**2).sum())
            self.__scs_ted = self.__k_0**4 / (6 * np.pi * const.epsilon_0**2) * ((abs(self.__p + 1j*self.__k_d / self.__v_d * self.__T)**2).sum())
            self.__scs_md = self.__k_0**4 * self.__eps_d * const.mu_0 / (6 * np.pi * const.epsilon_0) * (abs(self.__m)**2).sum()
            self.__scs_eq = (abs(self.__Q)**2).sum() * self.__eps_d**2 * self.__k_0**6 / (720 * np.pi * const.epsilon_0**2)
            self.__scs_mq = (abs(self.__M)**2).sum() * self.__k_0**6 * self.__eps_d**2 * const.mu_0 / (80*np.pi*const.epsilon_0)
            self.__scs_eoс = (abs(self.__O)**2).sum() * self.__k_0**8 * self.__eps_d**2 / (1890*np.pi*const.epsilon_0**2)
            self.__total_ed = abs(self.__scs_ed + self.__scs_md + self.__scs_eq + self.__scs_mq + self.__scs_eoс)
            self.__total_ted = abs(self.__scs_ted + self.__scs_md + self.__scs_eq + self.__scs_mq + self.__scs_eoс)



            self.__scs_ed_comp = self.__k_0**4 / (6 * np.pi * const.epsilon_0**2) * (abs(self.__p)**2)
            self.__scs_ted_comp = self.__k_0**4 / (6 * np.pi * const.epsilon_0**2) * ((abs(self.__p + 1j*self.__k_d / self.__v_d * self.__T)**2))
            self.__scs_md_comp = self.__k_0**4 * self.__eps_d * const.mu_0 / (6 * np.pi * const.epsilon_0) * (abs(self.__m)**2)


            return np.array([self.__scs_ed, self.__scs_ted, self.__scs_md, self.__scs_eq, self.__scs_mq, self.__scs_eoс, self.__total_ed, self.__total_ted,
             self.__scs_ed_comp [0], self.__scs_ed_comp [1], self.__scs_ed_comp [2], 
             self.__scs_ted_comp[0], self.__scs_ted_comp[1], self.__scs_ted_comp[2], 
             self.__scs_md_comp[0], self.__scs_md_comp[1], self.__scs_md_comp[2]])
        else:
            if self.__is_pec_subs:
                r_s = lambda theta: -1
                r_p = lambda theta: 1
            elif self.__is_multilayer:
                r_s = lambda theta: tmm.coh_tmm('s', self.__eps_list**0.5, self.__d_list, theta, self.__wl)['r']
                r_p = lambda theta: tmm.coh_tmm('p', self.__eps_list**0.5, self.__d_list, theta, self.__wl)['r']
            else:
                r_s = lambda theta: (self.__mu_s * (self.__eps_d - self.__eps_d  * np.sin(theta)**2)**0.5 - self.__mu_d * (self.__eps_s - self.__eps_d  * np.sin(theta)**2)**0.5) / (self.__mu_s * (self.__eps_d - self.__eps_d * np.sin(theta)**2)**0.5 + self.__mu_d * (self.__eps_s - self.__eps_d  * np.sin(theta)**2)**0.5)
                r_p = lambda theta: (self.__eps_s * (self.__eps_d - self.__eps_d  * np.sin(theta)**2)**0.5 - self.__eps_d * (self.__eps_s - self.__eps_d  * np.sin(theta)**2)**0.5) / (self.__eps_s * (self.__eps_d - self.__eps_d * np.sin(theta)**2)**0.5 + self.__eps_d *(self.__eps_s - self.__eps_d  * np.sin(theta)**2)**0.5)

            r_0 = self.__r0 * 1e-9
            
            n = lambda theta, phi: np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
            n_ = lambda theta, phi: np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), -np.cos(theta)])

            # limits = [[0, np.pi/2], [0, np.pi*2]]

            coef_e = (self.__k_0**2 / (4 * np.pi * const.epsilon_0))
            coef_m = ((const.mu_0 / const.epsilon_0)**0.5 * self.__k_0 * self.__k_d / (4 * np.pi))
            
        
            E_e_phi = lambda theta, phi, e_mom_d, e_mom_r: (e_mom_d[1] * np.cos(phi) - e_mom_d[0] * np.sin(phi)) + r_s(theta) * np.exp(2j * self.__k_d * r_0[2] * np.cos(theta))* (e_mom_r[1] * np.cos(phi) - e_mom_r[0] * np.sin(phi))
            E_e_theta = lambda theta, phi, e_mom_d, e_mom_r: e_mom_d[0] * np.cos(phi) * np.cos(theta) + e_mom_d[1] * np.sin(phi) * np.cos(theta) - r_p(theta) * np.exp(2j * self.__k_d * r_0[2] * np.cos(theta)) * (e_mom_r[0] * np.cos(phi) * np.cos(theta) + e_mom_r[1] * np.sin(phi) * np.cos(theta)) - (e_mom_d[2] * np.sin(theta) + e_mom_r[2] * np.sin(theta) *r_p(theta) * np.exp(2j * self.__k_d * r_0[2] * np.cos(theta)))
            E_e_mod_sq = lambda theta, phi, e_mom_d, e_mom_r: (abs(E_e_theta(theta, phi, e_mom_d, e_mom_r))**2 + abs(E_e_phi(theta,phi, e_mom_d, e_mom_r))**2) * np.sin(theta)

            E_m_phi = lambda theta, phi, m_mom_d, m_mom_r: (r_s(theta) * np.exp(2j * self.__k_d * r_0[2] * np.cos(theta)) * (m_mom_r[0] * np.cos(phi) * np.cos(theta) + m_mom_r[1] * np.sin(phi) * np.cos(theta)) - (m_mom_d[0] * np.cos(phi) * np.cos(theta) + m_mom_d[1] * np.sin(phi) * np.cos(theta))) + (m_mom_r[2] * np.sin(theta) * r_s(theta) * np.exp(2j * self.__k_d * r_0[2] * np.cos(theta)) + m_mom_d[2] * np.sin(theta))
            E_m_theta = lambda theta, phi, m_mom_d, m_mom_r: m_mom_d[1] * np.cos(phi) - m_mom_d[0] * np.sin(phi) + r_p(theta) * np.exp(2j * self.__k_d * r_0[2] * np.cos(theta)) * (m_mom_r[1] * np.cos(phi) - m_mom_r[0] * np.sin(phi))
            E_m_mod_sq = lambda theta, phi, m_mom_d, m_mom_r: (abs(E_m_theta(theta,phi,m_mom_d, m_mom_r))**2 + abs(E_m_phi(theta,phi,m_mom_d, m_mom_r))**2) * np.sin(theta)

            imag_one = -1j
            eq_mom_d = lambda theta, phi: np.dot(self.__Q, n(theta, phi)) * imag_one * self.__k_d / 6
            eq_mom_r = lambda theta, phi: np.dot(self.__Q, n_(theta, phi)) * imag_one * self.__k_d / 6
            mq_mom_d = lambda theta, phi: np.dot(self.__M, n(theta, phi)) * imag_one * self.__k_d / 2 
            mq_mom_r = lambda theta, phi: np.dot(self.__M, n_(theta, phi)) * imag_one * self.__k_d / 2 
            
            E_eq_mod_sq = lambda theta, phi: E_e_mod_sq(theta, phi, eq_mom_d(theta, phi), eq_mom_r(theta, phi))
            E_mq_mod_sq = lambda theta, phi: E_m_mod_sq(theta, phi, mq_mom_d(theta, phi), mq_mom_r(theta, phi))

            E_mod_sq_total = lambda theta, phi, ed_mom_d, ed_mom_r, md_mom_d, md_mom_r: (abs( coef_e * (E_e_theta(theta, phi, ed_mom_d, ed_mom_r) + E_e_theta(theta, phi, eq_mom_d(theta, phi), eq_mom_r(theta, phi))) + coef_m * (E_m_theta(theta, phi, md_mom_d, md_mom_r) + E_m_theta(theta, phi, mq_mom_d(theta, phi), mq_mom_r(theta, phi))))**2 + abs(coef_e * (E_e_phi(theta, phi, ed_mom_d, ed_mom_r) + E_e_phi(theta, phi, eq_mom_d(theta, phi), eq_mom_r(theta, phi))) + coef_m * (E_m_phi(theta, phi, md_mom_d, md_mom_r) + E_m_phi(theta, phi, mq_mom_d(theta, phi), mq_mom_r(theta, phi)))  )**2 ) * np.sin(theta)
            
            self.__scs_ed = coef_e**2 * integ.nquad(E_e_mod_sq, limits, args=(self.__p, self.__p))[0]
            self.__scs_ted = coef_e**2 * integ.nquad(E_e_mod_sq, limits, args=(self.__p + 1j * self.__k_d**2 / self.__omega * self.__T, self.__p + 1j * self.__k_d**2 / self.__omega * self.__T))[0]
            self.__scs_md = coef_m**2 * integ.nquad(E_m_mod_sq, limits, args=(self.__m, self.__m))[0]
            self.__scs_eq = coef_e**2 * integ.nquad(E_eq_mod_sq, limits)[0]
            self.__scs_mq = coef_m**2 * integ.nquad(E_mq_mod_sq, limits)[0]
            self.__scs_eoс = 0 ########
            

            # self.__scs_ed_x = coef_e**2 * integ.nquad(E_e_mod_sq, limits, args=(np.dot(x_mat,self.__p), np.dot(x_mat,self.__p)))[0]
            # self.__scs_ted_x = coef_e**2 * integ.nquad(E_e_mod_sq, limits, args=(np.dot(x_mat,self.__p) + 1j * self.__k_d**2 / self.__omega * np.dot(x_mat,self.__T), np.dot(x_mat,self.__p) + 1j * self.__k_d**2 / self.__omega * np.dot(x_mat,self.__T)))[0]
            # self.__scs_md_x = coef_m**2 * integ.nquad(E_m_mod_sq, limits, args=(np.dot(x_mat,self.__m), np.dot(x_mat,self.__m)))[0]

            # self.__scs_ed_y = coef_e**2 * integ.nquad(E_e_mod_sq, limits, args=(np.dot(y_mat,self.__p), np.dot(y_mat,self.__p)))[0]
            # self.__scs_ted_y = coef_e**2 * integ.nquad(E_e_mod_sq, limits, args=(np.dot(y_mat,self.__p) + 1j * self.__k_d**2 / self.__omega * np.dot(y_mat,self.__T), np.dot(y_mat,self.__p) + 1j * self.__k_d**2 / self.__omega * np.dot(y_mat,self.__T)))[0]
            # self.__scs_md_y = coef_m**2 * integ.nquad(E_m_mod_sq, limits, args=(np.dot(y_mat,self.__m), np.dot(y_mat,self.__m)))[0]

            # self.__scs_ed_z = coef_e**2 * integ.nquad(E_e_mod_sq, limits, args=(np.dot(z_mat,self.__p), np.dot(z_mat,self.__p)))[0]
            # self.__scs_ted_z = coef_e**2 * integ.nquad(E_e_mod_sq, limits, args=(np.dot(z_mat,self.__p) + 1j * self.__k_d**2 / self.__omega * np.dot(z_mat,self.__T), np.dot(z_mat,self.__p) + 1j * self.__k_d**2 / self.__omega * np.dot(z_mat,self.__T)))[0]
            # self.__scs_md_z = coef_m**2 * integ.nquad(E_m_mod_sq, limits, args=(np.dot(z_mat,self.__m), np.dot(z_mat,self.__m)))[0]



            # E_eq_mod_sq_x = lambda theta, phi: E_e_mod_sq(theta, phi, np.dot(x_mat, eq_mom_d(theta, phi)), np.dot(x_mat, eq_mom_r(theta, phi)))
            # E_mq_mod_sq_x = lambda theta, phi: E_m_mod_sq(theta, phi, np.dot(x_mat, mq_mom_d(theta, phi)), np.dot(x_mat, mq_mom_r(theta, phi)))
            # self.__scs_eq_x = coef_e**2 * integ.nquad(E_eq_mod_sq_x, limits)[0]
            # self.__scs_mq_x = coef_m**2 * integ.nquad(E_mq_mod_sq_x, limits)[0]


            # E_eq_mod_sq_y = lambda theta, phi: E_e_mod_sq(theta, phi, np.dot(y_mat, eq_mom_d(theta, phi)), np.dot(y_mat, eq_mom_r(theta, phi)))
            # E_mq_mod_sq_y = lambda theta, phi: E_m_mod_sq(theta, phi, np.dot(y_mat, mq_mom_d(theta, phi)), np.dot(y_mat, mq_mom_r(theta, phi)))
            # self.__scs_eq_y = coef_e**2 * integ.nquad(E_eq_mod_sq_y, limits)[0]
            # self.__scs_mq_y = coef_m**2 * integ.nquad(E_mq_mod_sq_y, limits)[0]


            # E_eq_mod_sq_z = lambda theta, phi: E_e_mod_sq(theta, phi, np.dot(z_mat, eq_mom_d(theta, phi)), np.dot(z_mat, eq_mom_r(theta, phi)))
            # E_mq_mod_sq_z = lambda theta, phi: E_m_mod_sq(theta, phi, np.dot(z_mat, mq_mom_d(theta, phi)), np.dot(z_mat, mq_mom_r(theta, phi)))
            # self.__scs_eq_z = coef_e**2 * integ.nquad(E_eq_mod_sq_z, limits)[0]
            # self.__scs_mq_z = coef_m**2 * integ.nquad(E_mq_mod_sq_z, limits)[0]




            self.__total_ed = integ.nquad(E_mod_sq_total, limits, args=(self.__p, self.__p, self.__m, self.__m))[0]
            self.__total_ted = integ.nquad(E_mod_sq_total, limits, args=(self.__p + 1j * self.__k_d**2 / self.__omega * self.__T, self.__p + 1j * self.__k_d**2 / self.__omega * self.__T, self.__m, self.__m))[0]
        return np.array([self.__scs_ed, self.__scs_ted, self.__scs_md, self.__scs_eq, self.__scs_mq, self.__scs_eoс, self.__total_ed, self.__total_ted])
        # , np.asarray( [[self.__scs_ed_x, self.__scs_ed_y, self.__scs_ed_z], [self.__scs_ted_x, self.__scs_ted_y, self.__scs_ted_z], [self.__scs_md_x, self.__scs_md_y,self.__scs_md_z]])
            # return np.array([self.__scs_ed, self.__scs_ted, self.__scs_md, self.__scs_eq, self.__scs_mq, self.__scs_eoс, self.__total_ed, self.__total_ted, 
                    # self.__scs_ed_x, self.__scs_ed_y, self.__scs_ed_z, self.__scs_ted_x, self.__scs_ted_y, self.__scs_ted_z, self.__scs_md_x, self.__scs_md_y,self.__scs_md_z,
                #    self.__scs_eq_x, self.__scs_eq_y, self.__scs_eq_z,  self.__scs_mq_x, self.__scs_mq_y, self.__scs_mq_z])

                
    def __CalculateSCS_SPP(self):
        z=0
        r_0 = self.__r0 * 1e-9
        a = (-self.__eps_d / self.__eps_s)**0.5
        k_p = (self.__eps_s * self.__eps_d / (self.__eps_s + self.__eps_d))**0.5 * self.__k_0
        C = 1j * a * k_p / (2 * (1 - a**2) * (1 - a**4))
        n_p = lambda phi: np.array([np.cos(phi),np.sin(phi), -1j*a])
        coef_spp_p =  (self.__mu_d / self.__eps_d)**0.5 * (1 - a**2) * (1 - a**4) / (2 * a * self.__k_0)
        coef = lambda phi: C * np.exp(-a * k_p * z) * (2 / (np.pi * k_p))**0.5 * np.exp(-1j * k_p * np.dot(n_p(phi), r_0))

        E_e_spp_z = lambda phi, e_mom: self.__k_0**2 / const.epsilon_0 * (e_mom[2] + 1j * a * (e_mom[0] * np.cos(phi) + e_mom[1] * np.sin(phi)))
        E_m_spp_z = lambda phi, m_mom: (const.mu_0/const.epsilon_0)**0.5 * self.__k_0 * k_p * (1 - a**2) * (m_mom[0] * np.sin(phi) - m_mom[1] * np.cos(phi))
        
        imag_one = -1j
        eq_mom = lambda phi: np.dot(self.__Q, n_p(phi)) * imag_one * self.__k_d / 6
        mq_mom = lambda phi: np.dot(self.__M, n_p(phi)) * imag_one * self.__k_d / 2



        E_spp_sq = lambda phi, e_mom, m_mom, eq_mom, mq_mom: abs((E_e_spp_z(phi, e_mom) + E_m_spp_z(phi, m_mom) + E_e_spp_z(phi, eq_mom(phi)) + E_m_spp_z(phi, mq_mom(phi))) * coef(phi) )**2 
        
        self.__scs_spp_total = integ.nquad(E_spp_sq, [[0, 2*np.pi]], args=(self.__p + 1j * self.__k_d**2 / self.__omega * self.__T, self.__m, eq_mom, mq_mom))[0] * coef_spp_p

        #return self.__scs_spp_total

        self.__scs_spp_total_1_1 = integ.nquad(E_spp_sq, [[0, np.pi]], args=(self.__p + 1j * self.__k_d**2 / self.__omega * self.__T, self.__m, eq_mom, mq_mom))[0] * coef_spp_p
        self.__scs_spp_total_1_2 = integ.nquad(E_spp_sq, [[np.pi, 2*np.pi]], args=(self.__p + 1j * self.__k_d**2 / self.__omega * self.__T, self.__m, eq_mom, mq_mom))[0] * coef_spp_p

        self.__scs_spp_total_2_1 = integ.nquad(E_spp_sq, [[np.pi/2, 3/2*np.pi]], args=(self.__p + 1j * self.__k_d**2 / self.__omega * self.__T, self.__m, eq_mom, mq_mom))[0] * coef_spp_p
        self.__scs_spp_total_2_2 = integ.nquad(E_spp_sq, [[-np.pi/2, np.pi/2]], args=(self.__p + 1j * self.__k_d**2 / self.__omega * self.__T, self.__m, eq_mom, mq_mom))[0] * coef_spp_p

        
        
        return np.array([self.__scs_spp_total, self.__scs_spp_total_1_1/self.__scs_spp_total_1_2, self.__scs_spp_total_2_1/self.__scs_spp_total_2_2])

    
    def PlotFarField(self, title="light"): #= r"$\lambda = "+ str(self.__wl)):
        if self.__is_pec_subs:
            r_s = lambda theta: -1
            r_p = lambda theta: 1
        elif self.__is_multilayer:
            r_s = lambda theta: tmm.coh_tmm('s', self.__eps_list**0.5, self.__d_list, theta, self.__wl)['r']
            r_p = lambda theta: tmm.coh_tmm('p', self.__eps_list**0.5, self.__d_list, theta, self.__wl)['r']
        else:
            r_s = lambda theta: (self.__mu_s * (self.__eps_d - self.__eps_d  * np.sin(theta)**2)**0.5 - self.__mu_d * (self.__eps_s - self.__eps_d  * np.sin(theta)**2)**0.5) / (self.__mu_s * (self.__eps_d - self.__eps_d * np.sin(theta)**2)**0.5 + self.__mu_d * (self.__eps_s - self.__eps_d  * np.sin(theta)**2)**0.5)
            r_p = lambda theta: (self.__eps_s * (self.__eps_d - self.__eps_d  * np.sin(theta)**2)**0.5 - self.__eps_d * (self.__eps_s - self.__eps_d  * np.sin(theta)**2)**0.5) / (self.__eps_s * (self.__eps_d - self.__eps_d * np.sin(theta)**2)**0.5 + self.__eps_d *(self.__eps_s - self.__eps_d  * np.sin(theta)**2)**0.5)

        r_0 = self.__r0 * 1e-9
        
        n = lambda theta, phi: np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
        n_ = lambda theta, phi: np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), -np.cos(theta)])

        limits = [[0, np.pi/2], [0, np.pi*2]]

        coef_e = (self.__k_0**2 / (4 * np.pi * const.epsilon_0))
        coef_m = ((const.mu_0 / const.epsilon_0)**0.5 * self.__k_0 * self.__k_d / (4 * np.pi))
        
    
        E_e_phi = lambda theta, phi, e_mom_d, e_mom_r: (e_mom_d[1] * np.cos(phi) - e_mom_d[0] * np.sin(phi)) + r_s(theta) * np.exp(2j * self.__k_d * r_0[2] * np.cos(theta))* (e_mom_r[1] * np.cos(phi) - e_mom_r[0] * np.sin(phi))
        E_e_theta = lambda theta, phi, e_mom_d, e_mom_r: e_mom_d[0] * np.cos(phi) * np.cos(theta) + e_mom_d[1] * np.sin(phi) * np.cos(theta) - r_p(theta) * np.exp(2j * self.__k_d * r_0[2] * np.cos(theta)) * (e_mom_r[0] * np.cos(phi) * np.cos(theta) + e_mom_r[1] * np.sin(phi) * np.cos(theta)) - (e_mom_d[2] * np.sin(theta) + e_mom_r[2] * np.sin(theta) *r_p(theta) * np.exp(2j * self.__k_d * r_0[2] * np.cos(theta)))
        E_e_mod_sq = lambda theta, phi, e_mom_d, e_mom_r: (abs(E_e_theta(theta, phi, e_mom_d, e_mom_r))**2 + abs(E_e_phi(theta,phi, e_mom_d, e_mom_r))**2) * np.sin(theta)

        E_m_phi = lambda theta, phi, m_mom_d, m_mom_r: (r_s(theta) * np.exp(2j * self.__k_d * r_0[2] * np.cos(theta)) * (m_mom_r[0] * np.cos(phi) * np.cos(theta) + m_mom_r[1] * np.sin(phi) * np.cos(theta)) - (m_mom_d[0] * np.cos(phi) * np.cos(theta) + m_mom_d[1] * np.sin(phi) * np.cos(theta))) + (m_mom_r[2] * np.sin(theta) * r_s(theta) * np.exp(2j * self.__k_d * r_0[2] * np.cos(theta)) + m_mom_d[2] * np.sin(theta))
        E_m_theta = lambda theta, phi, m_mom_d, m_mom_r: m_mom_d[1] * np.cos(phi) - m_mom_d[0] * np.sin(phi) + r_p(theta) * np.exp(2j * self.__k_d * r_0[2] * np.cos(theta)) * (m_mom_r[1] * np.cos(phi) - m_mom_r[0] * np.sin(phi))
        E_m_mod_sq = lambda theta, phi, m_mom_d, m_mom_r: (abs(E_m_theta(theta,phi,m_mom_d, m_mom_r))**2 + abs(E_m_phi(theta,phi,m_mom_d, m_mom_r))**2) * np.sin(theta)

        imag_one = -1j
        eq_mom_d = lambda theta, phi: np.dot(self.__Q, n(theta, phi)) * imag_one * self.__k_d / 6      #*0
        eq_mom_r = lambda theta, phi: np.dot(self.__Q, n_(theta, phi)) * imag_one * self.__k_d / 6     #*0
        mq_mom_d = lambda theta, phi: np.dot(self.__M, n(theta, phi)) * imag_one * self.__k_d / 2      #*0
        mq_mom_r = lambda theta, phi: np.dot(self.__M, n_(theta, phi)) * imag_one * self.__k_d / 2     #*0
        
        E_eq_mod_sq = lambda theta, phi: E_e_mod_sq(theta, phi, eq_mom_d(theta, phi), eq_mom_r(theta, phi))
        E_mq_mod_sq = lambda theta, phi: E_m_mod_sq(theta, phi, mq_mom_d(theta, phi), mq_mom_r(theta, phi))

        E_mod_sq_total = lambda theta, phi, ed_mom_d, ed_mom_r, md_mom_d, md_mom_r: (abs( coef_e * (E_e_theta(theta, phi, ed_mom_d, ed_mom_r) + E_e_theta(theta, phi, eq_mom_d(theta, phi), eq_mom_r(theta, phi))) + coef_m * (E_m_theta(theta, phi, md_mom_d, md_mom_r) + E_m_theta(theta, phi, mq_mom_d(theta, phi), mq_mom_r(theta, phi))))**2 + abs(coef_e * (E_e_phi(theta, phi, ed_mom_d, ed_mom_r) + E_e_phi(theta, phi, eq_mom_d(theta, phi), eq_mom_r(theta, phi))) + coef_m * (E_m_phi(theta, phi, md_mom_d, md_mom_r) + E_m_phi(theta, phi, mq_mom_d(theta, phi), mq_mom_r(theta, phi)))  )**2 ) #* np.sin(theta)
        
        theta = np.linspace(0, np.pi/2, num=101) 
        phi = np.linspace(0.001, np.pi*2, num=201)
        Far_tot = []
        Far_p = []
        Far_m = []
        for j in phi:
            far_tot = []
            far_p = []
            far_m = []
            for i in theta:
                far_tot.append(E_mod_sq_total(i, j, self.__p, self.__p, self.__m, self.__m))
                far_p.append(E_mod_sq_total(i, j, self.__p, self.__p, self.__m*0, self.__m*0))
                far_m.append(E_mod_sq_total(i, j, self.__p*0, self.__p*0, self.__m, self.__m))
            Far_tot.append(np.array(far_tot))
            Far_p.append(np.array(far_p))
            Far_m.append(np.array(far_m))
        Far_tot = np.array(Far_tot)*1e13
        Far_p = np.array(Far_p)*1e13
        Far_m = np.array(Far_m)*1e13    
        # Far_tot /= Far_tot.max()
        print("Finished")
        
        THETA, PHI = np.meshgrid(theta, phi)
        X = Far_tot * np.sin(THETA) * np.cos(PHI)
        Y = Far_tot * np.sin(THETA) * np.sin(PHI)
        Z = Far_tot * np.cos(THETA)

        fig = plt.figure(num=None, figsize=(12, 3), dpi=200)
        fig.suptitle('Far-fields ('+ title+')', y=1.05)
        
        ax = fig.add_subplot(131, projection='3d')
        ax.set_title("Total")
        ax.plot_surface(X,Y,Z,cmap=plt.cm.jet, rstride=1, cstride=1,antialiased=False)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        plt.tight_layout()

        X = Far_p * np.sin(THETA) * np.cos(PHI)
        Y = Far_p * np.sin(THETA) * np.sin(PHI)
        Z = Far_p * np.cos(THETA)


        ax = fig.add_subplot(132, projection='3d')
        ax.set_title("p")
        ax.plot_surface(X,Y,Z,cmap=plt.cm.jet, rstride=1, cstride=1,antialiased=False)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        plt.tight_layout()

        X = Far_m * np.sin(THETA) * np.cos(PHI)
        Y = Far_m * np.sin(THETA) * np.sin(PHI)
        Z = Far_m * np.cos(THETA)


        ax = fig.add_subplot(133, projection='3d')
        ax.set_title("m")
        ax.plot_surface(X,Y,Z,cmap=plt.cm.jet, rstride=1, cstride=1,antialiased=False)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        plt.tight_layout()
        
        # THETA, PHI = np.meshgrid(theta, phi)
        # ones = np.ones(theta.shape) 
        # f_test = E_mod_sq_total(THETA, PHI, self.__p*ones, self.__p*ones, self.__m*ones, self.__m*ones )

        # check = f_test-Far_tot


        # far_field_inter = RegularGridInterpolator((phi, theta), Far_tot, method='nearest' )


        # X = Far_tot * np.sin(THETA) * np.cos(PHI)
        # Y = Far_tot * np.sin(THETA) * np.sin(PHI)
        # Z = Far_tot * np.cos(THETA)


        


        # # for i in range(2):
        # i=0
        # ax = plt.subplot(111, polar=True)
        # theta_const_list = [np.pi/2,0]
        # phi = np.append(phi)
        # theta = np.ones(phi.shape[0])*theta_const_list[i]
        # pts = np.array([phi, theta]).transpose()
        # ax.plot(phi, far_field_inter(pts))

        # ax.set_theta_zero_location("N")
        # plt.tight_layout()



        plt.show()

        THETA, PHI = np.meshgrid(theta, phi)
        X = Far_tot * np.sin(THETA) * np.cos(PHI)
        Y = Far_tot * np.sin(THETA) * np.sin(PHI)
        Z = Far_tot * np.cos(THETA)
        return [Far_tot, theta, phi]
#         return np.array([X,Y,Z])


    
    def PlotFarFieldSPP(self):
        z=0
        r_0 = self.__r0 * 1e-9
        a = (-self.__eps_d / self.__eps_s)**0.5
        k_p = (self.__eps_s * self.__eps_d / (self.__eps_s + self.__eps_d))**0.5 * self.__k_0
        C = 1j * a * k_p / (2 * (1 - a**2 * (1 - a**4)))
        n_p = lambda phi: np.array([np.cos(phi),np.sin(phi), -1j*a])
        coef_spp_p =  (self.__mu_d / self.__eps_d)**0.5 * (1 - a**2) * (1 - a**4) / (2 * a * self.__k_0)
        coef = lambda phi: C * np.exp(-a * k_p * z) * (2 / (np.pi * k_p))**0.5 * np.exp(-1j * k_p * np.dot(n_p(phi), r_0))

        E_e_spp_z = lambda phi, e_mom: self.__k_0**2 / const.epsilon_0 * (e_mom[2] + 1j * a * (e_mom[0] * np.cos(phi) + e_mom[1] * np.sin(phi)))
        E_m_spp_z = lambda phi, m_mom: (const.mu_0/const.epsilon_0)**0.5 * self.__k_0 * k_p * (1 - a**2) * (m_mom[0] * np.sin(phi) - m_mom[1] * np.cos(phi))
        
        imag_one = -1j
        eq_mom = lambda phi: np.dot(self.__Q, n_p(phi)) * imag_one * self.__k_d / 6
        mq_mom = lambda phi: np.dot(self.__M, n_p(phi)) * imag_one * self.__k_d / 2



        E_spp_sq = lambda phi, e_mom, m_mom, eq_mom, mq_mom: abs((E_e_spp_z(phi, e_mom) + E_m_spp_z(phi, m_mom) + E_e_spp_z(phi, eq_mom(phi)) + E_m_spp_z(phi, mq_mom(phi))) * coef(phi))**2
        

        SCS_spp= lambda phi: E_spp_sq(phi,self.__p + 1j * self.__k_d**2 / self.__omega * self.__T, self.__m, eq_mom, mq_mom) * coef_spp_p

#         phi = np.linspace(0,2*np.pi, 101)
        phi = np.linspace(0.001, np.pi*2, num=201)
        SCS = SCS_spp(phi).real*1e13
        plt.polar(phi, SCS)
        plt.show()
        return [phi, SCS]

    
    def __CalculateMetaSurf(self, E0, D):
        S_L = D**2
        coef = 1j*self.__k_d/(2*E0*S_L*const.epsilon_0*self.__eps_d)
        self.__meta_ed = self.__p[1] * coef 
        self.__meta_md = self.__m[0] * coef * (1 / self.__v_d)
        self.__meta_eq = self.__Q[2,1] * coef * (1j * self.__k_d/6) 
        self.__meta_mq = self.__M[2,0] * coef * (1j * self.__k_d/2/self.__v_d) 
#           self.__r_eoс =(abs(self.__O)**2).sum()**0.5*coef*(self.__k_d**2/6)
        self.__r_total = self.__meta_ed + self.__meta_md + self.__meta_eq + self.__meta_mq
        self.__t_total = 1 + self.__meta_ed - self.__meta_md + self.__meta_eq + self.__meta_mq  

        return np.array([self.__meta_ed, self.__meta_md, self.__meta_eq, self.__meta_mq,self.__r_total, self.__t_total])
