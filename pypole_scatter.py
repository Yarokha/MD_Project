import numpy as np
import matplotlib.pyplot as plt
import pympole, os, time, re
import scipy as sp
import scipy.constants as const
import scipy.integrate as integ
from numba import jit


class pySEMPole(pympole.MultipoleDecomp):
    
    def __init__(self):
        self._MultipoleDecomp__is_pec_subs = False
        self._MultipoleDecomp__is_subs = False
        self._MultipoleDecomp__is_multilayer = False
        self._MultipoleDecomp__is_set_eff_point = False
        self.__is_multilayer = False
        self.__calc_scs_spp = False
        self.__is_multiparticles = False
        self.particles_moments = []
        


    def __WlNameExport(self, name):
        c = const.c*1e-3
        try:
            Lambda = c/float(re.findall(r'f=(\w+.\w+)',name)[0])
        except IndexError:
            try:
                Lambda = float((re.findall(r'(wl=\w+.\w+ )',name))[0][3:])
            except IndexError:
                Lambda = c/float(re.findall(r'_(\w+.\w+).txt',name)[0])*1e-3

        return Lambda
    

    def __LoadDispersion(self, path_csv, plotting=False, use_real_only=False):
        disp = np.genfromtxt(path_csv, delimiter=',')
        length_n_k = np.argwhere(np.isnan(disp[:,0]))
        if length_n_k.shape[0]==1:
            wl_n = wl_k= disp[1:,0]*1e3
            n_val = disp[1:,1]
            k_val = disp[1:,1]*0
            print('Dispersion without losses is loaded\n n from', wl_n[0], 'nm to', wl_n[-1],'nm')
        else :
            wl_n=disp[1:length_n_k[1][0],0]*1e3
            n_val = disp[1:length_n_k[1][0],1]
            wl_k = disp[length_n_k[1][0]+1:,0]*1e3
            k_val = disp[length_n_k[1][0]+1:,1]
            print('Dispersion with losses is loaded\n n from', wl_n[0], 'nm to', wl_n[-1],
                  'nm\n k from', wl_k[0], 'nm to', wl_k[-1], 'nm')

        n = lambda wl: np.interp(wl, wl_n, n_val) 
        k = lambda wl: np.interp(wl, wl_k, k_val) 

        eps_re = lambda wl: n(wl)**2 - k(wl)**2
        eps_im = lambda wl: 2 * n(wl) * k(wl)

        if plotting:
            plt.title("Dispersion")
            plt.xlabel(r'$\lambda, nm$')
            plt.ylabel(r"$\varepsilon$")  
            plt.plot(wl_n, eps_re(wl_n))
            plt.plot(wl_n, eps_im(wl_n))
            plt.legend(["Real", "Imag"])
            plt.show()
            
        return lambda wl: eps_re(wl)+1j*eps_im(wl)


    def LoadParticleDispersion(self, path_csv, plotting=False, use_real_only=False):
        self.__eps_p = self.__LoadDispersion(path_csv, plotting, use_real_only)

    def LoadMediumDispersion(self, path_csv, plotting=False, use_real_only=False):
        self.__eps_d = self.__LoadDispersion(path_csv, plotting, use_real_only)

    def LoadSubstrateDispersion(self, path_csv, plotting=False, use_real_only=False):
        self.__eps_s = self.__LoadDispersion(path_csv, plotting, use_real_only)
        self._MultipoleDecomp__is_subs = True

    def LoadParticleConstDispersion(self, eps_real, eps_imag):
        self.__eps_p = lambda wl: eps_real + 1j * eps_imag 

    def LoadMediumConstDispersion(self, eps_real, eps_imag):
        self.__eps_d = lambda wl: eps_real + 1j * eps_imag 

    def LoadSubstrateConstDispersion(self, eps_real, eps_imag):
        self.__eps_s = lambda wl: eps_real + 1j * eps_imag
        self._MultipoleDecomp__is_subs = True


    def MultiLayer(self):
        self.__eps_multi_list = []
        self.__d_multi_list = []
        self.__is_multilayer = True

    def AddLayerConstDispersion(self, eps_real, eps_imag, d):
        self.__eps_multi_list.append(lambda wl: eps_real + 1j * eps_imag)
        self.__d_multi_list.append(d*1e-9)

    def AddLayerDispersion(self, path_csv, d, plotting=False, use_real_only=False):
        self.__eps_multi_list.append(self.__LoadDispersion(path_csv, plotting, use_real_only))
        self._MultipoleDecomp__is_subs = True
        self._MultipoleDecomp__d_multi_list.append(d*1e-9)

    def LoadParticle(self, path, shift=[0,0,0], plotting=False):
        shift = np.array(shift)
        xyz = np.loadtxt(path, skiprows=2)
        self._MultipoleDecomp__LoadParticlePointList(xyz[:,0:3]+shift, plotting)
    
    def MultiParicle(self):
        self.__eps_multi_partilce_list = []
        self.__multi_particle_point_list = []
        self.__is_multiparticles= True

    def AddParticlePointList(self, path):
        self.__multi_particle_point_list.append(np.loadtxt(path, skiprows=2))
    
    
    def __DetermeineEffectiveDipolePoint(self):    
        r0 = np.zeros(3)
        for i in range(len(self.__multi_particle_point_list)):
            r0 += np.array([
                (self.__multi_particle_point_list[i][:,0].min()+self.__multi_particle_point_list[i][:,0].max())/2,
                (self.__multi_particle_point_list[i][:,1].min()+self.__multi_particle_point_list[i][:,1].max())/2,
                (self.__multi_particle_point_list[i][:,2].min()+self.__multi_particle_point_list[i][:,2].max())/2])
        r0/=3
        self.SetEffectiveDipolePoint(r0)
        print("Dipole Point is " + str(r0))


    def AddParticleDispersion(self, path_csv, plotting=False, use_real_only=False):
        self.__eps_multi_partilce_list.append(self.__LoadDispersion(path_csv, plotting, use_real_only))
    
 
    def AddParticleConstDispersion(self, eps_real, eps_imag):
        self.__eps_multi_partilce_list.append(lambda wl: eps_real + 1j * eps_imag)

 
    def PlotLoadedParticles(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title("Particle Points")
        part_center = []
        for i in self.__multi_particle_point_list:
            ax.scatter(i[:,0], i[:,1], i[:,2])
            part_center.append([[i[:,0].min(),i[:,0].max()], 
                                [i[:,1].min(),i[:,1].max()],
                                [i[:,2].min(),i[:,2].max()]                             
                               ])

        part_center = np.asarray(part_center)
        part_center = np.array([np.abs(part_center[:,0,:].max()-part_center[:,0,:].min()),
                                np.abs(part_center[:,1,:].max()-part_center[:,1,:].min()),
                                np.abs(part_center[:,2,:].max()-part_center[:,2,:].min())]) 
        part_center /= part_center.min()
    
        ax.set_xlabel("x, nm")
        ax.set_ylabel("y, nm")
        ax.set_zlabel("z, nm")
        ax.set_box_aspect(part_center)
        plt.tight_layout()
        plt.show()

    # def ShiftParticle(self, x,y,z, plotting=False):
    #     xyz[:,0:3] += np.array([x,y,z]) 
    #     self.LoadParticlePointList(xyz[:,0:3], plotting)

 
    def LoadElectricFields(self, path_e_fields):
        files_list = np.array(os.listdir(path_e_fields))
        # print(files_list)
        self.__e_field_files_list = np.sort(np.array([path_e_fields + i for i in files_list]))


    def __ReadElectricField(self, path_e_field):
        # print(path_e_field)
        self._MultipoleDecomp__LoadEField(np.loadtxt(path_e_field, skiprows=2))


    
    def StartCalculationMiltiPaticles(self, limits = [[0, np.pi/2], [0, np.pi*2]]):
        self.wl_list = []
        self.moments_list = []
        self.scs_list = []
        self.scs_spp_list = []
        self.scs_far_list = []
        self.scs_far_spp_list = []

        if not self._MultipoleDecomp__is_set_eff_point:
            self.__DetermeineEffectiveDipolePoint()


        for e_field_path in self.__e_field_files_list[::]:
            wl = self.__WlNameExport(e_field_path)
            self.__ReadElectricField(e_field_path)
            self._MultipoleDecomp__SetMediumParameters(wl, self.__eps_d(wl))
            self.particles_moments = []
            for i in range(len(self.__multi_particle_point_list)):
                self._MultipoleDecomp__LoadParticlePointList(self.__multi_particle_point_list[i])
                self._MultipoleDecomp__SetParticleParameters(self.__eps_multi_partilce_list[i](wl))
                self.particles_moments.append(self._MultipoleDecomp__CalculateMoments())
                # print("Particle " + str(i+1) + " is calculated")
            
            # moments = []
            # for i in range(len(self.particles_moments[0])):
            #     moments.append(self.particles_moments[0][i]+self.particles_moments[1][i])

            moments = self.particles_moments[0]
            for i in range(len(self.particles_moments[0])):
                for j in range(1, len(self.__multi_particle_point_list)):
                    moments[i] += self.particles_moments[j][i]

            self.moments_list.append(moments)
            
            self._MultipoleDecomp__p = moments[0] # electric dipole moment
            self._MultipoleDecomp__T = moments[1] # toroidal moment
            self._MultipoleDecomp__m = moments[2] # magnetic dipole moment
            self._MultipoleDecomp__Q = moments[3] # electric quadrapole moment
            self._MultipoleDecomp__M = moments[4] # magnetic quadrapole moment
            self._MultipoleDecomp__O = moments[5] # electric octapole moment
            
            if self._MultipoleDecomp__is_subs and not self._MultipoleDecomp__is_pec_subs:
                self._MultipoleDecomp__SetSubstrateParameters(self.__eps_s(wl))
            elif self.__is_multilayer:
                eps_list = []
                for i in self.__eps_multi_list:
                    eps_list.append(i(wl))
                eps_list = np.array(eps_list)
                self._MultipoleDecomp__LoadMultilayers(eps_list, np.array(self.__d_multi_list))

             
            self.scs_list.append(self._MultipoleDecomp__CalculateSCS(limits=limits))
            self.wl_list.append(wl)
            print(wl)
        self.wl_list = np.array(self.wl_list)
        self.scs_list  = np.array(self.scs_list)
        self.wl_list = np.array(self.wl_list)





    
    def StartCalculation(self, calc_scs_spp=False, plot_far_light=False, plot_far_spp=False, limits = [[0, np.pi/2], [0, np.pi*2]]):
        self.__calc_scs_spp = calc_scs_spp
        self._MultipoleDecomp__DetermeineEffectiveDipolePoint()
        self.wl_list = []
        self.moments_list = []
        self.scs_list = []
        self.scs_spp_list = []
        self.scs_far_list = []
        self.scs_far_spp_list = []
        if self.__is_multilayer:
            self.__eps_d = self.__eps_multi_list[0]
        
        for e_field_path in self.__e_field_files_list[::]:
            wl = self.__WlNameExport(e_field_path)
            self._MultipoleDecomp__SetParticleParameters(self.__eps_p(wl))
            self._MultipoleDecomp__SetMediumParameters(wl, self.__eps_d(wl))
            if self._MultipoleDecomp__is_subs and not self._MultipoleDecomp__is_pec_subs:
                # print(self.__eps_s(wl).real)
                self._MultipoleDecomp__SetSubstrateParameters(self.__eps_s(wl)) 
            elif self.__is_multilayer:
                eps_list = []
                for i in self.__eps_multi_list:
                    eps_list.append(i(wl))
                eps_list = np.array(eps_list)
                self._MultipoleDecomp__LoadMultilayers(eps_list, np.array(self.__d_multi_list))
            self.__ReadElectricField(e_field_path)
            self.moments_list.append(self._MultipoleDecomp__CalculateMoments())
            self.scs_list.append(self._MultipoleDecomp__CalculateSCS(limits = limits))
            if self.__calc_scs_spp:
                self.scs_spp_list.append(self._MultipoleDecomp__CalculateSCS_SPP())
            print(wl)
            if plot_far_light:
                self.scs_far_list.append(self.PlotFarField())
            if plot_far_spp:
                self.scs_far_spp_list.append(self.PlotFarFieldSPP())
            self.wl_list.append(wl)
        self.scs_list  = np.array(self.scs_list)
        self.wl_list = np.array(self.wl_list)
        self.scs_spp_list = np.array(self.scs_spp_list)





    
    def StartCalculationMetaSurf(self, E0 = 2.77279e7, D = 700e-9):
        self._MultipoleDecomp__DetermeineEffectiveDipolePoint()
        self.wl_list = []
        self.moments_list = []
        self.meta_r_t_list = []

        for e_field_path in self.__e_field_files_list[::]:
            wl = self.__WlNameExport(e_field_path)
            self._MultipoleDecomp__SetParticleParameters(self.__eps_p(wl))
            self._MultipoleDecomp__SetMediumParameters(wl, self.__eps_d(wl))
            self.__ReadElectricField(e_field_path)
            self.moments_list.append(self._MultipoleDecomp__CalculateMoments())
            self.meta_r_t_list.append(self._MultipoleDecomp__CalculateMetaSurf(E0, D))
            print(wl)
            self.wl_list.append(wl)
        self.meta_r_t_list  = np.array(self.meta_r_t_list)
        self.wl_list = np.array(self.wl_list)

        
    
    def PlotSCS(self, *args):
        scale = 1e12
        moment_name_list = {'ED':0, 'TED':1, 'MD':2, 'EQ':3, 'MQ':4, 'EOC':5, 'TOTAL(ED)':6, 'TOTAL(TED)':7}
        curve_style = ['-', '-', '-', '-', '-', '-', 'o-', '*-']
        if self._MultipoleDecomp__is_subs:
            plt.title("Scattering Cross Section (upper semi-space)")
        else:
            plt.title("Total Scattering Cross Section")
        for i in args:
            if i =='SPP':
                continue
            plt.plot(self.wl_list, self.scs_list[:,moment_name_list[i]].real * scale, curve_style[moment_name_list[i]],  label=i)
        if 'SPP' in args:
            plt.plot(self.wl_list, self.scs_spp_list.real * scale,'--', label="SPP")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.grid()
        plt.xlabel(r"$\lambda, nm$")
        plt.ylabel(r"$\sigma, \mu m^2$")
        plt.tight_layout()
        plt.show()
    


        ### Extinction for only x-polarized plane wave( normal incident)
        # ecs_coef = self.__k_d/(const.epsilon_0*self.__eps_d)
        # self.ecs_ed = ecs_coef * np.imag(self.p[0])
        # self.ecs_ted = ecs_coef * np.imag(self.p[0]+1j*self.__k_d/self.__v_d*self.T[0])
        # self.ecs_md = ecs_coef * np.imag(1/self.__v_d*self.m[1])
        # self.ecs_eq = ecs_coef * np.imag(-1j*self.__k_d/6*self.Q[0,2])
        # self.ecs_mq = ecs_coef * np.imag(-1j*self.__k_d/(2*self.__v_d)*self.M[1,2])
        # ### self.ecs_eoс = ecs_coef * np.imag(-self.__k_d**2/6*self.O[8])
        # self.ecs_eoс = ecs_coef * np.imag(-self.__k_d**2/6*self.O[0,-1,-1])

        ###  Extinction general case
        # ecs_coef = self.__k_d/(const.epsilon_0*self.__eps_d)#*self.__E0
        # self.ecs_ed = np.imag(np.dot(ecs_coef,self.p))
        # self.ecs_ted = np.imag(np.dot(ecs_coef,self.p+1j*self.__k_d/self.__v_d*self.T))
        # self.ecs_md = np.imag(np.dot(ecs_coef,1/self.__v_d*np.cross(self.m, self.__n)))
        # self.ecs_eq = np.imag(-1j*self.__k_d/6*np.dot(ecs_coef,np.dot (self.Q,self.__n) ))
        # self.ecs_mq = np.imag(np.dot(ecs_coef, 1j*self.__k_d/(2*self.__v_d) * np.cross(self.__n, np.dot(self.M, self.__n)) ))
        # self.ecs_eoс = np.imag(np.dot(ecs_coef,-self.__k_d**2/6 * np.dot(np.dot(self.O, self.__n), self.__n)))