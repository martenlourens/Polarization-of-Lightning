#!/usr/bin/env python3

import numpy as np
from numpy.linalg import norm
import scipy.signal as sg

#plotter imports
from matplotlib.pyplot import figure, show
from mpl_toolkits.mplot3d import Axes3D






#Main object containing all the Stokes' parameter operations
class stokes_params:
	def __init__(self,sampling_period=5E-9):
		self.sampling_period = sampling_period

	#function computing the different Stokes' parameters and returning the Stokes' vector
	def get_stokes_vector(self,Ex,Ey):
		I = np.abs(Ex)**2 + np.abs(Ey)**2
		Q = np.abs(Ex)**2 - np.abs(Ey)**2
		U = 2*np.real(Ex*np.conj(Ey))
		V = -2*np.imag(Ex*np.conj(Ey))

		#store Stokes' parameters in Stokes' vector
		self.S = np.array([I,Q,U,V])
		
		return self.S

	def get_pulseWidth(self,f_hp=0.5,f_σ=1):
		R = 1 - f_hp

		min_height = np.max(self.S[0])*0.25 #threshold is set to ~25% of the maximum height of a pulse in the dataset

		self.indices, self.peakDict = sg.find_peaks(self.S[0], height=min_height, rel_height=R, width=1, prominence=None)

		#sort all results by peak prominence (first entry is most prominent peak)
		#sortIndices = np.argsort(-self.peakDict['prominences'])[:1] #(temporary feature only most prominent pulse is selected)
		
		#filter out "central" peak (-2.724 is to compensate for the shift due to antenna response calibration)
		#shift 2*(sqrt(2)*1.380 + 0.090)/(2.99792458E8)/(5E-9) ~ 2.724 (i.e. twice the distance between the two outer ends of the antenna parallel to the ground)
		indx_offset = 2*(np.sqrt(2)*1.380 + 0.090)/(2.99792458E8)/self.sampling_period
		dist_from_center = self.indices-(self.S[0].size//2 - indx_offset)

		try:
			sortIndices = [np.argmin(np.abs(dist_from_center))]
		except:
			#print("Error [1]: Can't get pulse width!")
			return 0, 0, 0

		if dist_from_center[sortIndices] > -2 and dist_from_center[sortIndices] < 2:
			pass
		else:
			#print("Error [2]: Can't get pulse width!")
			return 0, 0, 0

		self.indices = self.indices[sortIndices]
		for k in self.peakDict.keys():
			self.peakDict[k] = self.peakDict[k][sortIndices]

		dev_l = f_σ*(2*np.log(1/(1-R)))**(-0.5)*(self.indices-self.peakDict['left_ips']) #index deviation left
		dev_r = f_σ*(2*np.log(1/(1-R)))**(-0.5)*(self.peakDict['right_ips']-self.indices) #index deviation right

		#uncomment for debugging
		"""
		for i in range(self.width.size):
			print("Peak {}:\nposition:{}\nheight:{}\nwidth:{}".format(i, self.sampling_period*self.indices[i], self.peakDict['peak_heights'][i], self.width[i]))
		"""

		#used to average Stokes' vector
		self.lps = self.indices - dev_l #"left position"
		self.rps = self.indices + dev_r #"right position"

		#used for plotting
		self.t_l = self.sampling_period*self.lps #left time
		self.t_r = self.sampling_period*self.rps #right time

		self.width = self.t_r - self.t_l

		return self.width, self.t_l, self.t_r

	#function returning a 2D array of the averaged Stokes' parameters (axis 0: different pulses (most prominent pulses first), axis 1: averaged Stokes' parameters)
	def average_stokes_parameters(self,S=None):
		if S is not None:
			self.S = S

		if 'width' not in dir(self):
			self.get_pulseWidth()
		
		self.S_avg = []
		for i in range(self.indices.size):
			average = np.average(self.S[:,int(self.lps[i]):int(self.rps[i])],axis=1)
			self.S_avg.append(average)

		self.S_avg = np.array(self.S_avg)
	
		return self.S_avg

	#function returning the degree of polarization
	def get_dop(self,S_avg=None):
		if 'S_avg' not in dir(self):
			if S_avg is None:
				print("Please run 'average_stokes_parameters(S)' first or provide 'S_avg'.")
				return
			self.S_avg = S_avg

		self.δ = []
		for S in self.S_avg:
			self.δ.append(norm(S[1:])/S[0])

		self.δ = np.array(self.δ)

		return self.δ

	def polarization_ellipse_parameters(self,S_avg=None,selection=None):
		#set selection to an array of indices to select particular pulses
		if 'δ' not in dir(self):
			if 'S_avg' not in dir(self):
				if S_avg is None:
					print("Please run 'average_stokes_parameters(S)' first or provide 'S_avg'.")
					return
				self.S_avg = S_avg
			self.get_dop()

		if selection is None:
			#we choose to read out all the pulses from S_avg
			selection = list(range(self.S_avg.shape[0]))

		S0 = self.S_avg[selection][:,0]**0.5

		divisor = self.S_avg[selection][:,0]*self.δ[selection]
		normalizedS = self.S_avg[selection]/divisor[:, np.newaxis]
		ε = 0.5*np.arcsin(normalizedS[:,3])
		τ = 0.5*np.arctan2(normalizedS[:,2], normalizedS[:,1])

		self.ell_params = np.array([S0,self.δ[selection],τ,ε]).T

		return self.ell_params





#Plotter object for making nice plots
class stokes_plotter:
	def __init__(self, plot=None, sampling_period=5E-9):
		self.sampling_period = sampling_period
		self.plot = plot

		#setup plot canvas
		if plot is not None:
			if 'stokes' in plot:
				self.fig = figure(figsize=(20,10))
				self.fig.tight_layout(pad = 0.5, w_pad = 0.5, h_pad = 0.5)

				self.frame01 = self.fig.add_subplot(2,2,1)
				self.frame01.grid()
				self.frame01.set_xlabel(r"$t\ [s]$",fontsize=16)
				self.frame01.set_ylabel(r"$I$",fontsize=16)

				self.frame02 = self.fig.add_subplot(2,2,2)
				self.frame02.grid()
				self.frame02.set_xlabel(r"$t\ [s]$",fontsize=16)
				self.frame02.set_ylabel(r"$Q/I$",fontsize=16)

				self.frame03 = self.fig.add_subplot(2,2,3)
				self.frame03.grid()
				self.frame03.set_xlabel(r"$t\ [s]$",fontsize=16)
				self.frame03.set_ylabel(r"$U/I$",fontsize=16)

				self.frame04 = self.fig.add_subplot(2,2,4)
				self.frame04.grid()
				self.frame04.set_xlabel(r"$t\ [s]$",fontsize=16)
				self.frame04.set_ylabel(r"$V/I$",fontsize=16)
			
			if 'polarization_ellipse' in plot:
				fig1 = figure(figsize=(20,10))

				self.frame11 = fig1.add_subplot(111, aspect='equal')
				self.frame11.grid()
				self.frame11.set_xlabel(r"$E_{az}$",fontsize=16)
				self.frame11.set_ylabel(r"$E_z$",fontsize=16)
			
			if 'poincare' in plot:
				fig2 = figure(figsize=(10,10))

				self.frame12 = fig2.add_subplot(111, projection='3d')
				self.frame12.set_xlim((-1,1))
				self.frame12.set_ylim((-1,1))
				self.frame12.set_zlim((-1,1))
				self.frame12.grid()
				self.frame12.set_xlabel(r"$Q/I$",fontsize=16)
				self.frame12.set_ylabel(r"$U/I$",fontsize=16)
				self.frame12.set_zlabel(r"$V/I$",fontsize=16)
		else:
			print("Please define a canvas to generate (i.e. 'stokes', 'polarization_ellipse' or 'poincare').")

	def plot_stokes_parameters(self,S,antenna_names,width=None):
		if 'stokes' not in self.plot:
			print("No canvas for 'stokes'!")
			return

		t = self.sampling_period*np.arange(S[0].size)

		p = self.frame01.plot(t,S[0],label=r"{}/{}".format(antenna_names[0],antenna_names[1]))
		c = p[0].get_color()

		self.frame02.plot(t,S[1]/S[0],label=r"{}/{}".format(antenna_names[0],antenna_names[1]))

		self.frame03.plot(t,S[2]/S[0],label=r"{}/{}".format(antenna_names[0],antenna_names[1]))

		self.frame04.plot(t,S[3]/S[0],label=r"{}/{}".format(antenna_names[0],antenna_names[1]))

		if width is not None:
			low = np.zeros(width[0].size)
			high = np.full_like(width[0], np.max(S[0]))

			self.frame01.plot([width[0],width[0]],[low,high],linestyle='dashed',color=c)
			self.frame01.plot([width[1],width[1]],[low,high],linestyle='dashed',color=c)

			low = np.full_like(width[0],-1)
			high = np.full_like(width[0],1)

			self.frame02.plot([width[0],width[0]],[low,high],linestyle='dashed',color=c)
			self.frame02.plot([width[1],width[1]],[low,high],linestyle='dashed',color=c)

			self.frame03.plot([width[0],width[0]],[low,high],linestyle='dashed',color=c)
			self.frame03.plot([width[1],width[1]],[low,high],linestyle='dashed',color=c)

			self.frame04.plot([width[0],width[0]],[low,high],linestyle='dashed',color=c)
			self.frame04.plot([width[1],width[1]],[low,high],linestyle='dashed',color=c)

	def plot_polarization_ellipse(self,data,errors=False):
		#data should look as follows: columns are different parameters (S_0,δ,τ,ε,S_0_err,δ_err,τ_err,ε_err) and rows are different pulses measured by antennae or antenna arrays
		if 'polarization_ellipse' not in self.plot:
			print("No canvas for 'polarization_ellipse'!")
			return

		if errors:
			for i in range(np.size(data,axis=0)):
				width = 2*data[i][0]*np.cos(data[i][3])
				width_err = 2*np.sqrt((np.cos(data[i][3])*data[i][4])**2 + (data[i][0]*np.sin(data[i][3])*data[i][7])**2)

				height = 2*data[i][0]*np.sin(data[i][3])
				height_err = 2*np.sqrt((np.sin(data[i][3])*data[i][4])**2 + (data[i][0]*np.cos(data[i][3])*data[i][7])**2)

				angle = data[i][2]
				angle_err = data[i][6]

				pos = (0,0)
				pos_cov = np.array([[0,0],[0,0]])

				r, r_err = Ellipse(pos, width, height, angle, pos_cov=pos_cov, width_err=width_err, height_err=height_err, angle_err=angle_err)
				
				self.frame11.errorbar(r[0],r[1],xerr=r_err[0],yerr=r_err[1], capsize=2, capthick=0.5, elinewidth=1, ecolor='k')

			return

		for i in range(np.size(data,axis=0)):
			width = 2*data[i][0]*np.cos(data[i][3])
			height = 2*data[i][0]*np.sin(data[i][3])
			angle = data[i][2]

			x, y = Ellipse((0,0),width,height,angle)
			self.frame11.plot(x,y)

	def plot_poincare(self,data):
		#data should look as follows: columns are different Stokes averaged parameters and rows are different pulses measured by antennae or antenna arrays
		if 'poincare' not in self.plot:
			print("No canvas for poincare'!")
			return

		for i in range(np.size(data,axis=0)):
			SI = data[i]/data[i][0] #normalize the Stokes parameters
			QI = SI[1]
			UI = SI[2]
			VI = SI[3]
			self.frame12.scatter(QI,UI,VI)

	def showPlots(self, legend=False, sname=None):
		if legend:
			if 'stokes' in self.plot:
				self.frame01.legend()
				self.frame02.legend()
				self.frame03.legend()
				self.frame04.legend()

		show()
		#extent = self.frame01.get_tightbbox(self.fig.canvas.get_renderer()).transformed(self.fig.dpi_scale_trans.inverted())
		#self.fig.savefig("1477200_all_stations_I.pdf", dpi=self.fig.dpi, bbox_inches=extent) #extent





#function that returns datapoints forming an Ellipse. (patches.Ellipse does not work properly for our purposes)
def Ellipse(pos, width, height, angle, pos_cov=np.array([[0,0],[0,0]]), width_err=None, height_err=None, angle_err=None):
	t = np.linspace(0,2*np.pi,100)
	x = np.array([width/2*np.cos(t),height/2*np.sin(t)])

	#rotation matrix anti-clockwise
	R = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])

	x = np.matmul(R,x)
	x = x + np.array([np.full_like(x[0],pos[0]),np.full_like(x[1],pos[1])])

	if not None in [width_err, height_err, angle_err]:

		σ = np.array([width_err, height_err, angle_err, pos_cov[0][0]**0.5, pos_cov[1][1]**0.5])
		
		#partials are initiated:
		#dx/dwidth
		part_width = np.matmul(R, np.array([1/2*np.cos(t),np.zeros(t.size)]))

		#dx/dheight
		part_height = np.matmul(R, np.array([np.zeros(t.size),1/2*np.sin(t)]))

		#dx/dangle
		#derivative of rotation matrix anti-clockwise
		part_R = np.array([[-np.sin(angle), -np.cos(angle)], [np.cos(angle), -np.sin(angle)]])
		part_angle = np.matmul(part_R, np.array([width/2*np.cos(t),height/2*np.sin(t)]))

		#dx/dpos0
		part_pos0 = np.array([np.ones(t.size),np.zeros(t.size)])

		#dx/dpos1
		part_pos1 = np.array([np.zeros(t.size),np.ones(t.size)])

		parts = np.array([part_width, part_height, part_angle, part_pos0, part_pos1])

		x_var = np.zeros((2, t.size))
		for i in range(5):
			x_var += np.multiply(parts[i]*σ[i], parts[i]*σ[i])

		return x, np.sqrt(x_var)

	return x