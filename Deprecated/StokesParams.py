#!/usr/bin/env python3

import numpy as np
from numpy.linalg import norm
import scipy.signal as sg
from matplotlib.pyplot import figure
from mpl_toolkits.mplot3d import Axes3D

class SP:
	def __init__(self, plot=None): #use ['stokes', 'polarization_ellipse', 'poincare'] to define what you want to plot 
		self.sampling_period = 5E-9

		self.StokesDict = {}
		self.δDict = {}
		self.ell_paramsDict = {}
		self.antennaNum = 0

		#setup plot canvas
		self.plot = plot

		if plot is not None:
			if 'stokes' in plot:
				fig = figure(figsize=(10,10))
				fig.tight_layout(pad = 0.5, w_pad = 0.5, h_pad = 0.5)

				self.frame1 = fig.add_subplot(2,2,1)
				self.frame1.grid()
				self.frame1.set_xlabel(r"$t\ [s]$",fontsize=16)
				self.frame1.set_ylabel(r"$I$",fontsize=16)

				self.frame2 = fig.add_subplot(2,2,2)
				self.frame2.grid()
				self.frame2.set_xlabel(r"$t\ [s]$",fontsize=16)
				self.frame2.set_ylabel(r"$Q/I$",fontsize=16)

				self.frame3 = fig.add_subplot(2,2,3)
				self.frame3.grid()
				self.frame3.set_xlabel(r"$t\ [s]$",fontsize=16)
				self.frame3.set_ylabel(r"$U/I$",fontsize=16)

				self.frame4 = fig.add_subplot(2,2,4)
				self.frame4.grid()
				self.frame4.set_xlabel(r"$t\ [s]$",fontsize=16)
				self.frame4.set_ylabel(r"$V/I$",fontsize=16)
			
			if 'polarization_ellipse' in plot:
				fig1 = figure(figsize=(10,10))

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

	def plot_stokes_parameters(self,plotWidth=False):
		if 'Stokes' not in dir(self):
			print("'self.Stokes' is not defined. Please run 'get_stokes_parameters()' first.")
			return

		if self.plot is None:
			print("Please generate the 'SP' object with 'plot=['stokes']'.")
			return
		
		if 'stokes' not in self.plot:
			print("Please put 'stokes' in the plot array of the 'SP' object.")
			return

		t = self.sampling_period*np.arange(self.Stokes[0].size)

		p = self.frame1.plot(t,self.Stokes[0],label=r"{}/{}".format(self.antenna_names[0],self.antenna_names[1]))
		c = p[0].get_color()

		self.frame2.plot(t,self.Stokes[1]/self.Stokes[0],label=r"{}/{}".format(self.antenna_names[0],self.antenna_names[1]))

		self.frame3.plot(t,self.Stokes[2]/self.Stokes[0],label=r"{}/{}".format(self.antenna_names[0],self.antenna_names[1]))

		self.frame4.plot(t,self.Stokes[3]/self.Stokes[0],label=r"{}/{}".format(self.antenna_names[0],self.antenna_names[1]))

		if plotWidth:
			self.frame1.plot([self.t_l,self.t_l],[np.zeros(self.t_l.size),self.peakDict['peak_heights']],linestyle='dashed',color=c)
			self.frame1.plot([self.t_r,self.t_r],[np.zeros(self.t_r.size),self.peakDict['peak_heights']],linestyle='dashed',color=c)

			low = np.full_like(self.t_l,-1)
			high = np.full_like(self.t_l,1)

			self.frame2.plot([self.t_l,self.t_l],[low,high],linestyle='dashed',color=c)
			self.frame2.plot([self.t_r,self.t_r],[low,high],linestyle='dashed',color=c)

			self.frame3.plot([self.t_l,self.t_l],[low,high],linestyle='dashed',color=c)
			self.frame3.plot([self.t_r,self.t_r],[low,high],linestyle='dashed',color=c)

			self.frame4.plot([self.t_l,self.t_l],[low,high],linestyle='dashed',color=c)
			self.frame4.plot([self.t_r,self.t_r],[low,high],linestyle='dashed',color=c)

	def plot_polarization_ellipse(self,data=None):
		#data should look as follows: columns are measurements of antennae or antenna arrays and rows are the different parameters
		if data is None:
			if 'ell_params' not in dir(self):
				print("'self.ell_params' is not defined. Please run 'polarization_ellipse_parameters(output=False)' first.")
				return
			data = self.ell_params[[0,2,3]]

		if self.plot is None:
			print("Please generate the 'SP' object with 'plot=['polarization_ellipse']'.")
			return
		
		if 'polarization_ellipse' not in self.plot:
			print("Please put 'polarization_ellipse' in the plot array of the 'SP' object.")
			return

		for i in range(np.size(data,axis=1)):
			width = 2*data[0][i]*np.cos(data[2][i])
			height = 2*data[0][i]*np.sin(data[2][i])
			angle = data[1][i]

			x, y = Ellipse((0,0),width,height,angle=angle)
			self.frame11.plot(x,y)

	def plot_poincare(self,data=None):
		#data should look as follows: columns are different Stokes averaged parameters and rows are measurements of antennae or antenna arrays
		if data is None:
			if 'Stokes_avg' not in dir(self):
				print("'self.Stokes_avg' is not defined. Please run 'average_stokes_parameters(output=False)' first.")
				return
			data = self.Stokes_avg

		if self.plot is None:
			print("Please generate the 'SP' object with 'plot=['polarization_ellipse']'.")
			return
		
		if 'poincare' not in self.plot:
			print("Please put 'poincare' in the plot array of the 'SP' object.")
			return

		for i in range(np.size(data,axis=0)):
			SI = data[i]/data[i][0]
			QI = SI[1]
			UI = SI[2]
			VI = SI[3]
			self.frame12.scatter(QI,UI,VI)

	def plotlegend(self,figure=None):
		if figure is None:
			print("Please define for which figure(s) you want a legend.")
			return

		if 'stokes' in figure:
			self.frame1.legend()
			self.frame2.legend()
			self.frame3.legend()
			self.frame4.legend()

		if 'polarization_ellipse' in figure:
			print("No legend available for 'polarization_ellipse'.")

		if 'poincare' in figure:
			print("No legend available for 'poincare'.")

	def get_stokes_parameters(self,Ex,Ey,antenna_names=None,output=False):
		if self.plot is not None:
			if antenna_names is None:
				print("Please specify the antenna name.")
				return
			self.antenna_names = antenna_names

		I = np.abs(Ex)**2 + np.abs(Ey)**2
		Q = np.abs(Ex)**2 - np.abs(Ey)**2
		p = Ex*np.conj(Ey)
		U = 2*np.real(p)
		V = -2*np.imag(p)
		self.Stokes = np.array([I,Q,U,V])

		if output:
			return self.Stokes

	def get_pulseWidth(self,f_hp=0.5,f_σ=2,output=False):
		if 'Stokes' not in dir(self):
			print("Please run 'get_stokes_parameters(Ex,Ey,output=False,antenna_names=None)' first.")
			return

		R = 1 - f_hp

		min_height = np.max(self.Stokes[0])*0.25 #threshold is set to ~25% of the maximum height of a pulse in the dataset

		self.indices, self.peakDict = sg.find_peaks(self.Stokes[0],height=min_height,prominence=None,rel_height=R,width=0)

		dev_l = f_σ*(2*np.log(1/(1-R)))**(-0.5)*(self.indices-self.peakDict['left_ips']) #index deviation left
		dev_r = f_σ*(2*np.log(1/(1-R)))**(-0.5)*(self.peakDict['right_ips']-self.indices) #index deviation right

		self.width = self.sampling_period*(dev_r + dev_l) #width of pulse in seconds

		"""
		#uncomment for debugging
		for i in range(self.width.size):
			print("Peak {}:\nposition:{}\nheight:{}\nwidth:{}".format(i, self.sampling_period*self.indices[i], self.peakDict['peak_heights'][i], self.width[i]))
		"""

		self.lps = self.indices - dev_l #"left position"
		self.rps = self.indices + dev_r #"right position"

		self.t_l = self.sampling_period*self.lps #left time
		self.t_r = self.sampling_period*self.rps #right time

		if output:
			return self.width

	def average_stokes_parameters(self,output=False):
		if 'peakDict' not in dir(self):
			print("Please run 'get_pulseWidth(output=False)' first.")
			return

		Stokes_avg = []
		for i in range(self.indices.size):
			avg = np.average(self.Stokes[:,int(self.lps[i]):int(self.rps[i])],axis=1)
			Stokes_avg.append(avg)

		self.Stokes_avg = np.array(Stokes_avg)

		self.StokesDict['{}'.format(self.antennaNum)] = self.Stokes_avg

		if output:
			return self.Stokes_avg

	def get_dop(self,output=False):
		if 'Stokes_avg' not in dir(self):
			print("Please run 'average_stokes_parameters(output=False)' first.")
			return

		δ = []
		for S in self.Stokes_avg:
			δ.append(norm(S[1:])/S[0])

		self.δ = np.array(δ)
		self.δDict['{}'.format(self.antennaNum)] = self.δ

		if output:
			return self.δ

	def polarization_ellipse_parameters(self,pulseNum=None,output=False): #overwrite pulseNum to select a particular pulse you want to show/save
		if 'Stokes_avg' not in dir(self):
			print("Please run 'average_stokes_parameters(output=False)' first.")
			return

		if 'δ' not in dir(self):
			print("Please run 'get_dop(output=False)' first.")

		if pulseNum is None:
			pulseNum = list(range(self.indices.size))

		divisor = self.Stokes_avg[pulseNum][:,0]*self.δ[pulseNum]

		normalizeS = self.Stokes_avg[pulseNum]/divisor[:, np.newaxis]

		S0 = self.Stokes_avg[pulseNum][:,0]**0.5
		ε = 0.5*np.arcsin(normalizeS[:,3])
		τ = 0.5*np.arcsin2(normalizeS[:,2], np.arccos(2*ε))

		self.ell_params = np.array([S0,self.δ[pulseNum],τ,ε])
		self.ell_paramsDict['{}'.format(self.antennaNum)] = self.ell_params[[0,2,3]]

		if output:
			return self.ell_params

	def next_antenna(self):
		self.antennaNum += 1

	#def average_over_antennas(self):

	def print_data_dicts(self):
		print(self.StokesDict)
		print(self.δDict)
		print(self.ell_paramsDict)

	#def save_data(self,data_dir='.'):
	#	print("Saving <data> to <dir/filename>")

#function that returns datapoints forming an ellipse (patches.Ellipse does not work properly for our purposes)
def Ellipse(pos,width,height,angle=0):
	t = np.linspace(0,2*np.pi,100)
	x = np.array([width/2*np.cos(t),height/2*np.sin(t)])

	#rotation matrix
	R = np.array([[np.cos(angle),np.sin(angle)],[-np.sin(angle),np.cos(angle)]])

	x = np.matmul(R,x)
	x = x + np.array([np.full_like(x[0],pos[0]),np.full_like(x[1],pos[1])])
	return x