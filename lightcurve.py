import numpy as np
import pandas as pd
from copy import deepcopy
from uncertainties import unumpy

class LightCurve:
	def __init__(self,x,y,yerr,xtype='time',ytype='flux'):
		if isinstance(x,np.ndarray)==False:
			x=np.array(x)
		if isinstance(y,np.ndarray)==False:
			y=np.array(y)
		if isinstance(yerr,np.ndarray)==False:
			yerr=np.array(yerr)

		x = pd.to_numeric(x,errors='coerce')
		y = pd.to_numeric(y,errors='coerce')
		yerr = pd.to_numeric(yerr,errors='coerce')

		self.x,self.y,self.yerr,self.xtype,self.ytype=x,y,yerr,xtype,ytype
		self.default_x,self.default_y,self.default_yerr,self.default_xtype,self.default_ytype=deepcopy(x),deepcopy(y),deepcopy(yerr),deepcopy(xtype),deepcopy(ytype)

	def Reset(self):
		self.x,self.y,self.yerr,self.xtype,self.ytype=self.default_x,self.default_y,self.default_yerr,self.default_xtype,self.default_ytype

	def Set_Default(self):
		self.default_x,self.default_y,self.default_yerr,self.default_xtype,self.default_ytype=deepcopy(self.x),deepcopy(self.y),deepcopy(self.yerr),deepcopy(self.xtype),deepcopy(self.ytype)

	def Mag2Flux(self,y=None,yerr=None):
		if self.ytype=='mag':
			if y==None:
				y=deepcopy(self.y)
			if yerr==None:
				yerr=deepcopy(self.yerr)
			y_uarray=10**(unumpy.uarray(y,np.abs(yerr))/-2.5)
			self.y,self.yerr=unumpy.nominal_values(y_uarray),unumpy.std_devs(y_uarray)
			self.ytype='flux'
		else:
			print('Type of y-values is ',self.ytype,' not mag, light curve unchanged...')

	def Remove_NanVals(self,rm_x=False,rm_y=True,rm_yerr=False,x=None,y=None,yerr=None):
		if x==None:
			x=deepcopy(self.x)
		if y==None:
			y=deepcopy(self.y)
		if yerr==None:
			yerr=self.yerr

		nan_index=[]

		if rm_x==True:
			nan_index+=list(np.where(pd.isnull(x))[0])
		if rm_y==True:
			nan_index+=list(np.where(pd.isnull(y))[0])
		if rm_yerr==True:
			nan_index+=list(np.where(pd.isnull(yerr))[0])

		nan_index=np.array(list(set(nan_index))).astype(int)

		self.x,self.y,self.yerr=np.delete(x,nan_index),np.delete(y,nan_index),np.delete(yerr,nan_index)


	def Flux2Mag(self,y=None,yerr=None):
		if self.ytype=='flux':
			if y==None:
				y=deepcopy(self.y)
			if yerr==None:
				yerr=deepcopy(self.yerr)
			y_uarray=-2.5*unumpy.log10(unumpy.uarray(y,yerr))
			self.y,self.yerr=unumpy.nominal_values(y_uarray),unumpy.std_devs(y_uarray)
			self.ytype='mag'
		else:
			print('Type of y-values is ',self.ytype,' not flux, light curve unchanged...')

	def Normalize(self,y=None,yerr=None,avg_type='median',magzero=0):
		if y==None:
			y=deepcopy(self.y)
		if yerr==None:
			yerr=deepcopy(self.yerr)

		if self.ytype=='mag':
			if avg_type=='median':
				self.y-=np.nanmedian(y)-magzero
				self.yerr=yerr
			elif avg_type=='mean':
				self.y-=np.nanmean(y)-magzero
				self.yerr=yerr

			else:
				print('No proper average type selected, light curve unchanged...')

		elif self.ytype=='flux':
			if avg_type=='median':
				self.y/=np.nanmedian(y)
				self.yerr/=np.nanmedian(y)

			elif avg_type=='mean':
				self.y/=np.nanmean(y)
				self.yerr/=np.nanmean(y)

			else:
				print('No proper average type selected, light curve unchanged...')

		else:
			print('No proper type for y-values, light curve unchanged...')
			self.y=y
			self.yerr=yerr

	def Bin(self,bin,method='bin_width',x=None,y=None,yerr=None):
		if x==None:
			x=deepcopy(self.x)
		if y==None:
			y=deepcopy(self.y)
		if yerr==None:
			yerr=deepcopy(self.yerr)

		xmin=np.nanmin(x)
		xmax=np.nanmax(x)

		if method=='bin_width':
			bin_width=bin
			Nbins=round((xmax-xmin)/bin_width)
		elif method=='Nbins':
			Nbins=bin
			bin_width=(xmax-xmin)/Nbins

		weights=np.zeros(Nbins+1)
		weighted_y=np.zeros(Nbins+1)
		for i in range(len(x)):
			if pd.notnull(x[i]):
				bin_num = round((x[i]-xmin)/bin_width)
			else:
				continue
			weights[bin_num]+=yerr[i]**-2
			weighted_y[bin_num]+=y[i]*yerr[i]**-2

		binned_x=(np.arange(Nbins+1)*bin_width)[:-1]+xmin
		binned_y=(weighted_y.astype(np.float64)/weights.astype(np.float64))[:-1]
		binned_yerr=(np.sqrt(1/weights.astype(np.float64)))[:-1]

		keep_index=np.where((pd.notnull(binned_x))&(pd.notnull(binned_y))&(pd.notnull(binned_yerr)))[0]
		binned_x,binned_y,binned_yerr=binned_x[keep_index],binned_y[keep_index],binned_yerr[keep_index]
		sort_index=np.argsort(binned_x)

		self.x,self.y,self.yerr=binned_x[sort_index],binned_y[sort_index],binned_yerr[sort_index]

	def Phase_Fold(self,period,zeropoint='min',x=None,y=None,yerr=None):
		if self.xtype=='time':
			if x==None:
				x=deepcopy(self.x)
			if y==None:
				y=deepcopy(self.y)
			if yerr==None:
				yerr=deepcopy(self.yerr)

			if zeropoint=='min':
				phase=(x-np.nanmin(x))/period % 1
			elif zeropoint=='None':
				phase=x/period % 1
			else:
				phase=(x-zeropoint)/period % 1

			sort_index=np.argsort(phase)

			self.x,self.y,self.yerr=phase[sort_index],y[sort_index],yerr[sort_index]
			self.xtype='phase'
		else:
			print('Type of x-values is ',self.xtype,' not time, light curve unchanged...')

	def Clip(self,avg_type='median',clip_type='rms',dim_clip=3.5,bri_clip=3.5,stop_cond='iterative',x=None,y=None,yerr=None):
		if x==None:
			x=deepcopy(self.x)
		if y==None:
			y=deepcopy(self.y)
		if yerr==None:
			yerr=deepcopy(self.yerr)

		if avg_type=='median':
			averagef=lambda y: np.nanmedian(y)
		elif avg_type=='mean':
			averagef=lambda y: np.nanmean(y)
		else:
			print('No average selected, light curve unchanged...')

		if clip_type=='mad':
			sigmaf=lambda y: np.nanmedian(np.abs(y - np.nanmedian(y)))
		elif clip_type=='rms':
			sigmaf=lambda y: np.nanstd(y)
		else:
			print('No clip type selected, light curve unchanged...')

		if self.ytype=='mag':
                        if stop_cond=='iterative':
                            end_clip=False
                            while end_clip==False:
                                average,sigma=averagef(y),sigmaf(y)
                                keep_index=np.where((y<=average+dim_clip*sigma)&(y>=average-bri_clip*sigma))[0]
                                if len(keep_index) != len(y):
                                    x,y,yerr=x[keep_index],y[keep_index],yerr[keep_index]
                                else:
                                    self.x,self.y,self.yerr=x[keep_index],y[keep_index],yerr[keep_index]
                                    end_clip=True

                        elif stop_cond=='single':
                            average,sigma=averagef(y),sigmaf(y)
                            keep_index=np.where((y<=average+dim_clip*sigma)&(y>=average-bri_clip*sigma))[0]
                            self.x,self.y,self.yerr=x[keep_index],y[keep_index],yerr[keep_index]



		elif self.ytype=='flux':
                        if stop_cond=='iterative':
                            end_clip=False
                            while end_clip==False:
                                average,sigma=averagef(y),sigmaf(y)
                                keep_index=np.where((y>=average-dim_clip*sigma)&(y<=average+bri_clip*sigma))[0]
                                if len(keep_index) != len(y):
                                    x,y,yerr=x[keep_index],y[keep_index],yerr[keep_index]
                                else:
                                    self.x,self.y,self.yerr=x[keep_index],y[keep_index],yerr[keep_index]
                                    end_clip=True

                        elif stop_cond=='single':
                            average,sigma=averagef(y),sigmaf(y)
                            keep_index=np.where((y>=average-dim_clip*sigma)&(y<=average+bri_clip*sigma))[0]
                            self.x,self.y,self.yerr=x[keep_index],y[keep_index],yerr[keep_index]

		else:
			print('No proper type for y-values, light curve unchanged...')

	def PolyDetrend(self,deg,weight=True,x=None,y=None,yerr=None):
		if x==None:
			x=deepcopy(self.x)
		if y==None:
			y=deepcopy(self.y)
		if yerr==None:
			yerr=deepcopy(self.yerr)

		if weight==True:
			coeff=np.polyfit(x,y,deg,w=1/yerr)
		else:
			coeff=np.polyfit(x,y,deg)

		polyf=np.poly1d(coeff)

		y=y-polyf(x)+np.nanmedian(y)

		self.x,self.y,self.yerr=x,y,yerr


	def Data(self):
		return (self.x,self.y,self.yerr)
