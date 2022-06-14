import tkinter as tk
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import re
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt
import os
import glob
import random
from copy import deepcopy
from lightcurve import LightCurve
from lcfuncs import *
import lcwindow_config as cfg

def load_file(file):
	if file[:2]==r'L:':
		global filelist
		filelist = load_list(file[2:])
		global filelist_index
		filelist_index=0

		import_entry.delete(0,tk.END)
		import_entry.insert(0,filelist[0])
		export_entry.delete(0,tk.END)
		export_entry.insert(0,filelist[0]+'.wlc')
		export_plot_entry.delete(0,tk.END)
		export_plot_entry.insert(0,filelist[0]+'.png')

		df = pd.read_csv(filelist[0],sep=cfg.delimiter,header=cfg.header)
		loaded_str.set('1/'+str(len(filelist)))
	else:
		df = pd.read_csv(file,sep=cfg.delimiter,header=cfg.header)
		export_entry.delete(0,tk.END)
		export_entry.insert(0,file+'.wlc')
		export_plot_entry.delete(0,tk.END)
		export_plot_entry.insert(0,file+'.png')

	x,y,yerr=np.array(df[df.columns[0]]), np.array(df[df.columns[1]]), np.array(df[df.columns[2]])
	global lc
	lc = LightCurve(x,y,yerr,xtype=cfg.xtype,ytype=cfg.ytype)
	lc.Remove_NanVals()
	lc.model=None
	plot_lc(lc)
	zoom_x_entry.delete(0,tk.END)
	zoom_y_entry.delete(0,tk.END)
	zoom_x_entry.insert(0,str(round(np.nanmin(lc.x),4))+':'+str(round(np.nanmax(lc.x),4)))
	zoom_y_entry.insert(0,str(round(np.nanmin(lc.y),4))+':'+str(round(np.nanmax(lc.y),4)))
	phasefold_entry.delete(0,tk.END)
	export_stats_entry.delete(0,tk.END)
	export_periodogram_entry.delete(0,tk.END)


def load_list(listpath):
	with open(listpath,'r') as f:
		filelist=[line.rstrip('\n') for line in f]

	return filelist

def plot_lc(lc,export=False):
	if export==True:
		plt.Figure(figsize=(10,8))
		plt.scatter(lc.x,lc.y,color='black',s=5,alpha=.33,zorder=0)
		plt.xlim(np.min(lc.x),np.max(lc.x))
		plt.ylim(np.min(lc.y),np.max(lc.y))

		if lc.xtype=='time':
			plt.xlabel('Time')
		elif lc.xtype=='phase':
			plt.xlabel('Phase')

		if lc.ytype=='flux':
			plt.ylabel('Flux')
		if lc.ytype=='mag':
			plt.ylabel('Magnitude')
			plt.gca().invert_yaxis()

		if isinstance(lc.model,type(lc)):
			plt.plot(lc.model.x,lc.model.y,color='red')
		plt.tight_layout()
		plt.savefig(export_plot_entry.get(),format='png')
		plt.close()

	elif export==False:
		sbp.clear()
		sbp.scatter(lc.x,lc.y,color='black',s=5,alpha=.33,zorder=0)
		sbp.set_xlim(np.min(lc.x),np.max(lc.x))
		sbp.set_ylim(np.min(lc.y),np.max(lc.y))

		if lc.xtype=='time':
			sbp.set_xlabel('Time')
		elif lc.xtype=='phase':
			sbp.set_xlabel('Phase')

		if lc.ytype=='flux':
			sbp.set_ylabel('Flux')
		if lc.ytype=='mag':
			sbp.set_ylabel('Magnitude')
			sbp.invert_yaxis()
		#sbp.set_ylabel('Normalized Flux')
		figure.tight_layout()
		figure.canvas.draw_idle()

def flux2mag(lc):
	lc.Flux2Mag()
	plot_lc(lc)
	if isinstance(lc.model,type(lc)):
		lc.model.Flux2Mag()
		sbp.plot(lc.model.x,lc.model.y,color='red')

def mag2flux(lc):
	lc.Mag2Flux()
	plot_lc(lc)
	if isinstance(lc.model,type(lc)):
		lc.model.Mag2Flux()
		sbp.plot(lc.model.x,lc.model.y,color='red')

def normalize(lc):
	lc.Normalize()
	plot_lc(lc)

def zoom_x():
	xlims=zoom_x_entry.get()
	xi,xf=xlims.split(':')
	sbp.set_xlim(float(xi),float(xf))

	figure.canvas.draw_idle()

def zoom_y():
	ylims=zoom_y_entry.get()
	yi,yf=ylims.split(':')
	sbp.set_ylim(float(yi),float(yf))

	figure.canvas.draw_idle()

def harmonic_check(f_test,f_prime,f_err):
	if f_test >= f_prime:
		if abs((f_test/f_prime)-round(f_test/f_prime))*f_prime<=f_err:
			harmonic=True
		else:
			harmonic=False

	elif f_test < f_prime:
		if abs((f_prime/f_test)-round(f_prime/f_test))*f_test<=f_err:
			harmonic=True
		else:
			harmonic=False

	return harmonic

def build_periodogram(lc,periodogram,export=False):
	if export==True:
		plt.Figure(figsize=(10,8))
		if periodogram.lower() == 'ls':
			frequencies,powers=Periodicity.LS_Periodogram(lc.x,lc.y,lc.yerr,min_period=cfg.min_period,max_period=cfg.max_period,nsamples=cfg.ls_nsamples)
			periods=1/frequencies
			plt.plot(frequencies,powers,color='black',lw=.75)
			plt.xlabel('Frequency')
			plt.ylabel('Normalized Power')

		if periodogram.lower() == 'bls':
			periods,powers=Periodicity.BLS_Periodogram(lc.x,lc.y,lc.yerr,min_period=cfg.min_period,max_period=cfg.max_period,nsamples=cfg.bls_nsamples,duration=0.02,min_n_trans=2)
			frequencies=1/periods
			plt.plot(frequencies,powers,color='black',lw=.75)
			plt.xlabel('Frequency')
			plt.ylabel('Power')

		sorted_power_index = np.argsort(powers)[::-1]
		frequencies_sorted, powers_sorted = frequencies[sorted_power_index], powers[sorted_power_index]
		freq_err=2/(np.median(lc.x[1:]-lc.x[:-1])*len(lc.x))

		peak_index=[0]

		count=1
		while len(peak_index)<cfg.npeaks:
			if np.min(np.abs(frequencies_sorted[count]-frequencies_sorted[peak_index])) > freq_err:
				peak_index.append(count)
			elif count==len(frequencies_sorted):
				break
			else:
				pass

			count+=1

		peak_frequencies, peak_powers = frequencies_sorted[peak_index], powers_sorted[peak_index]

		for n in range(len(peak_index)):
			if n==0:
				color='green'
			elif harmonic_check(peak_frequencies[n],peak_frequencies[0],freq_err):
				color='blue'
			else:
				color='orange'
			plt.scatter(peak_frequencies[n],peak_powers[n],marker='.',s=6,color=color,zorder=10)
			plt.annotate('{:.6f}'.format(1/peak_frequencies[n]),(peak_frequencies[n],peak_powers[n]),color=color,fontsize=5,zorder=10)


		plt.tight_layout()
		plt.savefig(export_periodogram_entry.get(),format='png')
		plt.close()

	elif export==False:
		ps_window = tk.Toplevel()
		ps_figure = plt.Figure(figsize=(4,3), dpi=200)
		ps_plot = FigureCanvasTkAgg(ps_figure, ps_window)
		ps_plot.get_tk_widget().pack()
		ps_sbp=ps_figure.add_subplot(111)

		if periodogram.lower() == 'ls':
			ps_window.title('Lomb-Scargle Periodogram')
			frequencies,powers=Periodicity.LS_Periodogram(lc.x,lc.y,lc.yerr,min_period=cfg.min_period,max_period=cfg.max_period,nsamples=cfg.ls_nsamples)
			periods=1/frequencies
			ps_sbp.plot(frequencies,powers,color='black',lw=.75)
			ps_sbp.set_xlabel('Frequency')
			ps_sbp.set_ylabel('Normalized Power')

		if periodogram.lower() == 'bls':
			ps_window.title('Box-Least-Squares Periodogram')
			periods,powers=Periodicity.BLS_Periodogram(lc.x,lc.y,lc.yerr,min_period=cfg.min_period,max_period=cfg.max_period,nsamples=cfg.bls_nsamples,duration=0.02,min_n_trans=2)
			frequencies=1/periods
			ps_sbp.plot(frequencies,powers,color='black',lw=.75)
			ps_sbp.set_xlabel('Frequency')
			ps_sbp.set_ylabel('Power')


		sorted_power_index = np.argsort(powers)[::-1]
		frequencies_sorted, powers_sorted = frequencies[sorted_power_index], powers[sorted_power_index]
		freq_err=2/(np.median(lc.x[1:]-lc.x[:-1])*len(lc.x))

		peak_index=[0]

		count=1
		while len(peak_index)<cfg.npeaks:
			if np.min(np.abs(frequencies_sorted[count]-frequencies_sorted[peak_index])) > freq_err:
				peak_index.append(count)
			elif count==len(frequencies_sorted):
				break
			else:
				pass

			count+=1

		peak_frequencies, peak_powers = frequencies_sorted[peak_index], powers_sorted[peak_index]

		for n in range(len(peak_index)):
			if n==0:
				color='green'
			elif harmonic_check(peak_frequencies[n],peak_frequencies[0],freq_err):
				color='blue'
			else:
				color='orange'
			ps_sbp.scatter(peak_frequencies[n],peak_powers[n],marker='.',s=6,color=color,zorder=10)
			ps_sbp.annotate('{:.6f}'.format(1/peak_frequencies[n]),(peak_frequencies[n],peak_powers[n]),color=color,fontsize=5,zorder=10)

		ps_figure.tight_layout()
		ps_figure.canvas.draw_idle()

		phasefold_entry.delete(0,tk.END)
		phasefold_entry.insert(0,'{:.6f}'.format(periods[sorted_power_index[0]]))

		export_periodogram_entry.delete(0,tk.END)
		export_periodogram_entry.insert(0,import_entry.get()+'_'+periodogram.lower()+'_pdgm.png')

def phase_fold(lc,period):
	lc.Phase_Fold(period)
	plot_lc(lc)
	if isinstance(lc.model,type(lc)):
		lc.model.Phase_Fold(period)
		sbp.plot(lc.model.x,lc.model.y,color='red')

def bin(lc,bin_params):
	bin_val,bin_method=bin_params.split(',')

	if bin_method=='Nbins':
		bin_val=int(bin_val)
	elif bin_method=='bin_width':
		bin_val=float(bin_val)

	lc.Bin(bin_val,bin_method)
	plot_lc(lc)

def clip(lc,clip_params):
	clip_min,clip_max,clip_type,clip_stop = clip_params.split(',')
	clip_min,clip_max = float(clip_min), float(clip_max)
	if lc.ytype=='flux':
		lc.Clip(avg_type='median',clip_type=clip_type,dim_clip=clip_min,bri_clip=clip_max,stop_cond=clip_stop)
	elif lc.ytype=='mag':
		lc.Clip(avg_type='median',clip_type=clip_type,dim_clip=clip_min,bri_clip=clip_max,stop_cond=clip_stop)

	plot_lc(lc)


def fit_model(lc,model):
	if model.lower() == '':
		plot_lc(lc)
		lc.model=None
		return

	elif model.lower() == 'linear':
		y_model,coeffs=Fitting.PolyFit(lc.x,lc.y,deg=1)

	elif model.lower() == 'poly':
		y_model,coeffs=Fitting.PolyFit(lc.x,lc.y,deg='infer',max_deg=10)

	elif (model.lower() == 'sine') | (model.lower() == 'cosine'):
		if phasefold_entry.get()!='':
			period=float(phasefold_entry.get())
		else:
			period='infer'

		y_model,coeffs,Nharm,Nsharm=Fitting.HarmFit(lc.x,lc.y,lc.yerr,period=period,Nharm=0,Nsharm=0)

	elif model.lower() == 'harmonic':
		if phasefold_entry.get()!='':
			period=float(phasefold_entry.get())
		else:
			period='infer'

		y_model,coeffs,Nharm,Nsharm=Fitting.HarmFit(lc.x,lc.y,lc.yerr,period=period,Nharm='infer',Nsharm='infer',max_Nharm=6,max_Nsharm=6)

	elif model.lower() == 'spline':
		y_model,coeffs=Fitting.SplineFit(lc.x,lc.y,3,Nknots='infer',min_Nknots=30,max_Nknots=3000)

	lc.model=LightCurve(lc.x,y_model,lc.yerr,ytype=lc.ytype)

	plot_lc(lc)
	sbp.plot(lc.model.x,lc.model.y,color='red')

def model_detrend(lc,lc_model):
	if lc_model != None:
		lc.y=lc.y-lc.model.y+np.nanmedian(lc.y)
		lc.model=None
		plot_lc(lc)
	else:
		pass

def set_checkpoint(lc):
	lc.Set_Default()

def undo_changes(lc):
	lc.Reset()
	plot_lc(lc)

def next_file():
	global filelist_index
	filelist_index=filelist_index+1
	if filelist_index > len(filelist)-1:
		filelist_index-=len(filelist)
	loaded_str.set(str(filelist_index+1)+'/'+str(len(filelist)))
	import_entry.delete(0,tk.END)
	import_entry.insert(0,filelist[filelist_index])
	load_file(filelist[filelist_index])

def previous_file():
	global filelist_index
	filelist_index=filelist_index-1
	if filelist_index < 0:
		filelist_index+=len(filelist)
	loaded_str.set(str(filelist_index+1)+'/'+str(len(filelist)))
	import_entry.delete(0,tk.END)
	import_entry.insert(0,filelist[filelist_index])
	load_file(filelist[filelist_index])

def view_stats(export=False):
	mean=Average.Mean(lc.y)
	median=Average.Median(lc.y)
	rms=Dispersion.Rms(lc.y)
	mad=Dispersion.Mad(lc.y)
	p2p_mean=Dispersion.P2P_Mean_Var(lc.y)
	p2p_median=Dispersion.P2P_Median_Var(lc.y)
	skew=Dispersion.Skew(lc.y)
	kurtosis=Dispersion.Kurtosis(lc.y)
	cov=Dispersion.CoV(lc.y)
	iqr=Dispersion.IQR(lc.y)
	dim_outliers=Dispersion.Noutliers_Dim(lc.y,ytype=lc.ytype)
	bri_outliers=Dispersion.Noutliers_Bri(lc.y,ytype=lc.ytype)
	bri_ratio=Depth.Brightness_Ratio(lc.y)
	amplitude=Depth.Amplitude(lc.y,peak=0.02)

	if export==False:
		st_window = tk.Toplevel()
		st_window.title('LC Stats')

		mean_label=tk.Label(st_window,text='Mean: '+'{:.4f}'.format(mean))
		median_label=tk.Label(st_window,text='Median: '+'{:.4f}'.format(median))
		std_label=tk.Label(st_window,text='Std: '+'{:.4f}'.format(rms))
		mad_label=tk.Label(st_window,text='MAD: '+'{:.4f}'.format(mad))
		p2p_mean_label=tk.Label(st_window,text='P2P Mean Var: '+'{:.4f}'.format(p2p_mean))
		p2p_median_label=tk.Label(st_window,text='P2P Median Var: '+'{:.4f}'.format(p2p_median))
		skew_label=tk.Label(st_window,text='Skew: '+'{:.4f}'.format(skew))
		kurtosis_label=tk.Label(st_window,text='Kurtosis: '+'{:.4f}'.format(kurtosis))
		cov_label=tk.Label(st_window,text='CoV: '+'{:.4f}'.format(cov))
		iqr_label=tk.Label(st_window,text='IQR: '+'{:.4f}'.format(Dispersion.IQR(lc.y)))
		dim_outliers_label=tk.Label(st_window,text='Dim Outliers: '+'{:.0f}'.format(dim_outliers))
		bri_outliers_label=tk.Label(st_window,text='Bri Outliers: '+'{:.0f}'.format(bri_outliers))
		bri_ratio_label=tk.Label(st_window,text='Bri Ratio: '+'{:.4f}'.format(bri_ratio))
		amplitude_label=tk.Label(st_window,text='2% Amplitude: '+'{:.4f}'.format(amplitude))

		mean_label.pack()
		median_label.pack()
		std_label.pack()
		mad_label.pack()
		p2p_mean_label.pack()
		p2p_median_label.pack()
		skew_label.pack()
		kurtosis_label.pack()
		cov_label.pack()
		iqr_label.pack()
		dim_outliers_label.pack()
		bri_outliers_label.pack()
		bri_ratio_label.pack()
		amplitude_label.pack()
		export_stats_entry.delete(0,tk.END)
		export_stats_entry.insert(0,import_entry.get()+'.stats')

	elif export==True:
		with open(export_stats_entry.get(),'w+') as f:
			f.write('Mean: '+'{:.4f}'.format(mean)+'\n')
			f.write('Median: '+'{:.4f}'.format(median)+'\n')
			f.write('Std: '+'{:.4f}'.format(rms)+'\n')
			f.write('MAD: '+'{:.4f}'.format(mad)+'\n')
			f.write('P2P Mean Var: '+'{:.4f}'.format(p2p_mean)+'\n')
			f.write('P2P Median Var: '+'{:.4f}'.format(p2p_median)+'\n')
			f.write('Skew: '+'{:.4f}'.format(skew)+'\n')
			f.write('Kurtosis: '+'{:.4f}'.format(kurtosis)+'\n')
			f.write('CoV: '+'{:.4f}'.format(cov)+'\n')
			f.write('IQR: '+'{:.4f}'.format(iqr)+'\n')
			f.write('Dim Outliers: '+'{:.0f}'.format(dim_outliers)+'\n')
			f.write('Bri Outliers: '+'{:.0f}'.format(bri_outliers)+'\n')
			f.write('Bri Ratio: '+'{:.4f}'.format(bri_ratio)+'\n')
			f.write('Amplitude: '+'{:.4f}'.format(amplitude)+'\n')


def export_lc(lc):
	with open(export_entry.get(),'w+') as f:
		for n in range(len(lc.x)):
			x,y,yerr='{:.4f}'.format(lc.x[n]),'{:.4f}'.format(lc.y[n]),'{:.4f}'.format(lc.yerr[n])+'\n'
			f.write(x+','+y+','+yerr)

def export_stats():
	view_stats(export=True)


root = tk.Tk()
root.rowconfigure(0, weight=1)
root.columnconfigure(0, weight=1)

plotframe=tk.Frame()
maniframe=tk.Frame(root)
fileframe=tk.Frame(root)
calcframe=tk.Frame(root)
tablframe=tk.Frame(root)

plotframe.config(bd=1, relief='solid')
maniframe.config(bd=1, relief='solid')
fileframe.config(bd=1, relief='solid')
calcframe.config(bd=1, relief='solid')
tablframe.config(bd=1, relief='solid')

plotframe.grid(row=0,column=0,rowspan=2,columnspan=2,sticky='nsew')
maniframe.grid(row=0,column=2,rowspan=2,columnspan=2,sticky='nsew')
fileframe.grid(row=2,column=0,rowspan=1,columnspan=1,sticky='nsew')
calcframe.grid(row=2,column=1,rowspan=1,columnspan=1,sticky='nsew')
tablframe.grid(row=2,column=2,rowspan=1,columnspan=2,sticky='nsew')

figure = plt.Figure(figsize=(5,3), dpi=200)
sbp=figure.add_subplot(111)
plt.ion()
lcplot = FigureCanvasTkAgg(figure, plotframe)
lcplot.draw()
lcplot.get_tk_widget().pack()

import_label=tk.Label(fileframe,text='Import LC(s): ',relief='solid')
import_entry=tk.Entry(fileframe)
import_button=tk.Button(fileframe,text='Load',bg='green',command=lambda: load_file(import_entry.get()))

export_label=tk.Label(fileframe,text='Export LC: ')
export_entry=tk.Entry(fileframe)
export_button=tk.Button(fileframe,text='Save',bg='green',command=lambda: export_lc(lc))

export_plot_label=tk.Label(fileframe,text='Export Plot: ',relief='solid')
export_plot_entry=tk.Entry(fileframe)
export_plot_button=tk.Button(fileframe,text='Save',bg='green',command=lambda: plot_lc(lc,export=True))

loaded_str=tk.StringVar()
loaded_str.set(r'0/0')
loaded_label=tk.Label(fileframe,textvariable=loaded_str)

cycle_label=tk.Label(fileframe,text='Previous/Next: ')
previous_button=tk.Button(fileframe,text='<',bg='green',command=lambda: previous_file())
next_button=tk.Button(fileframe,text='>',bg='green',command=lambda: next_file())

import_label.grid(row=0,column=0,sticky='nsew')
import_entry.grid(row=0,column=1,sticky='nsew')
import_button.grid(row=0,column=2,sticky='nsew')

export_label.grid(row=1,column=0,sticky='nsew')
export_entry.grid(row=1,column=1,sticky='nsew')
export_button.grid(row=1,column=2,sticky='nsew')

export_plot_label.grid(row=2,column=0,sticky='nsew')
export_plot_entry.grid(row=2,column=1,sticky='nsew')
export_plot_button.grid(row=2,column=2,sticky='nsew')

loaded_label.grid(row=0,column=3,rowspan=3,padx=(50,0),sticky='nsew')

cycle_label.grid(row=0,column=4,rowspan=3,padx=(50,0),sticky='nsew')
previous_button.grid(row=0,column=5,rowspan=3,columnspan=2,padx=(20,0),sticky='nsew')
next_button.grid(row=0,column=7,rowspan=3,columnspan=2,padx=(20,0),sticky='nsew')

flux_button=tk.Button(maniframe,text='Flux',bg='green',command=lambda: mag2flux(lc))
mag_button=tk.Button(maniframe,text='Magnitude',bg='green',command=lambda: flux2mag(lc))

normalize_button=tk.Button(maniframe,text='Normalize',bg='green',command=lambda: normalize(lc))

zoom_x_button=tk.Button(maniframe,text='Zoom-X',bg='green',command=lambda: zoom_x())
zoom_x_entry=tk.Entry(maniframe)

zoom_y_button=tk.Button(maniframe,text='Zoom-Y',bg='green',command=lambda: zoom_y())
zoom_y_entry=tk.Entry(maniframe)

phasefold_button=tk.Button(maniframe,text='Phase-Fold',bg='green',command=lambda: phase_fold(lc,float(phasefold_entry.get())))
phasefold_entry=tk.Entry(maniframe)

bin_button=tk.Button(maniframe,text='Bin',bg='green', command=lambda: bin(lc,bin_entry.get()))
bin_entry=tk.Entry(maniframe)
binvar=tk.StringVar()

clip_button=tk.Button(maniframe,text='Clip',bg='green',command=lambda: clip(lc,clip_entry.get()))
clip_entry=tk.Entry(maniframe)

clipvar=tk.StringVar()

model_detrend_button=tk.Button(maniframe,text='Model-Detrend',bg='green',command=lambda: model_detrend(lc,lc.model))

set_checkpoint_button=tk.Button(maniframe,text='Set Checkpoint',bg='green',command=lambda: set_checkpoint(lc))
undo_changes_button=tk.Button(maniframe,text='Undo Changes',bg='green',command=lambda: undo_changes(lc))

flux_button.grid(row=0,column=0,pady=20,sticky='nsew')
mag_button.grid(row=0,column=1,pady=20,sticky='nsew')

normalize_button.grid(row=1,column=0,columnspan=2,pady=20,sticky='nsew')

zoom_x_button.grid(row=2,column=0,pady=20,sticky='nsew')
zoom_x_entry.grid(row=2,column=1,pady=20,sticky='nsew')

zoom_y_button.grid(row=3,column=0,pady=20,sticky='nsew')
zoom_y_entry.grid(row=3,column=1,pady=20,sticky='nsew')

phasefold_button.grid(row=4,column=0,pady=20,sticky='nsew')
phasefold_entry.grid(row=4,column=1,pady=20,sticky='nsew')

bin_button.grid(row=5,column=0,pady=20,sticky='nsew')
bin_entry.grid(row=5,column=1,pady=20,sticky='nsew')

clip_button.grid(row=6,column=0,pady=20,sticky='nsew')
clip_entry.grid(row=6,column=1,pady=20,sticky='nsew')

model_detrend_button.grid(row=7,column=0,columnspan=2,pady=20,sticky='nsew')

set_checkpoint_button.grid(row=8,column=0,pady=20,sticky='nsew')
undo_changes_button.grid(row=8,column=1,pady=20,sticky='nsew')

periodogram_label=tk.Label(calcframe,text='Periodogram: ',relief='solid')
periodogram_entry=tk.Entry(calcframe)
periodogram_button=tk.Button(calcframe,text='Build',bg='green',command=lambda: build_periodogram(lc,periodogram_entry.get()))


export_periodogram_label=tk.Label(calcframe,text='Export Periodogram: ',relief='solid')
export_periodogram_entry=tk.Entry(calcframe)
export_periodogram_button=tk.Button(calcframe,text='Save',bg='green',command=lambda: build_periodogram(lc,periodogram_entry.get(),export=True))

model_label=tk.Label(calcframe,text='Model: ',relief='solid')
model_entry=tk.Entry(calcframe)
model_button=tk.Button(calcframe,text='Fit',bg='green',command=lambda: fit_model(lc,model_entry.get()))

periodogram_label.grid(row=0,column=0,sticky='nsew')
periodogram_entry.grid(row=0,column=1,sticky='nsew')
periodogram_button.grid(row=0,column=2,sticky='nsew')

export_periodogram_label.grid(row=1,column=0,sticky='nsew')
export_periodogram_entry.grid(row=1,column=1,sticky='nsew')
export_periodogram_button.grid(row=1,column=2,sticky='nsew')

model_label.grid(row=2,column=0,sticky='nsew')
model_entry.grid(row=2,column=1,sticky='nsew')
model_button.grid(row=2,column=2,sticky='nsew')

export_periodogram_label.grid(row=3,column=0,sticky='nsew')
export_periodogram_entry.grid(row=3,column=1,sticky='nsew')
export_periodogram_button.grid(row=3,column=2,sticky='nsew')

calcframe.grid_columnconfigure((0,2),weight=1)

view_stats_button = tk.Button(tablframe,text='View Stats',bg='green',command=lambda: view_stats())
export_stats_button = tk.Button(tablframe,text='Export Stats',bg='green',command=lambda: view_stats(export=True))
export_stats_entry = tk.Entry(tablframe)

view_stats_button.grid(row=0,column=0,rowspan=2,columnspan=2,sticky='nsew')
export_stats_button.grid(row=2,column=0,sticky='nsew')
export_stats_entry.grid(row=2,column=1,sticky='nsew')

tablframe.grid_rowconfigure((0,1),weight=1)
tablframe.grid_columnconfigure((0,1),weight=1)

root.title('Light Curve Window')
#root.geometry('1000x750')

root.mainloop()
