import numpy as np
from astropy.timeseries import BoxLeastSquares, LombScargle
from astropy import units as u
from scipy.optimize import curve_fit
import scipy.stats as spstats
from scipy.interpolate import UnivariateSpline, LSQUnivariateSpline
import scipy.signal as spsignal
import os

class Average:

    @staticmethod
    def Mean(y):
        mean = np.nanmean(y)
        return mean

    @staticmethod
    def Median(y):
        median = np.nanmedian(y)
        return median


class Dispersion:

    @staticmethod
    def Rms(y):
        rms = np.nanstd(y)
        return rms

    @staticmethod
    def Skew(y):
        skew = spstats.skew(y)
        return skew

    @staticmethod
    def Kurtosis(y):
        kurt=spstats.kurtosis(y)
        return kurt

    @staticmethod
    def Mad(y):
        mad = np.nanmedian(np.abs(y - np.nanmedian(y)))
        return mad

    @staticmethod
    def CoV(y):
        cov = np.nanstd(y)/np.nanmean(y)
        return cov

    @staticmethod
    def IQR(y):
        Q1=np.nanpercentile(y, 25)
        Q3=np.nanpercentile(y, 75)
        iqr=Q3-Q1
        return iqr

    @staticmethod
    def Noutliers_Dim(y,ytype='flux'):
        if ytype=='mag':
            noutliers_dim=len(np.where(y>np.nanpercentile(y, 75)+1.5*Dispersion.IQR(y))[0])
        elif ytype=='flux':
            noutliers_dim=len(np.where(y<np.nanpercentile(y, 25)-1.5*Dispersion.IQR(y))[0])
        else:
            return np.nan
        return noutliers_dim

    @staticmethod
    def Noutliers_Bri(y,ytype='flux'):
        if ytype=='mag':
            noutliers_bri=int(len(np.where(y<np.nanpercentile(y, 25)-1.5*Dispersion.IQR(y))[0]))
        elif ytype=='flux':
            noutliers_bri=int(len(np.where(y>np.nanpercentile(y, 75)+1.5*Dispersion.IQR(y))[0]))
        else:
            return np.nan
        return noutliers_bri

    @staticmethod
    def P2P_Median_Var(y):
        p2p_median_var=np.nanmedian([np.abs(y[n+1]-y[n]) for n in range(len(y)-1)])
        return p2p_median_var

    @staticmethod
    def P2P_Mean_Var(y):
        p2p_mean_var=np.nanmean([np.abs(y[n+1]-y[n]) for n in range(len(y)-1)])
        return p2p_mean_var

class Depth:

    @staticmethod
    def Amplitude(y,peak=.02):
        y = y[~np.isnan(y)]
        y = np.sort(y)
        amplitude=(np.nanmedian(y[:-int(peak*len(y))])-np.nanmedian(y[:int(peak*len(y))]))/2
        return amplitude

    @staticmethod
    def Brightness_Ratio(y,ytype='flux'):
        if ytype=='mag':
            brightness_ratio=(np.nanmin(y)-np.nanmedian(y))/(np.nanmin(y)-np.nanmax(y))
        elif ytype=='flux':
            brightness_ratio=(np.nanmax(y)-np.nanmedian(y))/(np.nanmax(y)-np.nanmin(y))
        return brightness_ratio

class Slope:

    @staticmethod
    def Slope(t,y,ytype='flux'):
        grad=np.gradient(y,(np.nanmax(t)-np.nanmin(t))/len(t))
        if ytype=='mag':
            slope_min,slope_max=np.nanmax(np.abs(grad)),np.nanmin(np.abs(grad))
        elif ytype=='flux':
            slope_min,slope_max=np.nanmin(np.abs(grad)),np.nanmax(np.abs(grad))
        else:
            return (np.nan,np.nan)
        return (slope_min,slope_max)

    @staticmethod
    def Inflection(t,y,ytype='flux'):
        grad=np.gradient(y,(np.nanmax(t)-np.nanmin(t))/len(t))
        grad2=np.gradient(grad,(np.nanmax(grad)-np.nanmin(grad))/len(grad))
        if ytype=='mag':
            inflection_min,inflection_max=np.nanmax(np.abs(grad2)),np.nanmin(np.abs(grad2))
        elif ytype=='flux':
            inflection_min,inflection_max=np.nanmin(np.abs(grad2)),np.nanmax(np.abs(grad2))
        else:
            return (np.nan,np.nan)
        return (inflection_min,inflection_max)

class Periodicity:
    @staticmethod
    def Lafler_Kinmann(y,ytype='flux'):
        y = y[~np.isnan(y)]
        y = np.sort(y)
        y_mean=np.mean(y)
        lafler_kinmann=(len(y)-1)/(2*len(y))*np.array([(y[i+1]-y[i])**2 for i in range(len(y)-1)]).sum()/np.array([(y[i]-y_mean)**2 for i in range(len(y)-1)]).sum()
        return lafler_kinmann

    @staticmethod
    def LS_Periodogram(t,y,yerr,min_period=.05,max_period=13.5,nsamples='infer'):
        min_freq,max_freq=1/max_period,1/min_period
        if nsamples=='infer':
            freq_width=1/(np.median(t[1:]-t[:-1])*len(t))
            nsamples=int((max_freq-min_freq)/freq_width)

        frequencies=np.linspace(min_freq,max_freq,int(nsamples))
        powers = LombScargle(t, y, yerr).power(frequency=frequencies,method='fast')
        return (frequencies,powers)

    @staticmethod
    def BLS_Periodogram(t,y,yerr,min_period=.05,max_period=13.5,nsamples='infer',duration=0.02,min_n_trans=2):
        min_freq,max_freq=1/max_period,1/min_period
        if nsamples=='infer':
            freq_width=1/(np.median(t[1:]-t[:-1])*len(t))
            nsamples=int((max_freq-min_freq)/freq_width)

        periods=1/np.linspace(min_freq,max_freq,int(nsamples))

        periodogram = BoxLeastSquares(t,y,yerr).power(period=periods,method='fast',duration=duration)
        powers = periodogram.power
        return (periods,powers)

class Fitting:
    @staticmethod
    def PolyFit(x,y,deg='infer',max_deg=10):
        if deg=='infer':
       	    deg=0
            coeff=np.polyfit(x,y,deg)
            polyf=np.poly1d(coeff)
            aic=len(x)*np.log(np.sum((y-polyf(x))**2)/len(x)) + 2*(deg+1)

            cont=True
       	    while (cont==True) & (deg<=max_deg):
                deg_n=deg+1
                coeff_n=np.polyfit(x,y,deg_n)
                polyf_n=np.poly1d(coeff_n)
                aic_n=len(x)*np.log(np.sum((y-polyf_n(x))**2)/len(x)) + 2*(deg_n+1)

       	       	if aic_n < aic:
                    deg=deg_n
                    coeff=coeff_n
                    polyf=polyf_n
                    aic=aic_n

                else:
                    cont=False

            return (polyf(x), coeff)


        else:
             coeff=np.polyfit(x,y,deg)
             polyf=np.poly1d(coeff)

             return (polyf(x), coeff)

    @staticmethod
    def SplineFit(x,y,deg=3,Nknots='infer',min_Nknots=30,max_Nknots=3000):
        if Nknots=='infer':
            knots=np.linspace(np.nanmin(x),np.nanmax(x),min_Nknots+2)
            knots=knots[1:-1]
            splf=LSQUnivariateSpline(x,y,t=knots,k=deg)
            aoc=len(x)*np.log(np.sum((y-splf(x))**2)/len(x)) + 2*(min_Nknots+deg+1)

            for n in range(min_Nknots+1,max_Nknots+1):
                knots_n=np.linspace(np.nanmin(x),np.nanmax(x),n+2)
                knots_n=knots_n[1:-1]
                splf_n=LSQUnivariateSpline(x,y,t=knots_n,k=deg)
                aoc_n=len(x)*np.log(np.sum((y-splf_n(x))**2)/len(x)) + 2*(n+deg+1)

                if aoc_n < aoc:
                    splf=splf_n
                    aoc=aoc_n

                else:
                    break
        else:
            knots=np.linspace(np.nanmin(x),np.nanmax(x),Nknots)
            splf=LSQUnivariateSpline(x,y,t=knots,k=deg)

        return (splf(x),splf.get_coeffs())


    @staticmethod
    def HarmFit(x,y,yerr=None,period='infer',Nharm='infer',Nsharm='infer',max_Nharm=12,max_Nsharm=0,ls_samples=1e4):
        if period=='infer':
            median_sep=np.median(x[1:]-x[:-1])
            if isinstance(yerr,type(None)):
                yerr=np.zeros(len(y))
            frequencies,powers=Periodicity.LS_Periodogram(x,y,yerr=yerr,min_period=2*median_sep,max_period=len(x)*median_sep/2,nsamples=ls_samples)
            best=np.argmax(powers)
            period=1/frequencies[best]

        def HarmF(*p,Nharm,Nsharm):
            x=p[0]
            p=p[1:]
            pi=p[:3]
            pharm=p[3:3*(Nharm+1)]
            psharm=p[3*(Nharm+1):-1]

            harm=1
            harmf=lambda xf,amp_a,amp_b,ps: amp_a*np.sin(2*np.pi/period*harm*(xf-ps)) + amp_b*np.cos(2*np.pi/period*harm*(xf-ps))
            yf=harmf(x,p[0],p[1],p[2])

            for n in range(len(pharm)//3):
                harm=n+2
                harmf=lambda xf,amp_a,amp_b,ps: amp_a*np.sin(2*np.pi/period*harm*(xf-ps)) + amp_b*np.cos(2*np.pi/period*harm*(xf-ps))
                yf+=harmf(x,pharm[0+3*n],pharm[1+3*n],pharm[2+3*n])

            for k in range(len(psharm)//3):
                harm=1/(k+2)
                harmf=lambda xf,amp_a,amp_b,ps: amp_a*np.sin(2*np.pi/period*harm*(xf-ps)) + amp_b*np.cos(2*np.pi/period*harm*(xf-ps))
                yf+=harmf(x,psharm[0+3*n],psharm[1+3*n],psharm[2+3*n],psharm[3+3*n])

            yf+=p[-1]

            return yf


        amp0,phase0=abs((np.nanmax(y)-np.nanmin(y))/2),0

        p0=[amp0,amp0,phase0]
        vshift0=np.nanmedian(y)

        if Nharm=='infer':
            wf=lambda *p: HarmF(*p,Nharm=0,Nsharm=0)
            params,params_covariance=curve_fit(wf,x,y,p0=p0+[vshift0])
            pf=[x]+list(params)
            yf=wf(*pf)
            aic=len(x)*np.log(np.sum((y-yf)**2)/len(x)) + 2*(len(params)+1)
            w_Nharm=0
            for n in range(1,max_Nharm+1):
                wf_n=lambda *p: HarmF(*p,Nharm=n,Nsharm=0)
                params_n,params_covariance_n=curve_fit(wf_n,x,y,p0=p0*(1+n)+[vshift0])
                pf_n=[x]+list(params_n)
                yf_n=wf_n(*pf_n)
                aic_n=len(x)*np.log(np.sum((y-yf_n)**2)/len(x)) + 2*(len(params_n)+1)
                w_Nharm=n

                if aic_n >= aic:
                    break
                else:
       	       	    aic=aic_n
                    w_Nharm=n

        else:
            w_Nharm=Nharm



        if Nsharm=='infer':
            wf=lambda *p: HarmF(*p,Nharm=w_Nharm,Nsharm=0)
            params,params_covariance=curve_fit(wf,x,y,p0=p0+[vshift0])
       	    pf=[x]+list(params)
       	    yf=wf(*pf)
       	    aic=len(x)*np.log(np.sum((y-yf)**2)/len(x)) + 2*(len(params)+1)
            w_Nsharm=0
            for n in range(max_Nsharm+1):
                wf_n=lambda *p: HarmF(*p,Nharm=w_Nharm,Nsharm=n)
                params_n,params_covariance_n=curve_fit(wf_n,x,y,p0=p0*(1+n)+[vshift0])
                pf_n=[x]+list(params_n)
                yf_n=wf_n(*pf_n)
                aic_n=len(x)*np.log(np.sum((y-yf_n)**2)/len(x)) + 2*(len(params_n)+1)
                w_Nsharm=n

                if aic_n >= aic:
                    break
                else:
       	       	    aic=aic_n
                    w_Nsharm=n

       	else:
            w_Nsharm=Nsharm

        wf=lambda *p: HarmF(*p,Nharm=w_Nharm,Nsharm=w_Nsharm)
        params,params_covariance=curve_fit(wf,x,y,p0=p0*(1+w_Nharm+w_Nsharm)+[vshift0])
        pf=[x]+list(params)
        yf=wf(*pf)

        return (yf, params,w_Nharm,w_Nsharm)

    @staticmethod
    def Correlation_Ratio(y1,y2):
        auto_corr_max=np.nanmax(spsignal.correlate(y1,y1))
        cross_corr_max=np.nanmax(spsignal.correlate(y1,y2))

        corr_ratio=cross_corr_max/auto_corr_max

        return corr_ratio
