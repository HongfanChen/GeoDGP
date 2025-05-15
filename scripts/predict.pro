;function max_curve, array, width
;
;  if width lt 2 then return, array
;
;  ; sliding maximum function (like smooth)
;  n = n_elements(array)
;  result = array
;
;  for i = 0, n - 1 do begin
;     i0 = i - width/2 > 0
;     i1 = i + width/2 - 1 < n-1
;     result(i) = max(array(i0:i1))
;  endfor
;
;  return, result
;end
;
; ============================================================================

function max_curve2, time, array, width

  if width le 0.0 then return, array

  ; sliding maximum function (like smooth)
  n = n_elements(array)
  result = array

  for i = 0, n - 2 do begin
     ; search for i1 so that time(i1) is at least t1
     t1 = time(i) + width + 1e-6
     i1 = i
     while time(i1+1<(n-1)) lt t1 and i1 lt n-1 do i1 = i1 + 1
     ; take maximum of points from i to i1
     result(i) = max(array(i:i1))
  endfor

  return, result
end

; ============================================================================

function exceeds, time, array, tmin, tmax, dt, threshold, mincount

  ;; time and array form a time series
  ;; bin data from tmin to tmax with dt bin size
  ;; return an array of 0-s and 1-s. 
  ;; 1 corresponds to bins where any data exceeds the threshold.
  ;; if mincount is present, eliminate bins with fewer data points

  if n_elements(mincount) eq 0 then mincount = 0
  
  n = n_elements(array)

  if n ne n_elements(time) then begin
     print, 'ERROR in function exceeds, number of elements differ in time and array'
     help, time, array
     retall
  endif
  
  nbin  = fix( (tmax - tmin)/dt ) ;;; + 1

  result = intarr(nbin)
  count  = intarr(nbin)

  for i = 0, n - 1 do begin
     t = time[i]
     if t lt tmin or t gt tmax then continue

     ibin = fix( (t - tmin)/dt ) < (nbin-1)

     count(ibin) += 1
     if array[i] ge threshold then result[ibin] = 1
  endfor

  ii = where(count ge mincount)

  ;;; help, time, array, dt, threshold, mincount, nbin, ii

  return, result(ii)

end

; ============================================================================

function fft_segment, time_obs, array_obs, time_sim, array_sim,    $
                      tmin, tmax, dt_window, slope_impedance,      $
                      plotfile, date_plot, title_plot,             $
                      integral_obs, integral_sim, model,           $
                      realization=realization, nsmooth=nsmooth

  ;; time_obs and array_obs form a time series of observations
  ;; time_sim and array_sim form a time series of simulations
  ;; bin data from tmin to tmax with the size of dt_window,
  ;; which is the window of the fft transform in minutes

  help, plotfile
  
  tmin=12
  tmax=66
  
  dt = dt_window/60 ; convert to hours
  
  if not keyword_set(plotfile) then plotfile =''
  if n_elements(realization) eq 0 then realization = -1
  if n_elements(nsmooth) eq 0 then nsmooth = 1
  
  if n_elements(array_obs) ne n_elements(time_obs) or            $
     n_elements(array_sim) ne n_elements(time_sim) then begin
     print, 'ERROR in function fft_segment, number of elements differ in time and array'
     help, time_obs, array_obs, time_sim, array_sim
     retall
  endif

  ;; adjust the array to within tmin and tmax
  index_obs       = where(time_obs le tmax and time_obs ge tmin)
  array_obs_local = array_obs(index_obs)
  time_obs_local  = time_obs(index_obs)
  cadence_obs     = time_obs_local(1:*)-time_obs_local(0:-2)

  index_sim       = where(time_sim le tmax and time_sim ge tmin)
  array_sim_local = array_sim(index_sim)
  time_sim_local  = time_sim(index_sim)
  cadence_sim     = time_sim_local(1:*)-time_sim_local(0:-2)

  if max(cadence_obs)/min(cadence_obs) ge 1.5 or                 $
     max(cadence_sim)/min(cadence_sim) ge 1.5 then begin
     print, 'ERROR in function fft_segment, missing data'
     print, 'min(cadence_obs), max(cadence_obs) = ', $
            min(cadence_obs), max(cadence_obs)
     print, 'min(cadence_sim), max(cadence_sim) = ', $
            min(cadence_sim), max(cadence_sim)
     ;; plot,cadence
     
     ;;return, 0
  endif

  ;; calculate the number of bins
  nbin  = fix( (tmax - tmin)/dt ) ;;; + 1

  integral_obs = fltarr(nbin)
  integral_sim = fltarr(nbin)

  ;; print, 'tmin, tmax =', tmin, tmax

  ;; loop though bin
  for i = 0, nbin - 1 do begin
     tmin_local = tmin + dt*i
     tmax_local = min([tmin + dt*(i+1),tmax])

     ;; obtain the index
     index_obs_tmp = where(time_obs_local ge tmin_local-1e-4 and time_obs_local lt tmax_local+1e-4)
     index_sim_tmp = where(time_sim_local ge tmin_local-1e-4 and time_sim_local lt tmax_local+1e-4)

     ;; the array to calculate the fft power spectrum
     array_obs_tmp = array_obs_local[index_obs_tmp]
     array_sim_tmp = array_sim_local[index_sim_tmp]

     ;; print, time_obs_local(index_obs_tmp(0)), time_obs_local(index_obs_tmp(-1))
     ;; print, time_sim_local(index_sim_tmp(0)), time_sim_local(index_sim_tmp(-1))

     power_obs_tmp = FFT_POWERSPECTRUM(array_obs_tmp, 60, freq=w_obs_tmp)
     power_sim_tmp = FFT_POWERSPECTRUM(smooth(array_sim_tmp, nsmooth, /edge_truncate), 60, freq=w_sim_tmp)

     ;; the heating effect is the power_spectrum / impedance with
     ;; some unknown constant coefficient...
     integral_obs(i) = total(power_obs_tmp*w_obs_tmp^(slope_impedance*2))
     integral_sim(i) = total(power_sim_tmp*w_sim_tmp^(slope_impedance*2))

     if 1 then begin
        ;; print, tmin_local
        set_device, plotfile + '_h'+string(tmin_local,format='(f5.2)')+'.eps', /land
        plot,  alog10(w_obs_tmp),alog10(power_obs_tmp), xtitle='log10(Frequency [Hz])', $
               ytitle='log10(dB power spectrum)', $
               xstyle=1, yrange = [min([alog10(power_obs_tmp), alog10(power_sim_tmp)])-1,      $
                                   max([alog10(power_obs_tmp), alog10(power_sim_tmp)])+1],     $
               ystyle=1, title='A ' + string(dt*60, format='(i3)') + '-min window from ' $
               + string(tmin_local,format='(f4.1)') + ' hours at ' +  title_plot
        oplot, alog10(w_sim_tmp),alog10(power_sim_tmp), color=250

        plot, [0.67, 0.72], [0.92, 0.92], /normal, xrange=[0,1], yrange=[0,1], $
              /noerase, xstyle=-1, ystyle=-1, thick=4
        plot, [0.67, 0.72], [0.84, 0.84], /normal, xrange=[0,1], yrange=[0,1], $
              /noerase, xstyle=-1, ystyle=-1, thick=4, color=250
        xyouts, 0.75, 0.90, 'Observations'
        xyouts, 0.75, 0.82, model, color=250
        close_device, /pdf

        tmp1 = power_obs_tmp*w_obs_tmp^(slope_impedance*2)
        tmp2 = power_sim_tmp*w_sim_tmp^(slope_impedance*2)

        set_device, plotfile + '_cont_h'+string(tmin_local,format='(f5.2)')+'.eps', /land
        plot,  w_obs_tmp(1:*),tmp1(1:*), xtitle='Frequency [Hz]', ytitle='E power spectrum', $
               xstyle=1, yrange = [min([tmp1(1:*), tmp2(1:*)])*0.8,   $
                                   max([tmp1(1:*), tmp2(1:*)])*1.2],  $
               ystyle=1, title='A ' + string(dt*60, format='(i3)') + '-min window from ' $
               + string(tmin_local,format='(f4.1)') + ' hours at ' +  title_plot
        oplot, w_sim_tmp(1:*),tmp2(1:*), color=250

        plot, [0.67, 0.72], [0.92, 0.92], /normal, xrange=[0,1], yrange=[0,1], $
              /noerase, xstyle=-1, ystyle=-1, thick=4
        plot, [0.67, 0.72], [0.84, 0.84], /normal, xrange=[0,1], yrange=[0,1], $
              /noerase, xstyle=-1, ystyle=-1, thick=4, color=250
        xyouts, 0.75, 0.90, 'Observations'
        xyouts, 0.75, 0.82, model, color=250
        close_device, /pdf
     endif
  endfor

  if 1 then begin
     t = dt*findgen(nbin) + tmin
     ;;;set_device, plotfile+'_integral.eps',/land
     ;;;plot, t, integral_obs, xtitle = 'Hours from ' + date_plot, $
     ;;;       ytitle = 'E!u2!n Hourly Integral',                                    $
     ;;;       yrange = [min([integral_obs, integral_sim])*0.8,                      $
     ;;;                 max([integral_obs, integral_sim])*1.2],                     $
     ;;;       title=title_plot
     ;;;oplot, t, integral_sim,color=250
     ;;;
     ;;;plot, [0.67, 0.72], [0.92, 0.92], /normal, xrange=[0,1], yrange=[0,1], $
     ;;;      /noerase, xstyle=-1, ystyle=-1, thick=4
     ;;;plot, [0.67, 0.72], [0.84, 0.84], /normal, xrange=[0,1], yrange=[0,1], $
     ;;;      /noerase, xstyle=-1, ystyle=-1, thick=4, color=250
     ;;;xyouts, 0.75, 0.90, 'Observations'
     ;;;xyouts, 0.75, 0.82, model, color=250
     ;;;close_device, /pdf

     help,realization, realization le 0

     if realization le 0 then $
        openw, lun_tmp, plotfile+'_integral.txt', /get_lun $
     else $
        openw, lun_tmp, plotfile+'_integral.txt', /get_lun, /append


     if realization eq -1 then $
        printf, lun_tmp, 'Time Integral_obs Integral_sim' $
     else if realization eq 0 then $
        printf, lun_tmp, 'Time Integral_obs Integral_sim realization'

     for ibin = 0, nbin-1 do begin
        if realization lt 0 then $
           printf, lun_tmp, dt*ibin + tmin, $
                   integral_obs(ibin), integral_sim(ibin) $
        else $
           printf, lun_tmp, dt*ibin + tmin, $
                   integral_obs(ibin), integral_sim(ibin), realization
     endfor
     close, lun_tmp & free_lun, lun_tmp

     ;;;exc_obs  = $
     ;;;   exceeds(findgen(nbin), integral_obs, 0., float(nbin-1), 2., 1., 0)
     ;;;
     ;;;exc_sim = $
     ;;;   exceeds(findgen(nbin), integral_sim, 0., float(nbin-1), 2., 1., 0)
     ;;;
     ;;;hit_fft   = total(exc_obs*exc_sim)
     ;;;false_fft = total((1-exc_obs)*(exc_sim))
     ;;;miss_fft  = total(exc_obs*(1-exc_sim))
     ;;;no_fft    = total((1-exc_obs)*(1-exc_sim))

     ;; get_skill_scores,hit_fft,false_fft,miss_fft,no_fft,fft_pod_local,fft_pof_local,fft_hss_local
     ;; print, plotfile + ' h,f,m,n,pod,pof,hss =', hit_fft,false_fft,miss_fft,no_fft,fft_pod_local,fft_pof_local,fft_hss_local

  endif

  return, 1
end

; ============================================================================
function scale_exp, x, a

 ; Used by lmfit. Applies the non-linear function f to x with parameters a
 ; returns f(x) and the partial derivatives of f with respect to a

  scale = a(0)
  exponent = a(1)
  f = (x/scale)^exponent

  return, [f, -exponent/scale * f, alog(x/scale) * f]

end

; ============================================================================

pro get_skill_scores,h,f,m,n,pod,pof,hss

  pod = h/(h+m)
  pof = f/(f+n)
  hss = 2*(h*n-m*f)/((h+m)*(m+n) + (h+f)*(f+n))

  if (~finite(pod)) then pod = -777.
  if (~finite(pof)) then pof = -777.
  if (~finite(hss)) then hss = -777.
end

; ============================================================================
pro read_station, file, t, data, date, tderiv, dataderiv, dn, $
                  realization=realization

  common start_date

  if not keyword_set(IsNbor) then IsNbor = 0
  
  ;; reset common block data
  start_year  = -1
  start_month = -1
  start_day   = -1

  ; read logfile, use hours for time unit
  get_log, file, wlog, wlogname, t, 'h'

  if n_elements(realization) gt 0 then begin
     if realization gt -1 then begin
        ;; keep only selected realization
        ii = where(wlog(*,-1) eq realization)
        help,ii
        wlog = wlog(ii,0:-2)
        t = t(ii)
     endif
  endif

  date = julday(wlog(0,1), wlog(0,2), wlog(0,0))

  ;; Calculate horizontal component (db or dbdt)
  i = where(wlogname eq 'B_NorthGeomag') & i = i(0)
  j = where(wlogname eq 'B_EastGeomag')  & j = j(0)

  if i lt 0 and j lt 0 then begin
     i = where(wlogname eq 'dBdt_NorthGeomag') & i = i(0)
     j = where(wlogname eq 'dBdt_EastGeomag')  & j = j(0)
  end

  if i lt 0 or j lt 0 then begin
     print, 'ERROR for file =', file
     print, 'Cannot find variables (B|dBdt)_(North|East) in wlogname='
     print, wlogname
     retall
  endif

  data = sqrt(wlog(*,i)^2 + wlog(*,j)^2)

  ;; Calculate time derivative
  n  = n_elements(data)

  tderiv = 0.5*(t(dn:n-1)+t(0:n-1-dn))

  dataderiv = sqrt( (wlog(dn:n-1,i)-wlog(0:n-1-dn,i))^2 + $
                    (wlog(dn:n-1,j)-wlog(0:n-1-dn,j))^2 ) $
              / (3600*(t(dn:n-1) - t(0:n-1-dn)))

end

; ============================================================================

function set_eventlist, events, mydir, model

  ; the format of events could be something like events='1,2-5,7'

  ;; replace // with /
  model_local = model.replace('//','/')

  ;; remove anything before deltaB
  if(strpos(model_local, 'deltaB/') ge 0) then model_local = strmid(model_local, strpos(model_local, 'deltaB/')+7)

  if strlen(events) lt 1 then begin
     ;; empty events string, the events in the  model_local dir
     events_dir = FILE_SEARCH(mydir+'/deltaB/'+model_local+'/Event*')

     ;; the string before the event number, and replace // with /
     tmp0 = ('deltaB/'+model_local+'/Event').replace('//','/')

     for i=0,n_elements(events_dir)-1 do begin
        ;; remove anything before the Event number
        events_dir(i) = strmid(events_dir(i),strpos(events_dir(i),tmp0)+tmp0.strlen())
     endfor
     event_I    = fix(events_dir)
  endif else begin
     ;; split the events string, as the  events now only contains numbers.
     tmp1 = events.split(',')
     event_I= fix(tmp1)
  endelse        

  return, event_I
end

; ============================================================================

pro set_stationlist,mydir=mydir,stationsFile=stationsFile,model=model, $
                    event=event,IsFromSim=IsFromSimIn,                 $
                    stations_I, station_orig_I

  ;; read the stationsFile containing the selected stations.
  ;; check how many simulated magnetometer files there if IsFromSim.
  ;; folder could be one or two digits

  ;; default is not to not to obtain the list of stations from the simulation.
  if not keyword_set(IsFromSimIn) then IsFromSim = 0
  if not keyword_set(event)       then event = 1

  model_local = model
  
  ;; obtain the mag-latitude info from supermag.dat
  get_log, mydir+'/SWMF/Param/supermag.dat',wlog,  wlognames, logtime, 'h', wlogrownames

  ;; start reading the station file...
  filename = mydir+'/'+stationsFile
  openr, lun, filename, /get_lun

  line=''
  i = 0
  ;; up to 1000 lines...
  station_orig_I = strarr(1000)

  while not eof(lun) do begin
     readf,lun,line
     if line eq 'IAGA' then continue
     if line ne '' then station_orig_I(i) = line

     ;; if any line contain this string, obtain the list of stations
     ;; from simulation
     if line eq 'all_stations'     or line eq 'mid_stations'      $
        or line eq 'low_stations'  or line eq 'veryhigh_stations' $
        or line eq 'high_stations' then IsFromSim = 1
     i++
  endwhile
  close,lun & free_lun,lun

  ;; adjust the size of station_orig_I
  station_orig_I = station_orig_I[0:i-1]

  if IsFromSim then begin
     ;; first remove any characters before deltaB (including deltaB)
     if(strpos(model_local,'deltaB/') ge 0) then $
        model_local = strmid(model_local,strpos(model_local,'deltaB/')+7)

     ;; check whether the foler of the Event is 1-digit or 2-digits
     if file_test(mydir+'/deltaB/'+model_local+'/Event'+string(event,format='(i2.2)')) then begin
        event_string_sim = 'Event'+string(event,format='(i2.2)')
     endif else begin
        event_string_sim = 'Event'+string(event,format='(i1.1)')
     endelse

     ;; set the patter for searching simulated magnetometer files
     patten_sim  = mydir+'/deltaB/'+model_local+'/'+event_string_sim+'/*.txt'

     file_sim_I = FILE_SEARCH(patten_sim, count=nFileSim)

     if nFileSim eq 0 then begin
        print, 'Error no simulation files found for patten_sim: ' $
               + patten_sim
        return
     endif
  
     str_stations_sim_all = ''
  
     for ifile = 1, nFileSim do begin
        filelocal = file_sim_I(ifile-1)
        strlocal  = strmid(filelocal,strpos(filelocal,event_string_sim)+strlen(event_string_sim)+1)
        strlocal  = strlocal.replace('.txt','')
        ;; DST is not in the real observations...
        if strlocal ne 'DST' then $
           str_stations_sim_all = str_stations_sim_all + strlocal + ' '
     endfor

     ;; upper case for all station names
     str_stations_sim_all = strupcase(strtrim(str_stations_sim_all))
  endif

  ;; the return list of stations.
  stations_I     = strarr(i)

  for i=0,n_elements(station_orig_I)-1 do begin
     if station_orig_I(i) eq 'all_ccmc' then begin
        stations_I[i] = 'frn frd fur wng new ott mea pbq abk ykc hrn iqa'
     endif else if station_orig_I(i) eq 'veryhigh_ccmc' then begin
        stations_I[i] = 'hrn iqa'
     endif else if station_orig_I(i) eq 'high_ccmc' then begin
        stations_I[i] = 'abk pbq ykc mea'
     endif else if station_orig_I(i) eq 'mid_ccmc' then begin
        stations_I[i] = 'wng new ott'
     endif else if station_orig_I(i) eq 'low_ccmc' then begin
        stations_I[i] = 'frd frn fur'
     endif else if station_orig_I(i) eq 'all_stations' then begin
        stations_I[i] = str_stations_sim_all
     endif else if station_orig_I(i) eq 'low_stations' then begin
        str_stations_I = strsplit(str_stations_sim_all,/extract)
        str_stations_tmp = ''
        for istation = 0, n_elements(str_stations_I)-1 do begin
           station_tmp = str_stations_I(istation)
           iloc = where(wlogrownames eq station_tmp)
           if iloc lt 0 then continue
           if (wlog(iloc, 2) le 50) then $
              str_stations_tmp = str_stations_tmp + station_tmp + ' '
        endfor
        stations_I[i] = strtrim(str_stations_tmp)
     endif else if station_orig_I(i) eq 'mid_stations' then begin
        str_stations_I = strsplit(str_stations_sim_all,/extract)
        str_stations_tmp = ''
        for istation = 0, n_elements(str_stations_I)-1 do begin
           station_tmp = str_stations_I(istation)
           iloc = where(wlogrownames eq station_tmp)
           if (wlog(iloc, 2) gt 50 and wlog(iloc, 2) le 60) then $
              str_stations_tmp = str_stations_tmp + station_tmp + ' '
        endfor
        stations_I[i] = strtrim(str_stations_tmp)
     endif else if station_orig_I(i) eq 'high_stations' then begin
        str_stations_I = strsplit(str_stations_sim_all,/extract)
        str_stations_tmp = ''
        for istation = 0, n_elements(str_stations_I)-1 do begin
           station_tmp = str_stations_I(istation)
           iloc = where(wlogrownames eq station_tmp)
           if (wlog(iloc, 2) gt 60 and wlog(iloc, 2) le 70) then $
              str_stations_tmp = str_stations_tmp + station_tmp + ' '
        endfor
        stations_I[i] = strtrim(str_stations_tmp)
     endif else if station_orig_I(i) eq 'veryhigh_stations' then begin
        str_stations_I = strsplit(str_stations_sim_all,/extract)
        str_stations_tmp = ''
        for istation = 0, n_elements(str_stations_I)-1 do begin
           station_tmp = str_stations_I(istation)
           iloc = where(wlogrownames eq station_tmp)
           if station_tmp eq 'IQA' then $
              print, 'IQA lat =', wlog(iloc, 2)
           if (wlog(iloc, 2) gt 70) then $
              str_stations_tmp = str_stations_tmp + station_tmp + ' '
        endfor
        stations_I[i] = strtrim(str_stations_tmp)
     endif else begin
        stations_I[i] = station_orig_I(i)
     endelse
  end

  stations_I = strupcase(stations_I)
end

; ============================================================================

function set_date_string, yyyy, mm, dd

  case fix(mm) of
     1:  mm_s = 'Jan.'
     2:  mm_s = 'Feb.'
     3:  mm_s = 'Mar.'
     4:  mm_s = 'Apr.'
     5:  mm_s = 'May'
     6:  mm_s = 'Jun.'
     7:  mm_s = 'Jul.'
     8:  mm_s = 'Aug.'
     9:  mm_s = 'Sep.'
     10: mm_s = 'Oct.'
     11: mm_s = 'Nov.'
     12: mm_s = 'Dec.'
  endcase

  date_string = mm_s + ' ' + string(dd,format='(i2.2)') + ' ' $
                + string(yyyy,format='(i4.4)')

  return, date_string
end

; ============================================================================

pro predict, choice,                                                  $
             model=model, namemodel=namemodel,                        $
             station_I=station_I, events=events,                      $
             deltat=deltat, mincount=mincount, stencil=stencil,       $
             threshold=threshold,                                     $
             scale=scale, exponent=exponent,                          $
             coeff=coeff, power=power,                                $
             db_pod=db_pod, db_pof=db_pof, db_hss=db_hss,             $
             dbdt_pod=dbdt_pod, dbdt_pof=dbdt_pof, dbdt_hss=dbdt_hss, $
             out_file=out_file, append=append, lunOut=lunOut,         $
             showinfo=showinfo, verbose=verbose,                      $
             showplot=showplot, saveplot=saveplot, mydir=mydir,       $
             stations_used=stations_used, DoPrintHeader=DoPrintHeader,$
             InputDir=InputDir,lunStation=lunStation,                 $
             DoPrintStationHeader=DoPrintStationHeader,               $
             integral_obs_III=integral_obs_III,                       $
             integral_sim_III=integral_sim_III,                       $
             realization=realization, nsmooth=nsmooth

  ;; choice = 'db', 'dbdt', 'corr'
  ;; model: the directorys containing results (not an array anymore)
  ;; namemodel: the name for printing results (default=model)
  ;; station_I: array of station names (default is all 12 stations)
  ;; deltat: width of bins in minutes (default=20)
  ;; mincount: minimum number of valid observations in each bin
  ;; stencil = 1 or 2: discretization for dB/dt (default=1, compact stencil)
  ;; scale, exponent: indirect dBdt = (dB/scale)^exponent (default=292,1.14)
  ;; coefff, power: artificial scaling of simulated db or dbdt reults:
  ;;                result_sim' = coeff*result_sim^power (default=1,1)
  ;; db_pod, db_pof, db_hss: array of probablity of true and false
  ;;                         detections and Heidke Skill Scores for dB
  ;; dbdt_pod, dbdt_pof, dbdt_hss: array of probablity of true and false
  ;;                         detections and Heidke Skill Scores for dB/dt
  ;; out_file: name of file to save results into (default is STDOUT)
  ;; append:   append to the out_file (default is overwrite)
  ;; verbose:  write verbose information to the screen (default is false)
  ;; showinfo: write information onto the plots (default is true)
  ;; showplot: plot results (default=saveplot)
  ;; saveplot: save plots into files (default=false)

  if choice eq 'fft' then begin
     if n_elements(exponent) lt 1 then exponent = 0.5
     if n_elements(deltat) lt 1 then deltat = 60.0 ; minutes
  endif else begin
     if n_elements(deltat) lt 1 then deltat = 20.0 ; seconds
     dt = deltat/60.0
  endelse

  if n_elements(mincount) lt 1 then mincount=0

  if not keyword_set(out_file) then out_file=''
  if not keyword_set(append)   then append=0
  if not keyword_set(verbose)  then verbose=0
  if not keyword_set(showplot) then showplot=0
  if not keyword_set(saveplot) then saveplot=0
  if not keyword_set(lunOut)   then lunOut=-1          ; -1 is STDOUT
  if not keyword_set(lunStation)    then lunStation=-1 ; -1 is STDOUT
  if not keyword_set(DoPrintHeader) then DoPrintHeader = 0
  if not keyword_set(DoPrintStationHeader) then DoPrintStationHeader = 0
  if not keyword_set(InputDir) then InputDir='Inputs'
  if not keyword_set(lun_fft)  then lun_fft=-1
  if saveplot then showplot=1

  ;; Width of stencil for time derivative
  if not keyword_set(stencil) then stencil = 1

  ;; threshold for dBdt [nT/s] or dB [nT]
  if not keyword_set(threshold) then begin
     if choice eq 'db' then threshold = 100.0 else threshold = 1.1
  endif

  ;; scaling and exponent for indirect db calculation:
  ;; dB/dt [nT/s] = (dB[nT]/scale)^exponent
  if n_elements(scale)    lt 1 then scale = 292.0
  if n_elements(exponent) lt 1 then exponent = 1.14

  ;; coefficient and power for artificial scaling of results
  ;; result' = coeff*result^power
  if n_elements(coeff) lt 1 then coeff = 1.0
  if n_elements(power) lt 1 then power = 1.0

  if n_elements(showinfo) lt 1 then showinfo=1

  if n_elements(namemodel) eq 0 then namemodel=model
  
  if model eq 'Observations' then begin
     namemodel = 'Observed'
  endif else if model eq 'SWMF_CCMC' then begin
     namemodel = 'SWMFccmc'
  endif else if model eq 'Results' then begin
     namemodel = 'SWMF_new'
  endif

  if not keyword_set(mydir)    then mydir='.'
  if strmid(mydir, mydir.strlen()-1)  ne '/' then mydir  = mydir+'/'

  ;; well, use the first model....
  event_I = set_eventlist(events,mydir,model)

  !p.charsize = 1.6

  nstation = n_elements(station_I)

  case nstation of
     1: !p.multi = [0,1,1]
     2: !p.multi = [0,2,1]
     3: !p.multi = [0,3,1]
     4: !p.multi = [0,2,2]
     5: !p.multi = [0,3,2]
     6: !p.multi = [0,3,2]
     7: !p.multi = [0,4,2]
     8: !p.multi = [0,4,2]
     9: !p.multi = [0,3,3]
     else: !p.multi = [0,4,3]
  endcase

  if verbose then print,'nstation=', nstation,' p.multi=', !p.multi

  ;; remove anything before deltaB
  if(strpos(model,    'deltaB/') ge 0) then model     = strmid(model,    strpos(model,    'deltaB/')+7)
  if(strpos(namemodel,'deltaB/') ge 0) then namemodel = strmid(namemodel,strpos(namemodel,'deltaB/')+7)

  ;; replace / with _
  namemodel = strjoin(strsplit(namemodel,'/',/extract),'_')

  if verbose then print,' model, namemodel=', model, ', ',namemodel

  ;; first index: hit, false, miss, nopred
  dbdt_score = fltarr(4,nstation)
  db_score   = fltarr(4,nstation)

  stations_used = strarr(n_elements(event_I))

  for iEvent = 0,n_elements(event_I)-1 do begin

     event = event_I(iEvent)

     ;; check whether the format in Observations is two digits or not.
     if file_test(mydir + InputDir +'/Event'+string(event,format='(i2.2)')) then begin
        event_string_obs = 'Event'+string(event,format='(i2.2)')
     endif else begin
        event_string_obs = 'Event'+string(event,format='(i1.1)')
     endelse

     ;; the path to the magnetometer data
     InputDir_deltaB = mydir + InputDir + '/' + event_string_obs + '/deltaB/'

     ;; check whether the format in SIMDIR/RESDIR is two digits or not.
     if file_test(mydir+'deltaB/'+model+'/Event'+string(event,format='(i2.2)')) then begin
        event_string_sim = 'Event'+string(event,format='(i2.2)')
     endif else begin
        event_string_sim = 'Event'+string(event,format='(i1.1)')
     endelse

     if verbose then print,' ievent, event=', event, '  Event'+string(event,format='(i2.2)')

     ;; setting the filename for the figure.
     if saveplot then begin
        plotfile  = choice + '_event' + string(event, format='(i2.2)')

        case choice of
           'db': plotfile += '_'+string(fix(threshold),format='(i0,"nT")')
           'dbdt': plotfile += '_'+string(threshold,format='(f3.1,"nTs")')
        endcase
        plotfile += '.eps'

        set_device, plotfile, /land
     endif

     for istation = 0, nstation-1 do begin
        station = strupcase(station_I[istation])

        if verbose then print,' istation=', istation, ' ', station

        if file_test(InputDir_deltaB + '/'+station+'.txt') then begin
           file_obs_db = InputDir_deltaB + '/'+station+'.txt'
        endif else if file_test(InputDir_deltaB + '/'+station+'.csv') then begin
           file_obs_db = InputDir_deltaB + '/'+station+'.csv'
        endif else begin
           if station ne 'PBQ' then begin
              print,'For event=',event,' there is no observation for station=',station
              continue
           endif else begin
              ;; well, for PBQ, it might also be SNK...
              if file_test(InputDir_deltaB + '/SNK.txt') then begin
                 file_obs_db = InputDir_deltaB + '/SNK.txt'
              endif else if file_test(InputDir_deltaB + '/SNK.csv') then begin
                 file_obs_db = InputDir_deltaB + '/SNK.csv'
              endif else begin
                 print,'For event=',event,' there is no observation for station= PBQ/SNK'
                 continue
              endelse
           endelse
        endelse

        file_sim_db  = mydir + 'deltaB/' + model + '/' + event_string_sim + '/' + station
        help,file_sim_db
        print,file_sim_db + '.txt', file_test(file_sim_db + '.txt')
        if file_test(file_sim_db + '.txt') then begin
           file_sim_db = file_sim_db + '.txt'
           help,file_sim_db
        endif else if file_test(file_sim_db + '.csv') then begin
           file_sim_db = file_sim_db + '.csv'
           help,file_sim_db
        endif else begin
           if station ne 'PBQ' then begin
              print,'For event=',event,' there is no result from model=',model,' for station=',station
              continue
           endif else begin
              file_sim_db  = mydir + 'deltaB/' + model + '/' + event_string_sim + '/SNK.txt'
              if not file_test(file_sim_db) then begin
                 print,'For event=',event,' there is no result from model=',model,' for station=PBQ/SNK.'
                 continue
              endif
           endelse
        endelse

        print,'finally:'

        if verbose then print,'reading observation and simulation data'

        ;; Read db and calculate db/dt from observations and simulations
        read_station, file_obs_db, t_db_obs, $
                      db_obs, date_obs, t_dbdt_obs, dbdt_obs, stencil
        read_station, file_sim_db, t_db_sim, $
                      db_sim, date_sim, t_dbdt_sim, dbdt_sim, stencil, $
                      realization=realization

        if date_obs ne date_sim then begin
           print, "adjust t_db_sim and t_dbdt_sim to align with observations for Event " $
                  + string(event,format='(i2.2)')
           t_db_sim   = t_db_sim   - (date_obs-date_sim)*24
           t_dbdt_sim = t_dbdt_sim - (date_obs-date_sim)*24
        endif

        ;; convert the Julday back to year, month, day
        JUL2GREG, date_obs, mm_date, dd_date, yyyy_date

        ;; set the date info
        date = string(yyyy_date,format='(i4.4)')+string(mm_date,format='(i2.2)') $
               +       string(dd_date,format='(i2.2)')

        ;; Station PBQ was replaced with SNK after 2007
        if yyyy_date gt 2007 and station eq 'PBQ' then station = 'SNK'

        stations_used[iEvent] = stations_used[iEvent] +  station + ' '

        ;; Calculate contingency tables and create plots
        case choice of
           'db':begin
              ;; artificial scaling
              db_sim = coeff*db_sim^power

              ;tmin = float(floor(t_db_obs(0)))
              ;tmax = float(ceil(t_db_obs(-1)))

              tmin = t_db_obs(0)
              tmax = t_db_obs(-1)

              if deltat le 1 then begin
                 exc_obs = db_obs ge threshold
                 exc_sim = db_sim ge threshold
              endif else begin
                 exc_obs  = $
                    exceeds(t_db_obs, db_obs, tmin, tmax, dt, threshold, mincount)

                 exc_sim = $
                    exceeds(t_db_sim, db_sim, tmin, tmax, dt, threshold, mincount)
              endelse
              
              all      = n_elements(exc_obs)
              if all ne n_elements(exc_sim) then begin
                 print,'ERROR: for station ',station
                 help,exc_obs, exc_sim
              endif

              obsyes   = total(exc_obs)
              obsno    = total(1-exc_obs)

              hit_db   = total(exc_obs*exc_sim)
              false_db = total((1-exc_obs)*(exc_sim))
              miss_db  = total(exc_obs*(1-exc_sim))
              no_db    = total((1-exc_obs)*(1-exc_sim))

              db_score[  *,istation] = db_score[*,istation] + $
                                       [hit_db, false_db, miss_db, no_db]

              get_skill_scores,hit_db,false_db,miss_db,no_db,db_pod_local,db_pof_local,db_hss_local

              if lunStation ne -1 and DoPrintStationHeader ne 0 then begin
                 printf, lunStation, 'db skill scores for model: ' + model
                 printf, lunStation, 'Event   =', event_I[iEvent]
                 printf, lunStation, 'Deltat  = ' + string(deltat,format='(f6.2)') + ', threshold unit = [nT]'
                 printf, lunStation, $
                         'name threshold  TP  TN  FP  FN  total  pod  far  hss'
                 printf, lunStation, '#START'
                 DoPrintStationHeader = 0
              endif

              if lunStation ne -1 then $
                 printf, lunStation, station, threshold, hit_db,no_db,false_db,miss_db,         $
                         hit_db+no_db+false_db+miss_db, db_pod_local,db_pof_local,db_hss_local, $
                         format='(a3,1x,f8.4, 5i6, 3f12.4)'

              if showplot then begin
                 title = strupcase(station) + ': H,F,M,N,HSS=' + $
                         string(hit_db, false_db, miss_db, no_db, db_hss_local, $
                                format='(4i7,f7.3)')

                 if verbose then print,'plotting into p.multi=',!p.multi

                 plot, t_db_obs, max_curve2(t_db_obs, db_obs, dt), $
                       xtitle = 'time [h]', ytitle='dB!DH!N', $
                       title = title, xrange = [tmin, tmax], xstyle=1, $
                       yrange = [0.,max([db_obs, db_sim])], thick = 3

                 oplot, [tmin, tmax], [threshold, threshold], linestyle=2
                 
                 oplot, t_db_sim, max_curve2(t_db_sim, db_sim, dt), $
                        color = 150, thick = 3
              endif
           end
           'dbdt':begin
              ;; artificial scaling
              dbdt_sim = coeff*dbdt_sim^power

              tmin = float(floor(t_db_obs(0)))
              tmax = float(ceil(t_db_obs(-1)))

              proxy_sim = (db_sim/scale)^exponent

              exc_obs = $
                 exceeds(t_dbdt_obs, dbdt_obs, tmin, tmax, dt, threshold, mincount)

              exc_sim = $
                 exceeds(t_dbdt_sim, dbdt_sim, tmin, tmax, dt, threshold, mincount)

              exc_db_sim = $
                 exceeds(t_db_sim, proxy_sim, tmin, tmax, dt, threshold, mincount)

              all      = n_elements(exc_obs)
              if all ne n_elements(exc_sim) then begin
                 print,'ERROR: for station ',station
                 help,exc_obs, exc_sim
              endif

              obsyes     = total(exc_obs)
              obsno      = total(1-exc_obs)

              hit_dbdt   = total(exc_obs*exc_sim)
              false_dbdt = total((1-exc_obs)*(exc_sim))
              miss_dbdt  = total(exc_obs*(1-exc_sim))
              no_dbdt    = total((1-exc_obs)*(1-exc_sim))

              hit_db     = total(exc_obs*exc_db_sim)
              false_db   = total((1-exc_obs)*(exc_db_sim))
              miss_db    = total(exc_obs*(1-exc_db_sim))
              no_db      = total((1-exc_obs)*(1-exc_db_sim))

              dbdt_score[*,istation] = dbdt_score[*,istation] + $
                                       [hit_dbdt, false_dbdt, $
                                        miss_dbdt, no_dbdt]
              db_score[  *,istation] = db_score[*,istation] + $
                                       [hit_db, false_db, $
                                        miss_db, no_db]
              
              title = strupcase(station) + ': H,F,M,N=' + $
                      string(hit_dbdt,false_dbdt,miss_dbdt,no_dbdt, format='(4i4)')

              if exponent gt 0.0 then title += $
                 '!C proxy: H,F,M,N='+ $
                 string(hit_db,false_db,miss_db,no_db,format='(4i4)')

              get_skill_scores,hit_dbdt,false_dbdt,miss_dbdt,no_dbdt,dbdt_pod_local,dbdt_pof_local,dbdt_hss_local
              get_skill_scores,hit_db,false_db,miss_db,no_db,db_pod_local,db_pof_local,db_hss_local

              if lunStation ne -1 and DoPrintStationHeader ne 0 then begin
                 printf, lunStation, 'db/dt skill scores for model: ' + model
                 printf, lunStation, 'Event   =', event_I[iEvent]
                 printf, lunStation, 'Deltat  = ' + string(deltat,format='(f6.2)') + ', threshold unit = [nT/s]'
                 printf, lunStation, $
                         'name threshold  TP  TN  FP  FN  total  TP_ind  TN_ind  FP_ind  FN_ind  total_ind  pod  far  hss  pod_ind  far_ind  hss_ind'
                 printf, lunStation, '#START'
                 DoPrintStationHeader = 0
              endif

              if lunStation ne -1 then $
                 printf, lunStation, station, threshold, hit_dbdt,no_dbdt,false_dbdt,miss_dbdt,$
                         hit_dbdt+no_dbdt+false_dbdt+miss_dbdt, $
                         hit_db,no_db,false_db,miss_db,hit_db+no_db+false_db+miss_db,      $
                         dbdt_pod_local,dbdt_pof_local,dbdt_hss_local, db_pod_local,db_pof_local,db_hss_local,       $
                         format='(a3,1x,f8.4, 10i6, 6f12.4)'

              if showplot then begin
                 if verbose then print,'plotting into p.multi=',!p.multi

                 plot, t_dbdt_obs, max_curve2(t_dbdt_obs, dbdt_obs, dt), $
                       xtitle = 'time [h]', ytitle='dB/dt!DH!N', $
                       title = title, xrange = [tmin, tmax], xstyle=1, $
                       thick=3

                 oplot, t_dbdt_sim, max_curve2(t_dbdt_sim, dbdt_sim, dt), $
                        color = 150, thick=3

                 oplot, [tmin, tmax], [threshold, threshold], linestyle=2

                 if exponent gt 0 then $
                    oplot, t_db_sim, max_curve2(t_db_sim, proxy_sim, dt), $
                           color = 250, thick = 3
              endif
           end
           'corr':begin

              ;; interpolate deltaB to the discrete times of dB/dt
              db_obs1 = interpol(db_obs, t_db_obs, t_dbdt_obs)

              a = max_curve2(t_dbdt_obs, dbdt_obs, dt)
              b = max_curve2(t_dbdt_obs, db_obs1,  dt)

              if n_elements(db_all) lt 1 then begin
                 dbdt_all = a
                 db_all   = b
              endif else begin
                 dbdt_all = [dbdt_all, a]
                 db_all   = [db_all,   b]
              endelse

              b = (b/scale)^exponent

              r = correlate(a,b)

              if showplot then begin
                 plot, t_dbdt_obs, a, $
                       xtitle = 'time [h]', ytitle='dB/dt!DH!N', $
                       title = station+', corr='+string(r,format='(f5.2)')

                 oplot, t_dbdt_obs, b, color = 250
              endif
           end
           'fft':begin

              !p.multi   = 0
              !p.thick   = 3
              !x.thick   = 3
              !y.thick   = 3
              !p.charsize = 2

              date_plot = set_date_string(yyyy_date(0),mm_date(0),dd_date(0))
              title_plot= station

              plotfile = choice + '_event' + string(event, format='(i2.2)') + $
                         '_' + model + '_' + station

              if showplot then begin
                 set_device, plotfile + '.eps', /land
                 plot, t_db_obs, db_obs, yrange=[0,max([db_obs,db_sim])*1.2], $
                       xtitle='Hours from ' + date_plot, ytitle='dB [nT]',    $
                       title=title_plot
                 oplot, t_db_sim, db_sim, color=250
                 plot, [0.67, 0.72], [0.92, 0.92], /normal, $
                       xrange=[0,1], yrange=[0,1], $
                       /noerase, xstyle=-1, ystyle=-1, thick=4
                 plot, [0.67, 0.72], [0.84, 0.84], /normal, $
                       xrange=[0,1], yrange=[0,1], $
                       /noerase, xstyle=-1, ystyle=-1, thick=4, color=250
                 xyouts, 0.75, 0.90, 'Observations'
                 xyouts, 0.75, 0.82, model, color=250
                 close_device, /pdf
                 
                 set_device, plotfile + '_dbdt.eps', /land
                 plot, t_dbdt_obs, dbdt_obs, $
                       yrange=[0,max([dbdt_obs,dbdt_sim])*1.2], $
                       xtitle='Hours from '+date_plot, ytitle='dBdt [nT/s]', $
                       title=title_plot
                 oplot, t_dbdt_sim, dbdt_sim, color=250
                 plot, [0.67, 0.72], [0.92, 0.92], /normal, $
                       xrange=[0,1], yrange=[0,1], $
                       /noerase, xstyle=-1, ystyle=-1, thick=4
                 plot, [0.67, 0.72], [0.84, 0.84], /normal, $
                       xrange=[0,1], yrange=[0,1], $
                       /noerase, xstyle=-1, ystyle=-1, thick=4, color=250
                 xyouts, 0.75, 0.90, 'Observations'
                 xyouts, 0.75, 0.82, model, color=250
                 close_device, /pdf
              endif
              
              tmin = float(floor(t_db_obs(0)))
              tmax = float(ceil(t_db_obs(-1)))

              help,deltat, exponent
              status = fft_segment(t_db_obs, db_obs, t_db_sim, db_sim, $
                                   tmin, tmax, deltat, exponent, plotfile, $
                                   date_plot, title_plot, $
                                   integral_obs, integral_sim, model, $
                                   realization=realization, nsmooth=nsmooth)

              if status then begin
                 if n_elements(integral_obs_III) gt 0 then begin
                    integral_obs_III[iEvent, istation, 0] = mean(integral_obs)
                    integral_obs_III[iEvent, istation, 1] = max(integral_obs)
                 endif
                 if n_elements(integral_sim_III) gt 0 then begin
                    integral_sim_III[iEvent, istation, 0] = mean(integral_sim)
                    integral_sim_III[iEvent, istation, 1] = max(integral_sim)
                 endif
              endif
           end
        endcase
     endfor                     ; istation

     if showinfo then begin
        info = 'Model=' + namemodel + $
               ', event' + string(event, format='(i2.2)') + '=' + date + $
               ', deltat=' + string(fix(deltat),format='(i0,"min")')
        if choice eq 'db' then info += $
           + ', threshold=' + string(fix(threshold),format='(i0,"nT")')
        if choice eq 'dbdt' then begin
           info += $
              + ', threshold=' + string(threshold,format='(f3.1,"nT/s")')

           if exponent gt 0 then info += $
              + ', proxy(red)=' + string(scale,exponent,format='(f5.1,"*dB!U",f5.2,"!N")')
           if coeff ne 1.0 then info += $
              + ', coeff=' + string(coeff) +', power=' + string(power)
        endif
        xyouts,0.0,1.03,/norm,charsize=1, info
     endif

     if saveplot then close_device, /pdf

  endfor                        ; iEvent for event_I

  !p.multi(0) = 0

  ;; Calculate various scores
  if choice eq 'dbdt' then begin

     h = total(dbdt_score(0,*))
     f = total(dbdt_score(1,*))
     m = total(dbdt_score(2,*))
     n = total(dbdt_score(3,*))

     h_ind = total(db_score(0,*))
     f_ind = total(db_score(1,*))
     m_ind = total(db_score(2,*))
     n_ind = total(db_score(3,*))

     get_skill_scores,h,f,m,n,dbdt_pod,dbdt_pof,dbdt_hss
     get_skill_scores,h_ind,f_ind,m_ind,n_ind,db_pod,db_pof,db_hss

     if lunOut ne -1 and DoPrintHeader ne 0 then begin
        printf, lunOut, 'db/dt skill scores for model: ' + model
        for iEvent = 0,n_elements(event_I)-1 do begin
           printf, lunOut, 'Event   =', event_I[iEvent]
           printf, lunOut, 'For Event '+string(event_I[iEvent],format='(i2)')+', Stations used: '+strtrim(stations_used[iEvent])
        endfor
        printf, lunOut, 'Deltat  = ' + string(deltat,format='(f6.2)') + ', threshold unit = [nT/s]'
        printf, lunOut, $
                'threshold  TP  TN  FP  FN  total  TP_ind  TN_ind  FP_ind  FN_ind  total_ind  pod  far  hss  pod_ind  far_ind  hss_ind'
        DoPrintHeader = 0
     endif
     
     if lunOut ne -1 then $
        printf, lunOut, threshold, h,n,f,m,h+n+f+m, h_ind,n_ind,f_ind,m_ind,h_ind+n_ind+f_ind+m_ind,    $
                dbdt_pod, dbdt_pof, dbdt_hss, db_pod, db_pof, db_hss,       $
                format='(f8.4, 10i6, 6f12.4)'
  endif

  if choice eq 'db' then begin

     h = total(db_score(0,*))
     f = total(db_score(1,*))
     m = total(db_score(2,*))
     n = total(db_score(3,*))

     get_skill_scores,h,f,m,n,db_pod,db_pof,db_hss

     if lunOut ne -1 and DoPrintHeader ne 0 then begin
        printf, lunOut, 'db skill scores for model: ' + model
        for iEvent = 0,n_elements(event_I)-1 do begin
           printf, lunOut, 'Event   =', event_I[iEvent]
           printf, lunOut, 'For Event '+string(event_I[iEvent],format='(i2)')+', Stations used: '+strtrim(stations_used[iEvent])
        endfor
        printf, lunOut, 'Deltat  = ' + string(deltat,format='(f6.2)') + ', threshold unit = [nT]'
        printf, lunOut, 'threshold  TP  TN  FP  FN  total  pod  far  hss'
        DoPrintHeader = 0
     endif

     if lunOut ne -1 then $
        printf, lunOut, threshold, h,n,f,m,h+n+f+m, db_pod, db_pof, db_hss, format='(f8.4,5i6, 3f12.4)'
  endif
  
  !p.multi = 0

  if choice eq 'corr' gt 0 then begin

     help, db_all, dbdt_all

     print,'correlate(db_all, dbdt_all)=', correlate(db_all, dbdt_all)

     param = [250., 1.0]
     dbdt_fit = lmfit(db_all, dbdt_all, param, FUNCTION_NAME='scale_exp', $
                      sigma=sigma)

     ii = where(db_all gt 10.0 and dbdt_all gt 0.1)
     db_all   = db_all(ii)
     dbdt_all = dbdt_all(ii)
     dbdt_fit = dbdt_fit(ii)

     print,'correlate(db_all>10, dbdt_all>0.1)=', correlate(db_all, dbdt_all)
     print,'correlate(log db_all>10, log dbdt_all>0.1)=', $
           correlate(alog(db_all), alog(dbdt_all))

     print,'parameter fit and 1 sigma errors=', param, sigma

     if showplot then begin
        if saveplot then set_device, 'corr_'+strjoin(station_I)+'.eps', /eps

        plot_oo, db_all, dbdt_all, psym=1, $
                 xtitle='dB!DH!N [nT]', ytitle='dB/dt!DH!N [nT/s]'

        oplot, db_all, dbdt_fit, color=250, thick=3

        oplot, [10,10000], 2/60.*[10,10000], thick=2, color=150

        if exponent gt 0 then $
           oplot, [1.,10000.], ([1.,10000.]/scale)^exponent, color=100
        
        if saveplot then close_device, /pdf
     endif
  endif

end

;==============================================================================

function func, arg

  common func_param, thresholds, stations, imodel, events, opt

  averagehss  = 0.0
  averagehss2 = 0.0

  n = n_elements(thresholds)

  for ithreshold = 0, n-1 do begin

     if opt eq 'db' then begin
        predict,'dbdt', threshold=thresholds[ithreshold], $
                stations=stations, imodel=imodel, $
                events=events,                               $
                scale=arg(0), exponent=arg(1), db_hss = hss, $
                dbdt_hss = hss2
     endif else begin
        predict,'dbdt', threshold=thresholds[ithreshold], $
                stations=stations, imodel=imodel, $
                events=events,                       $
                coeff=arg(0), power=arg(1), dbdt_hss = hss
     endelse

     averagehss  = averagehss + hss
     averagehss2 = averagehss2 + hss2

  endfor

  averagehss  = averagehss/n
  averagehss2 = averagehss2/n

  if opt eq 'db' then $
     print, 'Current scale, exponent, hss_db, hss_dbdt=', $
            arg, averagehss, averagehss2, format='(a,4f10.3)' $
  else $
     print, 'Current coeff, power, hss=', $
            arg, averagehss, format='(a,3f10.3)'


  ; Since amoebe finds the minimum, we take negative sign here
  return, -averagehss

end

;==============================================================================

pro optimize, thresholds=thresholds1, stations=stations1, imodel=imodel1, $
              events=events1, opt=opt1, $
              scale=scale, exponent=exponent

  common func_param, thresholds, stations, imodel, events, opt

  if keyword_set(thresholds1) then $
     thresholds = thresholds1 $
  else $
     thresholds = [0.3, 0.7, 1.1, 1.5]

  if keyword_set(stations1) then $
     stations  = stations1 $
  else $
     stations = ['abk','pbq','ykc']

  if keyword_set(imodel1) then $
     imodel = imodel1 $
  else $
     imodel = 0

  if keyword_set(events1) then $
     events=events1            $
  else $
     events='1-6'

  ; what shall we optimize for: db or dbdt
  if keyword_set(opt1) then $
     opt = opt1 $
  else $
     opt = 'db'

  ; initial values for scale and 
  if not keyword_set(scale)    then scale = 292.0
  if not keyword_set(exponent) then exponent = 1.14

  if opt eq 'db' then $
     results = amoeba(1.0e-3, scale=[100.0, 1.0], p0=[scale, exponent], $
                      function_value=averagehss) $
  else $
     results = amoeba(1.0e-3, scale=[0.25, 0.25], p0=[1.0, 1.0], $
                      function_value=averagehss)

  ; Change back sign
  averagehss = -averagehss
  
  print,'thresholds = ', thresholds
  print,'stations   = ', stations
  print,'imodel     = ', imodel
  print,'events     = ', events
 
  if opt eq 'db' then $
     print, 'Optimal scale, exponent, hss=', results, averagehss(0), $
            format='(a,3f10.3)' $
  else $
     print, 'Optimal coeff, power, hss=', results, averagehss(0), $
            format='(a,3f10.3)'

end

;==============================================================================

pro calc_hss_dt_table, stations, imodel, directs, indirects
  ; INPUTS:
  ; Stations is an array of station names
  ; imodel is the model number: 0 for obs, 1 for SWMF_old, 2 for SWMF_new.
  ; OUTPUTS:
  ; "directs", a nDeltaT-by-nThreshhold
  deltats    = [1, 2, 5, 10, 20, 40]
  thresholds = [0.3, 0.7, 1.1, 1.5]

  ndeltat    = n_elements(deltats)
  nthreshold = n_elements(thresholds)

  directs   = fltarr(ndeltat, nthreshold)
  indirects = fltarr(ndeltat, nthreshold)

  stencil = 1
  scale    = 292
  exponent = 1.14

;;  stencil = 2
;;  scale    = 379
;;  exponent = 1.14

  for ithreshold = 0, nthreshold-1 do begin
     threshold = thresholds[ithreshold]
     for ideltat = 0, ndeltat-1 do begin
        deltat = deltats[ideltat]
        predict,'dbdt',stations=stations,threshold=threshold,scale=scale, $
                exponent=exponent,deltat=deltat, dbdt_hss=dbdt_hss, $
                db_hss=db_hss, stencil=stencil
        directs[ideltat,ithreshold] = dbdt_hss
        indirects[ideltat,ithreshold] = db_hss
     endfor
  endfor

end
;==============================================================================
pro show_hss_dt_table, stations, directs, indirects

  deltats=[1,2,5,10,20,40]
  thresholds=[0.3, 0.7, 1.1, 1.5]

  s = size(directs)

  ndeltat    = s(1)
  nthreshold = s(2)

  print, 'stations=', stations
  print, '\tableline'
  for ithreshold = 0, nthreshold-1 do begin
     line1 = string(thresholds(ithreshold),format='(f3.1,"\,[nT/s]&")')
     line2 = line1
     line1 = line1 + ' direct   '
     line2 = line2 + ' indirect '
     for ideltat = 0, ndeltat-1 do begin
        line1 = line1 + string(  directs[ideltat,ithreshold], format='("& ",f5.3," ")')
        line2 = line2 + string(indirects[ideltat,ithreshold], format='("& ",f5.3," ")')
     endfor
     line1 = line1 + '\\'
     line2 = line2 + '\\'
     print, line1
     print, line2
     print, '\tableline'
  endfor

end

;==============================================================================
pro calc_all_events, choice=choice, models=models, events=events, $
                     mydir=mydir, InputDir=InputDir
  
;; Calculate scores for the models listed in the 'models' string array
;; for all station groups (low, mid, high, veryhigh, all).
;; Save scores into separate files for each station group and 
;; each event from the events string (default is 1,2,3,4,5,6).
;; Create plots of dB/dt per model and per event for the station group 'all'.

  ;; Compare deltaB/Results with deltaB/SWMF_CCMC by default
  if n_elements(models) lt 1 then models=['Results', 'SWMF_CCMC']

  ;; Set station groups and thresholds
  if not keyword_set(choice) then choice = 'dbdt'

  if choice eq 'dbdt' then begin
     thresholds  = [0.3, 0.7, 1.1, 1.5]
  endif else if choice eq 'db' then begin
     thresholds  = [101.6, 213.6, 317.5, 416.7]
  endif else begin
     print,'unknown choice =', choice
     return
  endelse

  ; These are default values
  deltat   = 20.
  stencil  = 1
  scale    = 292
  exponent = 1.14

  for imodel = 0, n_elements(models)-1 do begin
     model   = models(imodel)
     event_I = set_eventlist(events,mydir,model)

     ;; Create full metrics for each event:
     for iEvent = 0, n_elements(event_I)-1 do begin
        event=event_I(iEvent)
        set_stationlist, mydir=mydir, stationsFile='stations.csv',$
                         model=model, event=event,                $
                         stations_I, station_orig_I

        for istation = 0, n_elements(stations_I)-1 do begin
           stationIn_I = strsplit(stations_I(istation),/extract)
           if choice eq 'dbdt' then begin
              filename=string(station_orig_I(istation), event, $
                                 format='("metrics_lat_",a3,"_event",i2.2,".txt")')
              filenameStation=string(station_orig_I(istation), event, $
                                 format='("metrics_stations_lat_",a3,"_event",i2.2,".txt")')

              if strpos(station_orig_I(istation), 'all') ne 0 then begin
                 filenameInLocal='metrics_stations_lat_all_event' $
                                 +string(event,format='(i2.2)')+'.txt'

                 if file_test(filenameInLocal) then begin
                    print, filenameInLocal + ' exists, get scores from file.'
                    get_scores_from_file, choice, filenameInLocal,  $
                                          stations_I(istation),     $
                                          filename, filenameStation,$
                                          event,thresholds
                    continue
                 endif else begin
                    print, filenameInLocal + ' does not exist, calculate the scores...'
                 endelse

              endif
           endif else if choice eq 'db' then begin
              filename=string(station_orig_I(istation), event, $
                                 format='("metrics_db_lat_",a3,"_event",i2.2,".txt")')
              filenameStation=string(station_orig_I(istation), event, $
                                 format='("metrics_stations_db_lat_",a3,"_event",i2.2,".txt")')

              if strpos(station_orig_I(istation), 'all') ne 0 then begin
                 filenameInLocal='metrics_stations_db_lat_all_event' $
                                 +string(event,format='(i2.2)')+'.txt'

                 if file_test(filenameInLocal) then begin
                    print, filenameInLocal + ' exists, get scores from file.'
                    get_scores_from_file, choice, filenameInLocal,    $
                                          stations_I(istation),       $
                                          filename, filenameStation,  $
                                          event,thresholds
                    continue
                 endif else begin
                    print, filenameInLocal + ' does not exist, calculate the scores...'
                 endelse
              endif
           endif

           openw,  lunOut,     filename,        append=ithresh, /get_lun
           openw,  lunStation, filenameStation, append=ithresh, /get_lun
           DoPrintHeader        = 1
           DoPrintStationHeader = 1
           for ithresh = 0, n_elements(thresholds)-1 do begin
              threshold = thresholds[ithresh]
              predict,choice,                                     $
                      model=model,                                $
                      station_I=stationIn_I, threshold=threshold, $
                      scale=scale, exponent=exponent,             $
                      events=string(event,format='(i2.2)'),       $
                      deltat=deltat, stencil=stencil,             $
                      lunOut=lunOut, lunStation=lunStation,       $
                      saveplot=(station_orig_I(istation) eq 'all_ccmc'),             $
                      mydir=mydir, DoPrintHeader=DoPrintHeader,   $
                      InputDir=InputDir,DoPrintStationHeader=DoPrintStationHeader
           endfor
           close, lunOut     & free_lun, lunOut
           close, lunStation & free_lun, lunStation
        endfor
     endfor
  endfor
end
;==============================================================================

pro fft_all_events, models=models, events=events, mydir=mydir, InputDir=InputDir

;; to be continue

  if n_elements(models) lt 1 then models=['Results', 'SWMF_CCMC']

  for imodel = 0, n_elements(models)-1 do begin
     model   = models(imodel)
     event_I = set_eventlist(events,mydir,model)

     set_stationlist, mydir=mydir, stationsFile='stations.csv',$
                      model=model, stations_I, station_orig_I

     ;; Create full metrics for each event:
     for istation = 0, n_elements(station_orig_I)-1 do begin
        stationIn_I = strsplit(stations_I(istation),/extract)

        filename = string(station_orig_I(istation),  $
                          format='("fft_stats_",a3,".txt")')
        openw, lun_fft_all, filename, /get_lun
        str_event_I = strtrim(string(event_I),2)
        string_header = 'name ' + strjoin('mean_obs_'+str_event_I,' ')   $
                        + ' ' + strjoin('mean_sim_'+str_event_I,  ' ')   $
                        + ' ' + strjoin('max_obs_'+str_event_I,   ' ')   $
                        + ' ' + strjoin('max_sim_'+str_event_I,   ' ')
        printf,lun_fft_all, string_header

        filename = string(station_orig_I(istation),  $
                          format='("fft_ave_",a3,".txt")')
        openw, lun_fft_ave, filename, /get_lun
        printf, lun_fft_ave, 'name mean_obs mean_sim max_obs max_sim'

        integral_obs_III = fltarr(n_elements(event_I), n_elements(stationIn_I), 2)
        integral_sim_III = fltarr(n_elements(event_I), n_elements(stationIn_I), 2)
  
        predict,'fft', model=model, station_I=stationIn_I, events=events,  $
                mydir=mydir, integral_obs_III=integral_obs_III,            $
                integral_sim_III=integral_sim_III

        for itmp=0, n_elements(stationIn_I)-1 do begin
           printf, lun_fft_all, stationIn_I(itmp), integral_obs_III(*,itmp,0), integral_sim_III(*,itmp,0),       $
                   integral_obs_III(*,itmp,1), integral_sim_III(*,itmp,1), format='(a,100f17.2)'

           tmp_obs_ave = integral_obs_III(*,itmp,0)
           tmp_obs_max = integral_obs_III(*,itmp,1)
           tmp_sim_ave = integral_sim_III(*,itmp,0)
           tmp_sim_max = integral_sim_III(*,itmp,1)

           printf, lun_fft_ave, stationIn_I(itmp), mean(tmp_obs_ave(where(tmp_obs_ave gt 0))),             $
                   mean(tmp_sim_ave(where(tmp_sim_ave gt 0))), mean(tmp_obs_max(where(tmp_obs_max gt 0))), $
                   mean(tmp_sim_max(where(tmp_sim_max gt 0))), format='(a,100f17.2)'
        endfor
        close, lun_fft_all & free_lun, lun_fft_all
        close, lun_fft_ave & free_lun, lun_fft_ave
     endfor
  endfor
end

;==============================================================================

pro get_scores_from_file, choice, filenameInLocal, strStations, $
                          filename, filenameStation, event,thresholds

  get_log, filenameInLocal, wlog,  wlognames, logtime, 'h', $
           wlogrownames,headlines=headlinesLocal

  stationLocal_I = strsplit(strStations,/extract)

  strStationsUsed = strStations

  if choice eq 'dbdt' then begin
     formatOut = '(a3,1x,f8.4, 10i6, 6f12.4)'
  endif else if choice eq 'db' then begin
     formatOut ='(a3,1x,f8.4, 5i6, 3f12.4)'
  endif

  ;; saving the per station file
  ;; no idea why the new line char is missing???
  openw, lunOut, filenameStation, /get_lun
  for i=0,n_elements(headlinesLocal)-1 do begin
     printf,lunOut, headlinesLocal(i)
  endfor

  for istationLocal = 0, n_elements(stationLocal_I)-1 do begin
     index_local = where(wlogrownames eq stationLocal_I(istationLocal),nLocal)
     if nLocal eq 0 then begin
        ;; skip the station if not found in the file and removed it
        ;; from strStationsUsed
        strStationsUsed = strStationsUsed.replace(' '+stationLocal_I(istationLocal),'')
        continue
     endif
     for iIndex=0,n_elements(index_local)-1 do begin
        printf, lunOut, wlogrownames(index_local(iIndex)),     $
                wlog(index_local(iIndex),*),format=formatOut
     endfor
  endfor

  close, lunOut & free_lun, lunOut

  ;; saving the skill score file
  openw, lunOut, filename, /get_lun
  printf,lunOut, headlinesLocal(0)
  printf,lunOut, headlinesLocal(1)
  printf,lunOut, 'For Event '+string(event,format='(i2)')$
         +', Stations used: '+strStationsUsed

  for i=2,n_elements(headlinesLocal)-2 do begin
     printf,lunOut, headlinesLocal(i)
  endfor
  printf,lunOut,headlinesLocal(n_elements(headlinesLocal)-1).replace('name ','')

  for ithres=0,n_elements(thresholds)-1 do begin
     threshold   = thresholds(ithres)
     index_thres = where(abs(wlog(*,0)-threshold) le 1e-4)

     ;; reset all counts
     h=0     & f=0     & m=0     & n=0
     h_ind=0 & f_ind=0 & m_ind=0 & n_ind=0

     for istationLocal = 0, n_elements(stationLocal_I)-1 do begin
        index_stat = where(wlogrownames(index_thres) eq stationLocal_I(istationLocal),nLocal)
        if nLocal eq 0 then continue

        h = h + wlog(index_thres(index_stat),1)
        f = f + wlog(index_thres(index_stat),3)
        m = m + wlog(index_thres(index_stat),4)
        n = n + wlog(index_thres(index_stat),2)

        if choice eq 'dbdt' then begin
           h_ind = h_ind + wlog(index_thres(index_stat),6)
           f_ind = f_ind + wlog(index_thres(index_stat),8)
           m_ind = m_ind + wlog(index_thres(index_stat),9)
           n_ind = n_ind + wlog(index_thres(index_stat),7)
        endif
     endfor

     if choice eq 'dbdt' then begin
        get_skill_scores,h,f,m,n,dbdt_pod,dbdt_pof,dbdt_hss
        get_skill_scores,h_ind,f_ind,m_ind,n_ind,db_pod,db_pof,db_hss

        printf,lunOut, threshold, h,n,f,m,h+n+f+m, h_ind,n_ind,f_ind,m_ind,h_ind+n_ind+f_ind+m_ind,    $
               dbdt_pod, dbdt_pof, dbdt_hss, db_pod, db_pof, db_hss,       $
               format='(f8.4, 10i6, 6f12.4)'
     endif else if choice eq 'db' then begin
        get_skill_scores,h,f,m,n,db_pod,db_pof,db_hss

        printf, lunOut, threshold, h,n,f,m,h+n+f+m, db_pod, db_pof, db_hss, format='(f8.4,5i6, 3f12.4)'
     endif
  endfor

  close, lunOut & free_lun, lunOut
end

;==============================================================================

pro save_comp_dbdt_table, $
   stationlat, model1, model2, events=events, mydir=mydir, InputDir=InputDir

;; Compare deltaB/model1 and deltaB/model2 outputs 
;; for the stationlat station group:
;; stationlat = 'all', 'veryhigh', 'high', 'mid' or 'low'. Default is all.
;; If model1 is not given 'Results' is assumed. 
;; If model2 is not given 'SWMF_CCMC' are used.
;; Events starting from the event string (default is 1,2,3,4,5,6).

  if n_elements(stationlat) lt 1 then stationlat='all'
  if n_elements(model1)     lt 1 then model1 = 'Results'
  if n_elements(model2)     lt 1 then model2 = 'SWMF_CCMC'

  ;; check if the two models have the same events, if not, quit
  event1_I = set_eventlist(events,mydir,model1)
  event2_I = set_eventlist(events,mydir,model2)

  if not array_equal(event1_I, event2_I) then begin
     print, model1 + ' and ' + model2 + ' do not have the same event list.'
     retall
  endif

  thresholds=[0.3, 0.7, 1.1, 1.5]
  deltat = 20.

  ; Size of results arrays:
  nthresh = n_elements(thresholds)
  nmetric = 3 ; POD, FAR, HSS

  ; Arrays to store results:
  directs    = fltarr(nmetric, nthresh)
  indirects  = fltarr(nmetric, nthresh)
  directs2   = fltarr(nmetric, nthresh)
  indirects2 = fltarr(nmetric, nthresh)

  stencil  = 1
  scale    = 292
  exponent = 1.14

;;  stencil = 2
;;  scale    = 379
;;  exponent = 1.14

  for ithresh = 0, nthresh-1 do begin
     threshold = thresholds[ithresh]
     filename=string(stationlat, deltat, floor(threshold), $
                     floor(10*threshold-10*floor(threshold)), $
                     format='("metrics_",a,"_dt",i2.2,"_thresh",i1.1,"p",i2.2,".txt")')
     ; Results for model 1:
     predict,'dbdt',stationlat=stationlat,threshold=threshold,scale=scale, $
             exponent=exponent,deltat=deltat, stencil=stencil, $
             models=[model1], $
             events=events, $
             dbdt_hss=dbdt_hss, dbdt_pod=dbdt_pod, dbdt_pof=dbdt_pof,   $
             db_hss  =db_hss,   db_pod  =db_pod,   db_pof  =db_pof, mydir=mydir, $
             InputDir=InputDir

     directs[  *, ithresh] = [dbdt_pod, dbdt_pof, dbdt_hss]
     indirects[*, ithresh] = [db_pod,   db_pof,   db_hss  ]

     ; Results for model 2:
     predict,'dbdt',stationlat=stationlat,threshold=threshold,scale=scale, $
             exponent=exponent,deltat=deltat, stencil=stencil, $
             models=[model2], $
             events=events, $
             dbdt_hss=dbdt_hss, dbdt_pod=dbdt_pod, dbdt_pof=dbdt_pof,   $
             db_hss  =db_hss,   db_pod  =db_pod,   db_pof  =db_pof, mydir=mydir, $
             InputDir=InputDir

     directs2[  *, ithresh] = [dbdt_pod, dbdt_pof, dbdt_hss]
     indirects2[*, ithresh] = [db_pod,   db_pof,   db_hss  ]
  endfor

  ; print out comparison
  filename = "metric_table_" + stationlat + ".tex"
  openw, lun, filename, /get_lun
  
  printf, lun, 'models=', model1,' and (',model1,' - ',model2,')'
  printf, lun, 'events =' + strjoin(strtrim(string(event_I,format='(i)'),2),',')
  printf, lun, 'stationlat=', stationlat
  printf, lun, '$\Delta t_{window}=',deltat,'$'
  printf, lun, ''
  printf, lun, '\begin{table}[ht]'
  printf, lun, '\centering'
  printf, lun, '\begin{tabular}{l l|c c c}'
  printf, lun, 'Threshold&Method&POD&POFD&HSS\\'
  printf, lun, '\hline'
  for ithresh = 0, nthresh-1 do begin
     line1 = string(thresholds(ithresh),format='(f3.1,"\,[nT/s]&")')
     line2 = line1
     line1 += ' direct   '
     line2 += ' indirect '
     
     for imetric = 0, nmetric-1 do begin
        line1 += string( $
                directs[imetric,ithresh], $
                directs[imetric,ithresh] - directs2[ imetric, ithresh], $
                format='("& ",f5.3," (",f+6.3,") ")')
        line2 += string( $
                indirects[imetric,ithresh], $
                indirects[imetric,ithresh] - indirects2[imetric, ithresh], $
                format='("& ",f5.3," (",f+6.3,") ")')
     endfor
     line1 += '\\'
     line2 += '\\'
     printf, lun, line1
     printf, lun, line2
     printf, lun, '\hline'
  endfor
  
  printf, lun, '\end{tabular}'
  printf, lun, '\caption{Performance metrics for the SWMF.}'
  printf, lun, '\end{table}'
  
  free_lun, lun
end

;==============================================================================
pro save_comp_dbdt_tables, $
   model1, model2, events=events, mydir=mydir, InputDir=InputDir
  
  ;; Create comparison tables between results stored in
  ;; deltaB/model1 and deltaB/model2. Defaults are 'Results' and 'SWMF_CCMC'.
  ;; Combine scores for all events from the event string (default is 1
  ;; to 6). 
  ;; Create a separate table for the various station groups:
  ;; low, mid, high, veryhigh, all.

  ; Set station groups
  stationlats = ['all', 'veryhigh', 'high', 'mid', 'low']

  ; Create combined metrics for all events:
  for ilat=0, n_elements(stationlats)-1 do begin
     save_comp_dbdt_table, stationlats[ilat], $
                           model1, model2, $
                           events=events, $
                           mydir=mydir,   $
                           InputDir=InputDir
  endfor
end
;==============================================================================

pro save_comp_db_table, $
   stationlat, model1, model2, events=events, $
   mydir=mydir, InputDir=InputDir

;; Compare deltaB/model1 and deltaB/model2 outputs 
;; for the stationlat station group:
;; stationlat = 'all', 'veryhigh', 'high', 'mid' or 'low'. Default is all.
;; If model1 is not given 'Results' is assumed. 
;; If model2 is not given 'SWMF_CCMC' are used.
;; Events starting from the events string (default is 1 to 6).

  if n_elements(stationlat) lt 1 then stationlat='all'
  if n_elements(model1)     lt 1 then model1 = 'Results'
  if n_elements(model2)     lt 1 then model2 = 'SWMF_CCMC'

  ;; check if the two models have the same events, if not, quit
  event1_I = set_eventlist(events,mydir,model1)
  event2_I = set_eventlist(events,mydir,model2)

  if not array_equal(event1_I, event2_I) then begin
     print, model1 + ' and ' + model2 + ' do not have the same event list.'
     retall
  endif

  thresholds=[101.6, 213.6, 317.5, 416.7]
  deltat = 20.

  ; Size of results arrays:
  nthresh = n_elements(thresholds)
  nmetric = 3 ; POD, FAR, HSS

  ; Arrays to store results:
  directs    = fltarr(nmetric, nthresh)
  indirects  = fltarr(nmetric, nthresh)
  directs2   = fltarr(nmetric, nthresh)
  indirects2 = fltarr(nmetric, nthresh)

  stencil  = 1
  scale    = 292
  exponent = 1.14

;;  stencil = 2
;;  scale    = 379
;;  exponent = 1.14

  for ithresh = 0, nthresh-1 do begin
     threshold = thresholds[ithresh]
     filename=string(stationlat, deltat, floor(threshold), $
                     floor(10*threshold-10*floor(threshold)), $
                     format='("metrics_",a,"_dt",i2.2,"_thresh",i1.1,"p",i2.2,".txt")')
     ; Results for model 1:
     predict,'db',stationlat=stationlat,threshold=threshold,scale=scale, $
             exponent=exponent,deltat=deltat, stencil=stencil, $
             models=[model1], $
             events=events, $
             db_hss  =db_hss,   db_pod  =db_pod,   db_pof  =db_pof, mydir=mydir, $
             InputDir=InputDir

     directs[  *, ithresh] = [db_pod,   db_pof,   db_hss  ]

     ; Results for model 2:
     predict,'db',stationlat=stationlat,threshold=threshold,scale=scale, $
             exponent=exponent,deltat=deltat, stencil=stencil, $
             models=[model2], $
             events=events, $
             db_hss  =db_hss,   db_pod  =db_pod,   db_pof  =db_pof, mydir=mydir, $
             InputDir=InputDir

     directs2[  *, ithresh] = [db_pod,   db_pof,   db_hss  ]
  endfor

  ; print out comparison
  filename = "metric_table_db_" + stationlat + ".tex"
  openw, lun, filename, /get_lun

  printf, lun, 'models=', model1,' and (',model1,' - ',model2,')'
  printf, lun, 'events =' + strjoin(strtrim(string(event_I,format='(i)'),2),',')
  printf, lun, 'stationlat=', stationlat
  printf, lun, '$\Delta t_{window}=',deltat,'$'
  printf, lun, ''
  printf, lun, '\begin{table}[ht]'
  printf, lun, '\centering'
  printf, lun, '\begin{tabular}{l|c c c}'
  printf, lun, 'Threshold&POD&POFD&HSS\\'
  printf, lun, '\hline'
  for ithresh = 0, nthresh-1 do begin
     line1 = string(thresholds(ithresh),format='(f6.1,"\,[nT]")')
     line2 = line1
     ;line1 += ' direct   '
     ;line2 += ' indirect '
     
     for imetric = 0, nmetric-1 do begin
        line1 += string( $
                directs[imetric,ithresh], $
                directs[imetric,ithresh] - directs2[ imetric, ithresh], $
                format='("& ",f5.3," (",f+6.3,") ")')
        ;line2 += string( $
        ;        indirects[imetric,ithresh], $
        ;        indirects[imetric,ithresh] - indirects2[imetric, ithresh], $
        ;        format='("& ",f5.3," (",f+6.3,") ")')
     endfor
     line1 += '\\'
     line2 += '\\'
     printf, lun, line1
     ;printf, lun, line2
     printf, lun, '\hline'
  endfor

  printf, lun, '\end{tabular}'
  printf, lun, '\caption{Performance metrics for the SWMF.}'
  printf, lun, '\end{table}'
  
  free_lun, lun
  
end

;==============================================================================

pro save_comp_db_tables, $
   model1, model2, events=events, mydir=mydir, InputDir=InputDir
  
  ;; Create comparison tables between results stored in
  ;; deltaB/model1 and deltaB/model2. Defaults are 'Results' and 'SWMF_CCMC'.
  ;; Combine scores for all events from the event string.
  ;; (default is 1 to 6).
  ;; Create a separate table for the various station groups:
  ;; low, mid, high, veryhigh, all.

  ; Set station groups
  stationlats = ['all', 'veryhigh', 'high', 'mid', 'low']

  ; Create combined metrics for all events:
  for ilat=0, n_elements(stationlats)-1 do begin
     save_comp_db_table, stationlats[ilat], $
                         model1, model2, $
                         events=events, $
                         mydir=mydir, InputDir=InputDir
  endfor
end

;==============================================================================

pro save_tables, model=model, events=events, mydir=mydir

  ;; Create tables containing db/dt and db skill scores.
  ;; Defaults are 'Results'
  ;; Combine scores for all events from the event string
  ;; (default is 1 to 6).
  ;; Create a separate table for the various station groups:
  ;; low, mid, high, veryhigh, all.

  ; Set station groups
  stationlats = ['all', 'veryhigh', 'high', 'mid', 'low']

  ; Create combined metrics for all events:
  for ilat=0, n_elements(stationlats)-1 do begin
     save_table, stationlats[ilat], model, $
                 events=events, mydir=mydir
  endfor
end

;==============================================================================

pro save_table, stationlat, model, events=events, mydir=mydir

  ;; Create tables containing db/dt and db skill scores from
  ;; individual event files.
  ;; for the stationlat station group:
  ;; stationlat = 'all', 'veryhigh', 'high', 'mid' or 'low'. Default is all.
  ;; If model1 is not given 'Results' is assumed.
  ;; Events starting from the events strng (default is 1 to 6).
  
  if n_elements(stationlat) lt 1 then stationlat='all'
  if n_elements(model)      lt 1 then model = 'Results'

  event_I = set_eventlist(events,mydir,model)

  ;; 4 threshols,  9 vars, nEvents for db
  ;; 4 threshols, 17 vars, nEvents for db
  db_scores_III   = fltarr(4, 9,n_elements(event_I))
  dbdt_scores_III = fltarr(4,17,n_elements(event_I))

  ;; for db scores....
  fileRead_I = file_search( $
               "metrics_db_lat_"+strmid(stationlat,0,3)+"_event*.txt", count=nFileFound)

  if nFileFound gt 0 then begin
     filenameOut = "metric_table_db_" + stationlat + ".txt"
     openw, lunlocal, filenameOut, /get_lun
     printf,lunlocal, 'model: ' + model
     printf,lunlocal, 'Events = ' + events
  
     for iEvent = 0,n_elements(event_I)-1 do begin
        event = event_I(iEvent)
        filename=string(stationlat, event, $
                        format='("metrics_db_lat_",a3,"_event",i2.2,".txt")')
  
        if file_test(filename) then begin
           get_log, filename, wlog, wlogname, headlines=headlines
           db_scores_III(*,*,iEvent) = wlog
  
           printf, lunlocal, headlines(2)
        endif else begin
           printf, lunlocal, ' no file is found'
        endelse
     end
  
     printf, lunlocal,headlines(3)
     printf, lunlocal, 'threshold  pod      far     hss'
  
     for i=0,3 do begin
        ;; so stupid that the print out is not the same order as defined
        ;; in predict
        h = total(db_scores_III(i,1,*))
        n = total(db_scores_III(i,2,*))
        f = total(db_scores_III(i,3,*))
        m = total(db_scores_III(i,4,*))
  
        get_skill_scores,h,f,m,n,db_pod,db_pof,db_hss

        printf, lunlocal, wlog(i,0), db_pod, db_pof, db_hss, format='(4f8.4)'
     end
     free_lun, lunlocal
  endif

  ;; for dbdt scores....
  fileRead_I = file_search( $
               "metrics_lat_"+strmid(stationlat,0,3)+"_event*.txt", count=nFileFound)

  if nFileFound gt 0 then begin
     filenameOut = "metric_table_" + stationlat + ".txt"
     openw, lunlocal, filenameOut, /get_lun
     printf,lunlocal, 'model: ' + model
     printf,lunlocal, 'Events = ' + events

     for iEvent = 0,n_elements(event_I)-1 do begin
        event = event_I(iEvent)
        filename=string(stationlat, event, $
                        format='("metrics_lat_",a3,"_event",i2.2,".txt")')

        if file_test(filename) then begin
           get_log, filename, wlog, wlogname, headlines=headlines
           dbdt_scores_III(*,*,iEvent) = wlog

           printf, lunlocal, headlines(2)
        endif else begin
           printf, lunlocal, 'no event is found'
        endelse
     end

     printf, lunlocal,headlines(3)
     printf, lunlocal, 'threshold  pod  far  hss  pod_ind  far_ind  hss_ind'

     for i=0,3 do begin
        ;; so stupid that the print out is not the same order as defined
        ;; in predict
        h = total(dbdt_scores_III(i,1,*))
        n = total(dbdt_scores_III(i,2,*))
        f = total(dbdt_scores_III(i,3,*))
        m = total(dbdt_scores_III(i,4,*))

        h_ind = total(dbdt_scores_III(i,6,*))
        n_ind = total(dbdt_scores_III(i,7,*))
        f_ind = total(dbdt_scores_III(i,8,*))
        m_ind = total(dbdt_scores_III(i,9,*))

        get_skill_scores,h,f,m,n,dbdt_pod,dbdt_pof,dbdt_hss
        get_skill_scores,h_ind,f_ind,m_ind,n_ind,dbdt_pod_ind,dbdt_pof_ind,dbdt_hss_ind

        printf, lunlocal, wlog(i,0), dbdt_pod, dbdt_pof, dbdt_hss, $
                dbdt_pod_ind, dbdt_pof_ind, dbdt_hss_ind, format='(7f8.4)'
     end
     free_lun, lunlocal
  endif
end

;==============================================================================

pro save_tables_station, model=model, events=events, mydir=mydir

  ;; Create tables containing db/dt and db skill scores.
  ;; Defaults are 'Results'
  ;; Combine scores for all events from the event string
  ;; (default is 1 to 6).
  ;; Create a separate table for the various station groups:
  ;; low, mid, high, veryhigh, all.

  ; Set station groups
  stationlats = ['all']

  ; Create combined metrics for all events:
  for ilat=0, n_elements(stationlats)-1 do begin
     save_table_station, stationlats[ilat], model, 'dbdt', $
                         events=events, mydir=mydir
     save_table_station, stationlats[ilat], model, 'db',   $
                         events=events, mydir=mydir
  endfor
end

;==============================================================================

pro save_table_station, stationlat, model, choice, events=events, mydir=mydir

  if n_elements(stationlat) lt 1 then stationlat='all'
  if n_elements(model)      lt 1 then model = 'Results'
  if n_elements(choice)     lt 1 then model = 'dbdt'

  ;; set the list of the event
  event_I = set_eventlist(events,mydir,model)

  ;; set te list of stations
  set_stationlist, mydir=mydir, stationsFile='stations.csv',  $
                   model=model, stations_I, station_orig_I

  ;; find the corresponding string containing all the stations
  ;; and put it into an array.
  strStationsLocal = stations_I(where(strmatch(station_orig_I,stationlat+'*')))
  station_I = strsplit(strStationsLocal,/extract)

  nEventStation_I = intarr(n_elements(station_I))

  if choice eq 'dbdt' then begin
     thresholds  = [0.3, 0.7, 1.1, 1.5]
     dbdt_scores_stations_III = fltarr(4,n_elements(station_I),17)

     filenameOut = "metric_table_stations_" + stationlat + ".txt"
  endif else if choice eq 'db' then begin
     thresholds  = [101.6, 213.6, 317.5, 416.7]
     db_scores_stations_III = fltarr(4,n_elements(station_I),9)

     filenameOut = "metric_table_db_stations_" + stationlat + ".txt"
  endif

  ;; the string containg the Events used for the station
  strEventsUsed_I  = strarr(n_elements(station_I))

  ;; loop through all the thresholds
  for ithres=0,n_elements(thresholds)-1 do begin
     threshold   = thresholds(ithres)

     ;; loop through all the stations
     for istation=0,n_elements(station_I)-1 do begin
        ;; set the header
        if ithres eq 0 then strEventsUsed_I(istation) = $
           'For '+station_I(istation)+' Events used: '

        ;; reset all counts
        h=0     & f=0     & m=0     & n=0
        h_ind=0 & f_ind=0 & m_ind=0 & n_ind=0

        ;; loop through all the event files
        for iEvent = 0,n_elements(event_I)-1 do begin
           event = event_I(iEvent)

           if choice eq 'dbdt' then begin
              filenameInLocal=string(stationlat, event, $
                                     format='("metrics_stations_lat_",a3,"_event",i2.2,".txt")')
           endif else if choice eq 'db' then begin
              filenameInLocal=string(stationlat, event, $
                                     format='("metrics_stations_db_lat_",a3,"_event",i2.2,".txt")')
           endif

           get_log, filenameInLocal, wlog,  wlognames, logtime, 'h', $
                    wlogrownames,headlines=headlinesLocal

           ;; find the index for the threshold, for all stations
           index_thres = where(abs(wlog(*,0)-threshold) le 1e-4)

           ;; within the above index, find the index for the
           ;; corresponding station
           index_stat = where(wlogrownames(index_thres) eq station_I(istation),nLocal)

           ;; if the count of index_stat, meaning that this file
           ;; contains the skill scores for the station
           if nLocal gt 0 then begin
              h = h + wlog(index_thres(index_stat),1)
              f = f + wlog(index_thres(index_stat),3)
              m = m + wlog(index_thres(index_stat),4)
              n = n + wlog(index_thres(index_stat),2)

              if choice eq 'dbdt' then begin
                 h_ind = h_ind + wlog(index_thres(index_stat),6)
                 f_ind = f_ind + wlog(index_thres(index_stat),8)
                 m_ind = m_ind + wlog(index_thres(index_stat),9)
                 n_ind = n_ind + wlog(index_thres(index_stat),7)
              endif

              ;; put the event # into the header line
              if ithres eq 0 then strEventsUsed_I(istation) = strEventsUsed_I(istation) $
                 + string(event,format='(i3)')

              ;; if at least one threshold/event has this station
              nEventStation_I(istation) = nEventStation_I(istation) + 1
           endif
        endfor

        if choice eq 'dbdt' then begin
           get_skill_scores,h,f,m,n,dbdt_pod,dbdt_pof,dbdt_hss
           get_skill_scores,h_ind,f_ind,m_ind,n_ind,db_pod,db_pof,db_hss

           dbdt_scores_stations_III(ithres,istation,*) = $
              [threshold,h,n,f,m,h+n+f+m, h_ind,n_ind,f_ind,m_ind,h_ind+n_ind+f_ind+m_ind, $
               dbdt_pod, dbdt_pof, dbdt_hss, db_pod, db_pof, db_hss]
        endif else if choice eq 'db' then begin
           get_skill_scores,h,f,m,n,db_pod,db_pof,db_hss

           db_scores_stations_III(ithres,istation,*) =   $
              [threshold,h,n,f,m,h+n+f+m,db_pod,db_pof,db_hss]
        endif
     endfor
  endfor

  openw, lunlocal, filenameOut, /get_lun
  printf,lunlocal, 'model: ' + model
  for istation=0,n_elements(station_I)-1 do begin
     printf, lunlocal, strEventsUsed_I(istation)
  endfor

  if choice eq 'dbdt' then begin
     printf,lunlocal, 'Deltat  =  20.00, threshold unit = [nT/s]'
     printf,lunlocal, $
            'name threshold  TP  TN  FP  FN  total  TP_ind  TN_ind  FP_ind  FN_ind  total_ind  pod  far  hss  pod_ind  far_ind  hss_ind'
     printf,lunlocal, '#START'
     for istation=0,n_elements(station_I)-1 do begin
        for ithres=0,n_elements(thresholds)-1 do begin
           if nEventStation_I(istation) gt 0 then $
              printf, lunlocal, station_I(istation), dbdt_scores_stations_III(ithres,istation,*), $
                      format='(a3,1x,f8.4, 10i6, 6f12.4)'
        endfor
     endfor
  endif else if choice eq 'db' then begin
     printf,lunlocal, 'Deltat  =  20.00, threshold unit = [nT]'
     printf,lunlocal, 'name threshold  TP  TN  FP  FN  total  pod  far  hss'
     printf,lunlocal, '#START'
     for istation=0,n_elements(station_I)-1 do begin
        for ithres=0,n_elements(thresholds)-1 do begin
           if nEventStation_I(istation) gt 0 then $
              printf, lunlocal, station_I(istation), db_scores_stations_III(ithres,istation,*), $
                      format='(a3,1x,f8.4, 5i6, 3f12.4)'
        endfor
     endfor
  endif

  free_lun, lunlocal
end

;==============================================================================

pro make_2d_map, choice, model=models, mydir=mydir

  get_log, mydir+'/SWMF/Param/supermag.dat',wlogMag,  wlognamesMag, logtimeMag,     $
           'h', wlogrownamesMag
  
  if choice eq 'dbdt' then begin
     thresholds  = [0.3, 0.7, 1.1, 1.5]
     filenameScore = 'metric_table_stations_all.txt'
  endif else if choice eq 'db' then begin
     thresholds  = [101.6, 213.6, 317.5, 416.7]
     filenameScore = 'metric_table_db_stations_all.txt'
  endif

  get_log, filenameScore, wlogScore,  wlognamesScore, logtimeScore, 'h', $
           wlogrownamesScore

  nVarScore     = n_elements(wlognamesScore)
  ;; each station is repeated 4 times
  nStationScore = n_elements(wlogrownamesScore)/4

  nThres = n_elements(thresholds)
  
  ;; coordinates and vars
  x = dblarr(nStationScore,1,2)

  if choice eq 'dbdt' then begin
     w = dblarr(nStationScore, 1, 6*nThres)
     nameVariables = 'GeoLon GeoLat pod_1 far_1 hss_1 pod_ind_1 far_ind_1 hss_ind_1' $
                     + ' pod_2 far_2 hss_2 pod_ind_2 far_ind_2 hss_ind_2' $
                     + ' pod_3 far_3 hss_3 pod_ind_3 far_ind_3 hss_ind_3' $
                     + ' pod_4 far_4 hss_4 pod_ind_4 far_ind_4 hss_ind_4'
     fileout = 'score_map_dbdt.out'
  endif else if choice eq 'db' then begin
     w = dblarr(nStationScore, 1, 3*nThres)
     nameVariables = 'GeoLon GeoLat pod_1 far_1 hss_1 ' $
                     + ' pod_2 far_2 hss_2' $
                     + ' pod_3 far_3 hss_3' $
                     + ' pod_4 far_4 hss_4'
     fileout = 'score_map_db.out'
  endif

  for ithres=0,nThres-1 do begin
     threshold = thresholds(ithres)
     indexThres = where(abs(wlogScore(*,0)-threshold) le 1e-4, nStationFound)

     if nStationFound eq 0 then begin
        print, ' no stations found, skip'
        continue
     endif

     for iStation=0,nStationFound-1 do begin
        NameStation = wlogrownamesScore(indexThres(iStation))

        ;; find the location in GEO of the station in supermag.dat
        indexMag = where(wlogrownamesMag eq NameStation)

        x(iStation,0,0) = wlogMag(indexMag,1)
        x(iStation,0,1) = wlogMag(indexMag,0)
        if choice eq 'dbdt' then begin
           istart = 0+ithres*6
           w(iStation,0,istart:istart+5) = wlogScore(indexThres(iStation), nVarScore-6:*)
        endif  else if choice eq 'db' then begin
           istart = 0+ithres*3
           w(iStation,0,istart:istart+2) = wlogScore(indexThres(iStation), nVarScore-3:*)
        endif
     endfor

  endfor

  save_pict,fileout, 'skill score map', nameVariables, w, x, ndim=2, $
            gencoord=1
end
;==============================================================================
pro calc_kp_error, mydir=mydir, resdir=resdir, $
                   events=events, $
                   const=const, slope=slope

  ;; Calculate Kp error for multiple runs and events.
  ;; Fit a linear trend from run1: kpfit = slope*kp1 + const
  ;; Create one scatter plot for all events, and time series plots per event

  common getlog_param
  common log_data
  common plotlog_param

  if not keyword_set(mydir) then mydir='.'
  mydir += '/'

  if not keyword_set(resdir) then begin
     print,'Kp_error: keyword resdir is required'
     return
  endif
  
  ;; Use six events by default
  event_I = set_eventlist(events,mydir,resdir)

  ;; interpolate to observed times and collect data from all events
  for iEvent = 0,n_elements(event_I)-1 do begin
     ;; read in measured values
     event = event_I(iEvent)
     eventnumber = string(event,format='(i2.2)')
     eventnum    = strtrim(string(event),2)
     logfilename=mydir+'deltaB/'+ResDir+'/run*/Event'+eventnumber+'/geoindex*.log '+ $
                 mydir+'Kp/event_'+eventnumber+'.txt'
     read_log_data
     interpol_log,wlog,wlog1,kp,kp1,'kp',wlognames,wlognames1,logtime,timeunit='date'
     if event eq event_I(0) then begin
        kpall  = [kp]
        kp1all = [kp1]
     endif else begin
        kpall  = [kpall,kp]
        kp1all = [kp1all,kp1]
     endelse
  endfor

  ;; linear fit from simulated to observed kp
  n = n_elements(kpall)
  result   = linfit(kp1all, kpall)
  const    = result(0)
  slope    = result(1)
  kp1fit   = const + slope*kp1all ; detrended simulation Kp
  error    = total(abs(kpall-kp1all))/n
  errorfit = total(abs(kpall-kp1fit))/n
  print,'Best fit kp1 = ',slope,'*kp +', const
  print,'<|Kp-Kp1|>=', error,' <|Kp-Kp1fit|>=', errorfit
  set_device,mydir+'deltaB/'+ResDir+'/kp_scatter_all.eps', /square
  title = string(result,error,errorfit,format='("Fit const,slope=",f5.2,",",f5.2,"!C","Error orig,fit=",f5.2,",",f5.2)')
  plot, kpall, kp1all, xrange=[0,10], yrange=[0,10], $
        title=title, xtitle='Observed Kp', ytitle='Run1 Kp and fit (green)',$
        psym=1, thick=3,symsize=2
  oplot,kpall, kp1fit, psym=1, thick=3, symsize=2, color=150
  oplot,[0,10],[0,10], color=150, thick=3
  oplot,slope*[0,10]+const,[0,10], thick=3
  close_device, /pdf

  ;; plot the simulated Kp for all the runs and events
  colors=[255,50,250,150,200,100,25,220,125]
  timeunit = 'date'
  logfunc  = 'Kp'
  yranges = [[0,10]]
  for iEvent = 0, n_elements(event_I)-1 do begin
     event = event_I(iEvent)
     ;; read in measured values
     eventnumber = string(event,format='(i2.2)')
     eventnum    = strtrim(string(event),2)
     title = 'Event '+eventnum
     logfilename=mydir+'deltaB/'+ResDir+'/run*/Event'+eventnumber+'/geoindex*.log '+ $
                 mydir+'Kp/event_'+eventnumber+'.txt'
     read_log_data
     xrange = [logtime1(0), logtime1(-1)]
     set_device,mydir+'deltaB/'+ResDir+'/kp_plot_event'+eventnumber+'.eps', /land
     plot_log_data
     oplot,logtime1,slope*wlog1(*,8)+const,linestyle=2,thick=5
     close_device, /pdf
  endfor
  colors=[255,100,250,150,200,50,25,220,125] ; reset colors 

end

;==============================================================================
pro calc_dst_error, models=models, events=events, mydir=mydir, InputDir=InputDir

;; Calculate Dst (symH) error in nT
;; for the models listed in the 'models' string array
;; and save results into dst_error.txt

  common getlog_param
  common log_data
  common plotlog_param

  if not keyword_set(mydir)    then mydir='.'
  if not keyword_set(InputDir) then InputDir='Inputs'

  if strmid(mydir,mydir.strlen()-1)    ne '/' then mydir    = mydir+'/'
  if strmid(mydir,InputDir.strlen()-1) ne '/' then InputDir = InputDir+'/'

  modelnames = models

  ;; Default directory
  if n_elements(models) lt 1 then models=['Results']
  nmodel = n_elements(models)

  for imodel = 0, nmodel-1 do begin
     modelname = models[imodel]
     if(strpos(modelname,'deltaB/') ge 0) then modelname = strmid(modelname,strpos(modelname,'deltaB/')+7)
     modelnames[imodel] = strjoin(strsplit(modelname,'/',/extract),'_')
  endfor

  event_I = set_eventlist(events,mydir,models[0])

  errors = fltarr(nmodel)
  unit = 1
  openw, unit, 'dst_error.txt'
  printf, unit, 'L1 norm of Dst error in nT'
  printf, unit, 'event ', modelnames

  colors=[255,50,250,150,200,100,25,220,125]

  for iEvent = 0, n_elements(event_I)-1 do begin
     event=event_I(iEvent)
     ;; read in measured values, well, event number is in 2 digits
     ;; form now. 
     logfilename=mydir + InputDir + "/Event"+string(event,format='(i2.2)')+"/Dst.txt"
     logfilenameplot = logfilename
     legends = ['Observation']
     read_log_data
     wlog0 = wlog
     logtime0 = logtime
     wlognames0 = wlognames

     for imodel = 0, nmodel-1 do begin
        model = models[imodel]

        ;; remove anything before deltaB
        if(strpos(model,'deltaB/') ge 0) then model = strmid(model,strpos(model,'deltaB/')+7)

        ;; in case of processing the old 1 digit form of the event number
        if model eq 'run_test' then begin
           if file_test('run_test/run_event'+string(event,format='(i2.2)')+'/GM/IO2/log*.log') then begin
              logfilename = 'run_test/run_event'+string(event,format='(i2.2)')+'/GM/IO2/log*.log'
           endif else if file_test('run_test/run_event'+string(event,format='(i1.1)')+'/GM/IO2/log*.log') then begin
              logfilename = 'run_test/run_event'+string(event,format='(i1.1)')+'/GM/IO2/log*.log'
           endif else begin
              print, "Error: No simulation file is found. Stop"
              retall
           endelse
           legends    = [legends, 'Simulation']
        endif else begin
           if file_test(mydir+'deltaB/'+model+'/Event'+string(event,format='(i2.2)')+'/log*.log') then begin
              logfilename= mydir+'deltaB/'+model+'/Event'+string(event,format='(i2.2)')+'/log*.log'
           endif else if file_test(mydir+'deltaB/'+model+'/Event'+string(event,format='(i1.1)')+'/log*.log') then begin
              logfilename= mydir+'deltaB/'+model+'/Event'+string(event,format='(i1.1)')+'/log*.log'
           endif else begin
              print, "Error: No simulation file is found. Stop"
              retall
           endelse

           legends    =	[legends, modelnames[imodel]]
        endelse
        read_log_data
        logfilenameplot += ' ' + logfilenames
        if wlognames[19] eq 'dst' then wlognames[19] = 'dst_sm'
        interpol_log,wlog0,wlog,dst0,dst,'dst_sm',wlognames0,wlognames,logtime0
        errors[imodel] = total(abs(dst0-dst)) / n_elements(dst)
     endfor
     logfilename = logfilenameplot
     read_log_data
     set_device, 'dst_plot_event'+string(event,format='(i2.2)')+'.eps', /land
     logfunc='dst_sm'
     legendpos = [-0.05,0.02,0.05,0.23]
     xtitle   = 'Hours from ' + set_date_string(wlog(0,0),wlog(0,1),wlog(0,2))
     ytitles  = ['Dst [nT]']
     plot_log_data
     close_device, /pdf
     printf, unit, event, errors, format='(i2,10f8.2)'
  endfor
  colors=[255,100,250,150,200,50,25,220,125] ; reset colors
  close,unit

end

;==============================================================================
pro dst_stat_nRun, mydir=mydir, ResDir=ResDir, events=events, InputDir=InputDir

  common getlog_param
  common log_data
  common plotlog_param

  if not keyword_set(mydir)    then mydir='.'
  if not keyword_set(InputDir) then InputDir='Inputs'

  if strmid(mydir,mydir.strlen()-1)    ne '/' then mydir    = mydir+'/'
  if strmid(mydir,InputDir.strlen()-1) ne '/' then InputDir = InputDir+'/'

  ;; check how many runs
  file_I = FILE_SEARCH(mydir+'/deltaB/'+ResDir+'/run?/dst_error.txt', count=nRun)

  if nRun le 2 then begin
     print, ' nRun ='+string(nRun,format='(i2)')+' is less than 2'
     print, ' skipping combining dst error tables'
     print, mydir+'/deltaB/'+ResDir
     return
  end

  ;; take the event list from run1
  event_I = set_eventlist(events,mydir,ResDir+'/run1')

  print, ' combining the dst error tables for ' + ResDir

  fileOut = mydir+'/deltaB/'+ResDir+'/dst_error_all.txt'

  header_out = 'event     '
  StrRun     = ''
  line       = ''

  for iRun=0, nRun-1 do begin
     file_dst = file_I(iRun)

     ;; ------------------ start reading dst file ----------------------------
     openr, lunRead, file_dst, /get_lun

     ;; two lines of header
     readf, lunRead, line
     readf, lunRead, line

     ;; read the first line of data
     readf, lunRead, line
     DataTmp = float(STRSPLIT(line, ' ',/extract))
     nData   = n_elements(DataTmp)

     if (nData ne 2) then begin
        print, ' nData /= 2 ?????'
        print, ' line = ', line
        exit
     endif

     ;; maximum 1000 events
     dst_II = dblarr(nData, 1000)

     iEvent = 0
     dst_II(*,iEvent) = DataTmp

     while not eof(lunRead) do begin
        readf, lunRead, line

        DataTmp = float(STRSPLIT(line, ' ',/extract))

        iEvent = iEvent + 1
        dst_II(*, iEvent) = DataTmp
     endwhile

     close, lunRead
     free_lun, lunRead
     ;; ------------------ end reading dst file ----------------------------

     nEvent = iEvent + 1
     dst_II = dst_II(*,0:nEvent-1)

     if (iRun eq 0) then begin
        dst_III = fltarr(nData, nEvent, nRun)
     endif

     StrRun = StrRun + 'run_' +string(iRun+1, format='(i1)') + '    '
     dst_III(*,*,iRun) = dst_II
  endfor

  header_out= header_out + StrRun + 'average   stddev'

  openw,  lun, fileOut, /get_lun
  printf, lun, 'L1 norm of Dst error in nT for ' + ResDir
  printf, lun, header_out
  for iEvent = 0,nEvent-1 do begin
     printf, lun, dst_III(0,iEvent,0), dst_III(1,iEvent,*), mean(dst_III(1,iEvent,*)), $
             stddev(dst_III(1,iEvent,*)), format='(i2,5x, 10(f8.2,1x))'
  endfor
  close, lun
  free_lun, lun

  ;; plot the simulated dst for all the runs
  colors=[255,50,250,150,200,100,25,220,125]

  for iEvent = 0, n_elements(event_I)-1 do begin
     event = event_I(iEvent)
     ;; set the observed Dst file info
     logfilename = mydir + InputDir + "/Event"+string(event,format='(i2.2)')+"/Dst.txt"

     logfilenameplot = logfilename
     legends = ['Observation']

     if file_test(mydir+'/deltaB/'+ResDir+'/run?/Event'+string(event,format='(i2.2)')+'/log*.log') then begin
        eventnumber = string(event,format='(i2.2)')
     endif else if file_test(mydir+'/deltaB/'+ResDir+'/run?/Event'+string(event,format='(i1.1)')+'/log*.log') then begin
        eventnumber = string(event,format='(i1.1)')
     endif else begin
        print, "Error in dst_stat_nRun: No logfile matching"
        print, mydir+'/deltaB/'+ResDir+'/run?/Event'+string(event,format='(i2.2)')+'/log*.log' 
        retall
     endelse

     logfilename = mydir+'/deltaB/'+ResDir+'/run?/Event'+eventnumber+'/log*.log'
     legends     = [legends, 'run'+string(indgen(nRun)+1,format='(i1)')]

     read_log_data
     logfilenameplot = logfilenameplot + ' ' + strjoin(logfilenames, ' ')

     logfilename = logfilenameplot
     read_log_data
     set_device, 'dst_plot_event'+string(event,format='(i2.2)')+'.eps', /land
     logfunc='dst_sm'
     legendpos = [-0.05,0.02,0.05,0.28]
     xtitle   = 'Hours from ' + set_date_string(wlog(0,0),wlog(0,1),wlog(0,2))
     ytitles  = ['Dst [nT]']
     plot_log_data
     close_device, /pdf
  endfor

  colors=[255,100,250,150,200,50,25,220,125] ; reset colors
end

;==============================================================================
pro score_stat_nRun, mydir=mydir, ResDir=ResDir

  ;; check how many runs
  dir_I = FILE_SEARCH(mydir+'/deltaB/'+ResDir+'/run?', count=nRun)

  if nRun le 2 then begin
     print, ' nRun ='+string(nRun,format='(i2)')+' is less than 2'
     print, ' skipping combining the skill score tables'
     return
  end

  ;; go to run1 so that the file_I do not contain the path info
  cd, mydir+'/deltaB/'+ResDir+'/run1'
  file_I = FILE_SEARCH('metric*txt', count=nFile)

  if nFile eq 0 then begin
     print, 'Warning: no metric*txt, skipping combining the skill score tables'
     return
  end

  ;; go back to the ResDir, where the combined tables are save
  cd, mydir+'/deltaB/'+ResDir

  print, ' combining the skil score tables for ' + ResDir

  for iFile = 0, nFile-1 do begin
     fileOut = mydir+'/deltaB/'+ResDir+'/'+file_I(iFile)

     for iRun=0, nRun-1 do begin
        fileIn = mydir+'/deltaB/'+ResDir+'/run'+string(iRun+1, format='(i1)')+'/'+file_I(iFile)

        get_log, fileIn, wlog, wlogname, headlines=headlines

        table_II = wlog

        siz = size(table_II)
        nThreshold = siz[1]
        nData      = siz[2]

        ;; create the 3D metric and the 2D output metric
        if iRun eq 0 then begin
           table_III    = fltarr(nThreshold, nData,    nRun)
           table_out_II = fltarr(nThreshold, nData*2-1)
        endif

        table_III(*,*,iRun) = table_II
     endfor

     for iThreshold = 0, nThreshold-1 do begin
        table_local_II  = reform(table_III(iThreshold,*,*), nData, nRun)

        ;; the first element is the threshold, no need to calc the mean/stddev
        table_out_II(iThreshold)    = table_local_II(0,0)

        ;; calc the average and std with nRun
        for iData = 1, nData-1 do begin
           table_out_II(iThreshold, iData)         = mean(  table_local_II(iData,*))
           table_out_II(iThreshold, iData+nData-1) = stddev(table_local_II(iData,*))
        endfor
     endfor

     openw,  lunOut, fileOut, /get_lun
     ;; print the header info
     printf, lunOut, 'model: '+ mydir + '/deltaB/'+ResDir
     for i=1, n_elements(headlines)-2 do begin
        printf, lunOut, headlines[i]
     endfor

     printf, lunOut, strjoin(wlogname, ' ') + ' ' + strjoin('d'+wlogname(1:*), ' ')

     for iThreshold = 0, nThreshold-1 do begin
        printf, lunOut, table_out_II(iThreshold,*), format='(100f10.4)'
     endfor

     close, lunOut
     free_lun, lunOut
  endfor
end

;==============================================================================

pro check_calc_all, models=models, events=events, mydir=mydir, InputDir=InputDir

  if (not file_test('Event*',/DIRECTORY)) then begin
     print, "Error: no simulation results in dir:" + models
     retall
  endif

  print, models
  print, events
  print, mydir
  print, InputDir
  
  calc_all_events, choice='dbdt', models=models, events=events, $
                   mydir=mydir, InputDir=InputDir
  print,'----------------------------------------------------'
  print,'calc all dbdt done.'
  print,'----------------------------------------------------'

  calc_all_events, choice='db', models=models, events=events, $
                   mydir=mydir, InputDir=InputDir
  print,'----------------------------------------------------'
  print,'calc all db done.'
  print,'----------------------------------------------------'

  calc_dst_error,     models=models, events=events, mydir=mydir, InputDir=InputDir
  print,'----------------------------------------------------'
  print,'calc_dst_error done.'
  print,'----------------------------------------------------'

  save_tables,        model =models, events=events, mydir=mydir
  print,'----------------------------------------------------'
  print,'save_tables done.'
  print,'----------------------------------------------------'

  save_tables_station, model=models, events=events, mydir=mydir
  print,'----------------------------------------------------'
  print,'save_tables_station done.'
  print,'----------------------------------------------------'

  make_2d_map, 'dbdt', model=models, mydir=mydir
  make_2d_map, 'db',   model=models, mydir=mydir
  print,'----------------------------------------------------'
  print,'plot_2d_map done.'
  print,'----------------------------------------------------'

end

;==============================================================================

pro check_calc_fft, models=models, events=events, mydir=mydir, InputDir=InputDir

  if (not file_test('Event*',/DIRECTORY)) then begin
     print, "Error: no simulation results in dir:" + models
     retall
  endif

  fft_all_events, models=models, events=events,    $
                  mydir=mydir, InputDir=InputDir

  print,'----------------------------------------------------'
  print,'fft done.'
  print,'----------------------------------------------------'
end

;==============================================================================

pro stat_nRun, events=events, mydir=mydir, ResDir=ResDir, InputDir=InputDir

  dst_stat_nRun,   events=events, mydir=mydir, ResDir=ResDir, InputDir=InputDir
  score_stat_nRun, mydir=mydir, ResDir=ResDir

end

;==============================================================================

pro check_compare_all, models, events=events, mydir=mydir

  save_comp_dbdt_tables, models, events=events, mydir=mydir, InputDir=InputDir
  save_comp_db_tables,   models, events=events, mydir=mydir, InputDir=InputDir
  calc_dst_error, models=models, events=events, mydir=mydir, InputDir=InputDir

end
