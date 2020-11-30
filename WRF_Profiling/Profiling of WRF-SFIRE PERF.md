![Logo](/Support_Materials/Assets/Logo_MLJC.png)

# Profiling of WRF-SFIRE with Linux’s perf tool

## Foreground: how does perf works and what does it’s output means

perf is a performance counter. As stated on it’s [homepage](https://perf.wiki.kernel.org/index.php/Main_Page):

```
Performance counters are CPU hardware registers that count hardware events such as
instructions executed, cache-misses suffered, or branches mispredicted. They form a basis
for profiling applications to trace dynamic control flow and identify hotspots. perf
provides rich generalized abstractions over hardware specific capabilities. Among others, it
provides per task, per CPU and per-workload counters, sampling on top of these and source
code event annotation.
```
The output is the CPU overhead. This is calculated as following:

_overhead_ =
_subroutineCPUtime
totalCPUtimeoftheprogram_
and is displayed in percentage.

We use the test case located in WRF-SFIRE/test/em_fire/hill as an example. The result of perf
report (after perf record ./wrf.exe) is:


By running perf report --stdio after perf -g ./wrf.exe (notice that we added -g)
we obtain:


By looking at these datas we concluded that:

1. the subroutines in the following list are responsible of much of the computational time
    required to run the model
2. after having checked the available documentation about WRF-SFIRE and WRF itself, we
    found the equations the code implements, and therefore we understood that only two of the
    most computationally expensive subroutines are actually solving differential equations
3. some of the CPU workload is due to unavoidable explicit floating point calculations
    (__powf_fma) and therefore we do not consider those routines
4. the computational effort of the model is widespread in many different modules, making it
    difficult to outline a true bottleneck in the code. Otherwisely stated, there is not such as a
    formula (subroutine) that slows down the model, but the load is distributed in many different
    subroutines
5. a significant portion of the physics-induced slowdown is due to the WRF model itself and
    not the FIRE model itself

The equations reported below are either from [1] or [2]

## __module_fr_sfire_phys_MOD_fire_ros_cawfe

WRF-SFIRE/phys/module_fr_fire_phys.F
line 1413
_subroutine fire_ros(ros_base,ros_wind,ros_slope, &
propx,propy,i,j,fp)_

calculates fire spread rate with McArthur formula or Rothermel using fuel type of fuel cell.

if (ibeh .eq. 1) then! use Rothermel formula
! ... if wind is 0 or into fireline, phiw = 0, &this reduces to backing
ros.
spdms = max(speed,0.)!
umidm = min(spdms,30.)! max input wind spd is 30 m/s!
param!
umid = umidm * 196.850! m/s to ft/min
! eqn.: phiw = c * umid**bbb(i,j) * (fp%betafl(i,j)/betaop)**(-e)!
wind coef
phiw = umid**fp%bbb(i,j) * fp%phiwc(i,j)! wind coef


phis=0.
if (tanphi .gt. 0.) then
phis = 5.275 *(fp%betafl(i,j))**(-0.3) *tanphi**2! slope factor
endif
! rosm = fp%r_0(i,j)*(1. + phiw + phis) * .00508! spread rate, m/s
ros_base = fp%r_0(i,j) *.
ros_wind = ros_base*phiw
ros_slope= ros_base*phis

## __module_fr_sfire_core_MOD_check_lfn_tign_ij

WRF-SFIRE/phys/module_fr_sfire_core.F
line 419
_subroutine check_lfn_tign_ij(i,j,s,time_now,lfnij,tignij)_

!*** purpose: check consistency of lfn and ignition

We had some troubling understanding what does this function does, how, why does it requires so
much CPU time and how to (if possible) speed it up.

## __module_small_step_em_MOD_advance_w

WRF-SFIRE/dyn_em/module_small_step_em.F
line 1178
_SUBROUTINE advance_w( w, rw_tend, ww, w_save, u, v, &
mu1, mut, muave, muts, &
c1h, c2h, c1f, c2f, &
c3h, c4h, c3f, c4f, &
t_2ave, t_2, t_1, &
ph, ph_1, phb, ph_tend, &
ht, c2a, cqw, alt, alb, &
a, alpha, gamma, &
rdx, rdy, dts, t0, epssm, &
dnw, fnm, fnp, rdnw, rdn, &
cf1, cf2, cf3, msftx, msfty,&
config_flags, top_lid, &
ids,ide, jds,jde, kds,kde, &! domain dims
ims,ime, jms,jme, kms,kme, &! memory dims
its,ite, jts,jte, kts,kte )! tile dims
! We have used msfty for msft_inv but have not thought through w equation
! pieces properly yet, so we will have to hope that it is okay
! We think we have found a slight error in surface w calculation_

advance_w advances the implicit w and geopotential equations.

! calculation of phi and w equations
! Comments on map scale factors:
! phi equation is:
! partial d/dt(rho phi/my) = -mx partial d/dx(phi rho u/my)
! -mx partial d/dy(phi rho v/mx)
! - partial d/dz(phi rho w/my) + rho g w/my
! as with scalar equation, use uncoupled value (here phi) to find the
! coupled tendency (rho phi/my)
! here as usual rho -> ~'mu'
!


! w eqn [divided by my] is:
! partial d/dt(rho w/my) = -mx partial d/dx(w rho u/my)
! -mx partial d/dy(v rho v/mx)
! - partial d/dz(w rho w/my)
! +rho[(u*u+v*v)/a + 2 u omega cos(lat) -
! (1/rho) partial dp/dz - g + Fz]/my
! here as usual rho -> ~'mu'
!
! 'u,v,w' sent here must be coupled variables (= rho w/my etc.) by their usage

## __module_fr_sfire_util_MOD_print_3d_stats

WRF-SFIRE/phys/module_fr_sfire_util.F
line 1197
_subroutine print_3d_stats_by_slice(ips,ipe,kps,kpe,jps,jpe, &
ims,ime,kms,kme,jms,jme, &
a,name)_

This subroutine prints to file the datas, the overhead is therefore not computational in nature

## __module_small_step_em_MOD_advance_uv

WRF-SFIRE/dyn_em/module_small_step_em.F
line 654
_SUBROUTINE advance_uv ( u, ru_tend, v, rv_tend, &
p, pb, &
ph, php, alt, al, mu, &
muu, cqu, muv, cqv, mudf, &
c1h, c2h, c1f, c2f, &
c3h, c4h, c3f, c4f, &
msfux, msfuy, msfvx, &
msfvx_inv, msfvy, &
rdx, rdy, dts, &
cf1, cf2, cf3, fnm, fnp, &
emdiv, &
rdnw, config_flags, spec_zone, &
non_hydrostatic, top_lid, &
ids, ide, jds, jde, kds, kde, &
ims, ime, jms, jme, kms, kme, &
its, ite, jts, jte, kts, kte )_

! advance_uv advances the explicit perturbation horizontal momentum


! equations (u,v) by adding in the large-timestep tendency along with
! the small timestep pressure gradient tendency.

Comments on map scale factors:
! x pressure gradient: ADT eqn 44, penultimate term on RHS
! = -(mx/my)*(mu/rho)*partial dp/dx
! [i.e., first rho->mu; 2nd still rho; alpha=1/rho]
! Klemp et al. splits into 2 terms:
! mu alpha partial dp/dx + partial dp/dnu * partial dphi/dx
! then into 4 terms:
! mu alpha partial dp'/dx + nu mu alpha' partial dmubar/dx +
! + mu partial dphi/dx + partial dphi'/dx * (partial dp'/dnu - mu')
!
! first 3 terms:
! ph, alt, p, al, pb not coupled
! since we want tendency to fit ADT eqn 44 (coupled) we need to
! multiply by (mx/my):

Comments on map scale factors:
! 4th term:
! php, dpn, mu not coupled
! since we want tendency to fit ADT eqn 44 (coupled) we need to
! multiply by (mx/my):

Comments on map scale factors:
! y pressure gradient: ADT eqn 45, penultimate term on RHS
! = -(my/mx)*(mu/rho)*partial dp/dy
! [i.e., first rho->mu; 2nd still rho; alpha=1/rho]
! Klemp et al. splits into 2 terms:
! mu alpha partial dp/dy + partial dp/dnu * partial dphi/dy
! then into 4 terms:
! mu alpha partial dp'/dy + nu mu alpha' partial dmubar/dy +
! + mu partial dphi/dy + partial dphi'/dy * (partial dp'/dnu - mu')
!
! first 3 terms:
! ph, alt, p, al, pb not coupled
! since we want tendency to fit ADT eqn 45 (coupled) we need to
! multiply by (my/mx):
! mudf_xy is NOT a map scale factor coupling
! it is some sort of divergence damping

Comments on map scale factors:
! 4th term:
! php, dpn, mu not coupled
! since we want tendency to fit ADT eqn 45 (coupled) we need to
! multiply by (my/mx):

### REFERENCES:

**[1]** Janice L. Coen et al.[_WRF-Fire: Coupled Weather–Wildland Fire Modeling with the Weather Research
and Forecasting Model_](https://doi.org/10.1175/JAMC-D-12-023.1)

**[2]** William C. Skamarock et al. [_A Description of the Advanced Research WRF Model Version 4._](http://dx.doi.org/10.5065/1dfh-6p97), NCAR Technical Notes NCAR/TN-556+STR


