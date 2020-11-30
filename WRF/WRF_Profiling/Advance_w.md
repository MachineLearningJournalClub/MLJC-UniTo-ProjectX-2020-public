![Logo](/Support_Materials/Assets/Logo_MLJC.png)
<h1 align="center">
  SFIRE Profiling
</h1>


How WRF calls SFIRE

The defined interface to SFIRE is between WRFV3/phys/module_fr_sfire_driver_wrf.F and subroutine
sfire_driver_em in WRFV3/phys/module_fr_sfire_driver.F
WRF calls sfire_driver_em once at initialization, and then (with slightly different arguments) in every time
step.
The arguments of sfire_driver_em consist of two structures (called derived types in Fortran), grid, which
contains all state, input, and output variables, and config_flags, with all variables read from file
namelist.input, and some array dimensions.

# __module_small_step_em_MOD_advance_w

SMALL_STEP code for the geometric height coordinate model
WRF-SFIRE/dyn_em/module_small_step_em.F
line 1178

## SUBROUTINE advance_w

advances the implicit w (vertical velocity) and geopotential equations.

INPUT =

1) config_flags (da grid_config_rec_type) : the config_flags are informations taken from the namelist_imput file.
The namelist_input file is the file from the user editable. Here the user specifies his preferencies and desired parameters for the simulation.

2) declarations for the stuff coming in:

- ids,ide, jds,jde, kds,kde (domain dimensions)

- ims,ime, jms,jme, kms,kme (memory dimensions)

- its,ite, jts,jte, kts,kte (tile dimensions)

All these are indices taken from:

!-- ids start index for i in domain (WRF/frame/module_domain.F)

!-- ide end index for i in domain

!-- jds start index for j in domain

!-- jde end index for j in domain

!-- kds start index for k in domain

!-- kde end index for k in domain

!-- ims start index for i in memory

!-- ime end index for i in memory

!-- jms start index for j in memory

!-- jme end index for j in memory


!-- kms start index for k in memory

!-- kme end index for k in memory

!-- its start index for i in tile

!-- ite end index for i in tile

!-- jts start index for j in tile

!-- jte end index for j in tile

!-- kts start index for k in tile

!-- kte end index for k in tile


3) Others input of which dimension is specified (I am still trying to understand from where these input came from):

dimensions( ims:ime , kms:kme , jms:jme ) :: top_lid

dimensions( ims:ime , kms:kme, jms:jme ) ::

rw_tend,ww,w_save,u,v,t_2,t_1,ph_1,phb,ph_tend,alpha,gamma,a,c2a,cqw, alb, alt

dimensions( ims:ime , jms:jme ) :: mu1,mut,muave,muts,ht,msftx,msfty

dimensions( kms:kme ) :: fnp,fnm,rdnw,rdn,dnw

rdx,rdy,dts,cf1,cf2,cf3,t0,epssm

dimensions( kms:kme ) :: c1h, c2h, c1f, c2f,c3h, c4h, c3f, c4f (questi sono paramentri specificati nella
WRF/Registry/registry.hyb_coord, ma non ho capito cosa sono)

OUTPUT =

t_2ave,w,ph

What the subroutine makes:
Possible boundary conditions in the namelist.input file:

## specified (max_dom) .false. specified boundary conditions (only can be used for to domain 1)

## nested (max_dom) .false. nested boundary conditions (must be set to .true. for nests)

## periodic_x (max_dom) .false. periodic boundary conditions in x direction

Domain, memory e tile dimensions are set according to the periodic boundary conditions in namelist.input.

The subroutine calculates pi = 4 * atan(1.)

Extracts dampcoeff from the namelist.input file

## dampcoef (max_dom) 0. damping coefficient (see damp_opt)

and calculates dampmag = dts * dampcoef

Extracts zdamp from the namelist.input file

## zdamp (max_dom) 5000 damping depth (m) from model top

and calculates hdepth= zdamp

Now, the calulation of phi and w equations is executed (equations 2.23 e 2.32 from WRF Users' Guide)

1) phi equation is:

partial d/dt(rho phi/my) = -mx partial d/dx(phi rho u/my)-mx partial d/dy(phi rho v/mx)-
partial d/dz(phi rho w/my) + rho g w/my

2) w eqn [divided by my] is:
partial d/dt(rho w/my) = -mx partial d/dx(w rho u/my)-mx partial d/dy(v rho v/mx)- partial d/dz(w
rho w/my)+rho[(u*u+v*v)/a + 2 u omega cos(lat) - (1/rho) partial dp/dz - g + Fz]/my

Now building up RHS of phi equation:
...


