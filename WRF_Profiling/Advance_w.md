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
config_flags (da grid_config_rec_type) : le config_flags sono quelle info prese dal file namelist_input.
Namelist.input è un file che l’user può modificare in base alle propie preferenze per impostare molti
parametri del modello.

declarations for the stuff coming in:

- ids,ide, jds,jde, kds,kde (domain dimensions)
- ims,ime, jms,jme, kms,kme (memory dimensions)
- its,ite, jts,jte, kts,kte (tile dimensions)
questi sono indici tratti da:
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

Ora alenco altri input di cui è specificata la dimensione (questi input non ho ancora capito da dove vengono
presi e soprattuto cosa sono):
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

Cosa fa la subroutine:
Possibili condizioni al contorno specificate in namelist.input (file che l’user può modificare prima di far
girare il modello):

## specified (max_dom) .false. specified boundary conditions (only can be used for to domain 1)

## nested (max_dom) .false. nested boundary conditions (must be set to .true. for nests)

## periodic_x (max_dom) .false. periodic boundary conditions in x direction

Domain, memory e tile dimensions vengono impostate in base alle condizioni al contorno periodiche
specificate in namelist.input
Calcola pi = 4 * atan(1.)
Estrae dampcoeff dal file namelist.input

## dampcoef (max_dom) 0. damping coefficient (see damp_opt)


e calcola dampmag = dts * dampcoef
Estrae zdamp dal file namelist.input

## zdamp (max_dom) 5000 damping depth (m) from model top

e calcola hdepth= zdamp

Esegue ora il calcolo delle quazioni phi e w (le equazioni 2.23 e 2.32)
1) phi equation is:
partial d/dt(rho phi/my) = -mx partial d/dx(phi rho u/my)-mx partial d/dy(phi rho v/mx)-
partial d/dz(phi rho w/my) + rho g w/my
2) w eqn [divided by my] is:
partial d/dt(rho w/my) = -mx partial d/dx(w rho u/my)-mx partial d/dy(v rho v/mx)- partial d/dz(w
rho w/my)+rho[(u*u+v*v)/a + 2 u omega cos(lat) - (1/rho) partial dp/dz - g + Fz]/my

Now building up RHS of phi equation:
...


