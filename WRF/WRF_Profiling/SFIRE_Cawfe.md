![Logo](/Support_Materials/Assets/Logo_MLJC.png)
<h1 align="center">
  MODULO fr_sfire_phys_mod_fire_ros_cawfe
</h1>


FIRE SPREAD RATE
Fire spread rate comes from the modified Rothermel formula:
S = R_0(1+\phi W+\phi s)  (1)
with 
R_0 = \frac{I_R\varepsilon }{\rho_b \varepsilon Q_i_g)}

While the components of the equation (1) are computed from:
1) Fuel properties;
2) Wind speed component U, named mid-flame-level;
3) terrain slope: 
tan(\phi) = \bigtriangledown z\cdot n
(n = normal versor with respect of fire line)

Spread rate can also be provided from the equation:

S = max{S_0,R_0 + c min {e,max{0,U}^b} + d max {0,tan\phi}^2}

With all his coefficients derived from the fuel properties. These parameters are memorized for every grid point.

While U = U \cdot n is the normal component of wind with respect to the fire line.

Fortran code:
INTENT(IN) input only parameter (non possono essere usati per assegnare un valore)
INTENT(OUT) output only parameter
INTENT(INOUT) input/output parameter

subroutine fire_ros(ros_back,ros_wind,ros_slope, & propx,propy,i,j,fp,ierrx,msg)
computes the wind speed and the normal slope of the fire line. Finally, calls the fire_ros_cawfe subroutine.

OUTPUT = ros_back,ros_wind,ros_slope
INPUT=
1) i,j coordinates of the node
2) propx,propy = direction, must be normalized
3) fp = type(fire_params)

Firstly, the subroutine normalizes propx and propy.

From the module_fr_sfire_util module, the utility fire_advection is defined:

fire_advection=0, &! 0 = fire spread from normal wind/slope (CAWFE), 1 = full speed projected

if (fire_advection.ne.0):
1) the wind speed is the total velocity:
speed = windspeed and slope in the direction normal to the fireline
= sqrt(vx(i,j)*vx(i,j)+ vy(i,j)*vy(i,j))+tiny(speed)
In Fortran TINY(X) returns the smallest positive (non zero) number in the model of the type of X.

2) slope is the total slope.
tanphi = sqrt(dzdxf(i,j)* dzdxf(i,j) + dzdyf(i,j)*fp%dzdyf(i,j))+tiny(tanphi)
with dxdy,dzdx the terrain grad
3) calculates the cos of the wind and the slope (cor_wind and cor_slope)

if not:
1) the wind velocity is in the diffusion direction;
2) slope is in the diffusion direction;
3) cor_wind e cor_slope = 1)

endif
call fire_ros_cawfe(ros_back,ros_wind,ros_slope, & speed,tanphi,cor_wind,cor_slope,i,j,fp,ierrx,msg)

end of the fire_ros subroutine.


subroutine fire_ros_cawfe(ros_back,ros_wind,ros_slope, & speed,tanphi,cor_wind,cor_slope,i,j,fp,ierrx,msg)
Calculates the rate of spread of wind speed and slope.
OUTPUT = ros_back,ros_wind,ros_slope (these are the rate of spread: backing, due to wind, due to slope)
INPUT = speed,tanphi,cor_wind,cor_slope,i,j,fp
fp are the fire_params:
vx,vy! wind velocity (m/s)
zsf! terrain height (m)
dzdxf,dzdyf! terrain grad (1)


bbb, phisc, phiwc, r_0! spread formula coefficients
fgip! init mass of surface fuel (kg/m^2)
ischap! 1 if chapparal
fuel_time! time to burn to 1/e (s)
fmc_g! fuel moisture contents, ground (1)
nfuel_cat! fuel category (integer values)

The subroutines considers two different cases:

Not bushfires fire: (.not. fp%ischap(i,j) > 0.)
In this case there are two possibilities:
1) if (ibeh = 1) use Rothermel formula
2) 3) use the Rothermel formula modified.

ros_back,ros_wind,ros_slope are calculated, and than:
ros=ros_back+ros_wind+ros_slope
k = category number of fuel in the node (i,j)
if ros > 1e-6 and fmc_g >fuel moisture for extintion for fuel type k, so is generated the message:
'fire_ros_cawfe: at ',i,j,' rate of spread',ros,' moisture ', fmc_g(i,j),'> extinction =',fuelmce(k)

Bushfires fire:
In this case spread rate has no dependency on fuel character, only windspeed
ros_back,ros_wind,ros_slope are calculated.
Correcions of the 3 parameters are calculated:
Finally: is calculated:

## excess = ros_back + ros_wind + ros_slope - ros_max

If excess>0:

## ros_wind = ros_wind - excess*ros_wind/(ros_wind+ros_slope)

## ros_slope = ros_slope - excess*ros_slope/(ros_wind+ros_slope)

fine subroutine fire_ros_cawfe

###  References
<a id="1">[1]</a>
Rothermel, Richard C. (1972).
[_A Mathematical Model for Predicting Fire
Spread in Wildland Fires._](http://www.treesearch.fs.fed.us/pubs/32533)


