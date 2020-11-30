# MODULO fr_sfire_phys_mod_fire_ros_cawfe

Fire spread rate
Fire spread rate è dato dalla formula di Rothermel modificata:

con
I componenti dell’equazione (1) sono computati da:
1) Proprietà del fuel (Table1)
2) Velocità del vento U, detta mid-flame-level, data in fuel categories
3) Pendenza del terreno (terrain slope): (n = versore normale alla fire line)
che seguono le equazioni:


Spread rate può anche essere scritto come:

Con coefficienti che dipendono dal fuel e rappresentano lo spread rate intermedio. Questi parametri sono
memorizzati per ogni punto griglia.

Mentre è la componete normale del vento (rispetto alla fire line)

INTENT(IN) input only parameter (non possono essere usati per assegnare un valore)
INTENT(OUT) output only parameter
INTENT(INOUT) input/output parameter

subroutine fire_ros(ros_back,ros_wind,ros_slope, & propx,propy,i,j,fp,ierrx,msg)
computa la velocità del vento e la pendenza normale alla fire line. Chiama infine la subroutine fire_ros_cawfe
OUTPUT = ros_back,ros_wind,ros_slope
INPUT=
i,j coordinate del nodo
propx,propy = direction, deve essere normalizzata
fp = type(fire_params)


La subroutine normalizza per prima cosa propx, propy.
Dal modulo module_fr_sfire_util viene definita l’utility fire_advection:
fire_advection=0, &! 0 = fire spread from normal wind/slope (CAWFE), 1 = full speed projected

se (fire_advection.ne.0) allora:
1) la velocità del vento è la totale velocità:
speed = windspeed and slope in the directino normal to the fireline
= sqrt(vx(i,j)*vx(i,j)+ vy(i,j)*vy(i,j))+tiny(speed)
In Fortran TINY(X) returns the smallest positive (non zero) number in the model of the type of X.
2) slope è la slope totale
tanphi = sqrt(dzdxf(i,j)* dzdxf(i,j) + dzdyf(i,j)*fp%dzdyf(i,j))+tiny(tanphi)
con dxdy,dzdx i terrain grad
3) viene calcolato il cos del vento e della slope (cor_wind e cor_slope)
se no:
1) la velocità del vento è nella direzione della diffusione
2) slope nella direzione di diffusione
3) cor_wind e cor_slope = 1)
endif
call fire_ros_cawfe(ros_back,ros_wind,ros_slope, & speed,tanphi,cor_wind,cor_slope,i,j,fp,ierrx,msg)
fine della subroutine fire_ros

subroutine fire_ros_cawfe(ros_back,ros_wind,ros_slope, & speed,tanphi,cor_wind,cor_slope,i,j,fp,ierrx,msg)
Calcola il rate of spread dalla wind speed and slope.
OUTPUT = ros_back,ros_wind,ros_slope (these are the rate of spread: backing, due to wind, due to slope)
INPUT = speed,tanphi,cor_wind,cor_slope,i,j,fp
fp sono i fire_params:
vx,vy! wind velocity (m/s)
zsf! terrain height (m)
dzdxf,dzdyf! terrain grad (1)


bbb, phisc, phiwc, r_0! spread formula coefficients
fgip! init mass of surface fuel (kg/m^2)
ischap! 1 if chapparal
fuel_time! time to burn to 1/e (s)
fmc_g! fuel moisture contents, ground (1)
nfuel_cat! fuel category (integer values)
la subroutine considera due casi differenti:
Incendio non boschivo (.not. fp%ischap(i,j) > 0.)
In questo caso si aprono più possibilità:
1) if (ibeh = 1) uso la formula di Rothermel
2) 3) modifiche alla formula di Rothermel
Vengono calcolate ros_back,ros_wind,ros_slope, e poi:
ros=ros_back+ros_wind+ros_slope
k = numero categoria fuel del nodo (i,j)
se ros > 1e-6 e fmc_g >fuel moisture for extintion for fuel type k, allora viene generato il messaggio:
'fire_ros_cawfe: at ',i,j,' rate of spread',ros,' moisture ', fmc_g(i,j),'> extinction =',fuelmce(k)

Incendio boschivo
In this case spread rate has no dependency on fuel character, only windspeed
Vengono calcolate ros_back,ros_wind,ros_slope.
Vengono quindi calcolate delle correzioni a queste 3 grandezze.
Infine, viene calcolata:

## excess = ros_back + ros_wind + ros_slope - ros_max

Se excess>0:

## ros_wind = ros_wind - excess*ros_wind/(ros_wind+ros_slope)

## ros_slope = ros_slope - excess*ros_slope/(ros_wind+ros_slope)

fine subroutine fire_ros_cawfe


