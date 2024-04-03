## 1. Glossary

| Variable | Definition |
| ---- | ---- |
|$time$|date (en seconde)|
| $latitude$ | position de l'avion en latitude (en radian) |
| $longitude$ |  position de l'avion en latitude (en radian) |
| $z_p$ | altitude-pression, équivalent altitude de la $P_s$  |
| $V_a$ | norme de la vitesse avion dans le repére air  |
| $m$ | masse de l'avion en Kg  |
| $\psi_g$ | cap suivi par l'avion par rapport au Nord  |
| $C_L$ | coefficient de portance  |
| $CAS$ | "Calibrated Air Speed", équivalent vitesse de la pression dynamique  |
| $Ps$ | pression statique de l'air en Pascal  |
| $Re$ | rayon terrestre moyen (6371km)  |
| $S$ | surface de référence pour les modèles aérodynamiques  |
| $Ts$ | température statique de l'air en Kelvin  |
| $g$ | gravité terrestre  |
| $L$ | portance de l'avion  |
| $D$ | trainée de l'avion  |
| $Thr$ | poussée des moteurs  |
| $\mathbf{V_a}$ | vecteur vitesse NED de l'avion dans le repére air   |
| $\mathbf{V_g}$ | vecteur vitesse NED de l'avion dans le repére sol   |
| $\mathbf{W}$ | vecteur vitesse NED du vent dans le repére sol   |
| $\phi$ | inclinaison latérale de l'avion  |
| $\gamma_a$ | pente de l'avion par rapport à l'horizontal dans le repére air ($>0$ si $V_{a,D}<0$) |
| $\gamma_g$ | pente de l'avion par rapport à l'horizontal dans le repére sol  ($>0$ si $V_{g,D}<0$) |
| $\rho$ | masse volumique de l'air  |
| $\theta_c$ | angle entre l'orientation de l'avion et la route suivie (dérapage)  |
| $\psi_a$ | angle de l'avion par rapport au Nord  |
| $\psi_w$ | angle du vent par rapport au nord  |
|  $\mathbf{grad}(y)$ | gradient de la grandeur y dans le repére NED  $\left(\frac{\partial{y}}{\partial{x_N}}, \frac{\partial{y}}{\partial{x_E}}, \frac{\partial{y}}{\partial{x_D}}\right)$  |


## 2. CRUISE_SPEED mode simulation

In CRUISE_SPEED mode, the equilibrium point is determined at aircraft state $(time, latitude, longitude, z_p, V_a, m, \psi_g)$ by:
- calling the atmospheric model to get $(Ts,Ps,\mathbf{W}, \mathbf{grad}(Ts),\mathbf{grad}(Ps),\mathbf{grad}(\mathbf{W}))$
- computing $\mathbf{V_a}$ and $\mathbf{V_g}$ knowing that $\frac{dz_p}{dt}|_{Tgt}=0$

$$
\left\{
\begin{array}{ll}
V_{g,H}=&\frac{W_N\cdot\cos(\psi_g)+W_E\cdot\sin\psi_g+a_0\cdot(v_0-W_D)}{1+a_0^2}\\
&é\sqrt{\left(\frac{W_N\cdot\cos(\psi_g)+W_E\cdot\sin\psi_g+a_0\cdot(v_0-W_D)}{1+a_0^2}\right)^2+\frac{V_a^2-(W_N^2+W_E^2+(v_0-W_D)^2)}{1+a_0^2}}\\
V_{g,N}=&V_{g,H}\cdot\cos(\psi_g)\\
V_{g,E}=&V_{g,H}\cdot\sin(\psi_g)\\
V_{g,D}=&v_0-V_{g,H}\cdot{a_0}\\
\end{array}
\right
$$

with $v_0=\frac{\frac{dz_p}{dt}|_{Tgt}-\frac{\partial{z_p}}{\partial{t}}}{\frac{\partial{z_p}}{\partial{x_D}}}, a_0=\frac{\frac{\partial{z_p}}{\partial{x_N}}\cdot\cos(\psi_g)+\frac{\partial{z_p}}{\partial{x_E}}\cdot\sin(\psi_g)}{\frac{\partial{z_p}}{\partial{x_D}}}
$

- computing the acceleration to maintain the Mach number
	$$\frac{dV_a}{dt}=\frac{1}{2}\cdot\frac{V_a}{Ts}\cdot\frac{dT_s}{dt}$$
- computing the derivative of the ground path angle to stay on the great circle:
$$\frac{d\psi_g}{dt}=\sqrt{V_{g,N}^2+V_{g,E}^2}\cdot\frac{\sin\psi_g\cdot\tan\delta}{R_e+z_{Geo}}$$
- computing the bank angle from the moment equation:
$$
\tan\phi=\frac{V_g}{g\cdot\cos\theta_c}\cdot\frac{d\psi_{g}}{dt}+\left(\left(\frac{dV_a}{dt}\cdot{V_a}-g\cdot{V_{a,D}}+\mathbf{\frac{dW}{dt}}\cdot\mathbf{V_a}\right)\cdot\frac{\sqrt{V_a^2-V_{a,D}^2}}{V_a}+{g}\cdot{V_{a,D}}\right)\cdot\frac{\tan\theta_c}{g\cdot V_a} 
$$ 
where $\theta_c=\psi_g-\psi_a$ and $\psi_a=\arctan\frac{V_{a,E}}{V_{a,N}}$
- computing the lift from the lift equation:
$$L = \frac{m\cdot{g}\cdot{\cos\gamma_a}}{\cos\phi} = \frac{1}{2}\cdot\rho\cdot{V_a^2}\cdot{S_{ref}}\cdot{C_L}$$
where $\cos\gamma_a = \frac{\sqrt{V_{a,N}^2+V_{a,E}^2}}{V_a}$ 
- calling the aerodynamics module to get the drag:
$$D = \frac{1}{2}\cdot\rho\cdot{V_a^2}\cdot{S_{ref}}\cdot{C_D}=\frac{1}{2}\cdot\rho\cdot{V_a^2}\cdot{S_{ref}}\cdot{polar(C_L)}$$
- computing the requested thrust from the acceleration equation knowing $V_g$, $D=polar(L)$ et $\frac{dV_a}{dt}$:
$$Thr=D+m\cdot\left(\frac{dV_a}{dt}-g\cdot{\frac{V_{a,D}}{V_a}}+\frac{1}{V_a}\cdot\mathbf{\frac{dW}{dt}}\cdot\mathbf{V_a} \right)$$

## 3. Pairing fuel saving

### Constant saving $K_{saving}$

the saving was integrated as a vertical wind component in the final thrust equation above:
$$
Thr=D+m\cdot\left(\frac{dV_a}{dt}-g\cdot{\frac{V_{a,D}}{V_a}}+\frac{1}{V_a}\cdot\mathbf{\frac{dW}{dt}}\cdot\mathbf{V_a} \right)-m\cdot g\cdot{\frac{W^{saving}_{D}}{V_a}}
$$
with 
$$
W^{saving}_{D}=K_{saving}\cdot{V_a}\cdot{ \frac{D}{L}}
$$

### BEN0 model
$$\Gamma_0=\frac{m_l\cdot{g}}{\rho\cdot{b_{v0}}\cdot{V_{a}}}$$
$$V_w=\frac{K_{decay} \cdot{\Gamma_0}}{2\pi}\cdot{\left(\frac{DY}{DY^2+(K_{r_c}\cdot b_l)^2}-\frac{DY+K_{b_v}\cdot b_l}{(DY+K_{b_v}\cdot b_l)^2+(K_{r_c}\cdot b_l)^2}\right)}$$
$$\overline{V_w}=\frac{1}{b_f}\cdot\int_{-\frac{b_f}{2}}^{\frac{b_f}{2}}V_w\left(DY+\frac{b_f}{2}+y\right)\cdot\sqrt{1-\frac{4y^2}{b_f^2}}\cdot{dy}=f(geom_{follower},DY)\cdot g(geom_{leader},DY) \cdot{\frac{m_l\cdot{g}}{\rho\cdot{V_{a}}}}$$
$$DFC_{BEN0}=-\frac{sfc}{V_a}\cdot{SR_R}\cdot{\frac{m_f\cdot{g}}{V_a}\cdot\overline{V_w}}\cdot{K_{correction}(DY)}$$

So I propose to integrate the BEN0 model as following, with the assumption that the impact of the vortex wind is neglictible on the speed components and the lift

$$
Thr=D+m\cdot\left(\frac{dV_a}{dt}-g\cdot{\frac{V_{a,D}}{V_a}}+\frac{1}{V_a}\cdot\mathbf{\frac{dW}{dt}}\cdot\mathbf{V_a} \right)-m\cdot g\cdot{\frac{\overline{V_w^{corr}}}{V_a}}
$$
with
$$\overline{V_w^{corr}}=\frac{m_l\cdot{g}}{\rho\cdot{V_{a}}}\cdot{f(geom_{follower},DY)}\cdot g(geom_{leader},DY) \cdot{K_{correction}(DY)}$$
