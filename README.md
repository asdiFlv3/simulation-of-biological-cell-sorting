# simulation-of-biological-cell-sorting
Model originally from Graner, F. & Glazier, J. A. (1992) Phys. Rev. Lett. 69, 2013–2016.

cell sorting is a thermodynamic, simultaneous process, using Potts Model, driven by Monte Carlo step(MCS) Process 
## Monte Carlo Step
For temperature $T>0$, it looks lile *canonical ensemble* (partition function?)

$$
P(\sigma \to \sigma')=
\begin{cases} e^{ -\Delta \mathcal{H} /k_{B}T } & \Delta \mathcal{H} > 0 \\
1 & \Delta H \leq 0
\end{cases}
$$

cell at the boundary of the cluster will stochastically **attach** or **detach** by one lattice‐site per successful flip, driving the interface toward minimal. 
- If the energy is lowered, then the flip is always accepted
- If the energy $\Delta \mathcal{H}$ > 0, there is an acceptance probability. Algorithmly we just randomly choose a number from $[0,1]$ and if this number is smaller than $P_{\text{accept}}$ we accept the flip, otherwise rejected.
### Lattice anisotropy and neighbour range
- A simple nearest-neighbour square lattice pins boundaries along lattice directions, slowing sorting. Employing next-nearest neighbours reduces this anisotropy, giving more isotropic cell shapes and more biologically realistic dynamics

## Energy

*What is* $\Delta \mathcal{H}$?

$$
H_{sort} = \sum_{(i, j),(i',j')}J [ \tau(\sigma(i,j)),\tau (\sigma(i', j')) ] \left( 1- \delta_{\sigma,\sigma'}  \right) + \lambda \sum_{\sigma} \  [a(\sigma)-A_{\tau(\sigma)}]^{2} \theta(A_{\tau(\sigma)})
$$
 
 The first term is about **adhesion energy**
 
 - $J[\tau(\sigma),\tau(\sigma')]$: The *adhesion energy* per bond between a cell of type $\tau(\sigma)$ and one of type $\tau(\sigma')$.  Different types generally have higher interfacial cost. In our case we have:
	 - $\le J(d,d)\le\tfrac{J(d,d)+J(\ell,\ell)}2\le J(d,\ell)\le J(\ell,\ell)\le J(\ell,M)=J(d,M)$ such relationship
 - $1 - \delta_{\sigma,\sigma'}$: The Kronecker‐delta ensures *only* mismatched spins (i.e.\ true cell–cell boundaries) contribute.  If two neighbors have the **same** label, $\delta=1$ ⇒ no cost; if **different**, $\delta=0$ ⇒ cost $=J$.

 The second tern is about **area constraint**
 
- $a(\sigma)$ : area constraint. The *actual area* of cell $\sigma$ (i.e.\ the number of lattice sites carrying label $\sigma$).
- $A_{\tau(\sigma)}$ : The *target* (preferred) area for a cell of type $\tau(\sigma)$.  You often choose this so that all cells of a given type want roughly the same size. Biological aggregate are surrounded by fluid medium, ie $\tau = M$. We set target area $A_{M}$ negative
- $\bigl[a(\sigma)-A_{\tau(\sigma)}\bigr]^2$ : A *quadratic penalty* for deviating from the target area.
- $\lambda$: The Lagrange‐multiplier (or stiffness) that sets how strongly you enforce the area constraint.
	- **penalizes deviations** of each cell’s area from its target. A larger $\lambda$ makes it energetically very costly for a cell to grow or shrink away from target area​, effectively enforcing near‐constant cell size.

## Curret problem with the model
- Visualisation
- Took way too much steps to sort properly
