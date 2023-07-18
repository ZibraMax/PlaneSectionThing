import numpy as np
import matplotlib.pyplot as plt
import plyne

concrete_old = plyne.WhitneyConcrete(16000, color="gray")
concrete_new = plyne.WhitneyConcrete(28000, color="lightgray")
steel_new = plyne.SimplifiedSteel(
    200000000, 420000, 1.0*420000, 0.05, color="red")
steel_old = plyne.SimplifiedSteel(
    200000000, 235000, 1.0*235000, 0.05, color="red")


b = 0.4
h = 0.4

nb = 0.6
nh = 0.6

x = (nb-b)/2
y = (nh-h)/2


sections = []

section = plyne.Rectangular(nb, nh, concrete_new, x=-x, y=-y)
sections.append(section)

section = plyne.Rectangular(b, h, concrete_old)
sections.append(section)


composite = plyne.Composite(sections)

cover = 0.05
dbar = plyne.REBAR_DIAMETERS["6"]

composite.set_cover(cover)

composite.add_rebar("3#6", nb-2*cover+dbar, steel_new,
                    x=-x+cover, y=h+y-cover+dbar/2, n=20)
composite.add_rebar(
    "3#6", nb-2*cover+dbar, steel_new, x=-x+cover, y=-y+cover-dbar/2, n=20)

composite.add_rebar_point("#6", -x+cover, h/2, steel_new, 20)
composite.add_rebar_point("#6", h+y-cover, h/2, steel_new, 20)
composite.show()
plt.show()

composite.mesh(0.001)
composite.show("fibers")
plt.show()

composite.interaction_diagram(plot=True, n=200)
# composite.moment_curvature(phimax=0.33, n=200)
