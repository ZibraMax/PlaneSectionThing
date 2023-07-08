import numpy as np
import matplotlib.pyplot as plt
import plyne

concrete_base = plyne.WhitneyConcrete(28000, color="gray")
concrete_mander = plyne.ConcreteMander(28000, color="lightgray")
steel_base = plyne.SimplifiedSteel(
    200000000, 420000, 1.25*420000, 0.05, color="red")
steel_rebar = plyne.RebarSteel(color="yellow")
sections = []

section = plyne.Rectangular(0.3, 0.6, concrete_base)
sections.append(section)

composite = plyne.Composite(sections)
composite.set_cover(0.05)
composite.add_rebar("2#8", 0.3-2*0.05, steel_rebar, y=0.6-0.05, n=8)
composite.add_rebar("2#8", 0.3-2*0.05, steel_rebar, n=8)
composite.mesh()
composite.show("fibers")
plt.grid()
plt.show()
