import numpy as np
import matplotlib.pyplot as plt
import plyne

concrete_base = plyne.WhitneyConcrete(28000, color="gray")
concrete_mander = plyne.ConcreteMander(28000, color="lightgray")
steel_base = plyne.SimplifiedSteel(
    200000000, 420000, 1.0*420000, 0.05, color="red")
steel_rebar = plyne.RebarSteel(color="yellow")
aluminum = plyne.AL6061T6(color="cyan")
sections = []

section = plyne.Rectangular(0.33, 0.63, concrete_mander)
section = plyne.Section(
    section.coords-np.array([0.03/2, 0.03/2]), aluminum, True)
sections.append(section)

section = plyne.Rectangular(0.3, 0.6, concrete_mander)
sections.append(section)


composite = plyne.Composite(sections)
composite.set_cover(0.05)
composite.add_rebar("2#6", 0.3-2*0.05, steel_rebar, y=0.6-0.05, n=20)
composite.add_rebar("2#8", 0.3-2*0.05, steel_rebar, n=20)
print(composite.h)
composite.show()
plt.show()
i = 1
n_mat = len(composite.materials)
fig = plt.figure()
for m in composite.materials:
    ax = fig.add_subplot(1, n_mat, i)
    m.show_ss_curve()
    ax.legend()
    ax.grid()
    i += 1
plt.tight_layout()
plt.show()

composite.mesh_traingle()
composite.show("fibers")
plt.show()

composite.moment_curvature(True)
plt.show()
