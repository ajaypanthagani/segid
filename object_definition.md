# An Observational Definition of an Object

## Object

An object is a maximal set of sensed patches held together by interfaces that have never been observed to open. Its identity persists through motion, articulation, contact and occlusion, and is irreversibly divided when the interfaces holding it together are observed to open.

Nothing is excluded from the world: floor, walls and ceiling are ordinary patches. The whole scene may begin as one object; objects split off as they are observed to move apart.

## 1. Primitives

Let time be discrete, t ∈ {1, 2, …, T}.

**Patches.** Pₜ is the finite set of patches known at time t, with Pₜ ⊆ Pₜ₊₁. Patches are small surface elements; their correspondence across time is given.

**Visible patches.** Vₜ ⊆ Pₜ is the set of patches observed at time t.

**First observation.** f(a) = min { s : a ∈ Vₛ }.

**Distance.** dₜ(a, b) is the measured distance between the surfaces of patches a and b at time t, defined only when both are visible and the measurement is reliable.

**Adjacency observation.** At each time t, every pair a, b ∈ Pₜ is in exactly one of three states:

- **Adjacent**, (a, b) ∈ Aₜ: a, b ∈ Vₜ and dₜ(a, b) ≤ ε.
- **Apart**, (a, b) ∈ Nₜ: a, b ∈ Vₜ and dₜ(a, b) > ε, persisting for at least k consecutive observations.
- **Undetermined:** otherwise, including whenever either patch is not visible or the distance cannot be measured reliably.

**First determination.** τ(a, b) = min { s : (a, b) ∈ Aₛ ∪ Nₛ }.

**Preference.** π(a, b) is a score ranking candidate interfaces (for example, a material-continuity confidence). It is used only when interfaces are created.

## 2. Interfaces

A pair is a **candidate interface** if it was adjacent when first determined:

Cand(a, b) ⇔ (a, b) ∈ A_τ(a,b)

An interface is **open** by time t if the pair has since been observed apart:

Openₜ(a, b) ⇔ ∃ s with τ(a, b) ≤ s ≤ t and (a, b) ∈ Nₛ

Interfaces are never recreated. A pair whose first determination was apart, or that touches again after opening, never has an interface.

## 3. Construction

Let Oₜ be the set of objects at time t and Iₜ the set of admitted interfaces, with O₀ = ∅ and I₀ = ∅. At each time t:

1. **Open.** Iₜ starts as Iₜ₋₁ with every interface (a, b) ∈ Nₜ removed.
2. **Candidates.** Collect every pair with τ(a, b) = t and Cand(a, b).
3. **Admit.** In decreasing order of π, admit each candidate into Iₜ unless it would place two patches that belonged to different objects of Oₜ₋₁ into the same connected component of (Pₜ, Iₜ).
4. **Objects.** Oₜ is the set of connected components of (Pₜ, Iₜ). A patch with no admitted interface is an object on its own.

## 4. Division and Parts

An object U ∈ Oₜ′, where t′ < t, is **divided by time t** if its patches lie in more than one object of Oₜ.

The **parts** of U at time t are the objects of Oₜ containing patches of U:

Parts(U, t) = { O ∈ Oₜ : O ∩ U ≠ ∅ }

Each part retains a reference to its parent U. If U is not divided, its only part is the object containing it.

## 5. Sensing Requirements

- **(S1) Adjacency is geometric.** Aₜ and Nₜ depend only on measured surface distance — never on material, colour, motion or category.
- **(S2) Conservative apartness.** A pair enters Nₜ only when the separation is clearly measured with both patches visible and persists for k observations. Opening is permanent, so doubt means undetermined.
- **(S3) Fine patches.** Patches are small enough that none spans two physical objects.

## 6. Invariants

1. **Permanent separation:** patches in different objects at t′ are in different objects at every t ≥ t′.
2. **Monotonicity:** patches in the same object at t were in the same object at every earlier time when both were known.
3. **Partition:** every known patch belongs to exactly one object.
4. **Forest:** every object at t containing previously known patches has exactly one parent at any earlier time; objects and their parent–part links form a forest over time.
5. **Descent:** the parts of a divided object are disjoint and their patches partition the parent's patches.
6. **Motion independence:** relative motion changes identity only by opening an interface.
7. **Articulation:** patches of an object may move relative to each other, or open in places, as long as a chain of unopened interfaces joins them.
8. **Contact independence:** contact formed after a pair's first determination never affects identity.
9. **Occlusion independence:** an undetermined pair never opens.
10. **History dependence:** identity is determined by the full observation history, not the current scene alone.

## 7. Consequences

| Case | Result |
|---|---|
| Table and floor, never moved | one object |
| Table lifted or dragged | table–floor interfaces open; table becomes a separate object |
| Mug first seen on a table | inside the table–floor object |
| Mug lifted | mug–table interface opens; mug becomes separate |
| Mug put back down | stays separate; no interface is recreated |
| Hand grasping a mug | no interface (apart when first determined); no chain to person or floor |
| Person standing on the floor | no interface (walked in); no chain |
| Pen (nib and body) | one object; the interface never opens |
| Door on its hinge, laptop lid, scissors | one object (articulation) |
| Blanket folding over itself | one object; new self-contact adds nothing |
| Drawer pulled out, torn paper | divided |
| Rod half hidden behind a post | one object; occlusion opens nothing |

**Limit.** Patches never observed apart remain one object, since nothing observed shows they can separate. Any need to list objects before they move (for example, a room inventory) belongs in a separate layer above this definition, using boundaries, categories and support as labelled hypotheses that never change identity.
