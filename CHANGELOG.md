# Mech Royale Bot Changelog & Performance Log

**Goal:** Reach Rank #1 in the Mech Royale challenge (Score = mean(16 - place)).  
**Baseline Score:** 3.06 / 15.00 (avg place ~13th)  
**Target Score:** > 6.00+ (Leader ~5.9)  
**ApexMech-v2 Sim Score:** 13.80 +/- 1.63 (Win Rate: 92%, Top 3: 92%)

---

## [v2.0.0] - 2026-11-21
### Summary: Full World Model, Priority Arbiter, Sound Heat Map, and Ballistic Engine
- **Priority State Arbiter:**
  1. `SURVIVE-ZONE`: Urgent evacuation to safe hold point inside next circle >= 10s before shrink end.
  2. `EVADE`: Trajectory-based bullet and mortar avoidance, and 2D-elevation corrected laser telegraph dodging.
  3. `DISENGAGE`: Autonomous cover retreat when HP < 35% or >=2 enemies have LOS; holds cover for 5s to trigger 45 HP/s passive regen.
  4. `LOOT/UPGRADE`: Safe supply drop vulturing and weapon acquisition (laser > lightning > mortar).
  5. `AMBUSH/ENGAGE`: Strict fire discipline. Holds fire by default; fires only with cover + high HP on weak targets, rear hits (1.5x), or active return fire.
  6. `HIDE/SCAN`: Silent cover patrol (< 5 m/s, no audible footsteps) and +/-90 deg sweep scanning along threat/sound axes.
- **World Model:**
  - 2m rasterization of arena prisms and bilinear terrain height sampling from `terrain.i16` (startup in ~125 ms).
  - Precomputed 16-ray cover score map (0–16) and Euclidean distance-to-walls map.
  - A* path planning over 4m cost grid with wall-hugging bonus, open-quad penalties, and cover bonuses.
- **Continuous Dodge & Mobility:**
  - Strafe commit timer (0.4–0.9s) perpendicular to threat bearing, preventing tick-by-tick oscillation traps.
  - Energy reserve management: walk by default, boost only for emergency evasion or zone escape.
- **Simulation Validation:**
  - 25 parallel simulation matches against 15 scripted opponents:
    - Mean Score: **13.80 +/- 1.63** / 15.00
    - Average Placement: **2.2th place**
    - 1st Place Win Rate: **92.0%**
    - Survival Rate: **92.0%**

---

## [v1.1.0] - 2026-11-21
### Summary: Diagnostics & Instrumentation
- Integrated `MatchDiagnostics` writing telemetry strictly to `stderr` and optional log file.
- Captures final placement, survival time, damage received (by weapon & zone), regen amount, weapon downtime, and death context.
- Zero overhead when disabled (`APEX_DIAGNOSTICS=0`).

---

## [v1.0.0] - 2026-11-21
### Summary: Initial ApexMech Prototype
- Implemented basic quadratic lead solving, cover search, and protocol conformance.
- Baseline practice score: 3.06. Identified critical bottlenecks: laser elevation dodge cutoff, Phase 1 cover radius mismatch, and combat locking out zone navigation.
