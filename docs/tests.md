# Tests and pass marks

Two kinds of check: the **test suite** (`python -m pytest -q`, in the container) checks that the code does
what TECHNICAL.md says; the **experiment pass marks** say beforehand what result counts as the method
working. The marks below were fixed after the probe (`scripts/probe_model.py`) and before the results were
read; [`results.md`](results.md) reports each as met or **missed**.

## 1. Test suite

Fast tests need only NumPy and torch; tests marked `sim` build the LIBERO scene through Kangaroo_Libero.

| Test | What it checks | Pass mark |
| --- | --- | --- |
| `test_encoding::test_shape_and_layout` | the encoding is [q̃, sin(2⁰πq̃), cos(2⁰πq̃), …], 63 numbers for 7 joints | exact |
| `test_encoding::test_raw_is_identity` | L = 0 gives the raw normalised angle | exact |
| `test_encoding::test_numpy_and_torch_agree` | the training (torch) and analysis (NumPy) encodings are the same | 1e-12 |
| `test_encoding::test_encoding_separates_nearby_angles` | one fine step (0.005 rad) moves the encoded state much more than the raw angle | > 5 × |
| `test_mdp::test_bins_levels_and_zero` | 7 levels, per-step moves 0, ±0.005, ±0.015, ±0.05 rad | exact |
| `test_mdp::test_transition_integrates_and_clips` | q̂ = clip(q + Δt v), deterministic, clipped at limits | exact |
| `test_mdp::test_reward_terms` | the bonus, saturation, rejection and speed terms; bounds | exact |
| `test_mdp::test_reward_prefers_closer_points` | reward strictly decreases with distance inside 0.5 m | strict |
| `test_mdp::test_shaping_is_potential_based` | the optional shaping term telescopes along any path (cannot change the optimal policy) | 1e-9 |
| `test_mdp::test_factored_argmax_equals_brute_force` | per-joint argmax = argmax over all 7⁷ = 823,543 joint actions | exact |
| `test_agent::test_td_target_is_double_q_and_clipped` | y = r + γ Q_target(s', argmax_online), clipped to [r_min, r_max]/(1-γ) | 1e-6 |
| `test_agent::test_update_reduces_loss_on_fixed_batch` | the gradient step fits the Bellman target | loss < 20% of start |
| `test_agent::test_polyak_moves_target_towards_online` | θ⁻ ← (1-τ)θ⁻ + τθ | exact |
| `test_agent::test_explorer_repeats_random_actions` | exploring joints repeat their random bin, others are greedy | exact |
| `test_agent::test_replay_her_recomputes_rewards` | hindsight goals are achieved points of the same episode; rewards recomputed for them | 1e-6 |
| `test_agent::test_replay_without_her_keeps_goals` | without relabelling every goal is the episode's own | exact |
| `test_agent::test_n_step_return_and_truncation` | the 3-step return, recomputed per step for the goal, truncated at the episode end; bootstrap state and discount | 1e-5 |
| `test_agent::test_sequential_stages_are_consistent` | sequential agent: single-stage and batched paths agree; the greedy action is each stage's argmax | 1e-6 |
| `test_agent::test_sequential_update_fits_targets` | sequential agent: the gradient step fits its targets | loss < 30% of start |
| `test_dp::test_value_iteration_satisfies_bellman_equation` | V = max_a [r + γV(s')] after convergence | 1e-7 |
| `test_dp::test_rejected_moves_stay_and_are_penalised` | lattice transitions: walls reject, limits clip | exact |
| `test_dp::test_optimal_policy_goes_around_the_wall` | the exact policy reaches a goal behind an obstacle | 1e-9 m |
| `test_sim::test_scene_facts` (sim) | the 7 joints, actuator kinds, 0.25 m placement, reach point 5 cm along the fingers | exact |
| `test_sim::test_fk_parity_with_simulator_model` (sim) | model reach point = simulator model reach point, 100 random poses | 1e-9 m |
| `test_sim::test_collision_test_matches_full_contact_detection` (sim) | filtered collision test = mj_forward contacts (arm vs table/robot), 200 poses | 0 disagreements |
| `test_sim::test_physics_adapter_follows_the_command` (sim) | 20 decisions at +0.3 rad/s: q_ref moves 0.3 rad; the simulated elbow tracks it | 1e-9 / 0.02 rad |
| `test_sim::test_object_appears_under_the_goal` (sim) | the bowl is placed under the goal at its rest height | 1 cm |
| `test_sim::test_training_smoke` (sim) | 2,000 training steps write metrics and a checkpoint | runs |

Last run: 32 passed (26 fast, 6 sim), 17 s.

## 2. Experiment pass marks

| Check | Pass mark | Why this mark |
| --- | --- | --- |
| Model, final checkpoint (median over 3 seeds, 256 goals chained) | median ≤ 1 cm, p90 ≤ 3 cm, ≥ 85% within 2 cm | the binned Jacobian controller reaches 0.2 cm median: the bins allow it |
| Simulator, modes A and B (object goals) | median ≤ 1.5 cm | tracking adds about 0.5 cm (probe: 0.005 rad peak at the shoulder) |
| Gap between modes A and B | median ≤ 0.5 cm | velocity is not in the state; the gap measures what that costs |
| 2-joint exact check, final checkpoint | ≥ 90% of states with a near-optimal greedy action | the learning rule must solve the stated Bellman equation |
| Accuracy against training | final checkpoint better than the 100 k one in every seed | more training must help |
| Bowl displacement in the simulator | median < 1 cm | the hand aims at the object without touching it |
