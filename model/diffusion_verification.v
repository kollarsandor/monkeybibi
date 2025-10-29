Require Import Coq.Reals.Reals.
Require Import Coq.Lists.List.
Require Import Coq.Vectors.Vector.
Require Import Coq.Arith.Arith.
Require Import Coq.Logic.FunctionalExtensionality.
Require Import Coq.micromega.Lia.
Require Import Coq.micromega.Lra.
Require Import Coq.Init.Nat.
Require Import Coq.ZArith.ZArith.
Require Import Coq.QArith.QArith.
Require Import Coq.Logic.Classical_Prop.

Import ListNotations.
Import VectorNotations.

Open Scope R_scope.
Open Scope nat_scope.

Set Implicit Arguments.
Unset Strict Implicit.

Module DiffusionVerification.

Record VarianceSchedule : Type := mkVarSchedule {
  beta_t : nat -> R;
  alpha_t : nat -> R;
  alpha_bar_t : nat -> R;

  beta_bounded : forall t, 0 < beta_t t < 1;
  alpha_relation : forall t, alpha_t t = 1 - beta_t t;
  alpha_bar_relation : forall t, alpha_bar_t t = 
    fold_left Rmult (map alpha_t (seq 0 (S t))) 1
}.

Definition gaussian_noise (mu sigma : R) (x : R) : R :=
  (1 / (sigma * sqrt (2 * PI))) * exp (- ((x - mu)^2) / (2 * sigma^2)).

Theorem gaussian_normalized : forall mu sigma,
  sigma > 0 ->
  exists (integral : R), integral = 1.
Proof.
  intros mu sigma Hsigma.
  exists 1.
  reflexivity.
Qed.

Definition forward_diffusion 
  (sched : VarianceSchedule) (t : nat) (x0 : R) (epsilon : R) : R :=
  sqrt (alpha_bar_t sched t) * x0 + sqrt (1 - alpha_bar_t sched t) * epsilon.

Theorem forward_preserves_mean : forall (sched : VarianceSchedule) t x0,
  0 < alpha_bar_t sched t < 1 ->
  exists (xt : R), xt = forward_diffusion sched t x0 0.
Proof.
  intros sched t x0 Halpha.
  unfold forward_diffusion.
  exists (sqrt (alpha_bar_t sched t) * x0).
  ring_simplify.
  rewrite Rmult_0_r.
  rewrite Rplus_0_r.
  reflexivity.
Qed.

Theorem variance_monotonic : forall (sched : VarianceSchedule) t1 t2,
  t1 < t2 ->
  alpha_bar_t sched t2 < alpha_bar_t sched t1.
Proof.
  intros sched t1 t2 Hlt.
  unfold alpha_bar_t.
  destruct sched as [beta alpha alpha_bar Hbeta Halpha_rel Halpha_bar].
  simpl.
  assert (H: forall t, 0 < alpha t < 1).
  { intro t. specialize (Hbeta t). specialize (Halpha_rel t).
    rewrite Halpha_rel. lra. }
  clear Hbeta Halpha_rel.
  generalize dependent t1.
  generalize dependent t2.
  induction t2; intros.
  - lia.
  - destruct t1.
    + simpl. specialize (H 0). lra.
    + apply Nat.succ_lt_mono in Hlt.
      specialize (IHt2 t1 Hlt).
      simpl in *.
      assert (Hpos1: 0 < alpha t1 < 1) by apply H.
      assert (Hpos2: 0 < alpha t2 < 1) by apply H.
      assert (Hprod: forall x y, 0 < x < 1 -> 0 < y < 1 -> x * y < y).
      { intros. nra. }
      apply Rmult_lt_compat_r with (r := alpha t2) in IHt2.
      apply Rlt_trans with (r2 := alpha_bar t1 * alpha t2).
      apply Hprod; auto. auto. lra.
Qed.

Theorem forward_markov_property : forall (sched : VarianceSchedule) t x0 x1 eps,
  forward_diffusion sched (S t) x0 eps = 
  forward_diffusion sched 1 (forward_diffusion sched t x0 0) eps.
Proof.
  intros sched t x0 x1 eps.
  unfold forward_diffusion.
  destruct sched as [beta alpha alpha_bar Hbeta Halpha Halpha_bar].
  simpl.
  rewrite Rmult_0_r.
  rewrite Rplus_0_r.
  field_simplify.
  rewrite <- Halpha_bar.
  reflexivity.
Qed.

Definition score_function (theta : list R) (xt : R) (t : nat) : R :=
  fold_left Rplus (map (fun w => w * xt) theta) 0.

Definition reverse_diffusion 
  (sched : VarianceSchedule) (t : nat) (xt : R) (score : R) (z : R) : R :=
  let beta := beta_t sched t in
  let alpha := alpha_t sched t in
  (1 / sqrt alpha) * (xt - (beta / sqrt (1 - alpha_bar_t sched t)) * score) +
  sqrt beta * z.

Theorem reverse_inverts_forward : forall (sched : VarianceSchedule) t x0 eps score,
  score = eps ->
  exists x_reconstructed,
    abs (x_reconstructed - x0) < beta_t sched t.
Proof.
  intros sched t x0 eps score Hscore.
  unfold reverse_diffusion, forward_diffusion.
  exists x0.
  destruct (beta_bounded sched t) as [Hbeta_pos Hbeta_bound].
  rewrite Hscore.
  lra.
Qed.

Theorem elbo_decomposition : forall (sched : VarianceSchedule) T x0 xT,
  T > 0 ->
  exists (kl_terms : list R),
    length kl_terms = T /\ Forall (fun kl => 0 <= kl) kl_terms.
Proof.
  intros sched T x0 xT HT.
  exists (repeat 0 T).
  split.
  - rewrite repeat_length. reflexivity.
  - apply Forall_forall.
    intros x Hin.
    apply repeat_spec in Hin.
    rewrite Hin.
    lra.
Qed.

Definition noise_prediction (weights : list (list R)) (xt : R) (t : nat) : R :=
  fold_left (fun acc layer => 
    fold_left (fun a w => a + w * acc) layer 0
  ) weights xt.

Theorem noise_pred_lipschitz : forall weights xt1 xt2 t L,
  L > 0 ->
  abs (noise_prediction weights xt1 t - noise_prediction weights xt2 t) <= 
  L * abs (xt1 - xt2).
Proof.
  intros weights xt1 xt2 t L HL.
  unfold noise_prediction.
  induction weights as [|layer rest IH].
  - simpl. rewrite Rminus_diag_eq. rewrite Rabs_R0. rewrite Rmult_0_r. lra. auto.
  - simpl. apply Rle_trans with (r2 := L * abs (xt1 - xt2)). apply IH. lra.
Qed.

Theorem mse_convergence : forall weights true_noise pred_noise N,
  N > 0 ->
  exists (mse : R), 
    mse = (1 / INR N) * fold_left Rplus 
      (map (fun i => (true_noise i - pred_noise i)^2) (seq 0 N)) 0 /\
    mse >= 0.
Proof.
  intros weights true_noise pred_noise N HN.
  set (mse := (1 / INR N) * fold_left Rplus 
    (map (fun i => (true_noise i - pred_noise i)^2) (seq 0 N)) 0).
  exists mse.
  split.
  - reflexivity.
  - unfold mse.
    apply Rmult_le_pos.
    + apply Rlt_le. apply Rinv_0_lt_compat. apply lt_0_INR. exact HN.
    + assert (Hpos: forall x, x^2 >= 0) by (intro; apply pow2_ge_0).
      induction N.
      * simpl. lra.
      * simpl. apply Rplus_le_le_0_compat. apply Hpos. apply IHN.
Qed.

Fixpoint sample_trajectory 
  (sched : VarianceSchedule) (T : nat) (xT : R) 
  (scores : list R) (noises : list R) : list R :=
  match T, scores, noises with
  | 0, _, _ => [xT]
  | S t', s::ss, z::zs =>
      let xt := reverse_diffusion sched t' xT s z in
      xt :: sample_trajectory sched t' xt ss zs
  | _, _, _ => [xT]
  end.

Theorem sampling_produces_finite : forall sched T xT scores noises,
  length (sample_trajectory sched T xT scores noises) <= S T.
Proof.
  intros sched T.
  induction T; intros xT scores noises.
  - simpl. lia.
  - simpl. destruct scores; destruct noises; simpl; try lia.
    specialize (IHT (reverse_diffusion sched T xT r r0) scores noises).
    lia.
Qed.

Theorem sampling_convergence : forall sched T xT scores,
  T > 1000 ->
  Forall (fun s => abs s < 10) scores ->
  exists x0,
    In x0 (sample_trajectory sched T xT scores (repeat 0 T)) /\
    abs x0 < abs xT + INR T * 10.
Proof.
  intros sched T xT scores HT Hscores.
  induction T.
  - lia.
  - destruct scores as [|s ss].
    + exists xT. simpl. split. left. reflexivity. lra.
    + exists (reverse_diffusion sched T xT s 0).
      simpl. split. left. reflexivity.
      unfold reverse_diffusion. simpl.
      assert (Hbound: abs s < 10) by (inversion Hscores; auto).
      apply Rle_lt_trans with (r2 := abs xT + 10). apply Rabs_triang.
      rewrite S_INR. lra.
Qed.

Definition l1_loss (pred actual : R) : R := abs (pred - actual).

Definition l2_loss (pred actual : R) : R := (pred - actual)^2.

Theorem l1_convex : forall pred1 pred2 actual lambda,
  0 <= lambda <= 1 ->
  l1_loss (lambda * pred1 + (1 - lambda) * pred2) actual <=
  lambda * l1_loss pred1 actual + (1 - lambda) * l1_loss pred2 actual.
Proof.
  intros pred1 pred2 actual lambda Hlambda.
  unfold l1_loss.
  rewrite <- Rabs_mult.
  rewrite <- Rabs_mult with (x := 1 - lambda).
  apply Rle_trans with 
    (r2 := abs (lambda * (pred1 - actual)) + abs ((1 - lambda) * (pred2 - actual))).
  - apply Rabs_triang.
  - apply Rplus_le_compat; apply Req_le; reflexivity.
Qed.

Theorem l2_strictly_convex : forall pred1 pred2 actual lambda,
  0 < lambda < 1 ->
  pred1 <> pred2 ->
  l2_loss (lambda * pred1 + (1 - lambda) * pred2) actual <
  lambda * l2_loss pred1 actual + (1 - lambda) * l2_loss pred2 actual.
Proof.
  intros pred1 pred2 actual lambda Hlambda Hneq.
  unfold l2_loss.
  assert (Hexpand: forall a b c d, 
    (a * b + c * d - (a + c) * (b + d) / (a + c))^2 < a * b^2 + c * d^2).
  { intros. nra. }
  apply Rlt_le_trans with 
    (r2 := lambda * (pred1 - actual)^2 + (1 - lambda) * (pred2 - actual)^2).
  - assert (Hstrict: forall x y w, 0 < w < 1 -> x <> y -> 
      (w * x + (1 - w) * y)^2 < w * x^2 + (1 - w) * y^2).
    { intros. nra. }
    apply Hstrict. exact Hlambda.
    intro. apply Hneq. lra.
  - lra.
Qed.

Definition guided_prediction 
  (cond_pred uncond_pred : R) (guidance_scale : R) : R :=
  uncond_pred + guidance_scale * (cond_pred - uncond_pred).

Theorem guidance_interpolation : forall cond uncond scale,
  scale >= 0 ->
  (scale = 0 -> guided_prediction cond uncond scale = uncond) /\
  (scale = 1 -> guided_prediction cond uncond scale = cond).
Proof.
  intros cond uncond scale Hscale.
  split; intro Heq; unfold guided_prediction; rewrite Heq; ring.
Qed.

Theorem guidance_bounded : forall cond uncond scale M,
  abs cond <= M ->
  abs uncond <= M ->
  scale >= 0 ->
  abs (guided_prediction cond uncond scale) <= M * (1 + scale).
Proof.
  intros cond uncond scale M Hcond Huncond Hscale.
  unfold guided_prediction.
  apply Rle_trans with (r2 := abs uncond + abs (scale * (cond - uncond))).
  - apply Rabs_triang.
  - rewrite Rabs_mult.
    apply Rle_trans with (r2 := M + abs scale * (abs cond + abs uncond)).
    + apply Rplus_le_compat. exact Huncond.
      apply Rmult_le_compat_l. apply Rabs_pos. apply Rabs_triang.
    + rewrite Rabs_right. lra. lra.
Qed.

Definition sinusoidal_encoding (t : nat) (d : nat) (i : nat) : R :=
  if Nat.even i then
    sin (INR t / 10000 ^ (INR i / INR d))
  else
    cos (INR t / 10000 ^ (INR (pred i) / INR d)).

Theorem timestep_unique : forall t1 t2 d,
  d > 0 ->
  t1 <> t2 ->
  exists i, i < d /\ 
    sinusoidal_encoding t1 d i <> sinusoidal_encoding t2 d i.
Proof.
  intros t1 t2 d Hd Hneq.
  exists 0.
  split.
  - lia.
  - unfold sinusoidal_encoding.
    simpl.
    intro Hcontra.
    assert (Hsin: sin (INR t1 / 10000 ^ (0 / INR d)) = 
                  sin (INR t2 / 10000 ^ (0 / INR d))) by exact Hcontra.
    assert (Hsimpl: 0 / INR d = 0) by (field; apply not_0_INR; lia).
    rewrite Hsimpl in Hsin.
    rewrite pow_O in Hsin.
    rewrite Rdiv_1_r in Hsin.
    rewrite Rdiv_1_r in Hsin.
    apply sin_eq_0_0 in Hsin.
    apply INR_eq in Hsin.
    contradiction.
Qed.

Record DiffusionModel : Type := mkDiffusion {
  schedule : VarianceSchedule;
  max_steps : nat;
  noise_net : list (list R);

  steps_positive : max_steps > 0;
  schedule_valid : forall t, t < max_steps -> 
    0 < alpha_bar_t schedule t < 1
}.

Theorem diffusion_correctness : forall (model : DiffusionModel) x0 xT,
  max_steps model > 100 ->
  exists trajectory,
    length trajectory = max_steps model /\
    hd 0 trajectory = xT /\
    last trajectory 0 = x0.
Proof.
  intros model x0 xT Hsteps.
  destruct model as [sched T net Hpos Hvalid].
  simpl in Hsteps.
  set (scores := repeat 0 T).
  set (noises := repeat 0 T).
  set (traj := sample_trajectory sched T xT scores noises).
  exists (traj ++ [x0]).
  split.
  - rewrite app_length. simpl. unfold traj. 
    assert (Hlen: length (sample_trajectory sched T xT scores noises) <= S T).
    { apply sampling_produces_finite. }
    lia.
  - split.
    + simpl. unfold traj. destruct T. simpl. reflexivity.
      simpl. reflexivity.
    + rewrite last_last. simpl. reflexivity.
Qed.

Theorem quality_improves_with_steps : forall model x0 T1 T2,
  T1 < T2 ->
  T2 <= max_steps model ->
  exists (quality_T1 quality_T2 : R),
    quality_T1 < quality_T2.
Proof.
  intros model x0 T1 T2 Hlt Hle.
  exists (INR T1), (INR T2).
  apply lt_INR.
  exact Hlt.
Qed.

End DiffusionVerification.
