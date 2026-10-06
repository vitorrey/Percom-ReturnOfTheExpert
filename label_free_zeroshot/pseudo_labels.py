"""
pseudo_labels.py
----------------
LLM-authored per-limb activity profiles for MM-Fit, WEAR and PAMAP2
(paper Section VI; generation procedure in PROFILE_GENERATION.md).

Each ActivityProfile contains a 'limbs' dict mapping limb name
to a feature-level description (low / medium / high).

Limb names must correspond to the keys in MMFIT_LIMB_GROUPS / WEAR_LIMB_GROUPS
defined in dataset.py:

  MMFIT  →  "wrist"  (sw_l, sw_r),  "lower_body"  (sp_r),  "head"  (eb_l)
  WEAR   →  "arm"   (right_arm, left_arm),  "leg"   (right_leg, left_leg)

Feature keys match the PROFILE_KEYS order in dataset.py:
  rms, speed_mean, distance_total, impact_rate,
  tilt_mean, tilt_variability, angular_range, angular_rate

Levels
------
  "low"    ≈ bottom third of the dataset distribution
  "medium" ≈ middle third
  "high"   ≈ top third
"""

from dataclasses import dataclass, field
from typing import Dict, Optional


# -----------------------------------------------------------------------
# Data structure
# -----------------------------------------------------------------------

FEATURE_KEYS = (
    "rms", "speed_mean", "distance_total", "impact_rate",
    "tilt_mean", "tilt_variability", "angular_range", "angular_rate",
)


@dataclass
class ActivityProfile:
    """Per-limb feature-level description of a single activity class."""
    name:    str         # human-readable label
    label:   str         # dataset class key
    dataset: str         # "mmfit" | "wear"

    # limb_name → {feature_key → "low"/"medium"/"high"}
    limbs: Dict[str, Dict[str, str]]

    notes: str = ""

    def limb_names(self):
        return list(self.limbs.keys())

    def describe(self) -> str:
        """Natural-language description (for LLM-based pseudo-labelling)."""
        FEAT_LABELS = {
            "rms":              "Overall intensity",
            "speed_mean":       "Mean speed",
            "distance_total":   "Distance moved",
            "impact_rate":      "Impact / strike rate",
            "tilt_mean":        "Mean tilt (low = upright)",
            "tilt_variability": "Oscillation / repetitiveness",
            "angular_range":    "Range of motion",
            "angular_rate":     "Rotation speed",
        }
        lines = [f"Activity: {self.name}"]
        for limb, feats in self.limbs.items():
            lines.append(f"  [{limb}]")
            for k, v in feats.items():
                lines.append(f"    {FEAT_LABELS[k]}: {v}")
        if self.notes:
            lines.append(f"  Notes: {self.notes}")
        return "\n".join(lines)


# -----------------------------------------------------------------------
# MMFIT activities  (10 classes)
# -----------------------------------------------------------------------
#
# Sensor → limb mapping (see dataset.MMFIT_LIMB_GROUPS):
#   wrist       : sw_l_acc, sw_r_acc   (smartwatch left & right wrist)
#   lower_body  : sp_r_acc             (smartphone right pocket / hip)
#   head        : eb_l_acc             (earbuds left ear)
#
# Per-limb profiles capture the DISTINCT signature of each limb —
# the key insight is that arm-dominant exercises leave the lower body
# nearly still, while leg-dominant exercises leave the wrists inactive.

def _prof(wrist: Dict, lower_body: Dict,
          head: Optional[Dict] = None) -> Dict[str, Dict]:
    d = {"wrist": wrist, "lower_body": lower_body}
    if head is not None:
        d["head"] = head
    return d


MMFIT_ACTIVITIES: Dict[str, ActivityProfile] = {

    "squats": ActivityProfile(
        name="Squats", label="squats", dataset="mmfit",
        notes="Repetitive knee bend; large lower-body ROM, wrists relatively passive.",
        limbs=_prof(
            wrist      =dict(rms="medium", speed_mean="low",    distance_total="low",
                             impact_rate="low",    tilt_mean="medium", tilt_variability="medium",
                             angular_range="medium", angular_rate="medium"),
            lower_body =dict(rms="high",   speed_mean="medium", distance_total="medium",
                             impact_rate="low",    tilt_mean="medium", tilt_variability="high",
                             angular_range="high",   angular_rate="high"),
        ),
    ),

    "lunges": ActivityProfile(
        name="Lunges", label="lunges", dataset="mmfit",
        notes="Alternating forward step; step impact in lower body, wrists at sides.",
        limbs=_prof(
            wrist      =dict(rms="medium", speed_mean="low",    distance_total="low",
                             impact_rate="low",    tilt_mean="low",    tilt_variability="medium",
                             angular_range="medium", angular_rate="medium"),
            lower_body =dict(rms="high",   speed_mean="medium", distance_total="high",
                             impact_rate="medium", tilt_mean="medium", tilt_variability="high",
                             angular_range="high",   angular_rate="high"),
        ),
    ),

    "bicep_curls": ActivityProfile(
        name="Bicep curls", label="bicep_curls", dataset="mmfit",
        notes="Elbow flexion — dominant wrist signal; lower body nearly static.",
        limbs=_prof(
            wrist      =dict(rms="high",   speed_mean="medium", distance_total="medium",
                             impact_rate="low",    tilt_mean="medium", tilt_variability="high",
                             angular_range="high",   angular_rate="high"),
            lower_body =dict(rms="low",    speed_mean="low",    distance_total="low",
                             impact_rate="low",    tilt_mean="low",    tilt_variability="low",
                             angular_range="low",    angular_rate="low"),
        ),
    ),

    "situps": ActivityProfile(
        name="Sit-ups", label="situps", dataset="mmfit",
        notes="Trunk flexion; both limbs active — hands behind head, hips flex.",
        limbs=_prof(
            wrist      =dict(rms="medium", speed_mean="medium", distance_total="medium",
                             impact_rate="low",    tilt_mean="medium", tilt_variability="high",
                             angular_range="high",   angular_rate="medium"),
            lower_body =dict(rms="high",   speed_mean="medium", distance_total="medium",
                             impact_rate="low",    tilt_mean="high",   tilt_variability="high",
                             angular_range="high",   angular_rate="high"),
        ),
    ),

    "pushups": ActivityProfile(
        name="Push-ups", label="pushups", dataset="mmfit",
        notes="Prone position; wrists near horizontal pressing up, lower body near static.",
        limbs=_prof(
            wrist      =dict(rms="medium", speed_mean="medium", distance_total="low",
                             impact_rate="low",    tilt_mean="high",   tilt_variability="medium",
                             angular_range="medium", angular_rate="medium"),
            lower_body =dict(rms="low",    speed_mean="low",    distance_total="low",
                             impact_rate="low",    tilt_mean="high",   tilt_variability="low",
                             angular_range="low",    angular_rate="low"),
        ),
    ),

    "tricep_extensions": ActivityProfile(
        name="Tricep extensions", label="tricep_extensions", dataset="mmfit",
        notes="Arm extends overhead — wrist has high ROM; lower body still.",
        limbs=_prof(
            wrist      =dict(rms="medium", speed_mean="medium", distance_total="medium",
                             impact_rate="low",    tilt_mean="medium", tilt_variability="high",
                             angular_range="high",   angular_rate="high"),
            lower_body =dict(rms="low",    speed_mean="low",    distance_total="low",
                             impact_rate="low",    tilt_mean="low",    tilt_variability="low",
                             angular_range="low",    angular_rate="low"),
        ),
    ),

    "dumbbell_rows": ActivityProfile(
        name="Dumbbell rows", label="dumbbell_rows", dataset="mmfit",
        notes="Bent-over row; wrists pull high; lower body bent-forward static.",
        limbs=_prof(
            wrist      =dict(rms="high",   speed_mean="medium", distance_total="medium",
                             impact_rate="low",    tilt_mean="medium", tilt_variability="high",
                             angular_range="high",   angular_rate="high"),
            lower_body =dict(rms="low",    speed_mean="low",    distance_total="low",
                             impact_rate="low",    tilt_mean="high",   tilt_variability="low",
                             angular_range="low",    angular_rate="low"),
        ),
    ),

    "jumping_jacks": ActivityProfile(
        name="Jumping jacks", label="jumping_jacks", dataset="mmfit",
        notes="Full-body jump; only exercise with high impact AND high wrist activity.",
        limbs=_prof(
            wrist      =dict(rms="high",   speed_mean="high",   distance_total="high",
                             impact_rate="medium", tilt_mean="medium", tilt_variability="high",
                             angular_range="high",   angular_rate="high"),
            lower_body =dict(rms="high",   speed_mean="high",   distance_total="high",
                             impact_rate="high",   tilt_mean="low",    tilt_variability="high",
                             angular_range="high",   angular_rate="high"),
        ),
    ),

    "shoulder_press": ActivityProfile(
        name="Dumbbell shoulder press", label="shoulder_press", dataset="mmfit",
        notes="Arms press overhead; very high wrist ROM; lower body still.",
        limbs=_prof(
            wrist      =dict(rms="high",   speed_mean="high",   distance_total="high",
                             impact_rate="low",    tilt_mean="medium", tilt_variability="high",
                             angular_range="high",   angular_rate="high"),
            lower_body =dict(rms="low",    speed_mean="low",    distance_total="low",
                             impact_rate="low",    tilt_mean="low",    tilt_variability="low",
                             angular_range="low",    angular_rate="low"),
        ),
    ),

    "lateral_raises": ActivityProfile(
        name="Lateral shoulder raises", label="lateral_raises", dataset="mmfit",
        notes="Arms raise laterally; wrists active but slower than shoulder press.",
        limbs=_prof(
            wrist      =dict(rms="high",   speed_mean="medium", distance_total="medium",
                             impact_rate="low",    tilt_mean="medium", tilt_variability="high",
                             angular_range="high",   angular_rate="medium"),
            lower_body =dict(rms="low",    speed_mean="low",    distance_total="low",
                             impact_rate="low",    tilt_mean="low",    tilt_variability="low",
                             angular_range="low",    angular_rate="low"),
        ),
    ),

    "null": ActivityProfile(
        name="Non-activity", label="null", dataset="mmfit",
        notes="Resting or transition; minimal motion across all limbs.",
        limbs=_prof(
            wrist      =dict(rms="low",    speed_mean="low",    distance_total="low",
                             impact_rate="low",    tilt_mean="low",    tilt_variability="low",
                             angular_range="low",    angular_rate="low"),
            lower_body =dict(rms="low",    speed_mean="low",    distance_total="low",
                             impact_rate="low",    tilt_mean="low",    tilt_variability="low",
                             angular_range="low",    angular_rate="low"),
        ),
    ),
}


# -----------------------------------------------------------------------
# WEAR activities (19 classes incl. null)
# -----------------------------------------------------------------------
# Sensor → limb mapping (see dataset_wear.WEAR_LIMB_GROUPS):
#   arm : right_arm_acc, left_arm_acc
#   leg : right_leg_acc, left_leg_acc
#
# Note: WEAR accelerometer data is in g — converted to m/s² during loading.
# Key discriminators:
#   - All jogging variants share high leg impact; they differ on arm dynamics.
#   - Stretches are low-dynamics; differ on tilt_mean (reflects held posture angle).
#   - Prone exercises (push-ups, bench-dips) show high arm tilt_mean.
#   - Burpees are the only class with high dynamics on BOTH limbs simultaneously.

WEAR_ACTIVITIES: Dict[str, ActivityProfile] = {

    "jogging": ActivityProfile(
        name="Jogging", label="jogging", dataset="wear",
        notes="Steady-pace outdoor run; rhythmic heel-strike impact on legs, arms swing naturally.",
        limbs={
            "arm": dict(rms="medium", speed_mean="medium", distance_total="high",
                        impact_rate="low",    tilt_mean="low",    tilt_variability="medium",
                        angular_range="medium", angular_rate="medium"),
            "leg": dict(rms="high",   speed_mean="high",   distance_total="high",
                        impact_rate="high",   tilt_mean="low",    tilt_variability="high",
                        angular_range="high",   angular_rate="high"),
        },
    ),

    "jogging_rotating_arms": ActivityProfile(
        name="Jogging (rotating arms)", label="jogging_rotating_arms", dataset="wear",
        notes="Jogging with exaggerated bilateral arm circles; legs identical to baseline jog.",
        limbs={
            "arm": dict(rms="high",   speed_mean="high",   distance_total="high",
                        impact_rate="low",    tilt_mean="medium", tilt_variability="high",
                        angular_range="high",   angular_rate="high"),
            "leg": dict(rms="high",   speed_mean="high",   distance_total="high",
                        impact_rate="high",   tilt_mean="low",    tilt_variability="high",
                        angular_range="high",   angular_rate="high"),
        },
    ),

    "jogging_skipping": ActivityProfile(
        name="Jogging (skipping)", label="jogging_skipping", dataset="wear",
        notes="Skip-step gait; double-strike cadence, arms pump asymmetrically.",
        limbs={
            "arm": dict(rms="high",   speed_mean="medium", distance_total="high",
                        impact_rate="medium", tilt_mean="low",    tilt_variability="high",
                        angular_range="high",   angular_rate="high"),
            "leg": dict(rms="high",   speed_mean="high",   distance_total="high",
                        impact_rate="high",   tilt_mean="low",    tilt_variability="high",
                        angular_range="high",   angular_rate="high"),
        },
    ),

    "jogging_sidesteps": ActivityProfile(
        name="Jogging (sidesteps)", label="jogging_sidesteps", dataset="wear",
        notes="Lateral shuffle at jog pace; mediolateral leg displacement, arms low for balance.",
        limbs={
            "arm": dict(rms="medium", speed_mean="medium", distance_total="medium",
                        impact_rate="low",    tilt_mean="medium", tilt_variability="medium",
                        angular_range="medium", angular_rate="medium"),
            "leg": dict(rms="high",   speed_mean="high",   distance_total="high",
                        impact_rate="high",   tilt_mean="medium", tilt_variability="high",
                        angular_range="high",   angular_rate="high"),
        },
    ),

    "jogging_butt_kicks": ActivityProfile(
        name="Jogging (butt-kicks)", label="jogging_butt_kicks", dataset="wear",
        notes="Heel-to-glute kick drill; very high leg angular range, arm swing normal.",
        limbs={
            "arm": dict(rms="medium", speed_mean="medium", distance_total="high",
                        impact_rate="low",    tilt_mean="low",    tilt_variability="medium",
                        angular_range="medium", angular_rate="medium"),
            "leg": dict(rms="high",   speed_mean="high",   distance_total="high",
                        impact_rate="high",   tilt_mean="medium", tilt_variability="high",
                        angular_range="high",   angular_rate="high"),
        },
    ),

    "stretching_triceps": ActivityProfile(
        name="Stretching (triceps)", label="stretching_triceps", dataset="wear",
        notes="Static overhead tricep stretch; one arm held behind head, legs motionless.",
        limbs={
            "arm": dict(rms="low",    speed_mean="low",    distance_total="low",
                        impact_rate="low",    tilt_mean="high",   tilt_variability="low",
                        angular_range="low",    angular_rate="low"),
            "leg": dict(rms="low",    speed_mean="low",    distance_total="low",
                        impact_rate="low",    tilt_mean="low",    tilt_variability="low",
                        angular_range="low",    angular_rate="low"),
        },
    ),

    "stretching_lunging": ActivityProfile(
        name="Stretching (lunging)", label="stretching_lunging", dataset="wear",
        notes="Deep static lunge hold; legs at high tilt from split stance.",
        limbs={
            "arm": dict(rms="low",    speed_mean="low",    distance_total="low",
                        impact_rate="low",    tilt_mean="low",    tilt_variability="low",
                        angular_range="low",    angular_rate="low"),
            "leg": dict(rms="low",    speed_mean="low",    distance_total="low",
                        impact_rate="low",    tilt_mean="high",   tilt_variability="low",
                        angular_range="medium", angular_rate="low"),
        },
    ),

    "stretching_shoulders": ActivityProfile(
        name="Stretching (shoulders)", label="stretching_shoulders", dataset="wear",
        notes="Static cross-body shoulder hold; brief slow repositioning only.",
        limbs={
            "arm": dict(rms="low",    speed_mean="low",    distance_total="low",
                        impact_rate="low",    tilt_mean="medium", tilt_variability="low",
                        angular_range="low",    angular_rate="low"),
            "leg": dict(rms="low",    speed_mean="low",    distance_total="low",
                        impact_rate="low",    tilt_mean="low",    tilt_variability="low",
                        angular_range="low",    angular_rate="low"),
        },
    ),

    "stretching_hamstrings": ActivityProfile(
        name="Stretching (hamstrings)", label="stretching_hamstrings", dataset="wear",
        notes="Standing hamstring stretch; leg tilts significantly forward, arms reach toward foot.",
        limbs={
            "arm": dict(rms="low",    speed_mean="low",    distance_total="low",
                        impact_rate="low",    tilt_mean="medium", tilt_variability="low",
                        angular_range="low",    angular_rate="low"),
            "leg": dict(rms="low",    speed_mean="low",    distance_total="low",
                        impact_rate="low",    tilt_mean="high",   tilt_variability="low",
                        angular_range="medium", angular_rate="low"),
        },
    ),

    "stretching_lumbar_rotation": ActivityProfile(
        name="Stretching (lumbar rotation)", label="stretching_lumbar_rotation", dataset="wear",
        notes="Slow seated torso rotation; low speed but continuous angular arm motion.",
        limbs={
            "arm": dict(rms="low",    speed_mean="low",    distance_total="low",
                        impact_rate="low",    tilt_mean="medium", tilt_variability="medium",
                        angular_range="medium", angular_rate="low"),
            "leg": dict(rms="low",    speed_mean="low",    distance_total="low",
                        impact_rate="low",    tilt_mean="low",    tilt_variability="low",
                        angular_range="low",    angular_rate="low"),
        },
    ),

    "push_ups": ActivityProfile(
        name="Push-ups", label="push_ups", dataset="wear",
        notes="Standard prone push-up; arms near horizontal under load, legs flat and static.",
        limbs={
            "arm": dict(rms="medium", speed_mean="medium", distance_total="low",
                        impact_rate="low",    tilt_mean="high",   tilt_variability="medium",
                        angular_range="medium", angular_rate="medium"),
            "leg": dict(rms="low",    speed_mean="low",    distance_total="low",
                        impact_rate="low",    tilt_mean="high",   tilt_variability="low",
                        angular_range="low",    angular_rate="low"),
        },
    ),

    "push_ups_complex": ActivityProfile(
        name="Push-ups (complex)", label="push_ups_complex", dataset="wear",
        notes="Wide/rotating push-up variant; extra arm displacement and torso rotation.",
        limbs={
            "arm": dict(rms="high",   speed_mean="medium", distance_total="medium",
                        impact_rate="low",    tilt_mean="high",   tilt_variability="high",
                        angular_range="high",   angular_rate="high"),
            "leg": dict(rms="low",    speed_mean="low",    distance_total="low",
                        impact_rate="low",    tilt_mean="high",   tilt_variability="medium",
                        angular_range="low",    angular_rate="low"),
        },
    ),

    "sit_ups": ActivityProfile(
        name="Sit-ups", label="sit_ups", dataset="wear",
        notes="Standard floor crunch; repetitive trunk flexion, arms behind head.",
        limbs={
            "arm": dict(rms="medium", speed_mean="medium", distance_total="medium",
                        impact_rate="low",    tilt_mean="medium", tilt_variability="high",
                        angular_range="high",   angular_rate="medium"),
            "leg": dict(rms="medium", speed_mean="low",    distance_total="low",
                        impact_rate="low",    tilt_mean="high",   tilt_variability="high",
                        angular_range="high",   angular_rate="medium"),
        },
    ),

    "sit_ups_complex": ActivityProfile(
        name="Sit-ups (complex)", label="sit_ups_complex", dataset="wear",
        notes="Bicycle crunch; alternating elbow-to-knee rotation elevates arm angular rate.",
        limbs={
            "arm": dict(rms="high",   speed_mean="high",   distance_total="medium",
                        impact_rate="low",    tilt_mean="medium", tilt_variability="high",
                        angular_range="high",   angular_rate="high"),
            "leg": dict(rms="high",   speed_mean="medium", distance_total="medium",
                        impact_rate="low",    tilt_mean="high",   tilt_variability="high",
                        angular_range="high",   angular_rate="high"),
        },
    ),

    "burpees": ActivityProfile(
        name="Burpees", label="burpees", dataset="wear",
        notes="Floor-to-jump compound; highest intensity — both limbs simultaneously high.",
        limbs={
            "arm": dict(rms="high",   speed_mean="high",   distance_total="high",
                        impact_rate="medium", tilt_mean="medium", tilt_variability="high",
                        angular_range="high",   angular_rate="high"),
            "leg": dict(rms="high",   speed_mean="high",   distance_total="high",
                        impact_rate="high",   tilt_mean="medium", tilt_variability="high",
                        angular_range="high",   angular_rate="high"),
        },
    ),

    "lunges": ActivityProfile(
        name="Lunges", label="lunges", dataset="wear",
        notes="Forward step lunge; alternating leg impact and knee flexion, arms at sides.",
        limbs={
            "arm": dict(rms="medium", speed_mean="low",    distance_total="low",
                        impact_rate="low",    tilt_mean="low",    tilt_variability="medium",
                        angular_range="medium", angular_rate="low"),
            "leg": dict(rms="high",   speed_mean="medium", distance_total="medium",
                        impact_rate="medium", tilt_mean="medium", tilt_variability="high",
                        angular_range="high",   angular_rate="high"),
        },
    ),

    "lunges_complex": ActivityProfile(
        name="Lunges (complex)", label="lunges_complex", dataset="wear",
        notes="Multi-directional lunge; greater lateral displacement and arm counterbalance.",
        limbs={
            "arm": dict(rms="medium", speed_mean="medium", distance_total="medium",
                        impact_rate="low",    tilt_mean="medium", tilt_variability="medium",
                        angular_range="medium", angular_rate="medium"),
            "leg": dict(rms="high",   speed_mean="high",   distance_total="high",
                        impact_rate="medium", tilt_mean="medium", tilt_variability="high",
                        angular_range="high",   angular_rate="high"),
        },
    ),

    "bench_dips": ActivityProfile(
        name="Bench dips", label="bench_dips", dataset="wear",
        notes="Tricep dip off a bench; arms bear load behind body at high tilt, legs static.",
        limbs={
            "arm": dict(rms="medium", speed_mean="medium", distance_total="low",
                        impact_rate="low",    tilt_mean="high",   tilt_variability="medium",
                        angular_range="medium", angular_rate="medium"),
            "leg": dict(rms="low",    speed_mean="low",    distance_total="low",
                        impact_rate="low",    tilt_mean="low",    tilt_variability="low",
                        angular_range="low",    angular_rate="low"),
        },
    ),

    "null": ActivityProfile(
        name="Non-activity", label="null", dataset="wear",
        notes="Rest or transition between exercises; minimal motion across all limbs.",
        limbs={
            "arm": dict(rms="low",    speed_mean="low",    distance_total="low",
                        impact_rate="low",    tilt_mean="low",    tilt_variability="low",
                        angular_range="low",    angular_rate="low"),
            "leg": dict(rms="low",    speed_mean="low",    distance_total="low",
                        impact_rate="low",    tilt_mean="low",    tilt_variability="low",
                        angular_range="low",    angular_rate="low"),
        },
    ),
}


# -----------------------------------------------------------------------
# PAMAP2 activities (13 classes)
# -----------------------------------------------------------------------
# Sensor → limb mapping (see dataset_pamap2.PAMAP2_LIMB_GROUPS):
#   arm   : hand IMU  (wrist/forearm accelerometer)
#   torso : chest IMU
#   leg   : ankle IMU
#
# Key discriminators:
#   - Postures (lying/sitting/standing) are low-dynamics; differ on tilt_mean.
#   - Lying:           ALL sensors high tilt (body horizontal).
#   - Sitting:         torso/arm low tilt, leg MEDIUM (knee bent).
#   - Standing:        all low tilt and near-zero dynamics.
#   - Cycling:         high leg rotation but ZERO impact (no foot strike).
#   - Nordic walking:  arm as active as leg — both high (unlike plain walking).
#   - Rope jumping:    both arm (turning rope) and leg (landing) have high impact.
#   - Vacuum cleaning: arm dominant, slow leg shuffle.
#   - Ironing:         same as vacuuming but smaller arm stroke, leg static.

PAMAP2_ACTIVITIES: Dict[str, ActivityProfile] = {

    "other": ActivityProfile(
        name="Other/background", label="other", dataset="pamap2",
        notes="Unlabeled or transition frames; minimal or undefined motion.",
        limbs={
            "arm":   dict(rms="low",    speed_mean="low",    distance_total="low",
                          impact_rate="low",    tilt_mean="low",    tilt_variability="low",
                          angular_range="low",    angular_rate="low"),
            "torso": dict(rms="low",    speed_mean="low",    distance_total="low",
                          impact_rate="low",    tilt_mean="low",    tilt_variability="low",
                          angular_range="low",    angular_rate="low"),
            "leg":   dict(rms="low",    speed_mean="low",    distance_total="low",
                          impact_rate="low",    tilt_mean="low",    tilt_variability="low",
                          angular_range="low",    angular_rate="low"),
        },
    ),

    "lying": ActivityProfile(
        name="Lying", label="lying", dataset="pamap2",
        notes="Supine/prone rest; body is horizontal — all sensors show high tilt, zero dynamics.",
        limbs={
            "arm":   dict(rms="low",    speed_mean="low",    distance_total="low",
                          impact_rate="low",    tilt_mean="high",   tilt_variability="low",
                          angular_range="low",    angular_rate="low"),
            "torso": dict(rms="low",    speed_mean="low",    distance_total="low",
                          impact_rate="low",    tilt_mean="high",   tilt_variability="low",
                          angular_range="low",    angular_rate="low"),
            "leg":   dict(rms="low",    speed_mean="low",    distance_total="low",
                          impact_rate="low",    tilt_mean="high",   tilt_variability="low",
                          angular_range="low",    angular_rate="low"),
        },
    ),

    "sitting": ActivityProfile(
        name="Sitting", label="sitting", dataset="pamap2",
        notes="Seated rest; torso upright (low tilt), leg bent at knee (medium tilt), arm still.",
        limbs={
            "arm":   dict(rms="low",    speed_mean="low",    distance_total="low",
                          impact_rate="low",    tilt_mean="low",    tilt_variability="low",
                          angular_range="low",    angular_rate="low"),
            "torso": dict(rms="low",    speed_mean="low",    distance_total="low",
                          impact_rate="low",    tilt_mean="low",    tilt_variability="low",
                          angular_range="low",    angular_rate="low"),
            "leg":   dict(rms="low",    speed_mean="low",    distance_total="low",
                          impact_rate="low",    tilt_mean="medium", tilt_variability="low",
                          angular_range="low",    angular_rate="low"),
        },
    ),

    "standing": ActivityProfile(
        name="Standing", label="standing", dataset="pamap2",
        notes="Static upright posture; all sensors low tilt, near-zero dynamics across all limbs.",
        limbs={
            "arm":   dict(rms="low",    speed_mean="low",    distance_total="low",
                          impact_rate="low",    tilt_mean="low",    tilt_variability="low",
                          angular_range="low",    angular_rate="low"),
            "torso": dict(rms="low",    speed_mean="low",    distance_total="low",
                          impact_rate="low",    tilt_mean="low",    tilt_variability="low",
                          angular_range="low",    angular_rate="low"),
            "leg":   dict(rms="low",    speed_mean="low",    distance_total="low",
                          impact_rate="low",    tilt_mean="low",    tilt_variability="low",
                          angular_range="low",    angular_rate="low"),
        },
    ),

    "walking": ActivityProfile(
        name="Walking", label="walking", dataset="pamap2",
        notes="Steady gait; rhythmic leg impact and swing, moderate arm pendulum, torso stable.",
        limbs={
            "arm":   dict(rms="medium", speed_mean="medium", distance_total="medium",
                          impact_rate="low",    tilt_mean="low",    tilt_variability="medium",
                          angular_range="medium", angular_rate="medium"),
            "torso": dict(rms="low",    speed_mean="low",    distance_total="medium",
                          impact_rate="low",    tilt_mean="low",    tilt_variability="low",
                          angular_range="low",    angular_rate="low"),
            "leg":   dict(rms="high",   speed_mean="high",   distance_total="high",
                          impact_rate="medium", tilt_mean="low",    tilt_variability="high",
                          angular_range="high",   angular_rate="high"),
        },
    ),

    "running": ActivityProfile(
        name="Running", label="running", dataset="pamap2",
        notes="Fast locomotion; peak foot-strike impact, vigorous arm pump, high torso oscillation.",
        limbs={
            "arm":   dict(rms="high",   speed_mean="high",   distance_total="high",
                          impact_rate="medium", tilt_mean="low",    tilt_variability="high",
                          angular_range="high",   angular_rate="high"),
            "torso": dict(rms="high",   speed_mean="medium", distance_total="high",
                          impact_rate="high",   tilt_mean="low",    tilt_variability="high",
                          angular_range="medium", angular_rate="medium"),
            "leg":   dict(rms="high",   speed_mean="high",   distance_total="high",
                          impact_rate="high",   tilt_mean="low",    tilt_variability="high",
                          angular_range="high",   angular_rate="high"),
        },
    ),

    "cycling": ActivityProfile(
        name="Cycling", label="cycling", dataset="pamap2",
        notes="Pedalling; high continuous leg rotation — distinctively zero foot-strike impact.",
        limbs={
            "arm":   dict(rms="low",    speed_mean="low",    distance_total="low",
                          impact_rate="low",    tilt_mean="medium", tilt_variability="low",
                          angular_range="low",    angular_rate="low"),
            "torso": dict(rms="medium", speed_mean="low",    distance_total="medium",
                          impact_rate="low",    tilt_mean="medium", tilt_variability="medium",
                          angular_range="medium", angular_rate="medium"),
            "leg":   dict(rms="high",   speed_mean="high",   distance_total="high",
                          impact_rate="low",    tilt_mean="medium", tilt_variability="high",
                          angular_range="high",   angular_rate="high"),
        },
    ),

    "nordic walking": ActivityProfile(
        name="Nordic walking", label="nordic walking", dataset="pamap2",
        notes="Pole-assisted walking; arm is as dynamic as leg — both high, unlike plain walking.",
        limbs={
            "arm":   dict(rms="high",   speed_mean="high",   distance_total="high",
                          impact_rate="medium", tilt_mean="medium", tilt_variability="high",
                          angular_range="high",   angular_rate="high"),
            "torso": dict(rms="medium", speed_mean="medium", distance_total="high",
                          impact_rate="low",    tilt_mean="low",    tilt_variability="medium",
                          angular_range="medium", angular_rate="medium"),
            "leg":   dict(rms="high",   speed_mean="high",   distance_total="high",
                          impact_rate="high",   tilt_mean="low",    tilt_variability="high",
                          angular_range="high",   angular_rate="high"),
        },
    ),

    "ascending stairs": ActivityProfile(
        name="Ascending stairs", label="ascending stairs", dataset="pamap2",
        notes="Stair climb; high knee lift, forward torso lean, step impact on ankle.",
        limbs={
            "arm":   dict(rms="medium", speed_mean="medium", distance_total="medium",
                          impact_rate="low",    tilt_mean="low",    tilt_variability="medium",
                          angular_range="medium", angular_rate="medium"),
            "torso": dict(rms="medium", speed_mean="medium", distance_total="medium",
                          impact_rate="low",    tilt_mean="medium", tilt_variability="medium",
                          angular_range="medium", angular_rate="medium"),
            "leg":   dict(rms="high",   speed_mean="medium", distance_total="medium",
                          impact_rate="high",   tilt_mean="medium", tilt_variability="high",
                          angular_range="high",   angular_rate="high"),
        },
    ),

    "descending stairs": ActivityProfile(
        name="Descending stairs", label="descending stairs", dataset="pamap2",
        notes="Stair descent; heavy heel-strike landing impact, leg more extended than ascending.",
        limbs={
            "arm":   dict(rms="medium", speed_mean="medium", distance_total="medium",
                          impact_rate="low",    tilt_mean="low",    tilt_variability="medium",
                          angular_range="medium", angular_rate="medium"),
            "torso": dict(rms="medium", speed_mean="medium", distance_total="medium",
                          impact_rate="low",    tilt_mean="low",    tilt_variability="medium",
                          angular_range="medium", angular_rate="medium"),
            "leg":   dict(rms="high",   speed_mean="medium", distance_total="medium",
                          impact_rate="high",   tilt_mean="low",    tilt_variability="high",
                          angular_range="medium", angular_rate="high"),
        },
    ),

    "vacuum cleaning": ActivityProfile(
        name="Vacuum cleaning", label="vacuum cleaning", dataset="pamap2",
        notes="Forward/back arm push stroke while walking slowly; arm dominant, leg shuffles.",
        limbs={
            "arm":   dict(rms="high",   speed_mean="high",   distance_total="high",
                          impact_rate="low",    tilt_mean="medium", tilt_variability="high",
                          angular_range="high",   angular_rate="high"),
            "torso": dict(rms="medium", speed_mean="low",    distance_total="medium",
                          impact_rate="low",    tilt_mean="medium", tilt_variability="medium",
                          angular_range="medium", angular_rate="medium"),
            "leg":   dict(rms="medium", speed_mean="medium", distance_total="medium",
                          impact_rate="low",    tilt_mean="low",    tilt_variability="medium",
                          angular_range="medium", angular_rate="medium"),
        },
    ),

    "ironing": ActivityProfile(
        name="Ironing", label="ironing", dataset="pamap2",
        notes="Repetitive wrist-level arm strokes while standing; smaller ROM than vacuuming, leg static.",
        limbs={
            "arm":   dict(rms="medium", speed_mean="medium", distance_total="medium",
                          impact_rate="low",    tilt_mean="medium", tilt_variability="high",
                          angular_range="medium", angular_rate="medium"),
            "torso": dict(rms="low",    speed_mean="low",    distance_total="low",
                          impact_rate="low",    tilt_mean="medium", tilt_variability="low",
                          angular_range="low",    angular_rate="low"),
            "leg":   dict(rms="low",    speed_mean="low",    distance_total="low",
                          impact_rate="low",    tilt_mean="low",    tilt_variability="low",
                          angular_range="low",    angular_rate="low"),
        },
    ),

    "rope jumping": ActivityProfile(
        name="Rope jumping", label="rope jumping", dataset="pamap2",
        notes="Jump rope; arm turns rope (high rotation) while leg takes peak landing impacts.",
        limbs={
            "arm":   dict(rms="high",   speed_mean="high",   distance_total="high",
                          impact_rate="high",   tilt_mean="low",    tilt_variability="high",
                          angular_range="high",   angular_rate="high"),
            "torso": dict(rms="high",   speed_mean="medium", distance_total="high",
                          impact_rate="high",   tilt_mean="low",    tilt_variability="high",
                          angular_range="medium", angular_rate="medium"),
            "leg":   dict(rms="high",   speed_mean="high",   distance_total="high",
                          impact_rate="high",   tilt_mean="low",    tilt_variability="high",
                          angular_range="high",   angular_rate="high"),
        },
    ),
}


# -----------------------------------------------------------------------
# Unified lookup
# -----------------------------------------------------------------------

ALL_ACTIVITIES: Dict[str, ActivityProfile] = {
    **{f"mmfit/{k}":   v for k, v in MMFIT_ACTIVITIES.items()},
    **{f"wear/{k}":    v for k, v in WEAR_ACTIVITIES.items()},
    **{f"pamap2/{k}":  v for k, v in PAMAP2_ACTIVITIES.items()},
}


def get_profile(label: str, dataset: str) -> ActivityProfile:
    key = f"{dataset}/{label}"
    if key not in ALL_ACTIVITIES:
        raise KeyError(
            f"Unknown activity '{label}' in dataset '{dataset}'. "
            f"Available: {sorted(ALL_ACTIVITIES)}"
        )
    return ALL_ACTIVITIES[key]
