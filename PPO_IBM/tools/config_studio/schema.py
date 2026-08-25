"""
schema.py — declares which project constants config_studio can edit, and where each
one lives. This is the "content model" half of the Tina-CMS analogy: instead of blog
post fields, the editable schema here is the curriculum gate thresholds and the
scripted-expert control law, because those are the two knobs this project has actually
been tuning by hand (see experiments/bc_scaffold/).

Each field is one of two kinds:
  - "simple": a bare `NAME = <number>` assignment, matched by name.
  - "tuple2": a bare `NAME = (<a>, <b>)` assignment, exposed as two sub-fields.
  - "tier_dict": one entry inside curriculum_schedule.py's ADVANCE_TARGETS dict, keyed
    by (tier, dict-key-name).

Adding a field means adding one entry here — the server and UI both read this list,
nothing else needs touching.
"""

CURRICULUM_FILE = "training/curriculum_schedule.py"
EXPERT_FILE = "bc/bc_pretrain.py"

FIELDS = [
    # ── Curriculum advancement gates (ADVANCE_TARGETS in curriculum_schedule.py) ──
    {"id": "d0_harvest", "label": "D0 min harvest (mg)", "group": "D0 gate",
     "file": CURRICULUM_FILE, "kind": "tier_dict", "tier": 0, "key": "min_median_harvested_mg"},
    {"id": "d0_p25", "label": "D0 min p25 harvest (mg)", "group": "D0 gate",
     "file": CURRICULUM_FILE, "kind": "tier_dict", "tier": 0, "key": "min_p25_harvested_mg"},
    {"id": "d0_crash", "label": "D0 max crash rate", "group": "D0 gate",
     "file": CURRICULUM_FILE, "kind": "tier_dict", "tier": 0, "key": "max_crash_rate"},
    {"id": "d0_od", "label": "D0 min time_avg_od", "group": "D0 gate",
     "file": CURRICULUM_FILE, "kind": "tier_dict", "tier": 0, "key": "min_median_time_avg_od"},

    {"id": "d1_harvest", "label": "D1 min harvest (mg)", "group": "D1 gate",
     "file": CURRICULUM_FILE, "kind": "tier_dict", "tier": 1, "key": "min_median_harvested_mg"},
    {"id": "d1_p25", "label": "D1 min p25 harvest (mg)", "group": "D1 gate",
     "file": CURRICULUM_FILE, "kind": "tier_dict", "tier": 1, "key": "min_p25_harvested_mg"},
    {"id": "d1_crash", "label": "D1 max crash rate", "group": "D1 gate",
     "file": CURRICULUM_FILE, "kind": "tier_dict", "tier": 1, "key": "max_crash_rate"},
    {"id": "d1_od", "label": "D1 min time_avg_od", "group": "D1 gate",
     "file": CURRICULUM_FILE, "kind": "tier_dict", "tier": 1, "key": "min_median_time_avg_od"},

    {"id": "d2_harvest", "label": "D2 min harvest (mg)", "group": "D2 gate",
     "file": CURRICULUM_FILE, "kind": "tier_dict", "tier": 2, "key": "min_median_harvested_mg"},
    {"id": "d2_p25", "label": "D2 min p25 harvest (mg)", "group": "D2 gate",
     "file": CURRICULUM_FILE, "kind": "tier_dict", "tier": 2, "key": "min_p25_harvested_mg"},
    {"id": "d2_crash", "label": "D2 max crash rate", "group": "D2 gate",
     "file": CURRICULUM_FILE, "kind": "tier_dict", "tier": 2, "key": "max_crash_rate"},
    {"id": "d2_od", "label": "D2 min time_avg_od", "group": "D2 gate",
     "file": CURRICULUM_FILE, "kind": "tier_dict", "tier": 2, "key": "min_median_time_avg_od"},

    # ── Demotion / capability-abort tuning (curriculum_schedule.py module constants) ──
    {"id": "capability_demotion_chunks", "label": "Capability demotion chunks", "group": "Demotion",
     "file": CURRICULUM_FILE, "kind": "simple", "name": "CAPABILITY_DEMOTION_CHUNKS", "cast": "int"},
    {"id": "demotion_crash_rate", "label": "Demotion crash rate", "group": "Demotion",
     "file": CURRICULUM_FILE, "kind": "simple", "name": "DEMOTION_CRASH_RATE", "cast": "float"},
    {"id": "mastery_window", "label": "Mastery window (episodes)", "group": "Demotion",
     "file": CURRICULUM_FILE, "kind": "simple", "name": "MASTERY_WINDOW", "cast": "int"},
    {"id": "mastery_required_streak", "label": "Mastery required streak", "group": "Demotion",
     "file": CURRICULUM_FILE, "kind": "simple", "name": "MASTERY_REQUIRED_STREAK", "cast": "int"},

    # ── Scripted-expert control law (bc/bc_pretrain.py) ──
    {"id": "expert_od_setpoint", "label": "OD setpoint", "group": "Expert law",
     "file": EXPERT_FILE, "kind": "simple", "name": "EXPERT_OD_SETPOINT", "cast": "float"},
    {"id": "expert_gain", "label": "Gain", "group": "Expert law",
     "file": EXPERT_FILE, "kind": "simple", "name": "EXPERT_GAIN", "cast": "float"},
    {"id": "expert_frac_cap", "label": "Harvest fraction cap", "group": "Expert law",
     "file": EXPERT_FILE, "kind": "simple", "name": "EXPERT_FRAC_CAP", "cast": "float"},
    {"id": "expert_stir_min", "label": "Stir min (rpm)", "group": "Expert law",
     "file": EXPERT_FILE, "kind": "tuple2", "name": "EXPERT_STIR_RANGE", "slot": 0, "cast": "float"},
    {"id": "expert_stir_max", "label": "Stir max (rpm)", "group": "Expert law",
     "file": EXPERT_FILE, "kind": "tuple2", "name": "EXPERT_STIR_RANGE", "slot": 1, "cast": "float"},
    {"id": "expert_light_min", "label": "Light min (umol)", "group": "Expert law",
     "file": EXPERT_FILE, "kind": "tuple2", "name": "EXPERT_LIGHT_RANGE", "slot": 0, "cast": "float"},
    {"id": "expert_light_max", "label": "Light max (umol)", "group": "Expert law",
     "file": EXPERT_FILE, "kind": "tuple2", "name": "EXPERT_LIGHT_RANGE", "slot": 1, "cast": "float"},
]

FIELDS_BY_ID = {f["id"]: f for f in FIELDS}
