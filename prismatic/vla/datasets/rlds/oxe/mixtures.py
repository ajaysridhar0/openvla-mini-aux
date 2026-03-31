"""
mixtures.py

Defines a registry of dataset mixtures and weights for the Open-X Embodiment Datasets. Each dataset is associated with
a float "sampling weight"
"""

from typing import Dict, List, Tuple

# fmt: off
OXE_NAMED_MIXTURES: Dict[str, List[Tuple[str, float]]] = {
    # === Bridge V2 Dataset ===
    "bridge": [
        # ("bridge_oxe", 1.0),                                    # Version of Bridge V2 in Open-X GCP Bucket
        ("bridge_orig", 1.0),                                   # Original Version of Bridge V2 from Project Website
    ],


    # === [Moderate-Scale] Bridge++ Mixtures ===
    "bridge_rt_1": [
        # ("bridge_oxe", 1.0)                                   # Version of Bridge V2 in Open-X GCP Bucket
        ("bridge_orig", 1.0),                                   # Original Version of Bridge V2 from Project Website

        ("fractal20220817_data", 1.0),                          # Google RT-1 Robot Data (Large-Scale)
    ],

    # === RT-X Mixtures ===
    "rtx": [
        ("fractal20220817_data", 0.54087122203),                # Google RT-1 Robot Data (Large-Scale)
        ("kuka", 0.8341046294),
        # ("bridge_oxe", 1.0)                                   # Version of Bridge V2 in Open-X GCP Bucket
        ("bridge_orig", 1.0),                                   # Original Version of Bridge V2 from Project Website
        ("taco_play", 2.0),
        ("jaco_play", 2.0),
        ("berkeley_cable_routing", 3.0),
        ("roboturk", 1.0),
        # ("nyu_door_opening_surprising_effectiveness", 5.0),   # Note --> only contains wrist camera images (skip?)
        ("viola", 2.0),
        ("berkeley_autolab_ur5", 1.0),
        ("toto", 1.0),
    ],

    "rtx_franka": [
        ("fractal20220817_data", 0.54087122203),                # Google RT-1 Robot Data (Large-Scale)
        ("kuka", 0.8341046294),
        # ("bridge_oxe", 1.0)                                   # Version of Bridge V2 in Open-X GCP Bucket
        ("bridge_orig", 1.0),                                   # Original Version of Bridge V2 from Project Website
        ("taco_play", 2.0),
        ("jaco_play", 2.0),
        ("berkeley_cable_routing", 3.0),
        ("roboturk", 1.0),
        # ("nyu_door_opening_surprising_effectiveness", 5.0),   # Note --> only contains wrist camera images (skip?)
        ("viola", 2.0),
        ("berkeley_autolab_ur5", 1.0),
        ("toto", 1.0),

        ("taco_play", 1.0),
        ("berkeley_cable_routing", 1.0),
        ("viola", 1.0),
        ("toto", 1.0),
        ("stanford_hydra_dataset_converted_externally_to_rlds", 1.0),
        ("austin_buds_dataset_converted_externally_to_rlds", 3.0),
        ("nyu_franka_play_dataset_converted_externally_to_rlds", 3.0),
        ("maniskill_dataset_converted_externally_to_rlds", 0.1),
        ("furniture_bench_dataset_converted_externally_to_rlds", 0.1),
        ("cmu_franka_exploration_dataset_converted_externally_to_rlds", 5.0),
        ("austin_sailor_dataset_converted_externally_to_rlds", 1.0),
        ("austin_sirius_dataset_converted_externally_to_rlds", 1.0),
        ("berkeley_rpt_converted_externally_to_rlds", 1.0),
        ("kaist_nonprehensile_converted_externally_to_rlds", 3.0),
        ("stanford_robocook_converted_externally_to_rlds", 1.0),
        ("iamlab_cmu_pickup_insert_converted_externally_to_rlds", 1.0),
        ("utaustin_mutex", 1.0),
        ("cmu_play_fusion", 1.0),
    ],

    # === Open-X Magic Soup ===
    "oxe_magic_soup": [
        ("fractal20220817_data", 0.54087122203),                # Google RT-1 Robot Data (Large-Scale)
        ("kuka", 0.8341046294),
        # ("bridge_oxe", 1.0)                                   # Version of Bridge V2 in Open-X GCP Bucket
        ("bridge_orig", 1.0),                                   # Original Version of Bridge V2 from Project Website
        ("taco_play", 2.0),
        ("jaco_play", 1.0),
        ("berkeley_cable_routing", 1.0),
        ("roboturk", 2.0),
        # ("nyu_door_opening_surprising_effectiveness", 1.0),   # Note --> only contains wrist camera images (skip?)
        ("viola", 2.0),
        ("berkeley_autolab_ur5", 2.0),
        ("toto", 1.0),
        ("language_table", 0.1),
        ("stanford_hydra_dataset_converted_externally_to_rlds", 2.0),
        ("austin_buds_dataset_converted_externally_to_rlds", 1.0),
        ("nyu_franka_play_dataset_converted_externally_to_rlds", 3.0),
        ("furniture_bench_dataset_converted_externally_to_rlds", 0.1),
        ("ucsd_kitchen_dataset_converted_externally_to_rlds", 2.0),
        ("austin_sailor_dataset_converted_externally_to_rlds", 1.0),
        ("austin_sirius_dataset_converted_externally_to_rlds", 1.0),
        # ("bc_z", 0.2),                                        # Note --> raw data is broken!
        ("dlr_edan_shared_control_converted_externally_to_rlds", 1.0),
        ("iamlab_cmu_pickup_insert_converted_externally_to_rlds", 1.0),
        # ("uiuc_d3field", 1.0),                                # Note --> raw data is broken!
        ("utaustin_mutex", 1.0),
        ("berkeley_fanuc_manipulation", 2.0),
        ("cmu_stretch", 1.0),
    ],

    # === Open-X Magic Soup++ ===
    "oxe_magic_soup_plus": [
        ("fractal20220817_data", 0.54087122203),                # Google RT-1 Robot Data (Large-Scale)
        ("kuka", 0.8341046294),
        ("bridge_orig", 1.0),                                   # Original Version of Bridge V2 from Project Website
        ("taco_play", 2.0),
        ("jaco_play", 1.0),
        ("berkeley_cable_routing", 1.0),
        ("roboturk", 2.0),
        ("viola", 2.0),
        ("berkeley_autolab_ur5", 2.0),
        ("toto", 1.0),
        ("language_table", 0.1),
        ("stanford_hydra_dataset_converted_externally_to_rlds", 2.0),
        ("austin_buds_dataset_converted_externally_to_rlds", 1.0),
        ("nyu_franka_play_dataset_converted_externally_to_rlds", 3.0),
        ("furniture_bench_dataset_converted_externally_to_rlds", 0.1),
        ("ucsd_kitchen_dataset_converted_externally_to_rlds", 2.0),
        ("austin_sailor_dataset_converted_externally_to_rlds", 1.0),
        ("austin_sirius_dataset_converted_externally_to_rlds", 1.0),
        ("dlr_edan_shared_control_converted_externally_to_rlds", 1.0),
        ("iamlab_cmu_pickup_insert_converted_externally_to_rlds", 1.0),
        ("utaustin_mutex", 1.0),
        ("berkeley_fanuc_manipulation", 2.0),
        ("cmu_stretch", 1.0),
        ## New Datasets in MagicSoup++
        ("bc_z", 0.2),                                          # Note: use v0.1.0 --> later versions broken
        ("fmb_dataset", 1.0),
        ("dobbe", 0.2),
        ("droid", 0.06),
    ],

    "oxe_magic_soup_plus_minus": [
        ("fractal20220817_data", 1.0),                          # Google RT-1 Robot Data (Large-Scale)
        ("kuka", 0.8341046294),
        ("bridge_orig", 1.0),                                   # Original Version of Bridge V2 from Project Website
        ("taco_play", 2.0),
        ("jaco_play", 1.0),
        ("berkeley_cable_routing", 1.0),
        ("roboturk", 2.0),
        ("viola", 2.0),
        ("berkeley_autolab_ur5", 2.0),
        ("toto", 1.0),
        # ("language_table", 0.1),
        ("stanford_hydra_dataset_converted_externally_to_rlds", 2.0),
        ("austin_buds_dataset_converted_externally_to_rlds", 1.0),
        ("nyu_franka_play_dataset_converted_externally_to_rlds", 3.0),
        ("furniture_bench_dataset_converted_externally_to_rlds", 0.1),
        ("ucsd_kitchen_dataset_converted_externally_to_rlds", 2.0),
        ("austin_sailor_dataset_converted_externally_to_rlds", 1.0),
        ("austin_sirius_dataset_converted_externally_to_rlds", 1.0),
        ("dlr_edan_shared_control_converted_externally_to_rlds", 1.0),
        ("iamlab_cmu_pickup_insert_converted_externally_to_rlds", 1.0),
        ("utaustin_mutex", 1.0),
        ("berkeley_fanuc_manipulation", 2.0),
        ("cmu_stretch", 1.0),
        ## New Datasets in MagicSoup++
        ("bc_z", 0.2),                                          # Note: use v0.1.0 --> later versions broken
        ("fmb_dataset", 1.0),
        ("dobbe", 0.2),
        # ("droid", 0.06),
    ],

    # === T-DROID Dataset ===
    "tdroid_carrot_in_bowl": [
        ("tdroid_carrot_in_bowl", 1.0),
    ],
    "tdroid_pour_corn_in_pot": [
        ("tdroid_pour_corn_in_pot", 1.0),
    ],
    "tdroid_flip_pot_upright": [
        ("tdroid_flip_pot_upright", 1.0),
    ],
    "tdroid_move_object_onto_plate": [
        ("tdroid_move_object_onto_plate", 1.0),
    ],
    "tdroid_knock_object_over": [
        ("tdroid_knock_object_over", 1.0),
    ],
    "tdroid_cover_object_with_towel": [
        ("tdroid_cover_object_with_towel", 1.0),
    ],

    # === DROID Finetuning Datasets ===
    "droid_wipe": [
        ("droid_wipe", 1.0),
    ],


    # === RoboCasa-X Release Mixtures ===
    "robocasa-x-xp3k-pnp": [
        ("robocasa_x_xp3k_pnp", 1.0),
    ],
    "robocasa-x-xp900-pnp": [
        ("robocasa_x_xp900_pnp", 1.0),
    ],
    "robocasa-x-xp3k-turn-on-sink-faucet": [
        ("robocasa_x_xp3k_turn_on_sink_faucet", 1.0),
    ],
    "robocasa-x-xp900-turn-on-sink-faucet": [
        ("robocasa_x_xp900_turn_on_sink_faucet", 1.0),
    ],
    "robocasa-x-xp3k-flip-mug-upright": [
        ("robocasa_x_xp3k_flip_mug_upright", 1.0),
    ],
    "robocasa-x-xp900-flip-mug-upright": [
        ("robocasa_x_xp900_flip_mug_upright", 1.0),
    ],
    "robocasa-x-target-panda-pnp": [
        ("robocasa_x_target_panda_pnp", 1.0),
    ],
    "robocasa-x-target-panda-og-pnp": [
        ("robocasa_x_target_panda_og_pnp", 1.0),
    ],
    "robocasa-x-target-jaco-pnp": [
        ("robocasa_x_target_jaco_pnp", 1.0),
    ],
    "robocasa-x-target-panda-turn-on-sink-faucet": [
        ("robocasa_x_target_panda_turn_on_sink_faucet", 1.0),
    ],
    "robocasa-x-target-panda-og-turn-on-sink-faucet": [
        ("robocasa_x_target_panda_og_turn_on_sink_faucet", 1.0),
    ],
    "robocasa-x-target-jaco-turn-on-sink-faucet": [
        ("robocasa_x_target_jaco_turn_on_sink_faucet", 1.0),
    ],
    "robocasa-x-target-panda-flip-mug-upright": [
        ("robocasa_x_target_panda_flip_mug_upright", 1.0),
    ],
    "robocasa-x-target-panda-og-flip-mug-upright": [
        ("robocasa_x_target_panda_og_flip_mug_upright", 1.0),
    ],
    "robocasa-x-target-jaco-flip-mug-upright": [
        ("robocasa_x_target_jaco_flip_mug_upright", 1.0),
    ],
    "robocasa-x-xp3k-panda-pnp-mix": [
        ("robocasa_x_xp3k_pnp", 0.01),
        ("robocasa_x_target_panda_pnp", 1.0),
    ],
    "robocasa-x-xp3k-panda-og-pnp-mix": [
        ("robocasa_x_xp3k_pnp", 0.01),
        ("robocasa_x_target_panda_og_pnp", 1.0),
    ],
    "robocasa-x-xp3k-jaco-pnp-mix": [
        ("robocasa_x_xp3k_pnp", 0.01),
        ("robocasa_x_target_jaco_pnp", 1.0),
    ],
    "robocasa-x-xp900-panda-pnp-mix": [
        ("robocasa_x_xp900_pnp", 0.05),
        ("robocasa_x_target_panda_pnp", 1.0),
    ],
    "robocasa-x-xp900-panda-og-pnp-mix": [
        ("robocasa_x_xp900_pnp", 0.05),
        ("robocasa_x_target_panda_og_pnp", 1.0),
    ],
    "robocasa-x-xp900-jaco-pnp-mix": [
        ("robocasa_x_xp900_pnp", 0.05),
        ("robocasa_x_target_jaco_pnp", 1.0),
    ],
    "robocasa-x-xp3k-panda-turn-on-sink-faucet-mix": [
        ("robocasa_x_xp3k_turn_on_sink_faucet", 0.01),
        ("robocasa_x_target_panda_turn_on_sink_faucet", 1.0),
    ],
    "robocasa-x-xp3k-panda-og-turn-on-sink-faucet-mix": [
        ("robocasa_x_xp3k_turn_on_sink_faucet", 0.01),
        ("robocasa_x_target_panda_og_turn_on_sink_faucet", 1.0),
    ],
    "robocasa-x-xp3k-jaco-turn-on-sink-faucet-mix": [
        ("robocasa_x_xp3k_turn_on_sink_faucet", 0.01),
        ("robocasa_x_target_jaco_turn_on_sink_faucet", 1.0),
    ],
    "robocasa-x-xp900-panda-turn-on-sink-faucet-mix": [
        ("robocasa_x_xp900_turn_on_sink_faucet", 0.05),
        ("robocasa_x_target_panda_turn_on_sink_faucet", 1.0),
    ],
    "robocasa-x-xp900-panda-og-turn-on-sink-faucet-mix": [
        ("robocasa_x_xp900_turn_on_sink_faucet", 0.05),
        ("robocasa_x_target_panda_og_turn_on_sink_faucet", 1.0),
    ],
    "robocasa-x-xp900-jaco-turn-on-sink-faucet-mix": [
        ("robocasa_x_xp900_turn_on_sink_faucet", 0.05),
        ("robocasa_x_target_jaco_turn_on_sink_faucet", 1.0),
    ],
    "robocasa-x-xp3k-panda-flip-mug-upright-mix": [
        ("robocasa_x_xp3k_flip_mug_upright", 0.01),
        ("robocasa_x_target_panda_flip_mug_upright", 1.0),
    ],
    "robocasa-x-xp3k-panda-og-flip-mug-upright-mix": [
        ("robocasa_x_xp3k_flip_mug_upright", 0.01),
        ("robocasa_x_target_panda_og_flip_mug_upright", 1.0),
    ],
    "robocasa-x-xp3k-jaco-flip-mug-upright-mix": [
        ("robocasa_x_xp3k_flip_mug_upright", 0.01),
        ("robocasa_x_target_jaco_flip_mug_upright", 1.0),
    ],
    "robocasa-x-xp900-panda-flip-mug-upright-mix": [
        ("robocasa_x_xp900_flip_mug_upright", 0.05),
        ("robocasa_x_target_panda_flip_mug_upright", 1.0),
    ],
    "robocasa-x-xp900-panda-og-flip-mug-upright-mix": [
        ("robocasa_x_xp900_flip_mug_upright", 0.05),
        ("robocasa_x_target_panda_og_flip_mug_upright", 1.0),
    ],
    "robocasa-x-xp900-jaco-flip-mug-upright-mix": [
        ("robocasa_x_xp900_flip_mug_upright", 0.05),
        ("robocasa_x_target_jaco_flip_mug_upright", 1.0),
    ],
    "robocasa-x-sp900-panda-pnp": [
        ("robocasa_x_sp900_panda_pnp", 1.0),
    ],
    "robocasa-x-sp900-panda-og-pnp": [
        ("robocasa_x_sp900_panda_og_pnp", 1.0),
    ],
    "robocasa-x-sp900-jaco-pnp": [
        ("robocasa_x_sp900_jaco_pnp", 1.0),
    ],
    "robocasa-x-sp900-panda-turn-on-sink-faucet": [
        ("robocasa_x_sp900_panda_turn_on_sink_faucet", 1.0),
    ],
    "robocasa-x-sp900-panda-og-turn-on-sink-faucet": [
        ("robocasa_x_sp900_panda_og_turn_on_sink_faucet", 1.0),
    ],
    "robocasa-x-sp900-jaco-turn-on-sink-faucet": [
        ("robocasa_x_sp900_jaco_turn_on_sink_faucet", 1.0),
    ],
    "robocasa-x-sp900-panda-flip-mug-upright": [
        ("robocasa_x_sp900_panda_flip_mug_upright", 1.0),
    ],
    "robocasa-x-sp900-panda-og-flip-mug-upright": [
        ("robocasa_x_sp900_panda_og_flip_mug_upright", 1.0),
    ],
    "robocasa-x-sp900-jaco-flip-mug-upright": [
        ("robocasa_x_sp900_jaco_flip_mug_upright", 1.0),
    ],
    "robocasa-x-sp900-panda-pnp-mix": [
        ("robocasa_x_target_panda_pnp", 1.0),
        ("robocasa_x_sp900_panda_pnp", 0.05),
    ],
    "robocasa-x-sp900-panda-og-pnp-mix": [
        ("robocasa_x_target_panda_og_pnp", 1.0),
        ("robocasa_x_sp900_panda_og_pnp", 0.05),
    ],
    "robocasa-x-sp900-jaco-pnp-mix": [
        ("robocasa_x_target_jaco_pnp", 1.0),
        ("robocasa_x_sp900_jaco_pnp", 0.05),
    ],
    "robocasa-x-sp900-panda-turn-on-sink-faucet-mix": [
        ("robocasa_x_target_panda_turn_on_sink_faucet", 1.0),
        ("robocasa_x_sp900_panda_turn_on_sink_faucet", 0.05),
    ],
    "robocasa-x-sp900-panda-og-turn-on-sink-faucet-mix": [
        ("robocasa_x_target_panda_og_turn_on_sink_faucet", 1.0),
        ("robocasa_x_sp900_panda_og_turn_on_sink_faucet", 0.05),
    ],
    "robocasa-x-sp900-jaco-turn-on-sink-faucet-mix": [
        ("robocasa_x_target_jaco_turn_on_sink_faucet", 1.0),
        ("robocasa_x_sp900_jaco_turn_on_sink_faucet", 0.05),
    ],
    "robocasa-x-sp900-panda-flip-mug-upright-mix": [
        ("robocasa_x_target_panda_flip_mug_upright", 1.0),
        ("robocasa_x_sp900_panda_flip_mug_upright", 0.05),
    ],
    "robocasa-x-sp900-panda-og-flip-mug-upright-mix": [
        ("robocasa_x_target_panda_og_flip_mug_upright", 1.0),
        ("robocasa_x_sp900_panda_og_flip_mug_upright", 0.05),
    ],
    "robocasa-x-sp900-jaco-flip-mug-upright-mix": [
        ("robocasa_x_target_jaco_flip_mug_upright", 1.0),
        ("robocasa_x_sp900_jaco_flip_mug_upright", 0.05),
    ],

    # === Supplemental Internal RoboCasa Datasets ===
    "mg_kinova_pnp": [
        ("mg_kinova_pnp", 1.0),
    ],
    "mg_kinova_pnp_lite": [
        ("mg_kinova_pnp_lite", 1.0),
    ],
    "mg_ur5e_pnp": [
        ("mg_ur5e_pnp", 1.0),
    ],
    "mg_ur5e_pnp_lite": [
        ("mg_ur5e_pnp_lite", 1.0),
    ],
    "mg_iiwa_pnp": [
        ("mg_iiwa_pnp", 1.0),
    ],
    "mg_iiwa_pnp_lite": [
        ("mg_iiwa_pnp_lite", 1.0),
    ],
    "panda_real": [
        ("franka_pnpcountertosink_aux", 1.0),
        ("franka_pnpsinktocounter_aux", 1.0)
    ],
    "panda_real_vary": [
        ("franka_pnpcountertosink_vary_aux", 1.0),
        ("franka_pnpsinktocounter_aux", 1.0)
    ],
    "viper_real": [
        ("viper_pnpcountertosink_aux", 1.0),
        ("viper_pnpsinktocounter_aux", 1.0)
    ],
    "viper_real_vary": [
        ("viper_pnpcountertosink_vary_aux", 1.0),
        ("viper_pnpsinktocounter_aux", 1.0)
    ],
    "panda_viper_real_mg": [
        ("franka_pnpcountertosink_aux", 1.0),
        ("franka_pnpsinktocounter_aux", 1.0),
        ("viper_pnpcountertosink_aux", 1.0),
        ("viper_pnpsinktocounter_aux", 1.0),
        ("robocasa_x_xp3k_pnp", 0.01)  # 0.002 for tokenizer
    ],
    "panda_viper_real_vary_mg": [
        ("franka_pnpcountertosink_vary_aux", 1.0),
        ("franka_pnpsinktocounter_aux", 1.0),
        ("viper_pnpcountertosink_vary_aux", 1.0),
        ("viper_pnpsinktocounter_aux", 1.0),
        ("robocasa_x_xp3k_pnp", 0.01)  # 0.002 for tokenizer
    ],

    # Supplemental internal single-camera datasets
    "mg_panda_pnp_single_cam": [
        ("mg_panda_pnp_single_cam", 1.0),
    ],
    "panda+mg_panda_pnp_single_cam": [
        ("robocasa_x_target_panda_pnp", 1.0),
        ("mg_panda_pnp_single_cam", 0.05),
    ],
    "mg_jaco_pnp_single_cam": [
        ("mg_jaco_pnp_single_cam", 1.0),
    ],
    "jaco+mg_jaco_pnp_single_cam": [
        ("robocasa_x_target_jaco_pnp", 1.0),
        ("mg_jaco_pnp_single_cam", 0.05),
    ],
    # TODO (ajaysri): add partial held-out mixture
}
# fmt: on
