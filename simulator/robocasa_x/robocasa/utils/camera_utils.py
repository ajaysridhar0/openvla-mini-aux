"""
Collection of constants for cameras / robots / etc
in kitchen environments
"""

# default free cameras for different kitchen layouts
LAYOUT_CAMS = {
    0: dict(
        lookat=[2.26593463, -1.00037131, 1.38769295],
        distance=3.0505089839567323,
        azimuth=90.71563812375285,
        elevation=-12.63948837207208,
    ),
    1: dict(
        lookat=[2.66147999, -1.00162429, 1.2425155],
        distance=3.7958766287746255,
        azimuth=89.75784013699234,
        elevation=-15.177406642875091,
    ),
    2: dict(
        lookat=[3.02344359, -1.48874618, 1.2412914],
        distance=3.6684844368165512,
        azimuth=51.67880851867874,
        elevation=-13.302619131542388,
    ),
    # 3: dict(
    #     lookat=[11.44842548, -11.47664723, 11.24115989],
    #     distance=43.923271794728187,
    #     azimuth=227.12928449329333,
    #     elevation=-16.495686334624907,
    # ),
    4: dict(
        lookat=[1.6, -1.0, 1.0],
        distance=5,
        azimuth=89.70301806083651,
        elevation=-18.02177994296577,
    ),
}

DEFAULT_LAYOUT_CAM = {
    "lookat": [2.25, -1, 1.05312667],
    "distance": 5,
    "azimuth": 89.70301806083651,
    "elevation": -18.02177994296577,
}


CAM_CONFIGS = dict(
    robot0_agentview_center=dict(
        pos=[-0.6, 0.0, 0.95],
        quat=[
            0.636945903301239,
            0.3325185477733612,
            -0.3199238181114197,
            -0.6175596117973328,
        ],
        parent_body="mobilebase0_support",
    ),
    robot0_agentview_wide=dict(
        pos=[-0.6, 0, 0.95],
        quat=[
            0.636945903301239,
            0.3325185477733612,
            -0.3199238181114197,
            -0.6175596117973328,
        ],
        camera_attribs=dict(fovy="60"),
        parent_body="mobilebase0_support",
    ),
    robot0_agentview_wide_75=dict(
        pos=[-0.6, 0, 0.95],
        quat=[
            0.636945903301239,
            0.3325185477733612,
            -0.3199238181114197,
            -0.6175596117973328,
        ],
        camera_attribs=dict(fovy="75"),
        parent_body="mobilebase0_support",
    ),
    robot0_agentview_wide_90=dict(
        pos=[-0.6, 0, 0.95],
        quat=[
            0.636945903301239,
            0.3325185477733612,
            -0.3199238181114197,
            -0.6175596117973328,
        ],
        camera_attribs=dict(fovy="90"),
        parent_body="mobilebase0_support",
    ),
    old_robot0_agentview_left=dict(
        # pos=[-0.5, 0.35, 1.05],
        # pos=[-0.4, 0.90, 1.00], good
        pos=[-0.4, 0.85, 0.8],
        #quat=[0.55623853, 0.29935253, -0.37678665, -0.6775092],
        quat=[ 0.36193285,  0.19163277, -0.44142606, -0.79838871], # rotate z -30
        # quat=[ 0.32676315, 0.17219564, -0.44936483, -0.81341611], #rotate z -35
        # quat=[ 0.29097143,  0.15243073, -0.4564482,  -0.82689512], #rotate z -40
        camera_attribs=dict(fovy="30"),
        parent_body="mobilebase0_support",
    ),
    robot0_agentview_left=dict(
        # pos=[-0.35, 0.40, 0.60], from ajay
        # pos=[-0.23,  0.4,  0.64], original panda position
        pos=[-0.17,  0.4,  0.64],
        quat=[ 0.48874131,  0.26186095, -0.40374364, -0.72770313],
        # quat=[0.55623853, 0.29935253, -0.37678665, -0.6775092], original
        #quat=[ 0.4630471,  0.24761098, -0.41263651, -0.74431666],# rotate z -15
        camera_attribs=dict(fovy="45"),
        parent_body="mobilebase0_support",
    ),
    panda_agentview_left=dict(
        pos=[-0.17,  0.4,  0.64],
        quat=[ 0.48874131,  0.26186095, -0.40374364, -0.72770313],
        camera_attribs=dict(fovy="45"),
        parent_body="mobilebase0_support",
    ),
    jaco_agentview_left=dict(
        pos=[-0.23, 0.4, 0.64],
        quat=[ 0.48874131,  0.26186095, -0.40374364, -0.72770313],
        camera_attribs=dict(fovy="45"),
        parent_body="mobilebase0_support",
    ),
    kinova_agentview_left=dict(
        pos=[-0.23, 0.4, 0.74],
        quat=[ 0.48874131,  0.26186095, -0.40374364, -0.72770313],
        camera_attribs=dict(fovy="45"),
        parent_body="mobilebase0_support",
    ),
    ur5e_agentview_left=dict(
        pos=[-0.13, 0.4, 0.64],
        quat=[ 0.48874131,  0.26186095, -0.40374364, -0.72770313],
        camera_attribs=dict(fovy="45"),
        parent_body="mobilebase0_support",
    ),
    iiwa_agentview_left=dict(
        pos=[-0.13, 0.4, 0.84],
        quat=[ 0.48874131,  0.26186095, -0.40374364, -0.72770313],
        camera_attribs=dict(fovy="45"),
        parent_body="mobilebase0_support",
    ),
    debug_left=dict(
        pos=[0.1, 1.2, 0.2],
        # quat=[-0.00939108, -0.01287645, -0.48105538, -0.87654533],
        quat=[-0.01240376, -0.01000711, -0.69153044, -0.72217148],
        camera_attribs=dict(fovy="60"),
        parent_body="mobilebase0_support",
    ),
    robot0_agentview_right=dict(
        # pos=[-0.5, -0.35, 1.05],
        pos=[-0.5, -0.75, 1.05],
        # quat=[
        #     0.6775091886520386,
        #     0.3767866790294647,
        #     -0.2993525564670563,
        #     -0.55623859167099,
        # ],
        quat=[ 0.79838867,  0.44142608, -0.19163278, -0.3619329 ],
        camera_attribs=dict(fovy="60"),
        parent_body="mobilebase0_support",
    ),
    fixed_sink=dict(
        # pos=[ 1.14999147, -2.19999871,  1.5       ], #original from ajay
        # pos=[ 1.0, -2.2,  1.5],
        pos=[ 1.03, -2.2,  1.54],
        # pos=[ 1.14999147, -2.19999871,  1.55       ],
        # quat=[-0.67750813, -0.37678607, -0.2993532 , -0.55623974], # original
        # quat=[-0.72340949634489, -0.401442636648453, -0.2653750015103631, -0.4950743566901294], # rotate z -10
        quat=[-0.7277022350759109, -0.4037431553139996, -0.2618616934241865, -0.48874264712590476], # rotate z -11
        # quat=[-0.7319395564643171, -0.40601292737511396, -0.2583284435562168, -0.4823737179168574], # rotate z -12
        # quat=[-0.7259449040520018, -0.41684159170790175, -0.269033738976401, -0.47630655149360535], # rotate y 2, z -12
        # quat=[-0.7443158109379011, -0.412636047198726, -0.2476117414382804, -0.4630484765572257], # rotate z -15
        # quat=[-0.6488396580005492, -0.4238317738947667, -0.357262795794145, -0.5212840112494309], # rotate y 10
        # quat=[-0.6918035223947313, -0.45335647036032967, -0.3189639299417053, -0.4627502658895065], # rotate y 10, z -10
        # quat=[-0.6997742630867096, -0.45885411011818833, -0.31100318880853794, -0.45060615053505315], # rotate y 10, z -12
        # quat=[-0.708280635592756, -0.4278067307178245, -0.29244781124120656, -0.479368563943315], # rotate y 5, z -10
        # quat=[-0.7285162724512191, -0.4401559478670662, -0.27350879822135543, -0.4480175438825311], # rotate y 5, z -15
        # quat=[-0.7165388960325166, -0.43284549174408327, -0.28493701308195873, -0.4669343522651304], # rotate y 5, z -12
        camera_attribs=dict(fovy="45"),
    ),
    robot0_frontview=dict(
        pos=[-0.50, 0, 0.95],
        quat=[
            0.6088936924934387,
            0.3814677894115448,
            -0.3673907518386841,
            -0.5905545353889465,
        ],
        camera_attribs=dict(fovy="60"),
        parent_body="mobilebase0_support",
    ),
    robot0_eye_in_hand=dict(
        pos=[0.05, 0, 0],
        quat=[0, 0.707107, 0.707107, 0],
        parent_body="robot0_right_hand",
    ),
)
