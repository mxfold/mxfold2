import numpy as np

score_base_pair = np.array(
    [
        [0.0, 0.0, 0.0, 0.597912],
        [0.0, 0.0, 1.5442907, 0.0],
        [0.0, 1.5442907, 0.0, -0.01304755],
        [0.597912, 0.0, -0.01304755, 0.0],
    ],
    dtype=np.float32,
)

score_terminal_mismatch = np.array(
    [
        [
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            [
                [-0.18454607, -0.11818442, -0.44614697, -0.61752546],
                [0.00478846, 0.08319395, -0.224948, -0.3981327],
                [0.51911104, -0.35241193, -0.40564296, -0.7733932],
                [-0.01574404, 0.26857004, -0.09343887, 0.33737114],
            ],
        ],
        [
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.08386423, -0.25207168, -0.6711842, -0.381635],
                [0.11178522, -0.17043936, -0.21799877, -0.45926765],
                [0.852064, -0.9332489, -0.32895517, -0.7778822],
                [-0.24223399, -0.03780509, -0.43223342, -0.24199761],
            ],
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
        ],
        [
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            [
                [-0.1703136, -0.09154056, -0.2522413, -0.85203147],
                [0.04763224, -0.24286543, -0.20792751, -0.187427],
                [0.6540034, -0.7823989, 0.19958982, -0.44321695],
                [-0.17369218, 0.28849435, -0.01638238, 0.6757989],
            ],
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            [
                [-0.48716077, 0.1105032, 0.3633739, -0.6193199],
                [0.34510562, 0.0314945, -0.3799173, -0.03222973],
                [0.49486387, -0.28219527, -0.27022272, -0.06658395],
                [-0.43061545, -0.09497864, -0.31307945, -0.2283243],
            ],
        ],
        [
            [
                [0.01153639, -0.3923408, 0.05661064, -0.12514853],
                [-0.06545075, -0.31672007, 0.00225838, -0.42221773],
                [0.5458417, -0.2085888, -0.1971766, -0.472241],
                [-0.17796426, 0.16434543, -0.5005617, 0.13338676],
            ],
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.12187413, 0.19902602, 0.04681894, 0.32562646],
                [0.11868123, -0.18510652, -0.04311513, -0.6150608],
                [0.75493324, -0.31507084, 0.1569583, -0.51497],
                [-0.2926246, 0.13730681, -0.05422333, 0.03086777],
            ],
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
        ],
    ],
    dtype=np.float32,
)

score_hairpin_length = np.array(
    [
        -5.9931803,
        -3.1081057,
        0.41689762,
        2.205419,
        1.9267497,
        -0.58732456,
        -0.08275717,
        0.578389,
        -0.72208834,
        -0.17258747,
        -0.30250898,
        -0.02963159,
        -0.9268996,
        -0.03157754,
        -0.10224721,
        0.19014074,
        -0.0928091,
        0.16904484,
        -0.08172566,
        -0.3445939,
        -0.10915029,
        -0.29035237,
        -0.33937135,
        -0.19153641,
        -0.0501921,
        -0.03874621,
        0.04751471,
        0.06744322,
        0.09721876,
        0.16731317,
        0.23299372,
    ],
    dtype=np.float32,
)

score_internal_explicit = np.array(
    [
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, -0.1754591, 0.03083787, -0.17156543, -0.22946809],
        [0.0, 0.03083787, -0.13040727, -0.0773033, 0.2782767],
        [0.0, -0.17156543, -0.0773033, -0.0289895, 0.31123507],
        [0.0, -0.22946809, 0.2782767, 0.31123507, -0.32263482],
    ],
    dtype=np.float32,
)

score_bulge_length = np.array(
    [
        0.0,
        -2.3995485,
        -0.8945183,
        -0.9088551,
        -0.8412475,
        -0.43654794,
        -0.56991875,
        0.20028342,
        0.75387615,
        -0.6045045,
        -0.7200948,
        -0.5136722,
        -0.36147267,
        -0.26144543,
        -0.15939268,
        -0.08624668,
        -0.03107091,
        -0.01097222,
        0.0300122,
        0.04759124,
        -0.04296172,
        -0.017919,
        -0.07800552,
        -0.07099327,
        -0.05767953,
        -0.04633795,
        -0.03559421,
        -0.02674934,
        -0.01818958,
        -0.01052301,
        -0.00515363,
    ],
    dtype=np.float32,
)

score_internal_length = np.array(
    [
        0.0,
        0.0,
        -0.42906144,
        -0.35321116,
        -0.39637974,
        -0.3111199,
        -0.25519454,
        -0.05149117,
        -0.04319002,
        0.00198549,
        -0.17615132,
        -0.26396862,
        -0.34606135,
        -0.2926603,
        -0.0362425,
        -0.11999538,
        -0.04354772,
        -0.08209293,
        -0.00711323,
        0.02354825,
        0.03066974,
        -0.06618241,
        -0.13160923,
        -0.14079955,
        -0.06600292,
        -0.07779205,
        -0.05084201,
        -0.04139876,
        0.00327658,
        0.00592458,
        0.00687574,
    ],
    dtype=np.float32,
)

score_internal_symmetry = np.array(
    [
        0.0,
        -0.5467083,
        -0.38547015,
        -0.25884664,
        -0.23408367,
        0.14505778,
        -0.6562933,
        -0.30210882,
        -0.03032275,
        -0.3517944,
        -0.21591325,
        -0.12282705,
        -0.15522087,
        -0.08541121,
        -0.0459211,
        -0.02232234,
    ],
    dtype=np.float32,
)

score_internal_asymmetry = np.array(
    [
        0.0,
        -2.1056466,
        -0.55201405,
        -0.5770708,
        -0.6136668,
        -0.30571568,
        -0.1155052,
        -0.21056122,
        -0.3145743,
        -0.31489617,
        -0.09018189,
        -0.22000268,
        -0.14064832,
        -0.21624112,
        -0.17255314,
        -0.15589118,
        -0.10408586,
        -0.06967684,
        -0.04105977,
        -0.01570624,
        0.01382001,
        0.04131988,
        0.03594186,
        0.02822186,
        0.01636586,
        0.02550056,
        0.03348033,
        0.03971924,
        -0.00254511,
    ],
    dtype=np.float32,
)

score_bulge_0x1 = np.array(
    [-0.12168617, -0.07111241, 0.00894703, -0.00268576], dtype=np.float32
)

score_internal_1x1 = np.array(
    [
        [0.29444048, 0.08641361, -0.36641973, -0.2053107],
        [0.08641361, -0.15825436, 0.41752738, 0.13687626],
        [-0.36641973, 0.41752738, -0.11935148, -0.41881013],
        [-0.2053107, 0.13687626, -0.41881013, 0.14714065],
    ],
    dtype=np.float32,
)

score_helix_stacking = np.array(
    [
        [
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.14820053],
                [0.0, 0.0, 0.43434972, 0.0],
                [0.0, 0.70796424, 0.0, -0.10107776],
                [0.24325666, 0.0, 0.16236542, 0.0],
            ],
        ],
        [
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.48787078],
                [0.0, 0.0, 0.848132, 0.0],
                [0.0, 0.47842485, 0.0, -0.18112682],
                [0.70796424, 0.0, 0.4849351, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
        ],
        [
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.5551786],
                [0.0, 0.0, 0.50083244, 0.0],
                [0.0, 0.848132, 0.0, 0.21659625],
                [0.43434972, 0.0, 0.48646036, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, -0.04665365],
                [0.0, 0.0, 0.48646036, 0.0],
                [0.0, 0.4849351, 0.0, 0.18334474],
                [0.16236542, 0.0, -0.28589708, 0.0],
            ],
        ],
        [
            [
                [0.0, 0.0, 0.0, 0.3897594],
                [0.0, 0.0, 0.5551786, 0.0],
                [0.0, 0.48787078, 0.0, -0.11573338],
                [0.14820053, 0.0, -0.04665365, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, -0.11573338],
                [0.0, 0.0, 0.21659625, 0.0],
                [0.0, -0.18112682, 0.0, 0.12029654],
                [-0.10107776, 0.0, 0.18334474, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
        ],
    ],
    dtype=np.float32,
)

score_helix_closing = np.array(
    [[ 0.        ,  0.        ,  0.        , -0.97708935],
       [ 0.        ,  0.        , -0.45746508,  0.        ],
       [ 0.        , -0.82659954,  0.        , -1.0516789 ],
       [-0.9246141 ,  0.        , -0.3698708 ,  0.        ]],
      dtype=np.float32)

score_multi_base = np.array([-1.1990551], dtype=np.float32)

score_multi_unpaired = np.array([-0.19833004], dtype=np.float32)

score_multi_paired = np.array([-0.9253884], dtype=np.float32)

score_dangle_left = np.array(
    [
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [-0.12510377, 0.04416067, -0.02541879, 0.00785099],
        ],
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.07224382, 0.05279282, 0.10095543, -0.1515059],
            [0.0, 0.0, 0.0, 0.0],
        ],
        [
            [0.0, 0.0, 0.0, 0.0],
            [-0.1829535, 0.03393, 0.13353391, -0.16042745],
            [0.0, 0.0, 0.0, 0.0],
            [-0.06517512, -0.04250883, 0.02875972, -0.04359727],
        ],
        [
            [-0.03373848, -0.00507032, -0.11868612, -0.01162358],
            [0.0, 0.0, 0.0, 0.0],
            [-0.08047139, 0.001608, 0.10162722, -0.09200843],
            [0.0, 0.0, 0.0, 0.0],
        ],
    ],
    dtype=np.float32,
)

score_dangle_right = np.array(
    [
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.03232578, -0.09096819, -0.0740751, -0.01621157],
        ],
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.21339644, -0.06234811, -0.07008531, -0.21419123],
            [0.0, 0.0, 0.0, 0.0],
        ],
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.01581958, 0.00564432, -0.00943298, -0.2597793],
            [0.0, 0.0, 0.0, 0.0],
            [-0.04480272, -0.07321213, 0.01270495, -0.05717034],
        ],
        [
            [-0.16319185, 0.06769305, -0.08789074, -0.0552557],
            [0.0, 0.0, 0.0, 0.0],
            [0.04105458, -0.00813664, -0.03808592, -0.08629373],
            [0.0, 0.0, 0.0, 0.0],
        ],
    ],
    dtype=np.float32,
)

score_external_unpaired = np.array([-0.00972883], dtype=np.float32)

score_external_paired = np.array([-0.00096741], dtype=np.float32)