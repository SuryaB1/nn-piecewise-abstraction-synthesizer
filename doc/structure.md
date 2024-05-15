.
├── README.md
├── data
│   ├── inputs
│   │   ├── init_datapoints
│   │   │   ├── diagonal-split-sample-data.json
│   │   │   ├── diagonal-split-sample-data.txt
│   │   │   ├── diagonal-split_sample-data_structured.txt
│   │   │   └── empty.txt
│   │   └── models
│   │       ├── model_saving_scripts
│   │       │   ├── output_3d-unit-sqr_classif_nnet_no-sigmoid.py
│   │       │   ├── output_4d-unit-sqr_classif_nnet_no-sigmoid.py
│   │       │   ├── output_basic_nnet.py
│   │       │   ├── output_concave-poly_classif_nn.py
│   │       │   ├── output_convex-poly_classif_nn.py
│   │       │   ├── output_diagonal-split_classif_nn.py
│   │       │   ├── output_disjoint-polys_classif_nn.py
│   │       │   ├── output_quadrant1_classif_nnet.py
│   │       │   ├── output_reluplex_fig2_nnet.py
│   │       │   ├── output_sign_classif_nnet.py
│   │       │   ├── output_sign_classif_nnet_no-softmax.py
│   │       │   ├── output_three_in_two_out_nnet.py
│   │       │   ├── output_unit-sqr_classif_nnet.py
│   │       │   └── output_unit-sqr_classif_nnet_no-sigmoid.py
│   │       └── saved_models
│   │           ├── 3d-unit-sqr_classif_nnet_no-sigmoid
│   │           │   ├── assets
│   │           │   ├── saved_model.pb
│   │           │   └── variables
│   │           │       ├── variables.data-00000-of-00001
│   │           │       └── variables.index
│   │           ├── 4d-unit-sqr_classif_nnet_no-sigmoid
│   │           │   ├── assets
│   │           │   ├── saved_model.pb
│   │           │   └── variables
│   │           │       ├── variables.data-00000-of-00001
│   │           │       └── variables.index
│   │           ├── basic_nnet
│   │           │   ├── assets
│   │           │   ├── saved_model.pb
│   │           │   └── variables
│   │           │       ├── variables.data-00000-of-00001
│   │           │       └── variables.index
│   │           ├── concave-poly_classif_nnet
│   │           │   ├── assets
│   │           │   ├── saved_model.pb
│   │           │   └── variables
│   │           │       ├── variables.data-00000-of-00001
│   │           │       └── variables.index
│   │           ├── convex-poly_classif_nnet
│   │           │   ├── assets
│   │           │   ├── saved_model.pb
│   │           │   └── variables
│   │           │       ├── variables.data-00000-of-00001
│   │           │       └── variables.index
│   │           ├── diagonal-split_classif_nnet
│   │           │   ├── assets
│   │           │   ├── saved_model.pb
│   │           │   └── variables
│   │           │       ├── variables.data-00000-of-00001
│   │           │       └── variables.index
│   │           ├── disjoint-polys_classif_nnet
│   │           │   ├── assets
│   │           │   ├── saved_model.pb
│   │           │   └── variables
│   │           │       ├── variables.data-00000-of-00001
│   │           │       └── variables.index
│   │           ├── reluplex_fig2_nnet
│   │           │   ├── assets
│   │           │   ├── saved_model.pb
│   │           │   └── variables
│   │           │       ├── variables.data-00000-of-00001
│   │           │       └── variables.index
│   │           ├── sign_classif_nn_no-softmax
│   │           │   ├── assets
│   │           │   ├── saved_model.pb
│   │           │   └── variables
│   │           │       ├── variables.data-00000-of-00001
│   │           │       └── variables.index
│   │           ├── sign_classif_nnet
│   │           │   ├── assets
│   │           │   ├── saved_model.pb
│   │           │   └── variables
│   │           │       ├── variables.data-00000-of-00001
│   │           │       └── variables.index
│   │           ├── three-in_two-in_nnet
│   │           │   ├── assets
│   │           │   ├── fingerprint.pb
│   │           │   ├── saved_model.pb
│   │           │   └── variables
│   │           │       ├── variables.data-00000-of-00001
│   │           │       └── variables.index
│   │           ├── unit-sqr_classif_nn
│   │           │   ├── assets
│   │           │   ├── saved_model.pb
│   │           │   └── variables
│   │           │       ├── variables.data-00000-of-00001
│   │           │       └── variables.index
│   │           ├── unit-sqr_classif_nnet
│   │           │   ├── assets
│   │           │   ├── saved_model.pb
│   │           │   └── variables
│   │           │       ├── variables.data-00000-of-00001
│   │           │       └── variables.index
│   │           └── unit-sqr_classif_nnet_no-sigmoid
│   │               ├── assets
│   │               ├── saved_model.pb
│   │               └── variables
│   │                   ├── variables.data-00000-of-00001
│   │                   └── variables.index
│   └── outputs
│       └── voronoi_cegis_output_gifs
│           ├── tess_form_gif_10
│           │   ├── cegis_iteration_0.png
│           │   ├── cegis_iteration_1.png
│           │   ├── ...
│           │   └── tessellation_formation.gif
│           ├── tess_form_gif_100
│           │   ├── cegis_iteration_0.png
│           │   ├── cegis_iteration_1.png
│           │   ├── ...
│           │   └── tessellation_formation.gif
│           ├── tess_form_gif_100_bfs
│           │   ├── cegis_iteration_0.png
│           │   ├── cegis_iteration_1.png
│           │   ├── ...
│           │   └── tessellation_formation.gif
│           ├── tess_form_gif_100_structured_input
│           │   ├── cegis_iteration_0.png
│           │   ├── cegis_iteration_1.png
│           │   ├── ...
│           │   └── tessellation_formation.gif
│           ├── tess_form_gif_100_workaround
│           │   ├── cegis_iteration_0.png
│           │   ├── cegis_iteration_1.png
│           │   ├── ...
│           │   └── tessellation_formation.gif
│           └── tess_form_gif_structured_bfs
│               ├── cegis_iteration_0.png
│               ├── cegis_iteration_1.png
│               ├── ...
│               └── tessellation_formation.gif
├── docs
│   ├── TODO.md
│   ├── marabou_issues.md
│   ├── ridge_hyperplane_calculations.ipynb
│   └── scipy_voronoi.ipynb
├── legacy
│   ├── function_abstraction_cegis
│   │   ├── main.py
│   │   ├── multiple_output_function_cegis.out
│   │   └── single_output_function_cegis.out
│   └── rectangular_piecewise_abstraction_cegis
│       ├── multi_dim_input
│       │   ├── main.py
│       │   ├── multi_in_rect_piecewise.out
│       │   └── tests
│       │       ├── test_unit-square-nn_2d-input.py
│       │       ├── test_unit-square-nn_3d-input.py.py
│       │       └── test_unit-square-nn_4d-input.py.py
│       └── single_dim_input
│           ├── main.py
│           └── single_in_rect_piecewise.out.out
├── requirements.txt
├── setup.py
├── src
│   └── nn_piecewise_abstraction_synthesizer
│       ├── __init__.py
│       ├── main.py
│       └── utils
│           ├── voronoi_cell.py
│           └── voronoi_plot_2d.py
└── tests
    ├── standalone_tests
    │   ├── test_individual_query.py
    │   └── test_loaded_tf_nn.py
    └── synthesizer_tests
