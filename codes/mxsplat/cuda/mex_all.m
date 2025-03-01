clc
clear

setenv("NVCC_APPEND_FLAGS", '-allow-unsupported-compiler')

file_name = "fused_2dgs_map_tointersects";
mexcuda("" + file_name + ".cu");

file_name = "fused_2dgs_project_backward";
mexcuda("" + file_name + ".cu");

file_name = "fused_2dgs_project_forward";
mexcuda("" + file_name + ".cu");

file_name = "fused_2dgs_rasterize_backward";
mexcuda("" + file_name + ".cu");

file_name = "fused_2dgs_rasterize_forward";
mexcuda("" + file_name + ".cu");

file_name = "fused_get_tilebin_edges";
mexcuda("" + file_name + ".cu");
