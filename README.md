# Single Cell Analyser

Codes destined to automation of single cell nuclei
analysis in brightfield microscopy images.

Works in parallel with BRightfield Artificial Intelligence Nucleus Detector (BRAIND),
a fine-tuned version of the R2CNN model: https://drive.google.com/file/d/1ElM2jHpjaitJFfyVDfos6o2DAwbdECvU/view?usp=sharing

BRAIND usage:
*sudo ./run_analysis_pipeline.sh $input_folder $images_extension $output_folder*

single_cell_analyser codes usage example:
*python -m src.utils.analysers.add_nucleus_overlays -h*

(all single_cell_analyser codes are documented and
have a '--help' option that specifies input parameters)
