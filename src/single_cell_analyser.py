# single cell analyser module

print('initializing...')  # noqa

# code destined to automating single cell
# feature extraction and analyses.

######################################################################
# importing required libraries

print('importing required libraries...')  # noqa
from time import sleep
from src.utils.aux_funcs import get_data_from_consolidated_df
print('all required libraries successfully imported.')  # noqa
sleep(0.8)

######################################################################
# defining auxiliary functions

######################################################################
# defining main function

def main():
    """
    Runs main code.
    """
    # getting data from consolidated df csv
    print('getting data from consolidated df...')
    main_df = get_data_from_consolidated_df(consolidated_df_file_path=CONSOLIDATED_DATAFRAME_PATH)

    pass

######################################################################
# running main function


if __name__ == '__main__':
    main()

######################################################################
# end of current module
