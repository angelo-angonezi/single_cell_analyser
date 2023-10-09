# move hard files module

# Code destined to moving images classified as "hard" between folders.

######################################################################
# imports

BASE_PATH = ('/home'
             '/angelo'
             '/dados'
             '/pycharm_projects'
             '/single_cell_analyser'
             '/data'
             '/')
all_path = os.path.join(path, 'all')
selected_path = os.path.join(path, 'selected')
hard_path = os.path.join(path, 'hard')

all_files = os.listdir(all_path)
print(all_files)
total = len(all_files)
selected_files = os.listdir(selected_path)
print(selected_files)

for i_index, i in enumerate(all_files, 1):
    print(f"Analisando arquivo {i_index} de {total}")
    if i in selected_files:
        print(f"Arquivo {i} é selecionado")
        continue
    else:
        src = os.path.join(all_path, i)
        dest = os.path.join(hard_path, i)
        print(f"Arquivo {i} é dificil")
        # shutil.copy(src, dest)