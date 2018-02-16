#### Basic Work Flow of Data Processing Modules ####

 - Write you one treemaker in ./scripts for example Fun(TreeMaker) in fun.py
 - Test your treemaker using treemaker_testing like ```python treemaker_testing.py ./scripts fun Fun```
 - Integrate your treemaker into data processing chain by add it into data_processing_process.py
 - Test data_processing_process.py out like ```python data_processing_process.py [indir] [outdir] [run_name]```
 - Mass processing data by write a [name].ini file into ./config, and running data_processing_manager and give it your config name