#!/bin/bash 

screen -AdmS ex2_kgof -t tab0 bash 
# launch each problem in parallell, each in its own screen tab
# See http://unix.stackexchange.com/questions/74785/how-to-open-tabs-windows-in-gnu-screen-execute-commands-within-each-one
# http://stackoverflow.com/questions/7120426/invoke-bash-run-commands-inside-new-shell-then-give-control-back-to-user

#screen -S ex2_kgof -X screen -t tab1 bash -lic "python ex2_prob_params.py gmd"
#screen -S ex2_kgof -X screen -t tab3 bash -lic "python ex2_prob_params.py gvinc_d5"
#screen -S ex2_kgof -X screen -t tab3 bash -lic "python ex2_prob_params.py gvsub1_d1"
#screen -S ex2_kgof -X screen -t tab4 bash -lic "python ex2_prob_params.py gmd_d10_ms"
#screen -S ex2_kgof -X screen -t tab5 bash -lic "python ex2_prob_params.py gvd"


#screen -S ex2_kgof -X screen -t tab6 bash -lic "python ex2_prob_params.py gbrbm_dx50_dh10"
screen -S ex2_kgof -X screen -t tab6 bash -lic "python ex2_prob_params.py gbrbm_dx50_dh40"
#screen -S ex2_kgof -X screen -t tab7 bash -lic "python ex2_prob_params.py glaplace"


