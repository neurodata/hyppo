#!/bin/bash 

screen -AdmS ex1_kgof -t tab0 bash 
# launch each problem in parallell, each in its own screen tab
# See http://unix.stackexchange.com/questions/74785/how-to-open-tabs-windows-in-gnu-screen-execute-commands-within-each-one
# http://stackoverflow.com/questions/7120426/invoke-bash-run-commands-inside-new-shell-then-give-control-back-to-user

#screen -S ex1_kgof -X screen -t tab6 bash -lic "python ex1_vary_n.py gmd_p03_d10_ns"


screen -S ex1_kgof -X screen -t tab7 bash -lic "python ex1_vary_n.py gbrbm_dx50_dh10_h0"
screen -S ex1_kgof -X screen -t tab7 bash -lic "python ex1_vary_n.py gbrbm_dx50_dh40_h0"
screen -S ex1_kgof -X screen -t tab7 bash -lic "python ex1_vary_n.py gbrbm_dx50_dh10_vp1"
screen -S ex1_kgof -X screen -t tab7 bash -lic "python ex1_vary_n.py gbrbm_dx50_dh40_vp1"

