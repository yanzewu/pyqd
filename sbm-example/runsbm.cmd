@echo off
REM run spin boson model using Ehrenfest dynamics
REM Usage: runsbm [filename_prefix] [step/10000] (timestep)
REM Please make sure you are under this foler
set fprefix=sbm-example/%~1
set nstep=%~2
shift & shift
if "%1"=="" (set timestep=1e-3) else (set timestep=%1)
cd ..
@echo on
python -m pyqd -x @%fprefix%-x.txt -p @%fprefix%-p.txt -a %fprefix%-model.txt -s %nstep%0000 -d %timestep% -r 100 -m sbm -M 1 -t ehrenfest -O population -n 0