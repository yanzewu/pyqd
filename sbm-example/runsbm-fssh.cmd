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
python -m pyqd --x0=@%fprefix%-x.txt --k0=@%fprefix%-p.txt --args=%fprefix%-model.txt --nstep=%nstep%0000 --dt=%timestep% --dstep=100 --model=sbm --m=1 --task=fssh --obj=population --box=-10000,10000 --batch=0