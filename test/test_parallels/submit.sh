
#!/bin/bash

#----------------------------------------#
#- IGM SUBMISSION SCRIPT, multi-threaded JOB
#----------------------------------------#

#$ -M fmusella@g.ucla.edu
#$ -m ea
#$ -N volume
#$ -l h_data=40G,h_vmem=INFINITY
#$ -l h_rt=96:00:00
#$ -l highp
#$ -cwd
#$ -o out_submit
#$ -e err_submit
#$ -V 
#$ -pe shared 2

export PATH="$PATH"
ulimit -s 8192

# -----------------------
# print JOB ID, can be useful for keeping track of status
echo $JOB_ID

# print PATH pointing to directory the job is run from
echo $SGE_O_WORKDIR


# shared memory parallelization: same node, more cores, export number of threads
export OMP_NUM_THREADS=2
echo "submitting code in parallel...."

# execute job
python3 test_parallel_ctenv.py
