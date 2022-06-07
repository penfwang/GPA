#!/bin/sh
#!/bin/bash -e

#SBATCH --account=vuw03334
#SBATCH --time=240:00
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=2000M
#SBATCH --partition=large
#SBATCH --output=log.%j.out # Include the job ID in the names of
#SBATCH --error=log.%j.err # the output and error files
#SBATCH --array=1-30                 # Array jobs

file_path=/nesi/project/vuw03334/GP/algorithms1

# module purge
module load Python/3.8.1-gimkl-2018b
python $file_path/SGP.py $1 ${SLURM_ARRAY_TASK_ID}

mv *.txt  /nesi/project/vuw03334/GP/results/SGP/$1
mv *.npy  /nesi/project/vuw03334/GP/results/SGP/$1
mv *.pickle  /nesi/project/vuw03334/GP/results/SGP/$1
