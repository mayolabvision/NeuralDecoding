import sys

mtfef_pairs = helpers.get_neuronCombos(int(sys.argv[1]))
mtfef_pairs.remove((0,0))


PARAMETERS_COUNT=$(len(mtfef_pairs))
print(PARAMETERS_COUNT)
print(blah)
# Extract maximum array size from Slurm configuration.
MAX_ARRAY_SIZE=$(scontrol show config | grep MaxArraySize | awk '{split($0, a, "="); print a[2]}' | sed 's/^ *//g')
ARRAY_TASK_COUNT=$((MAX_ARRAY_SIZE < PARAMETERS_COUNT ? MAX_ARRAY_SIZE : PARAMETERS_COUNT))
ARRAY=0-$((ARRAY_TASK_COUNT - 1))

sbatch --input="$PARAMETERS" --array="$ARRAY" run.sh


ARRAY_TASK_COUNT = len(mtfef_pairs)
ARRAY=0-$((ARRAY_TASK_COUNT - 1))

print(ARRAY)
#sbatch --input="$PARAMETERS" --array="$ARRAY" run.sh

