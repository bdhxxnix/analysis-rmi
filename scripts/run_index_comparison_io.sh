# set -x
#!/usr/bin/env bash

trap "exit" SIGINT

EXPERIMENT="index comparison I/O"

DIR_DATA="/home/andy/Projects/Datasets/SOSD"
DIR_RESULTS="results"
FILE_RESULTS="${DIR_RESULTS}/index_comparison_io.csv"

BIN="build/bin/index_comparison"

# Set number of repetitions and samples
N_REPS="3"
N_SAMPLES="20000000"
PARAMS="--io --n_reps ${N_REPS} --n_samples ${N_SAMPLES}"

# Set which indexes to run on datasets
declare -A flags
# flags['books_200M_uint64']="--rmi --alex --pgm --fitting-tree --rs --cht --ref"
flags['fb_200M_uint64']="--pgm --ref --bin"
# flags['osm_cellids_200M_uint64']="--rmi --alex --pgm --fitting-tree --rs --cht --ref"
# flags['wiki_ts_200M_uint64']="--rmi --alex --pgm --fitting-tree --rs --ref" # ART and CHT do not support duplicates

run() {
    DATASET=$1
    DATA_FILE="${DIR_DATA}/${DATASET}"
    ${BIN} ${PARAMS} ${flags[${DATASET}]} ${DATA_FILE} >> ${FILE_RESULTS}
}

# Create results directory
if [ ! -d "${DIR_RESULTS}" ];
then
    mkdir -p "${DIR_RESULTS}";
fi

# Check data downloaded
if [ ! -d "${DIR_DATA}" ];
then
    >&2 echo "Please download datasets first."
    return 1
fi

# Run experiments
echo "dataset,n_keys,index,config,size_in_bytes,rep,n_queries,mean_ns,p50_ns,p90_ns,p95_ns,p99_ns,bytes_read" > ${FILE_RESULTS} # Write csv header
for dataset in ${!flags[@]};
do
    echo "Performing ${EXPERIMENT} on '${dataset}'..."
    run $dataset
done
