#!/bin/bash

# These are scripts to run minigo.


################################################################

# run it on cloud

# set your bucket on cloud
export BUCKET_NAME=your-bucket_name
# set the board size, either 9x9 or 19 x19
export BOARD_SIZE=9

rm -rf ~/work_dir

# initialize a random model
export MODEL_NAME=000000-bootstrap
python3 main.py bootstrap ~/work_dir gs://$BUCKET_NAME/models/$MODEL_NAME

# do several self-play games with latest model
for iter in {1..5}
do
        python3 main.py selfplay gs://$BUCKET_NAME/models/$MODEL_NAME   --readouts 100   -v 3   --output-dir=gs://$BUCKET_NAME/data/selfplay/$MODEL_NAME/local_worker   --output-sgf=gs://$BUCKET_NAME/sgf/$MODEL_NAME/local_worker
done

#groups games played with the same model into larger files of tfexamples
python3 main.py gather   --input-directory=gs://$BUCKET_NAME/data/selfplay/000000-bootstrap   --output-directory=gs://$BUCKET_NAME/data/training_chunks

# train a new model with the selfplay results
python3 main.py train ~/work_dir gs://$BUCKET_NAME/data/training_chunks gs://$BUCKET_NAME/models/000001-training --generation-num=1


# evaluate the latest model with bootstrap model
python3 main.py evaluate gs://$BUCKET_NAME/models/000000-bootstrap gs://$BUCKET_NAME/models/000001-training --readouts 100 --games 8 --verbose 3





##############################################################

# run it locally on workstation

rm -rf ~/work_dir

export BASE_DIR=/tmp/minigo
# initialize a random model
export MODEL_NAME=000000-bootstrap
python3 main.py bootstrap ~/work_dir $BASE_DIR/models/$MODEL_NAME

# do several self-play games with latest model
for iter in {1..5}
do
        python3 main.py selfplay $BASE_DIR/models/$MODEL_NAME   --readouts 100   -v 3   --output-dir=$BASE_DIR/data/selfplay/$MODEL_NAME/local_worker   --output-sgf=$BASE_DIR/sgf/$MODEL_NAME/local_worker
done

#groups games played with the same model into larger files of tfexamples
python3 main.py gather   --input-directory=$BASE_DIR/data/selfplay/000000-bootstrap   --output-directory=$BASE_DIR/data/training_chunks

# train a new model with the selfplay results
python3 main.py train ~/work_dir $BASE_DIR/data/training_chunks $BASE_DIR/models/000001-training --generation-num=1


# evaluate the latest model with bootstrap model
python3 main.py evaluate $BASE_DIR/models/000000-bootstrap $BASE_DIR/models/000001-training --readouts 100 --games 8 --verbose 3
