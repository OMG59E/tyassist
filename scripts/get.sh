set -e

WORK_DIR=$(dirname $0)/..

# get datasets
wget ftp://192.168.33.100//date/ftp-intellif/DEngine/data/datasets --ftp-user=lijiao --ftp-password=lj --directory-prefix=$WORK_DIR -nH --cut-dirs=4 -r

# get models
wget ftp://192.168.33.100//date/ftp-intellif/DEngine/origin_models_dp2000/* --ftp-user=lijiao --ftp-password=lj --directory-prefix=$WORK_DIR/models -nH --cut-dirs=5 -r

echo "download all tyassist package files successfully."