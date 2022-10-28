WORK_DIR=$(cd $(dirname $0); pwd)/..

apt install wget

wget ftp://192.168.33.100//date/ftp-intellif/DEngine/origin_models_dp2000/* --ftp-user=lijiao --ftp-password=lj --directory-prefix=$WORK_DIR/models -nH --cut-dirs=5 -r

echo "download models from ftp successfully."
