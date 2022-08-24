WORK_DIR=$(cd $(dirname $0); pwd)/..

apt install wget

wget ftp://192.168.33.100//date/ftp-intellif/DEngine/data/datasets --ftp-user=lijiao --ftp-password=lj --directory-prefix=$WORK_DIR -nH --cut-dirs=4 -r

echo "download data from ftp successfully."
