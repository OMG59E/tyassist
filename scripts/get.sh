set -e

WORK_DIR=$(dirname $0)/..

# get datasets
wget $PACK_FTP_HOME/DEngine/data/datasets --ftp-user=$PACK_FTP_USER --ftp-password=$PACK_FTP_PWD --directory-prefix=$WORK_DIR -nH --cut-dirs=4 -r

# get pic
wget $PACK_FTP_HOME/DEngine/data/pic --ftp-user=$PACK_FTP_USER --ftp-password=$PACK_FTP_PWD --directory-prefix=$WORK_DIR -nH --cut-dirs=4 -r

# get models
wget $PACK_FTP_HOME/DEngine/models/* --ftp-user=$PACK_FTP_USER --ftp-password=$PACK_FTP_PWD --directory-prefix=$WORK_DIR/models -nH --cut-dirs=5 -r

echo "download all tyassist package files successfully."