PPATH=$(dirname $(readlink -f "$0"))
echo ${PPATH}
DPATH=${PPATH/preprocess/data}
echo ${DPATH}