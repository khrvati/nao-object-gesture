qibuild configure --release -c atom114
qibuild make --release -c atom114
if [ "$#" -eq 1 ] ; then
scp build-atom114-release/sdk/lib/naoqi/* nao@$1:/home/nao/naoqi/modules/
fi
